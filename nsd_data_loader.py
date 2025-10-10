# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
import typing as tp
from pathlib import Path

import nibabel
import nilearn.signal
import numpy as np
import pandas as pd
import pydantic
import torch
from torch.utils.data import DataLoader
import h5py
from PIL import Image
import gc

TR_s = 4 / 3




class NSDDataset(torch.utils.data.Dataset):
    """
    A Dataset that holds fMRI time steps.
    """
    def __init__(self, cached_data: dict[str, torch.Tensor], img_path: Path):
        self.data = cached_data
        self.stimuli_path = img_path
        self.stim_file = None
        self.img_brick = None

    def __len__(self) -> int:
        # Get length from the first dimension of the fmri tensor
        return self.data["fmri"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Slicing every tensor in the data dictionary is fast and correct.
        return {key: tensor[idx] for key, tensor in self.data.items()}

    def _open_stim_file(self):
        """Opens the HDF5 file and gets the imgBrick dataset."""
        if self.stim_file is None:
            self.stim_file = h5py.File(self.stimuli_path, 'r')
            self.img_brick = self.stim_file['imgBrick']

    def __get_image__(self, idx: int) -> Image.Image:
        """
        Special method to retrieve a PIL image from the HDF5 brick.
        """
        self._open_stim_file()  # Ensure the HDF5 file is open
        image_data = self.img_brick[idx]
        return Image.fromarray(image_data)

    def get_sequence_item(self, idx: int) -> tp.Optional[dict[str, torch.Tensor]]:
        """
        Finds the full 6-sample sequence and corresponding image for a given
        timestep index.
        """
        
        # 1. Identify stimulus ID from the history tensor. Use .item() to get Python numbers.
        stim_id = self.data['stimulus_history'][idx][-1].item()
        
        if stim_id == -1:
            print("No prior images seen for this timestep (resting state).")
            return None
        
        target_run = self.data['run'][idx].item()
        target_session = self.data['session'][idx].item()

        # 2. Search backwards to find the start of the sequence.
        sequence_start_index = -1
        for i in range(idx, -1, -1):
            current_run = self.data['run'][i].item()
            current_session = self.data['session'][i].item()
            
            # Stop if we cross a run/session boundary.
            if current_run != target_run or current_session != target_session:
                sequence_start_index = i + 1
                break
            
            # Stop when the stimulus ID changes. The next sample is the start.
            current_stim_id = self.data['stimulus_history'][i][-1].item()
            if current_stim_id != stim_id:
                sequence_start_index = i + 1
                break
            
            # Handle the case where the sequence starts at the very beginning.
            if i == 0:
                sequence_start_index = 0
                break
        
        # 3. Define the 6-sample slice and check its boundaries.
        sequence_end_index = sequence_start_index + 6
        if sequence_end_index > len(self):
            return None # Sequence is truncated at the end of the dataset.

        # Ensure the sequence does not cross a run boundary.
        last_sample_run = self.data['run'][sequence_end_index - 1].item()
        last_sample_session = self.data['session'][sequence_end_index - 1].item()
        if last_sample_run != target_run or last_sample_session != target_session:
            return None # Sequence is truncated by the end of the run.

        # 4. Extract data and format it into the desired dictionary.
        # This is now a simple, fast tensor slice!
        fmri = self.data['fmri'][sequence_start_index:sequence_end_index]
        
        image_pil = self.__get_image__(stim_id)
        image = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0

        subject_id = self.data['subject_idx'][idx] # Already a tensor
        
        return {"brain": fmri, "img": image, "subject_idx": subject_id}


class NsdSaeDataConfig(pydantic.BaseModel):
    """
    Configuration and builder for the time-step-wise NSD dataset for SAE training.
    
    Implements a two-stage caching system:
    1.  Preprocesses full-brain data into a set of per-session files
    2.  When an ROI is requested, it creates a downsampled cache for that ROI
    """
    nsddata_path: str
    subject_id: int
    offset: float = 4.6  # Hemodynamic response offset in seconds
    history_length: int = 4 # How many previous images to store as history
    roi: str = ""

    def _get_preproc_dir_path(self) -> Path:
        """Gets the directory for storing preprocessed full-brain data, run by run."""
        nsddata_path = Path(self.nsddata_path).resolve()
        cache_dir = nsddata_path / ".cache"
        cache_dir.mkdir(exist_ok=True)
        preproc_dir = cache_dir / f"subj{self.subject_id:02d}_preprocessed"
        return preproc_dir

    def _get_roi_cache_filepath(self) -> Path:
        """Determines the standardized cache file path for a given ROI."""
        nsddata_path = Path(self.nsddata_path).resolve()
        cache_dir = nsddata_path / ".cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Sanitize ROI name for the filename
        roi_name = Path(self.roi).stem
        
        cache_filename = f"subj{self.subject_id:02d}_roi-{roi_name}_offset{self.offset}.pt"
        return cache_dir / cache_filename

    def _preprocess_full_brain(self, preproc_dir: Path):
        """
        Processes fMRI data from raw NIfTI files and saves preprocessed,
        full-brain data for each session into its own file. This avoids
        creating one giant cache file and is the slowest step, run only once per subject.
        """
        print(f"No preprocessed data found. Processing full-brain data into {preproc_dir}...", flush=True)
        preproc_dir.mkdir(exist_ok=True, parents=True)
        
        nsddata_path = Path(self.nsddata_path).resolve()
        subject_to_14run_sessions = {1: (21, 38), 5: (21, 38), 2: (21, 30), 7: (21, 30)}
        offset_in_TRs = int(round(self.offset / TR_s))

        for session in range(1, 41):
            session_samples = []
            runs_range = (
                range(2, 14)
                if self.subject_id in subject_to_14run_sessions and subject_to_14run_sessions[self.subject_id][0] <= session <= subject_to_14run_sessions[self.subject_id][1]
                else range(1, 13)
            )
            for run in runs_range:
                run_id = f"session{session:02d}_run{run:02d}"
                print(f"Preprocessing: {run_id}", flush=True)
                
                path_to_df = nsddata_path / f"nsddata_timeseries/ppdata/subj{self.subject_id:02d}/func1pt8mm/design/design_{run_id}.tsv"
                im_ids = pd.read_csv(path_to_df, header=None).iloc[:, 0]
                
                nifti_fp = nsddata_path / f"nsddata_timeseries/ppdata/subj{self.subject_id:02d}/func1pt8mm/timeseries/timeseries_{run_id}.nii.gz"
                nifti = nibabel.load(nifti_fp, mmap=True)
                
                nifti_data_T = nifti.get_fdata(dtype=np.float16)[..., :225].T
                nifti.uncache()
                shape = nifti_data_T.shape
                
                cleaned_data = nilearn.signal.clean(
                    nifti_data_T.reshape(shape[0], -1), detrend=True, high_pass=None, t_r=TR_s, standardize="zscore_sample"
                )
                
                run_fmri_data = cleaned_data.reshape(shape).T.astype(np.float16)
                
                stimulus_history_queue = deque([-1] * self.history_length, maxlen=self.history_length)
                last_event_id, time_of_last_event = 0, None
                
                for t in range(run_fmri_data.shape[-1]):
                    stim_t = t - offset_in_TRs
                    if 0 <= stim_t < len(im_ids):
                        current_event_id = im_ids.iloc[stim_t]
                        if current_event_id > 0 and current_event_id != last_event_id:
                            stimulus_history_queue.append(current_event_id - 1)
                            last_event_id = current_event_id
                            time_of_last_event = t

                    sample = {
                        "fmri": run_fmri_data[..., t],
                        "stimulus_history": list(stimulus_history_queue),
                        "subject_idx": self.subject_id,
                        "session": session,
                        "run": run,
                        "time": t,
                    }
                    session_samples.append(sample)
            
            session_filepath = preproc_dir / f"session_{session:02d}.pt"
            torch.save(session_samples, session_filepath)
            del session_samples
            gc.collect()
            print(f"✅ Saved preprocessed data for session {session:02d}")

    def _create_roi_cache(self, preproc_dir: Path, roi_cache_path: Path) -> dict:
        """
        Creates a specific cache for an ROI by loading preprocessed full-brain data,
        applying the ROI mask, stacking all data into large tensors, and saving the result.
        """
        roi_name = Path(self.roi).name
        print(f"Creating new cache for ROI '{roi_name}' at {roi_cache_path}", flush=True)

        roi_path = Path(self.roi)
        roi_mask = nibabel.load(roi_path, mmap=True).get_fdata() > 0
        
        # Use lists to collect data before stacking
        all_fmri = []
        all_stim_hist = []
        all_subj_idx = []
        all_session = []
        all_run = []
        all_time = []

        session_files = sorted(preproc_dir.glob("session_*.pt"))
        print(f"Found {len(session_files)} preprocessed session files. Applying ROI...", flush=True)
        for session_filepath in session_files:
            session_samples = torch.load(session_filepath)
            for sample in session_samples:
                # Append numpy arrays and python scalars, which is fast
                all_fmri.append(sample["fmri"][roi_mask])
                all_stim_hist.append(sample["stimulus_history"])
                all_subj_idx.append(sample["subject_idx"])
                all_session.append(sample["session"])
                all_run.append(sample["run"])
                all_time.append(sample["time"])
            del session_samples
            gc.collect()

        print(f"Stacking {len(all_fmri)} samples into tensors...", flush=True)
        # Convert lists into single, large tensors
        cached_data = {
            "fmri": torch.from_numpy(np.array(all_fmri, dtype=np.float32)),
            "stimulus_history": torch.tensor(all_stim_hist, dtype=torch.long),
            "subject_idx": torch.tensor(all_subj_idx, dtype=torch.long),
            "session": torch.tensor(all_session, dtype=torch.long),
            "run": torch.tensor(all_run, dtype=torch.long),
            "time": torch.tensor(all_time, dtype=torch.float),
        }

        print(f"Saving consolidated ROI-specific data to cache...", flush=True)
        torch.save(cached_data, roi_cache_path)
        print(f"✅ Saved ROI cache: {roi_cache_path}", flush=True)
        
        return cached_data

    def build(self, roi: str= "") -> 'NSDDataset':
        """
        Orchestrates data loading and caching. It creates an ROI-specific cache 
        on demand from preprocessed full-brain data.
        """
        self.roi = roi
        img_path = Path(self.nsddata_path).resolve() / "nsd_stimuli.hdf5"
        roi_cache_filepath = self._get_roi_cache_filepath()

        if roi_cache_filepath.exists():
            print(f"✅ Loading cached ROI data from: {roi_cache_filepath}", flush=True)
            final_data = torch.load(roi_cache_filepath)
        else:
            print(f"ROI cache not found. Attempting to build it.", flush=True)
            preproc_dir = self._get_preproc_dir_path()
            if not preproc_dir.exists() or not any(preproc_dir.iterdir()):
                self._preprocess_full_brain(preproc_dir)
            else:
                print(f"Found existing preprocessed data at {preproc_dir}", flush=True)
            final_data = self._create_roi_cache(preproc_dir, roi_cache_filepath)
        
        # Correctly get voxel count from the stacked fmri tensor's second dimension
        print(f"Dataset ready. Voxel count per sample: {final_data['fmri'].shape[1]}", flush=True)
        
        # Pass the dictionary of tensors to the corrected NSDDataset constructor
        return NSDDataset(cached_data=final_data, img_path=img_path)


def create_nsd_dataloader(
    config: NsdSaeDataConfig, 
    roi: str, 
    batch_size: int, 
    workers: int = 4, 
    shuffle: bool = True, 
    pin_memory: bool = True
) -> DataLoader:
    """
    Builds the dataset using the given configuration and returns a DataLoader.

    Args:
        config (NsdSaeDataConfig): The main data configuration object.
        roi (str): Path to the region of interest NIfTI file.
        batch_size (int): The batch size for the DataLoader.
        workers (int): The number of workers for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        pin_memory (bool): Whether to use pinned memory.

    Returns:
        DataLoader: A configured PyTorch DataLoader.
    """
    dataset = config.build(roi=roi)
    print(f"Number of samples in dataset: {len(dataset)}", flush=True)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )