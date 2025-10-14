# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this tree.

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
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import mmap
import struct

TR_s = 4 / 3
NUM_TRS_PER_RUN = 225

class NSDDataset(torch.utils.data.Dataset):
    def __init__(self, fmri_cache_path: Path, meta_cache_path: Path, img_path: Path, return_fmri_only: bool = True):
        self.stimuli_path = img_path
        self.return_fmri_only = return_fmri_only

        # Load data using memory-mapping
        # mmap_mode='r' is crucial for fast, shared, read-only access
        self.fmri_data = np.load(fmri_cache_path, mmap_mode='r')
        
        # Metadata is small, just load it into RAM
        # allow_pickle=True is needed for structured arrays
        self.metadata = np.load(meta_cache_path, allow_pickle=True)
        
        self._length = self.fmri_data.shape[0]
        
        print(f"Dataset initialized with {self._length} samples", flush=True)
        print(f"Voxel count per sample: {self.fmri_data.shape[1]}", flush=True)
    
    def change_return_type(return_fmri_only: bool):
        self.return_fmri_only = return_fmri_only

    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> tp.Union[torch.Tensor, dict[str, torch.Tensor]]:
        # Reading from a memory-map is extremely fast and process-safe
        fmri_tensor = torch.from_numpy(self.fmri_data[idx].astype(np.float32))
        
        if self.return_fmri_only:
            return fmri_tensor
        
        metadata = self.metadata[idx]
        return {
            "fmri": fmri_tensor,
            "stimulus_history": torch.from_numpy(metadata['stimulus_history']).long(),
            "subject_idx": torch.tensor(metadata['subject_idx'], dtype=torch.long),
            "session": torch.tensor(metadata['session'], dtype=torch.long),
            "run": torch.tensor(metadata['run'], dtype=torch.long),
            "time": torch.tensor(metadata['time'], dtype=torch.float),
        }
    
    def __get_image__(self, idx: int) -> Image.Image:
        """Special method to retrieve a PIL image from the HDF5 brick. THIS IS THE IMAGE INDEX STORED IN stimulus_history, not sample index"""
        # This function is still inefficient. If you plan to use it
        # in training, apply the worker-init pattern from Solution 1
        # to self.stimuli_path.
        with h5py.File(self.stimuli_path, 'r') as stim_file:
            img_brick = stim_file['imgBrick']
            image_data = img_brick[idx]
        return Image.fromarray(image_data)

    def get_sequence_item(self, idx: int) -> tp.Optional[dict[str, torch.Tensor]]:
        """
        Finds the full 6-sample sequence and corresponding image for a given
        timestep index.
        """
        if not (0 <= idx < len(self)):
            raise IndexError("Index out of range.")
        
        # 1. Get target sample metadata
        target_metadata = self.metadata[idx]
        stim_id = target_metadata['stimulus_history'][-1]
        
        # If stim_id is -1, it's a resting state, not part of a sequence
        if stim_id == -1:
            return None
        
        target_run = target_metadata['run']
        target_session = target_metadata['session']
        
        # 2. Search backwards to find sequence start
        # Use a sliding window to minimize HDF5 reads
        search_window = min(24, idx)  # Look back at most 50 samples
        start_search_idx = max(0, idx - search_window)
        
        # Read metadata chunk for backward search
        metadata_chunk = self.metadata[start_search_idx:idx+1]
        
        sequence_start_index = -1
        for i in range(len(metadata_chunk) - 1, -1, -1):
            abs_idx = start_search_idx + i
            current_meta = metadata_chunk[i]
            
            # Stop if we cross a run/session boundary
            if current_meta['run'] != target_run or current_meta['session'] != target_session:
                sequence_start_index = abs_idx + 1
                break
            
            # Stop when stimulus ID changes
            if current_meta['stimulus_history'][-1] != stim_id:
                sequence_start_index = abs_idx + 1
                break
            
            # Handle the case where sequence starts at beginning of data
            if abs_idx == 0:
                sequence_start_index = 0
                break
        
        # If we didn't find a boundary in the window, it's at the start
        if sequence_start_index == -1:
            sequence_start_index = start_search_idx
        
        # 3. Define the 6-sample slice and check boundaries
        sequence_end_index = sequence_start_index + 6
        if sequence_end_index > len(self):
            return None  # Sequence truncated at end of dataset
        
        # Check if sequence crosses run boundary
        last_metadata = self.metadata[sequence_end_index - 1]
        if last_metadata['run'] != target_run or last_metadata['session'] != target_session:
            return None  # Sequence truncated by end of run
        
        # 4. Extract fMRI data for the sequence
        fmri_sequence = self.fmri_data[sequence_start_index:sequence_end_index]

        fmri = torch.from_numpy(fmri_sequence.astype(np.float32)).float() 
        
        # 5. Get the stimulus image
        image_pil = self.__get_image__(stim_id)
        image = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0
        
        subject_id = torch.tensor(target_metadata['subject_idx'], dtype=torch.long)
        
        return {"brain": fmri, "img": image, "subject_idx": subject_id}

    def close(self):
        """Close any open file handles."""
        # Check if fmri_data and its _mmap attribute exist and are not None
        if hasattr(self, 'fmri_data') and hasattr(self.fmri_data, '_mmap') and self.fmri_data._mmap:
            self.fmri_data._mmap.close()
            self.fmri_data._mmap = None

    def __del__(self):
        self.close()


def _process_single_file(session_filepath: Path, roi_mask_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads one HDF5 file, applies the ROI mask, and returns the results.
    This is the target function for each worker process.
    """
    print(f"  Worker {os.getpid()} processing {session_filepath.name}...", flush=True)
    with h5py.File(session_filepath, 'r') as src:
        fmri_full_session = src['fmri'][:]
        metadata_full_session = src['metadata'][:]
        
        # Apply ROI mask in memory
        roi_fmri = fmri_full_session[:, roi_mask_flat]
        
        return roi_fmri, metadata_full_session

class NsdSaeDataConfig(pydantic.BaseModel):

    nsddata_path: str
    subject_id: int
    offset: float = 4.6
    history_length: int = 4
    
    # Chunking parameters for optimal I/O
    chunk_size: int = 1000  # Number of samples per chunk during processing
    hdf5_chunk_rows: int = 100  # HDF5 chunk size for better I/O performance
    
    def _get_preproc_dir_path(self) -> Path:
        """Gets the directory for storing preprocessed full-brain data."""
        nsddata_path = Path(self.nsddata_path).resolve()
        cache_dir = nsddata_path / ".cache"
        cache_dir.mkdir(exist_ok=True)
        preproc_dir = cache_dir / f"subj{self.subject_id:02d}_preprocessed_hdf5"
        return preproc_dir
    
    def _get_roi_cache_filepath(self, roi) -> Path:
        """Determines the standardized *base* cache file path for a given ROI."""
        nsddata_path = Path(self.nsddata_path).resolve()
        cache_dir = nsddata_path / ".cache"
        cache_dir.mkdir(exist_ok=True)
        roi_name = Path(roi).stem
        # NOTE: We use .h5 as a *base* name to derive the .npy paths
        cache_filename = f"subj{self.subject_id:02d}_roi-{roi_name}_offset{self.offset}.h5"
        return cache_dir / cache_filename
    
    def _create_session_hdf5(self, session_filepath: Path, n_voxels: int, estimated_samples: int):
        """Create an HDF5 file for a session with proper chunking."""
        with h5py.File(session_filepath, 'w') as f:
            # Create datasets with chunking for better I/O
            f.create_dataset(
                'fmri',
                shape=(0, n_voxels),
                maxshape=(None, n_voxels),
                dtype=np.float16,
                chunks=(self.hdf5_chunk_rows, n_voxels),
                compression='gzip',
                compression_opts=4  # Medium compression, good speed/size tradeoff
            )
            
            # Metadata stored as compound datatype for efficiency
            metadata_dtype = np.dtype([
                ('stimulus_history', np.int32, (self.history_length,)),
                ('subject_idx', np.int16),
                ('session', np.int16),
                ('run', np.int16),
                ('time', np.float32)
            ])
            
            f.create_dataset(
                'metadata',
                shape=(0,),
                maxshape=(None,),
                dtype=metadata_dtype,
                chunks=(self.hdf5_chunk_rows,)
            )
    
    def _append_to_hdf5(self, filepath: Path, fmri_batch: np.ndarray, metadata_batch: list):
        with h5py.File(filepath, 'a') as f:
            fmri_dset = f['fmri']
            metadata_dset = f['metadata']
            
            current_size = fmri_dset.shape[0]
            new_size = current_size + len(fmri_batch)
            
            # Resize datasets
            fmri_dset.resize(new_size, axis=0)
            metadata_dset.resize(new_size, axis=0)
            
            # Write data
            fmri_dset[current_size:new_size] = fmri_batch
            metadata_dset[current_size:new_size] = np.array(metadata_batch, dtype=metadata_dset.dtype)
    
    def preprocess_full_brain(self, preproc_dir: Path):
        """
        Process fMRI data and save to HDF5 files
        """
        preproc_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Identify all expected session files
        expected_session_files = {preproc_dir / f"session_{session:02d}.h5" for session in range(1, 41)}
        
        # 2. Check the existence of each file
        existing_session_files = {f for f in expected_session_files if f.exists()}
        
        num_existing = len(existing_session_files)
        num_expected = len(expected_session_files)

        # 3. Determine the state of the cache
        if num_existing == num_expected:
            print(f"Found all {num_expected} preprocessed session files at {preproc_dir}. Skipping creation.", flush=True)
            return

        elif 0 < num_existing < num_expected:
            # 4. Case: Subset missing, print warning
            missing_files = expected_session_files - existing_session_files
            print("WARNING: Full-brain cache is partially built.", flush=True)
            print(f"Missing {len(missing_files)} of {num_expected} session files (e.g., {next(iter(missing_files)).name}). Rebuilding or completing the cache.", flush=True)
            
        elif num_existing == 0:
            print(f"No existing preprocessed full-brain data found at {preproc_dir}. Starting creation.", flush=True)

        # Proceed with creation/completion of the cache
        print(f"Processing full-brain data into {preproc_dir}...", flush=True)
        
        nsddata_path = Path(self.nsddata_path).resolve()
        subject_to_14run_sessions = {1: (21, 38), 5: (21, 38), 2: (21, 30), 7: (21, 30)}
        offset_in_TRs = int(round(self.offset / TR_s))
        
        for session in range(1, 41):
            session_filepath = preproc_dir / f"session_{session:02d}.h5"
            
            # Skip session if the file already exists (handles partial build completion)
            if session_filepath.exists():
                continue
                
            print(f"\nProcessing session {session:02d}", flush=True)
            
            runs_range = (
                range(2, 14)
                if self.subject_id in subject_to_14run_sessions and 
                   subject_to_14run_sessions[self.subject_id][0] <= session <= subject_to_14run_sessions[self.subject_id][1]
                else range(1, 13)
            )
            
            first_run = True
            
            for run in runs_range:
                run_id = f"session{session:02d}_run{run:02d}"
                print(f"     Processing: {run_id}", flush=True)
                
                # Load stimulus info
                path_to_df = nsddata_path / f"nsddata_timeseries/ppdata/subj{self.subject_id:02d}/func1pt8mm/design/design_{run_id}.tsv"
                try:
                    im_ids = pd.read_csv(path_to_df, header=None).iloc[:, 0]
                except FileNotFoundError:
                    print(f"Design file not found for {run_id}. Skipping run.", flush=True)
                    continue
                
                # Load and process fMRI data
                nifti_fp = nsddata_path / f"nsddata_timeseries/ppdata/subj{self.subject_id:02d}/func1pt8mm/timeseries/timeseries_{run_id}.nii.gz"
                try:
                    nifti = nibabel.load(nifti_fp, mmap=True)
                except FileNotFoundError:
                    print(f"Nifti file not found for {run_id}. Skipping run.", flush=True)
                    continue

                nifti_data_T = nifti.get_fdata(dtype=np.float16)[..., :NUM_TRS_PER_RUN].T
                shape = nifti_data_T.shape
                n_voxels = np.prod(shape[1:])
                
                # Initialize HDF5 file on first run *of this session*
                if first_run:
                    estimated_samples = len(runs_range) * NUM_TRS_PER_RUN
                    self._create_session_hdf5(session_filepath, n_voxels, estimated_samples)
                    first_run = False
                
                # Clean signal
                cleaned_data = nilearn.signal.clean(
                    nifti_data_T.reshape(shape[0], -1),
                    detrend=True,
                    high_pass=None,
                    t_r=TR_s,
                    standardize="zscore_sample"
                )
                
                run_fmri_data = cleaned_data.reshape(shape).T.astype(np.float16)
                del nifti_data_T, cleaned_data
                
                nifti.uncache()
                
                # Process in chunks to reduce memory
                fmri_batch = []
                metadata_batch = []
                
                stimulus_history_queue = deque([-1] * self.history_length, maxlen=self.history_length)
                last_event_id = 0
                
                for t in range(run_fmri_data.shape[-1]):
                    stim_t = t - offset_in_TRs
                    if 0 <= stim_t < len(im_ids):
                        current_event_id = im_ids.iloc[stim_t]
                        if current_event_id > 0 and current_event_id != last_event_id:
                            stimulus_history_queue.append(current_event_id - 1)
                            last_event_id = current_event_id
                    
                    fmri_batch.append(run_fmri_data[..., t].flatten())
                    metadata_batch.append((
                        list(stimulus_history_queue),
                        self.subject_id,
                        session,
                        run,
                        float(t)
                    ))
                    
                    # Write in chunks
                    if len(fmri_batch) >= self.chunk_size:
                        self._append_to_hdf5(
                            session_filepath,
                            np.array(fmri_batch, dtype=np.float16),
                            metadata_batch
                        )
                        fmri_batch = []
                        metadata_batch = []
                        gc.collect()
                
                # Write remaining data
                if fmri_batch:
                    self._append_to_hdf5(
                        session_filepath,
                        np.array(fmri_batch, dtype=np.float16),
                        metadata_batch
                    )
                
                del run_fmri_data
                gc.collect()
            
            if session_filepath.exists():
                print(f"Saved session {session:02d}", flush=True)
    
    def load_roi_cache(self, roi_base_path: Path, roi: Path):
            """
            Builds ROI cache using a pool of processes for maximum speed.
            """
            fmri_cache_path = roi_base_path.with_suffix(".fmri.npy")
            meta_cache_path = roi_base_path.with_suffix(".meta.npy")

            if fmri_cache_path.exists() and meta_cache_path.exists():
                print(f"Found cached ROI data: {fmri_cache_path}", flush=True)
                return

            print(f"Building ROI cache at {fmri_cache_path} and {meta_cache_path}", flush=True)
            preproc_dir = self._get_preproc_dir_path()
            self.preprocess_full_brain(preproc_dir)
            
            roi_path = Path(roi)
            roi_mask = nibabel.load(roi_path, mmap=True).get_fdata() > 0
            roi_mask_flat = roi_mask.flatten()
            n_roi_voxels = roi_mask_flat.sum()
            
            print(f"ROI has {n_roi_voxels} voxels", flush=True)
            
            session_files = sorted(preproc_dir.glob("session_*.h5"))
            
            # Determine number of workers, e.g., number of CPUs available
            num_workers = min(len(session_files), os.cpu_count() or 1)
            print(f"Processing {len(session_files)} sessions in parallel with {num_workers} workers...", flush=True)
            
            # Use a ProcessPoolExecutor to parallelize the work
            fmri_chunks = []
            meta_chunks = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all file processing tasks to the pool
                future_to_file = {executor.submit(_process_single_file, fp, roi_mask_flat): fp for fp in session_files}
                
                for future in as_completed(future_to_file):
                    try:
                        # Retrieve the result (masked_fmri, metadata)
                        roi_fmri, metadata = future.result()
                        fmri_chunks.append(roi_fmri)
                        meta_chunks.append(metadata)
                    except Exception as exc:
                        print(f'{future_to_file[future]} generated an exception: {exc}')

            print("Concatenating data from all workers...")
            final_fmri = np.concatenate(fmri_chunks, axis=0).astype(np.float16)
            final_meta = np.concatenate(meta_chunks, axis=0)
            
            print(f"Saving fMRI data ({final_fmri.shape}) to {fmri_cache_path}")
            np.save(fmri_cache_path, final_fmri)
            
            print(f"Saving metadata ({final_meta.shape}) to {meta_cache_path}")
            np.save(meta_cache_path, final_meta)

            del fmri_chunks, meta_chunks, final_fmri, final_meta
            gc.collect()

            print(f"ROI cache created: {fmri_cache_path.parent}", flush=True)
    
    def load_roi_NSDDataset(self, roi: str = "", return_fmri_only: bool = True) -> NSDDataset:
        img_path = Path(self.nsddata_path).resolve() / "nsd_stimuli.hdf5"
        
        # Get the base path (e.g., .../cache.h5)
        roi_base_cache_path = self._get_roi_cache_filepath(roi)
        
        # This will create/load the .npy cache files
        self.load_roi_cache(roi_base_cache_path, roi)
        
        # Derive the new .npy paths
        fmri_cache_path = roi_base_cache_path.with_suffix(".fmri.npy")
        meta_cache_path = roi_base_cache_path.with_suffix(".meta.npy")
        
        # Dataset uses memory-mapping internally
        return NSDDataset(
            fmri_cache_path=fmri_cache_path,
            meta_cache_path=meta_cache_path,
            img_path=img_path,
            return_fmri_only=return_fmri_only
        )