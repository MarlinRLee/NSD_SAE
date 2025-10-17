# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this tree.

from typing import Optional, Union
import torch
from collections import deque
from pathlib import Path
import numpy as np
import shutil
import nibabel
import nilearn.signal
import pandas as pd
import pydantic
import h5py
import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
# Assuming 'data' is a package with the NSDDataset class
from data import NSDDataset 

# Global constants for NSD fMRI acquisition
TR_s = 4 / 3 # Repetition Time in seconds
NUM_TRS_PER_RUN = 225 # Number of time points (TRs) in one fMRI run

class NsdSaeDataConfig(pydantic.BaseModel):
    """
    Configuration class for NSD fMRI data preprocessing and caching.
    Uses Pydantic for validation and structured configuration.
    """
    nsddata_path: str # Path to the root of the NSD dataset
    subject_id: int
    offset: float = 4.6 # Time shift (in seconds) to account for Hemodynamic Response Function (HRF) delay
    history_length: int = 4 # Number of past stimuli stored
    
    # Chunking parameters for optimal I/O
    chunk_size: int = 1000 # Number of samples (TRs) to process before writing to HDF5
    hdf5_chunk_rows: int = 100 # HDF5 chunk size for better I/O performance (read/write access)
    
    def _get_preproc_dir_path(self) -> Path:
        """Gets the directory for storing intermediate, full-brain (unmasked) HDF5 data."""
        nsddata_path = Path(self.nsddata_path).resolve()
        cache_dir = nsddata_path / ".cache"
        cache_dir.mkdir(exist_ok=True)
        preproc_dir = cache_dir / f"subj{self.subject_id:02d}_preprocessed_hdf5"
        return preproc_dir
    
    def _get_roi_cache_filepath(self, roi_id_name: str, roi_is_custom: bool) -> Path:
        """
        Determines the standardized *base* cache file path for a given ROI identifier.
        The final fMRI and metadata cache files will derive from this base path.
        
        Args:
            roi_id_name: The name/identifier for the ROI (e.g., 'V1' or 'my_custom_roi').
            roi_is_custom: True if the ROI was passed as a raw array, influencing naming.
        """
        nsddata_path = Path(self.nsddata_path).resolve()
        cache_dir = nsddata_path / ".cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Use .h5 as a *base* name to derive the .npy paths
        cache_filename = f"subj{self.subject_id:02d}_roi-{roi_id_name}_offset{self.offset}.h5"
        return cache_dir / cache_filename
    
    @staticmethod
    def _validate_roi_mask(roi_mask: np.ndarray, ref_roi_path: Path) -> None:
        """
        Checks if a given ROI mask is valid by comparing its shape to a reference 
        ROI and ensuring its values are binary (0s and 1s).
        
        Raises:
            ValueError: If the shape does not match or the ROI is not binary.
        """
        ref_roi_shape = nibabel.load(ref_roi_path, mmap=True).shape

        # 2. Make sure shape matches
        if roi_mask.shape != ref_roi_shape:
            raise ValueError(f"ROI shape mismatch: Input ROI shape {roi_mask.shape} does not match reference shape {ref_roi_shape}.")

        # 3. Make sure mask is binary (i.e., only contains 0 and 1)
        unique_values = np.unique(roi_mask)
        is_binary = np.all(np.isin(unique_values, [0, 1]))
        
        if not is_binary:
            raise ValueError(f"ROI is not binary: Found unique values {unique_values}. Only 0 and 1 are allowed.")
    
    def _get_and_validate_roi_mask(self, roi: Union[str, np.ndarray]) -> np.ndarray:
        """
        Loads the ROI mask from path or validates the provided array.
        
        Args:
            roi: Either a string path to a NIfTI file or a pre-loaded np.ndarray.
            
        Returns:
            The flattened, binary, boolean ROI mask (np.ndarray of bools).
        """
        # The reference file for shape check is assumed to be a known, consistent file
        ref_roi_path = Path(self.nsddata_path) / "nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz"

        if isinstance(roi, str) or isinstance(roi, Path):
            # Case 1: ROI is a path to a NIfTI file
            roi_path = Path(roi)
            # Load the NIfTI data
            roi_mask_data = nibabel.load(roi_path, mmap=True).get_fdata() > 0
            # Validate the loaded data
            self._validate_roi_mask(roi_mask_data, ref_roi_path)
        elif isinstance(roi, np.ndarray):
            # Case 2: ROI is a pre-loaded array
            roi_mask_data = roi
            # Validate the provided array
            self._validate_roi_mask(roi_mask_data, ref_roi_path)
        else:
            raise TypeError("ROI must be a file path (str/Path) or a NumPy array (np.ndarray).")

        # Binarize the data (True for voxels in the ROI, False otherwise) and flatten
        roi_mask_flat = (roi_mask_data > 0).flatten()
        return roi_mask_flat

    def _create_session_hdf5(self, session_filepath: Path, n_voxels: int, estimated_samples: int = 0):
        """Creates an extendable HDF5 file for a session"""
        with h5py.File(session_filepath, 'w') as f:
            # Create datasets with chunking for better I/O
            f.create_dataset(
                'fmri',
                shape=(0, n_voxels),
                maxshape=(None, n_voxels),
                dtype=np.float16,
                chunks=(self.hdf5_chunk_rows, n_voxels),
                compression='gzip',
                compression_opts=4 # Medium compression, good speed/size tradeoff
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
        """Appends a batch of fMRI and metadata samples to the session HDF5 file."""
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
    
    def preprocess_full_brain(self, preproc_dir: Path, num_workers: int = None):
        """
        Processes all NSD sessions to create intermediate full-brain HDF5 files.
        This is a computationally intensive step that is skipped if files exist.
        """
        preproc_dir.mkdir(exist_ok=True, parents=True)

        # Identify which session files are missing.
        sessions_to_process = []
        for session_num in range(1, 41):
            if not (preproc_dir / f"session_{session_num:02d}.h5").exists():
                sessions_to_process.append(session_num)

        # If all files exist, we're done.
        if not sessions_to_process:
            print(f"Brain data already preprocessed at: {preproc_dir}", flush=True)
            return
        
        print(f"Found {40 - len(sessions_to_process)} existing session files. Processing the {len(sessions_to_process)} missing sessions...", flush=True)

        if num_workers is None:
            # Determine number of workers based on number of sessions to process
            num_workers = min(len(sessions_to_process)//4, os.cpu_count() or 1)
            
        if num_workers > 0 and len(sessions_to_process) > 0:
             print(f"Beginning parallel preprocessing with {num_workers} workers...", flush=True)
             with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit only the missing sessions to the pool.
                futures = [executor.submit(_process_single_session, session, self, preproc_dir) for session in sessions_to_process]
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"A session processing task generated an exception: {exc}")
        
        print("All missing sessions have been preprocessed successfully.", flush=True)

    def generate_roi_cache(self, roi_base_path: Path, roi_mask_flat: np.ndarray, num_workers: Optional[int] = None):
        """
        Aggregates preprocessed full-brain data, applies the ROI mask, and saves
        the final, masked fMRI and metadata to .npy cache files.
        """
        fmri_cache_path = roi_base_path.with_suffix(".fmri.npy")
        meta_cache_path = roi_base_path.with_suffix(".meta.npy")

        # Skip if the final ROI cache files already exist
        if fmri_cache_path.exists() and meta_cache_path.exists():
            print(f"Found cached ROI data: {fmri_cache_path}", flush=True)
            return

        print(f"Building ROI cache at {fmri_cache_path} and {meta_cache_path}", flush=True)
        preproc_dir = self._get_preproc_dir_path()

        self.preprocess_full_brain(preproc_dir, num_workers=num_workers)
        
        n_roi_voxels = roi_mask_flat.sum() # Total count of voxels in the ROI
        print(f"ROI has {n_roi_voxels} voxels", flush=True)
        
        session_files = sorted(preproc_dir.glob("session_*.h5"))
        
        if num_workers is None:
            # Determine number of workers, e.g., number of CPUs available
            # Note: os.cpu_count() might return None in restricted environments
            num_workers = min(len(session_files)//4, os.cpu_count() or 1)
        
        print(f"Processing {len(session_files)} sessions in parallel with {num_workers} workers...", flush=True)
        
        # Use a ProcessPoolExecutor to parallelize the work
        fmri_chunks = []
        meta_chunks = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(_process_single_file, fp, roi_mask_flat): fp for fp in session_files}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    roi_fmri, metadata = future.result()
                    fmri_chunks.append(roi_fmri)
                    meta_chunks.append(metadata)
                except Exception as exc:
                    print(f'{file_path} generated an exception: {exc}')

        print("Concatenating data from all workers...")
        final_fmri = np.concatenate(fmri_chunks, axis=0).astype(np.float32)
        final_meta = np.concatenate(meta_chunks, axis=0)
        
        print(f"Saving fMRI data ({final_fmri.shape}) to {fmri_cache_path}")
        np.save(fmri_cache_path, final_fmri)
        
        print(f"Saving metadata ({final_meta.shape}) to {meta_cache_path}")
        np.save(meta_cache_path, final_meta)

        del fmri_chunks, meta_chunks, final_fmri, final_meta
        gc.collect()

        print(f"ROI cache created: {fmri_cache_path.parent}", flush=True)
    
    def load_roi_NSDDataset(self, 
                           roi: Union[str, Path, np.ndarray], 
                           custom_name: str = None,
                           return_fmri_only: bool = True, 
                           num_workers: Optional[int] = None, 
                           local_file: str = "") -> NSDDataset:
        """
        Primary interface to load the final PyTorch Dataset.
        Handles ROI loading/validation, cache creation, file copying, 
        and initialization of the NSDDataset.
        
        Args:
            roi: A path (str or Path) to a NIfTI ROI file, OR a pre-loaded np.ndarray 
                 representing the 3D ROI mask.
            custom_name: An optional string to use as the unique identifier in the 
                         cache filename. If 'roi' is a path, its stem is used by default.
                         If 'roi' is an array, 'custom_name' is REQUIRED.
            return_fmri_only: Passed to NSDDataset.
            num_workers: Number of workers for cache generation.
            local_file: Local directory path for copying cache files.
        """
        
        # --- 1. Determine ROI Identifier and Type ---
        roi_is_custom_array = isinstance(roi, np.ndarray)
        if roi_is_custom_array:
            if not custom_name:
                raise ValueError("When 'roi' is a NumPy array, 'custom_name' must be provided for the cache file.")
            roi_id_name = custom_name
        else:
            roi_path = Path(roi)
            roi_id_name = roi_path.stem
        
        # --- 2. Load and Validate ROI Mask ---
        # This will raise a ValueError if the ROI is invalid
        roi_mask_flat = self._get_and_validate_roi_mask(roi)
        
        # --- 3. Get Cache Paths and Generate Cache ---
        img_path = Path(self.nsddata_path).resolve() / "nsd_stimuli.hdf5"
        
        # Get the base path
        roi_base_cache_path = self._get_roi_cache_filepath(roi_id_name, roi_is_custom_array)
        
        # This will create/load the .npy cache files
        self.generate_roi_cache(roi_base_cache_path, roi_mask_flat, num_workers=num_workers)
        
        # Derive the final .npy paths
        fmri_cache_path = roi_base_cache_path.with_suffix(".fmri.npy")
        meta_cache_path = roi_base_cache_path.with_suffix(".meta.npy")

        # --- 4. Handle Local File Copying ---
        current_fmri_path = fmri_cache_path
        current_meta_path = meta_cache_path
        
        if local_file:
            print(f"Copying files to local disk ({local_file}).", flush=True)
            LOCAL_DIR = Path(local_file)
            LOCAL_DIR.mkdir(exist_ok=True)
            
            fmri_cache_path_local = LOCAL_DIR / current_fmri_path.name
            meta_cache_path_local = LOCAL_DIR / current_meta_path.name

            if not fmri_cache_path_local.exists():
                shutil.copy(current_fmri_path, fmri_cache_path_local)

            if not meta_cache_path_local.exists():
                shutil.copy(current_meta_path, meta_cache_path_local)
            
            # Update the paths to the local copies for the Dataset initializer
            current_fmri_path = fmri_cache_path_local
            current_meta_path = meta_cache_path_local
        
        # --- 5. Initialize Dataset ---
        return NSDDataset(
            fmri_cache_path=current_fmri_path,
            meta_cache_path=current_meta_path,
            img_path=img_path,
            return_fmri_only=return_fmri_only
        )
    
def _process_single_session(session_num: int, config: 'NsdSaeDataConfig', preproc_dir: Path):
    """
    Processes all fMRI runs for a single session: loads, cleans, applies
    time-shift (offset), generates metadata, and saves the result to a
    session-specific HDF5 file.
    """
    
    session_filepath = preproc_dir / f"session_{session_num:02d}.h5"
    nsddata_path = Path(config.nsddata_path).resolve()
    
    # Logic to determine which runs belong to this session
    subject_to_14run_sessions = {1: (21, 38), 5: (21, 38), 2: (21, 30), 7: (21, 30)}
    runs_range = (
        range(2, 14)
        if config.subject_id in subject_to_14run_sessions and
           subject_to_14run_sessions[config.subject_id][0] <= session_num <= subject_to_14run_sessions[config.subject_id][1]
        else range(1, 13)
    )

    first_run = True
    for run in runs_range:
        run_id = f"session{session_num:02d}_run{run:02d}"
        
        # Load stimulus info
        path_to_df = nsddata_path / f"nsddata_timeseries/ppdata/subj{config.subject_id:02d}/func1pt8mm/design/design_{run_id}.tsv"
        im_ids = pd.read_csv(path_to_df, header=None).iloc[:, 0]
        
        # Load and process fMRI data
        nifti_fp = nsddata_path / f"nsddata_timeseries/ppdata/subj{config.subject_id:02d}/func1pt8mm/timeseries/timeseries_{run_id}.nii.gz"
        nifti = nibabel.load(nifti_fp, mmap=True)
        
        # Take up to NUM_TRS_PER_RUN (225) time points
        nifti_data_T = nifti.get_fdata(dtype=np.float16)[..., :NUM_TRS_PER_RUN].T 
        shape = nifti_data_T.shape
        n_voxels = np.prod(shape[1:])
        
        if first_run:
            estimated_samples = len(list(runs_range)) * NUM_TRS_PER_RUN
            # Merged: includes estimated_samples in the call to _create_session_hdf5
            config._create_session_hdf5(session_filepath, n_voxels, estimated_samples) 
            first_run = False
        
        # Preprocessing: detrend and z-score standardization
        cleaned_data = nilearn.signal.clean(nifti_data_T.reshape(shape[0], -1), detrend=True, high_pass=None, t_r=TR_s, standardize="zscore_sample")
        run_fmri_data = cleaned_data.reshape(shape).T.astype(np.float16)
        del nifti_data_T, cleaned_data
        nifti.uncache()
        
        fmri_batch, metadata_batch = [], []
        stimulus_history_queue = deque([-1] * config.history_length, maxlen=config.history_length)
        last_event_id = 0
        offset_in_TRs = int(round(config.offset / TR_s))

        for t in range(run_fmri_data.shape[-1]):
            stim_t = t - offset_in_TRs
            if 0 <= stim_t < len(im_ids):
                current_event_id = im_ids.iloc[stim_t]
                if current_event_id > 0 and current_event_id != last_event_id:
                    # Note: NSD image IDs are 1-indexed, but the dataset expects 0-indexed indices
                    stimulus_history_queue.append(current_event_id - 1) 
                    last_event_id = current_event_id
            
            fmri_batch.append(run_fmri_data[..., t].flatten())
            metadata_batch.append((list(stimulus_history_queue), config.subject_id, session_num, run, float(t)))
            
            if len(fmri_batch) >= config.chunk_size:
                config._append_to_hdf5(session_filepath, np.array(fmri_batch, dtype=np.float16), metadata_batch)
                fmri_batch, metadata_batch = [], []
        
        # Append any remaining samples
        if fmri_batch:
            config._append_to_hdf5(session_filepath, np.array(fmri_batch, dtype=np.float16), metadata_batch)
        
        del run_fmri_data
        gc.collect()

    print(f" Finished session {session_num:02d}. Saved to {session_filepath.name}", flush=True)
    return session_filepath

def _process_single_file(session_filepath: Path, roi_mask_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads one full-brain session HDF5 file, applies the ROI mask, and returns
    only the fMRI data and metadata for the ROI.
    """
    with h5py.File(session_filepath, 'r') as src:
        fmri_full_session = src['fmri'][:]
        metadata_full_session = src['metadata']
        
        # Apply ROI mask in memory
        roi_fmri = fmri_full_session[:, roi_mask_flat]
        
        return roi_fmri, metadata_full_session