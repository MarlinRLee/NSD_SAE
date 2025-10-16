# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this tree.
import typing as tp
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py
from PIL import Image


class NSDDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading preprocessed NSD fMRI and stimulus data.

    Utilizes memory-mapping (mmap) for fMRI data for highly efficient,
    process-safe, and low-RAM access across multiple data workers.
    """
    def __init__(self, fmri_cache_path: Path, meta_cache_path: Path, img_path: Path, return_fmri_only: bool = True, verbose = True):
        self.stimuli_path = img_path
        self.return_fmri_only = return_fmri_only

        # Load data using memory-mapping
        self.fmri_data = np.load(fmri_cache_path, mmap_mode='r')
        
        # Metadata is small, just load it into RAM
        # allow_pickle=True is needed for structured arrays
        self.metadata = np.load(meta_cache_path, allow_pickle=True)
        
        self._length = self.fmri_data.shape[0]  # Total number of samples (TRs)
        self.voxel_dim = self.fmri_data.shape[1] # Number of voxels in the ROI
        
        if verbose:
            print(f"Dataset initialized with {self._length} samples", flush=True)
            print(f"Voxel count per sample: {self.fmri_data.shape[1]}", flush=True)
    
    def change_return_type(self, return_fmri_only: bool):
        """Allows switching the return format (fMRI only vs. fMRI + metadata)."""
        self.return_fmri_only = return_fmri_only

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self._length
    
    def __getitem__(self, idx: int) -> tp.Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Retrieves one sample (TR) from the dataset.

        Args:
            idx: Index of the sample (TR).

        Returns:
            fMRI tensor or a dictionary of fMRI tensor and metadata tensors.
        or if return_fmri_only is false it returns a dict with keys:
            "fmri": fMRI tensor
            "stimulus_history": last n images seen before this timestep
            "subject_idx": what subject the data is from
            "session": what session the sample is from
            "run": what run in the session the sample is from
            "time": what time in the run the sample is from
        """

        fmri_tensor = torch.from_numpy(self.fmri_data[idx])
        
        if self.return_fmri_only:
            return fmri_tensor
        
        # Extract metadata and convert to PyTorch tensors
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
        """
        Special method to retrieve a PIL image from the main NSD HDF5 file.
        
        NOTE: 'idx' here is the **image ID (stimulus_history value)**, not the sample index.
        """
        with h5py.File(self.stimuli_path, 'r') as stim_file:
            img_brick = stim_file['imgBrick']
            image_data = img_brick[idx]
        return Image.fromarray(image_data)

    def get_sequence_item(self, idx: int) -> tp.Optional[dict[str, torch.Tensor]]:  
        """
        Finds the full 6-sample fMRI sequence (TRs) and corresponding image for a given target timestep index (`idx`). 
        This is mainly meant to extract the corresponding dynadiff sample.
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
        """Safely close any open memory-map handles to release resources."""
        # Check if fmri_data and its _mmap attribute exist and are not None
        if hasattr(self, 'fmri_data') and hasattr(self.fmri_data, '_mmap') and self.fmri_data._mmap:
            self.fmri_data._mmap.close()
            self.fmri_data._mmap = None

    def __del__(self):
        """Ensure file handles are closed when the object is deleted."""
        self.close()