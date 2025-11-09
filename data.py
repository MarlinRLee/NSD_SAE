import typing as tp
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py
from PIL import Image, ImageDraw, ImageFont


class NSDDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading preprocessed NSD fMRI and stimulus data.

    Uses lazy, memory-mapped loading for fMRI data for highly efficient,
    process-safe access across multiple data workers.
    """
    def __init__(self, fmri_cache_path: Path, meta_cache_path: Path, img_path: Path, return_fmri_only: bool = True, verbose = True):
        # Store paths
        self.fmri_cache_path = fmri_cache_path
        self.meta_cache_path = meta_cache_path
        self.stimuli_path = img_path
        self.return_fmri_only = return_fmri_only

        # The actual memory-mapped data will be loaded in the worker processes
        self.fmri_data = None  # Will be loaded on first __getitem__ call in each worker
        self.voxel_dim = None

        # Load metadata in the main process (it's small and process-safe)
        # allow_pickle=True is needed for structured arrays
        self.metadata = np.load(self.meta_cache_path, allow_pickle=True)
        
        self._length = self.metadata.shape[0]  # Total number of samples (TRs)
        
        if verbose:
            print(f"Dataset initialized with {self._length} samples (fMRI data will be loaded lazily).", flush=True)

    def _load_fmri_data(self):
        """Helper to load fmri data using memory-mapping, only called once per process."""
        if self.fmri_data is None:
            self.fmri_data = np.load(self.fmri_cache_path, mmap_mode='r')
            self.voxel_dim = self.fmri_data.shape[1]
    
    def change_return_type(self, return_fmri_only: bool):
        """Allows switching the return format (fMRI only vs. fMRI + metadata)."""
        self.return_fmri_only = return_fmri_only

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self._length
    
    def __getitem__(self, idx: int) -> tp.Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Retrieves one sample (TR) from the dataset.
        """
        # Load the fMRI data if it hasn't been loaded in this process/worker yet
        if self.fmri_data is None:
            self._load_fmri_data()
            
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

        If the index is -1, it returns a gray image with text indicating no image was seen.
        
        NOTE: 'idx' here is the **image ID (stimulus_history value)**, not the sample index.
        """
        if idx != -1:
            # File is opened and closed for each callg
            with h5py.File(self.stimuli_path, 'r') as stim_file:
                img_brick = stim_file['imgBrick']
                image_data = img_brick[idx]
            return Image.fromarray(image_data)
        
        # --- Create a placeholder image for "no seen image" ---
        width, height = 425, 425
        bg_color = (128, 128, 128)  # Gray
        text = "no seen image"
        
        # Create a new gray image
        img = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Load a font (using default as a fallback)
        try:
            font = ImageFont.truetype("arial.ttf", size=30)
        except IOError:
            font = ImageFont.load_default()
            
        # Calculate text position to center it
        _, _, text_width, text_height = draw.textbbox((0, 0), text, font=font)
        text_x = (width - text_width) / 2
        text_y = (height - text_height) / 2
        
        # Draw the text on the image
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255)) # White text
        
        return img

    def __get_recent_image__(self, idx: int) -> Image.Image:
            """
            Retrieves the PIL image for the most recently seen stimulus at a given sample index.

            Args:
                idx: The sample index (TR) to query.
            
            Returns:
                Image.Image: The corresponding PIL image. Returns a "no seen image" 
                             placeholder if the most recent stimulus ID is -1.
            """
            # 1. Get the metadata for the requested sample index
            if not (0 <= idx < len(self)):
                raise IndexError("Index out of range.")
                
            metadata = self.metadata[idx]
            
            # 2. Get the stimulus history from the metadata
            stimulus_history = metadata['stimulus_history']
            
            # 3. The most recent image ID is the last one in the history array
            recent_image_id = stimulus_history[-1]
            
            return self.__get_image__(recent_image_id)

    def get_sequence_item(self, idx: int) -> tp.Optional[dict[str, torch.Tensor]]:  
        """
        Finds the full 6-sample fMRI sequence (TRs) and corresponding image for a given target timestep index (`idx`). 
        """
        # Ensure data is loaded for sequence retrieval as well
        if self.fmri_data is None:
            self._load_fmri_data()

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
        # Only close if the data was actually loaded
        if self.fmri_data is not None and hasattr(self.fmri_data, '_mmap') and self.fmri_data._mmap:
            self.fmri_data._mmap.close()
            self.fmri_data._mmap = None
            self.fmri_data = None # Clear reference

    def __del__(self):
        """Ensure file handles are closed when the object is deleted."""
        self.close()