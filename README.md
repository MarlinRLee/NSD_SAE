# NSD PyTorch Data Loader (SAE & Decoding Ready)

This repository provides an optimized, parallelized PyTorch `DataLoader` pipeline for the **Natural Scenes Dataset (NSD)**. It is specifically designed to support the **Sparse Autoencoder (SAE)** project by providing fast access to individual fMRI time-steps, while retaining the capability to retrieve the corisponding **6-TR fMRI sequences** for runing it on dynadiff.

The pipeline features a robust, two-stage, multiprocessing-enabled caching system using `h5py` for intermediate storage and memory-mapped NumPy arrays (`mmap`) for the final dataset, ensuring highly efficient and process-safe data access.

-----

## Quick Start & Usage

### 1\. Setup

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 2\. Data Download

Use the provided `download.py` script to fetch the necessary NSD files.

1.  **Configure `download.py`**: Update the S3 bucket URL, local `path`, and the list of `subjects` you require.

2.  **Run**:

    ```bash
    python download.py
    ```

### 3\. Core Workflow: `NsdSaeDataConfig`

The entire pipeline is managed by the **`NsdSaeDataConfig`** class, which handles all preprocessing, parallelization, and cache management.

```python
from nsd_data_loader import NsdSaeDataConfig
from torch.utils.data import DataLoader

# 1. Configuration
config = NsdSaeDataConfig(
    nsddata_path="/path/to/nsddata/", # Root directory of your NSD download
    subject_id=1,
    offset=4.6,     # HRF delay offset in seconds. This is what Dynadiff uses and seems standard so we should not mess with this without good reason.
    history_length=4, # Past stimuli seen.
)

# 2. Define ROI
# Use a NIfTI path. Example: full-brain general mask
roi_path = "/path/to/nsddata/nsddata/ppdata/subj01/func1pt8mm/roi/general.nii.gz"

# 3. Build the dataset
# This function automatically triggers cache creation if necessary.
dataset = config.load_roi_NSDDataset(
    roi=roi_path, 
    return_fmri_only=True, # Can be changed after loading but better during training
    num_workers=8 
)

# 4. Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=4096,
    num_workers=8,
    shuffle=True
)
```

-----

## Feature Set

### Mode 1: SAE Training (Time-step Mode)

This is the default and most efficient mode, optimized for training SAEs voxel-wise on individual TRs.

  * Set `return_fmri_only=True` in `load_roi_NSDDataset()`.
  * **Output:** The `DataLoader` yields a single **`torch.Tensor`** of shape `[batch_size, num_voxels]`.

### Mode 2: Decoding/Dynadiff (Sequence Mode)

The underlying **`NSDDataset`** object supports retrieval of the full 6-TR sequence and the corresponding stimulus image for a stimulus presentation event.

  * Use the **`.get_sequence_item(idx)`** method directly on the `dataset` object (not the `DataLoader`).

<!-- end list -->

```python
# Assuming 'dataset' is loaded...
sequence = dataset.get_sequence_item(5000)

if sequence:
    fmri_sequence = sequence["brain"] # Shape: [6, num_voxels]
    stimulus_image = sequence["img"] # Shape: [3, 425, 425]
    
    print(f"6-TR sequence retrieved: {fmri_sequence.shape}")
```

### Metadata Retrieval

To retrieve fMRI data *with* its corresponding metadata (e.g., session, run, time, stimulus history), set **`return_fmri_only=False`**.

  * **Output:** The `DataLoader` yields a dictionary with keys: `fmri`, `stimulus_history`, `subject_idx`, `session`, `run`, and `time`.

-----

## 🧠 Caching Workflow in Detail

The pipeline is designed to be run once per subject and once per unique ROI, leveraging multiprocessing to minimize wait times.

### 1\. Stage 1: Full-Brain Preprocessing (Intermediate HDF5 Cache)

| Step | Purpose | Output | Cost (Subj 01) |
| :--- | :--- | :--- | :--- |
| **`preprocess_full_brain()`** | Cleans raw NIfTI data: applies **detrending, standardization (z-score)**, and **HRF time-shift**. | Per-session **`.h5` files** containing full-brain fMRI data and metadata. | $\sim 25$ min, $\sim 80$ GB disk |
| **Notes** | This is a **one-time process per subject**. We use HDF5 for high-performance chunked I/O. | | |

### 2\. Stage 2: ROI Cache Generation (Final Memory-Mapped Cache)

| Step | Purpose | Output | Cost (Subj 01, General ROI) |
| :--- | :--- | :--- | :--- |
| **`load_roi_cache()`** | Loads Stage 1 data, applies the **ROI mask**, and concatenates all sessions into final arrays. | Final **`.fmri.npy`** and **`.meta.npy`** files, saved for memory-mapping. | $\sim 3$ min, $\sim 20$ GB disk |
| **Notes** | This is a **one-time process per unique ROI**. Future loads of this ROI are **instantaneous** via `numpy.mmap_mode='r'`. | | |

```
```