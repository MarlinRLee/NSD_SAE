# NSD Data Loader for PyTorch

This repository provides a flexible and efficient dataloader for the Natural Scenes Dataset (NSD), designed for PyTorch-based machine learning. It features a robust two-stage caching system to dramatically speed up data loading after the initial setup.

The dataloader supports two primary modes:

1.  **Time-step Mode**: Ideal for training models like Sparse Autoencoders (SAEs), where each sample is an individual fMRI time-step.
2.  **Sequence Mode**: Designed for brain decoding tasks, allowing you to retrieve the full 6-TR fMRI response sequence corresponding to a single stimulus image.

-----

## How to Use

### 1\. Prerequisites and Data Download

First, clone the repository and download the necessary NSD data.

1.  **Install dependencies:**
    ```bash
    pip install torch pydantic pandas numpy nibabel nilearn h5py Pillow
    ```
2.  **Download NSD data:**
    You will need to download the preprocessed fMRI time-series data (`nsddata_timeseries`), the stimulus images (`nsd_stimuli.hdf5`), and the ROI masks from the [official NSD data repository](https://naturalscenesdataset.org/). Organize them into a main `nsddata` directory.

### 2\. Running Your Analysis

You will primarily interact with the `NsdSaeDataConfig` class to configure and build your dataset. The caching process is handled automatically.

#### Example 1: Loading Individual fMRI Time-steps (for SAEs)

This is the default behavior, optimized for speed by returning only fMRI data.

```python
from nsd_data_loader import NsdSaeDataConfig
from torch.utils.data import DataLoader

# 1. Configure the data handler
config = NsdSaeDataConfig(
    nsddata_path="/path/to/your/nsddata/", # Directory where you downloaded the data
    subject_id=1,
)

# 2. Define the path to your desired ROI
# This can be a specific ROI like 'V1' or the whole brain 'general' mask
roi_path = "nsddata/ppdata/subj01/func1pt8mm/roi/general.nii.gz"

# 3. Build the dataset. This triggers caching on the first run.
# By default, `return_fmri_only=True` for maximum efficiency.
print("Building dataset for SAE training...")
dataset = config.build(roi=roi_path, return_fmri_only=True)
print(f"Dataset ready with {len(dataset)} samples.")

# 4. Create a standard PyTorch DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=4096,
    num_workers=8,
    shuffle=True
)

# 5. Use the dataloader in your training loop
# The batch is a single tensor of fMRI data.
first_batch = next(iter(train_loader))

print("Batch type:", type(first_batch))
print("fMRI data batch shape:", first_batch.shape) # Shape: [batch_size, num_voxels_in_roi]
```

#### Example 2: Retrieving Full fMRI Sequences and Images (for Decoding)

The `NSDDataset` object includes, `get_sequence_item(idx)`, which allows you to retrieve the 6-TR brain response dynadiff was trained to predict and the corresponding stimulus image for any time-step within that response window.

```python
from nsd_data_loader import NsdSaeDataConfig
import matplotlib.pyplot as plt

# Build the dataset as before
config = NsdSaeDataConfig(nsddata_path="/path/to/your/nsddata/", subject_id=1)
roi_path = "nsddata/ppdata/subj01/func1pt8mm/roi/general.nii.gz"
dataset = config.build(roi=roi_path)

# Use the .get_sequence_item() method on the dataset object
# Let's get the sequence corresponding to the 5000th time-step in the data
sequence_item = dataset.get_sequence_item(5000)


# The method returns a dictionary with the brain scan sequence and the image
fmri_sequence = sequence_item["brain"]
stimulus_image = sequence_item["img"]
subject_id = sequence_item["subject_idx"]

print("--- Sequence Item ---")
print(f"Subject Index: {subject_id.item()}")
print(f"fMRI sequence shape: {fmri_sequence.shape}") # Shape: [6, num_voxels_in_roi]
print(f"Stimulus image shape: {stimulus_image.shape}") # Shape: [3, 425, 425]

# Display the stimulus image
plt.imshow(stimulus_image.permute(1, 2, 0))
plt.title("Stimulus Image")
plt.axis('off')
plt.show()

```

-----

## The Caching Workflow Explained 🧠

The data loader uses a two-stage caching system to avoid reprocessing data.

**Stage 1: Initial Preprocessing (Full Brain)**

  * **Trigger:** Runs automatically if the preprocessed cache directory (`.cache/subj{ID}_preprocessed/`) is not found.
  * **Process:** Iterates through every NIfTI file for the subject, applies detrending and z-score standardization, and saves the cleaned, full-brain data for each session into its own `.pt` file.
  * **Cost:** This is a one-time process per subject.
      * **Time:** \~3 hours
      * **Disk Space:** \~250 GB
      * **Peak RAM:** \~10 GB

**Stage 2: ROI Cache Generation**

  * **Trigger:** Runs if the preprocessed data exists, but a cache for the **specific requested ROI** does not.
  * **Process:** Loads the preprocessed full-brain data session-by-session, applies the ROI mask to extract only the relevant voxels, and saves all samples into a **single, final cache file**.
  * **Cost:** This is a one-time process per ROI.
      * **Time:** \~15 minutes (for the large `general` ROI)
      * **Peak RAM:** \~25 GB

Once an ROI-specific cache is built, any future experiment using that same ROI will load almost instantly. ✅

-----

## Data Sample Format

The format of the data yielded by the `DataLoader` depends on the `return_fmri_only` flag set during the `config.build()` call.

#### Default: `return_fmri_only=True`

The `DataLoader` yields a single **torch.Tensor**.

  * **Shape**: `(batch_size, num_voxels)`
  * **Content**: The fMRI brain data for one time-step, filtered to the ROI and flattened.

#### Dictionary Mode: `return_fmri_only=False`

If you need metadata alongside the fMRI data, set the flag to `False`. The `DataLoader` will yield a dictionary with the following keys for each time-step:

  * `fmri`: A `torch.Tensor` containing the fMRI data.
  * `stimulus_history`: A `torch.Tensor` of shape `(history_length,)` with the indices of the last N images shown.
  * `subject_idx`: The integer index of the subject (e.g., `1`).
  * `session`: The session number.
  * `run`: The run number within the session.
  * `time`: The time-step index within the run.