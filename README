# NSD Data Loader for SAE Training

This repository provides a set of tools for efficiently loading and preprocessing the Natural Scenes Dataset (NSD). It is designed to prepare individual fMRI time-steps for training models like Sparse Autoencoders (SAEs) in a standard PyTorch environment.

-----

## How to Use

### 1\. Prerequisites and Data Download

First, clone the repository and set up your environment. You will then need to download the necessary NSD data.

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Download NSD data:**
    Run the provided script to download the required timeseries and stimulus data. This can take over 2 hours and requires significant disk space.
    ```bash
    python prepare_data.py
    ```

### 2\. Running Your Analysis

To use the data loader, you'll import `NsdSaeDataConfig` to build a `Dataset` object, which you can then pass to a standard `torch.utils.data.DataLoader`. The caching process is triggered automatically the first time you build the dataset with a specific ROI.

**Example Usage:**

```python
from nsd_data_loader import NsdSaeDataConfig
from torch.utils.data import DataLoader

# 1. Configure the data handler
config = NsdSaeDataConfig(
    nsddata_path="/path/to/your/nsddata/", # Directory where you downloaded the data
    subject_id=1,
    offset=4.6,#this was what dynadiff had so should probably not be changed
    history_length=4
)

# 2. Define the path to your desired ROI
roi_path = "nsddata/ppdata/subj01/func1pt8mm/roi/general.nii.gz"

# 3. Build the dataset. This will trigger the caching process on the first run.
print("Building dataset...")
dataset = config.build(roi=roi_path)
print(f"Dataset ready with {len(dataset)} samples.")

# 4. Create a standard PyTorch DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=4096,
    num_workers=8,
    shuffle=True
)

# 5. Use the dataloader in your training loop
first_batch = next(iter(train_loader))

print("Batch keys:", first_batch.keys())
print("fMRI data shape:", first_batch["fmri"].shape)
```

-----

## The Caching Workflow Explained

The data loading process has two main stages of caching

**Stage 1: Initial Preprocessing (Full Brain)**

  * **Trigger:** This runs automatically if the preprocessed cache directory (`.cache/subj{subject_id}_preprocessed/`) is not found.
  * **Process:** It iterates through every session's NIfTI files, applies detrending and standardization, and saves the cleaned full-brain data for each session into its own `.pt` file.
  * **Time/Resource Cost:**
      * **Time:** \~3 hours of CPU time.
      * **Disk Space:** \~250 GB.
      * **Peak RAM:** \~10 GB.

**Stage 2: ROI Cache Generation**

  * **Trigger:** Runs if the preprocessed data exists, but a cache for the *specific requested ROI* does not.
  * **Process:** It loads the preprocessed full-brain data session-by-session, applies the ROI mask to extract the relevant voxels, and saves all samples into a **single, final cache file**.
  * **Time/Resource Cost:**
      * **Time:** \~15 minutes for the largest ROIs (e.g., `general`).
      * **Peak RAM:** \~25 GB.

Once the ROI-specific cache is built, any future experiment using that same ROI will load almost instantly

-----

## Data Sample Format

Each sample fetched from the `DataLoader` is a dictionary with the following keys:

  * `fmri`: A `torch.Tensor` containing the fMRI brain data for one time-step, filtered to the ROI and flattened.
  * `stimulus_history`: A `torch.Tensor` of shape `(history_length,)` containing the indices of the last N images shown to the subject (offset by the hemodynamic response).
  * `time_since_last_stimulus`: A `torch.Tensor` indicating how many TRs have passed since the most recent stimulus was presented.
  * `subject_idx`: The integer index of the subject (e.g., `1`).