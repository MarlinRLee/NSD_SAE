#!/bin/bash -l
#SBATCH --time=10:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30g
#SBATCH --tmp=20g
#SBATCH --gres=gpu:a100:1
#SBATCH -p a100-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu

module purge

# Load CUDA
module load cuda/12.0

# Explicitly ensure the newer GCC's bin directory is at the very front of PATH
export PATH="/common/software/install/migrated/gcc/9.2.0/bin:$PATH"
export LD_LIBRARY_PATH="/common/software/install/migrated/gcc/9.2.0/lib64:$LD_LIBRARY_PATH"

export TRITON_CACHE_DIR="/scratch.local/lee02328/.triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

# Set HDF5 performance environment variables
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

# Use tmpfs for faster I/O if available
export TMPDIR=/scratch.local/lee02328
mkdir -p "$TMPDIR"

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh

conda activate dynadiff_final3

cd /users/9/lee02328/Ada_Comp/NSDSAE/

echo "=================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=================================="

# Optimized test with GPU and tuned parameters
python test_dataloader.py \
  --nsddata_path nsddata \
  --roi_path nsddata/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz \
  --subject_id 1 \
  --batch_size 512 \
  --workers 8 \
  --chunk_size 2000 \
  --hdf5_chunk_rows 256 \
  --epochs 10 \
  --warmup 5\
  --profile

echo "=================================="
echo "Job finished at: $(date)"
echo "=================================="