#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=30g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1

module purge

# Load the CUDA module first (which is tied to GCC 7.2.0)
module load cuda/12.0

# Explicitly ensure the newer GCC's bin directory is at the very front of PATH.
export PATH="/common/software/install/migrated/gcc/9.2.0/bin:$PATH"

# Explicitly add the newer GCC's lib64 directory to LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="/common/software/install/migrated/gcc/9.2.0/lib64:$LD_LIBRARY_PATH"

export TRITON_CACHE_DIR="/scratch.local/lee02328/.triton_cache"

mkdir -p "$TRITON_CACHE_DIR"


source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh


conda info --envs

conda activate dynadiff_final3

cd /users/9/lee02328/Ada_Comp/NSDSAE/

python test_dataloader.py