import argparse
import time
import cProfile
import pstats
import io

import torch
from torch.utils.data import DataLoader

# Assuming 'overcomplete' is a library you have
from overcomplete.sae import TopKSAE, train_sae

# Import the optimized data loading class
from nsd_data_loader import NsdSaeDataConfig

def criterion(x, x_hat, pre_codes, codes, dictionary):
    """A simple MSE loss for reconstruction."""
    mse = (x - x_hat).square().mean()
    return mse

def main(args):
    """Main function to setup and run the profiling."""
    print("--- ⚙️ Test Configuration ---")
    print(f"  Subject ID:         {args.subject_id}")
    print(f"  Batch Size:         {args.batch_size}")
    print(f"  Num Workers:        {args.workers}")
    print(f"  Training Epochs:    {args.epochs}")
    print(f"  Expansion Factor:   {args.expansion_factor}")
    print(f"  Sparsity:           {args.sparsity}")
    print(f"  Learning Rate:      {args.lr}")
    print(f"  NSD Data Path:      '{args.nsddata_path}'")
    print(f"  ROI Path:           '{args.roi_path}'")
    print(f"  Chunk Size:         {args.chunk_size}")
    print(f"  HDF5 Chunk Rows:    {args.hdf5_chunk_rows}")
    print("----------------------------\n")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Setup Data using the optimized config
    config = NsdSaeDataConfig(
        nsddata_path=args.nsddata_path,
        subject_id=args.subject_id,
        offset=args.offset,
        history_length=args.history_length,
        chunk_size=args.chunk_size,
        hdf5_chunk_rows=args.hdf5_chunk_rows
    )
    
    print("Initializing and building dataset...")
    cache_build_start = time.perf_counter()
    
    # Build the dataset - this will create/load the cache
    dataset = config.load_roi_NSDDataset(
        roi=args.roi_path,
        return_fmri_only=True
    )
    
    cache_build_time = time.perf_counter() - cache_build_start
    print(f"Dataset initialized in {cache_build_time:.2f}s")
    
    # 2. Get input dimension from the dataset
    input_dim = dataset.fmri_data.shape[1] # Read from .npy shape
    print(f"   Input dimension: {input_dim} voxels")
    
    # 3. Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=(device == 'cuda'),
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None
    )
    print(f"DataLoader created with {len(dataset)} samples.")

    # 4. Setup Model
    nb_concepts = int(input_dim * args.expansion_factor)
    top_k = int(nb_concepts * args.sparsity)
    
    model = TopKSAE(
        input_dim,
        nb_concepts=nb_concepts,
        top_k=top_k,
        device=device
    )

    
    print(f"TopKSAE Model initialized:")
    print(f"   - Input dim: {input_dim}")
    print(f"   - Concepts: {nb_concepts}")
    print(f"   - Top-k: {top_k}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Setup Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 6. Warm-up iteration to ensure everything is loaded
    if args.warmup:
        print("\nRunning warm-up iteration...")
        warmup_start = time.perf_counter()
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= 5:  # Just do a few batches
                    break
                batch = batch.to(device)
                _ = model(batch)
        warmup_time = time.perf_counter() - warmup_start
        print(f"Warm-up complete in {warmup_time:.2f}s")
    
    # 7. Profile training
    print("\nStarting training with profiling...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    training_start = time.perf_counter()
    
    train_sae(
        model,
        train_loader,
        criterion,
        optimizer,
        nb_epochs=args.epochs,
        device=device
    )
    
    training_time = time.perf_counter() - training_start
    
    profiler.disable()
    
    print("\nTraining complete.")

    # 8. Report Results
    total_samples = len(dataset) * args.epochs
    throughput = total_samples / training_time
    samples_per_epoch = len(dataset)
    time_per_epoch = training_time / args.epochs
    
    print(f"\n--- Performance Metrics ---")
    print(f"  Cache build time:        {cache_build_time:.2f}s")
    print(f"  Training time:         {training_time:.2f}s")
    print(f"  Time per epoch:          {time_per_epoch:.2f}s")
    print(f"  Total samples processed: {total_samples:,}")
    print(f"  Samples per epoch:       {samples_per_epoch:,}")
    print(f"  Overall throughput:      {throughput:.2f} samples/sec")
    print(f"  Throughput per epoch:    {samples_per_epoch/time_per_epoch:.2f} samples/sec")
    
    # Calculate batches per second
    total_batches = (samples_per_epoch // args.batch_size) * args.epochs
    batches_per_sec = total_batches / training_time
    print(f"  Batches per second:      {batches_per_sec:.2f}")
    print("------------------------------")
    
    # 9. Memory usage report (if CUDA)
    if device == 'cuda':
        print(f"\n--- GPU Memory Usage ---")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print("---------------------------")
    
    # 10. Profile report
    if args.profile:
        print("\n--- cProfile Report (Top 20 functions) ---")
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        print(s.getvalue())
        print("---------------------------------------------")
    
    # 11. Clean up
    dataset.close()
    print("\nCleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile the optimized NSD SAE DataLoader with a TopKSAE model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data loading arguments
    parser.add_argument(
        "--nsddata_path",
        type=str,
        default="nsddata",
        help="Path to the root of the NSDdata directory."
    )
    parser.add_argument(
        "--roi_path",
        type=str,
        default="nsddata/nsddata/ppdata/subj01/func1pt8mm/roi/streams.nii.gz",
        help="Path to the .nii.gz ROI file."
    )
    parser.add_argument(
        "--subject_id",
        type=int,
        default=1,
        help="Subject ID to process (e.g., 1)."
    )
    
    # DataLoader arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for the DataLoader."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading."
    )
    
    # Cache optimization arguments
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of samples per chunk during cache creation (higher = faster but more RAM)."
    )
    parser.add_argument(
        "--hdf5_chunk_rows",
        type=int,
        default=100,
        help="HDF5 chunk size for I/O optimization (tune based on batch size)."
    )
    
    # Data preprocessing arguments
    parser.add_argument(
        "--offset",
        type=float,
        default=4.6,
        help="Hemodynamic response offset in seconds."
    )
    parser.add_argument(
        "--history_length",
        type=int,
        default=4,
        help="Number of previous stimulus images to store in history."
    )
    
    # Model and Training arguments
    parser.add_argument(
        "--expansion_factor",
        type=float,
        default=0.1,
        help="Factor to determine nb_concepts (nb_concepts = input_dim * expansion_factor)."
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.1,
        help="Sparsity ratio for top-k activation (k = nb_concepts * sparsity)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train for."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the Adam optimizer."
    )
    
    # Profiling arguments
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed cProfile output."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warm-up batches before profiling (0 = no warmup)."
    )
    
    args = parser.parse_args()
    main(args)