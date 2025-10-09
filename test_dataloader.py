import argparse
import time
import cProfile
import pstats
import io

import torch
from torch.utils.data import DataLoader

# Assuming 'overcomplete' is a library you have
from overcomplete.sae import TopKSAE, train_sae

# Import the new data loading class (NSDDataModule is no longer needed)
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
    print(f"  Top-K:              {args.top_k}")
    print(f"  Learning Rate:      {args.lr}")
    print(f"  NSD Data Path:      '{args.nsddata_path}'")
    print(f"  ROI Path:           '{args.roi_path}'")
    print("----------------------------\n")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Setup Data using the config and a standard DataLoader
    config = NsdSaeDataConfig(nsddata_path=args.nsddata_path, subject_id=args.subject_id)
    
    print("🚀 Initializing and building dataset...")
    # Directly build the dataset using the configuration object
    dataset = config.build(roi=args.roi_path)
    print("✅ Dataset initialized.")
    
    # Create a standard PyTorch DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True  # Good practice for GPU training
    )
    print(f"✅ DataLoader created with {len(dataset)} samples.")

    # 2. Setup Model
    # Get the input dimension directly from the created dataset
    input_dim = dataset[0]['fmri'].shape[0]
    nb_concepts = int(input_dim * args.expansion_factor)
    
    model = TopKSAE(
        input_dim,
        nb_concepts=nb_concepts,
        top_k=args.top_k,
        device=device
    ).float()
    
    print(f"✅ TopKSAE Model initialized with input_dim={input_dim}, nb_concepts={nb_concepts}, top_k={args.top_k}")

    # 3. Setup Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # This helper adapts our dictionary-based loader to the tuple-based format train_sae expects
    def get_tensor_iterator(dataloader, device):
        for batch in dataloader:
            yield (batch['fmri'].to(device),)

    tensor_loader = get_tensor_iterator(train_loader, device)
    
    # 4. Run Profiling and Training
    print(f"\n🔥 Starting training for {args.epochs} epoch(s)...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.perf_counter()
    
    train_sae(
        model,
        tensor_loader,
        criterion,
        optimizer,
        nb_epochs=args.epochs,
        device=device
    )
    
    total_time = time.perf_counter() - start_time
    
    profiler.disable()
    
    print("\n🏁 Training complete.")

    # 5. Report Results
    total_samples = len(dataset) * args.epochs
    throughput = total_samples / total_time
    print(f"\n--- ⏱️ Throughput ---")
    print(f"  Processed {total_samples} samples in {total_time:.2f} seconds.")
    print(f"  Overall Throughput: {throughput:.2f} samples/sec.")
    print("------------------------")
    
    print("\n--- 🔬 cProfile Report (Top 15 functions) ---")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(15)
    print(s.getvalue())
    print("---------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile the NSD SAE DataLoader with a TopKSAE model.")
    # Data loading arguments
    parser.add_argument("--nsddata_path", type=str, default = "nsddata", help="Path to the root of the NSDdata directory.")
    parser.add_argument("--roi_path", type=str, default = "nsddata/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz", help="Path to the .nii.gz ROI file.")
    parser.add_argument("--subject_id", type=int, default=1, help="Subject ID to process (e.g., 1).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for the DataLoader.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading.")
    
    # Model and Training arguments
    parser.add_argument("--expansion_factor", type=float, default=0.01, help="Factor to determine nb_concepts (nb_concepts = input_dim * expansion_factor).")
    parser.add_argument("--top_k", type=int, default=32, help="Number of concepts to activate (k in Top-K).")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the Adam optimizer.")
    
    args = parser.parse_args()
    main(args)