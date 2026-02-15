"""
Evaluation script for TransVAE model
Computes reconstruction metrics (PSNR, SSIM, LPIPS, MSE) and generates visual comparisons
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import yaml
from tqdm import tqdm
import json

from transvae import TransVAE, TransVAELoss

# Import your dataset classes
from train_working import COCODataset, get_coco_root


def parse_args():
    p = argparse.ArgumentParser("Evaluate TransVAE")
    
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--config", type=str, required=True, help="Path to model config")
    p.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for results")
    p.add_argument("--dataset", type=str, default="coco", choices=["coco"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate")
    p.add_argument("--num_visual_samples", type=int, default=16, help="Number of visual comparison samples")
    p.add_argument("--device", type=str, default="cuda")
    
    return p.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """Calculate SSIM between two images (simplified version)"""
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def create_evaluation_dataloader(args):
    """Create dataloader for evaluation"""
    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ])
    
    if args.dataset == "coco":
        coco_root = get_coco_root()
        eval_dataset = COCODataset(
            root=coco_root,
            transform=transform,
            max_samples=args.num_samples
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return dataloader


@torch.no_grad()
def evaluate_model(model, dataloader, loss_fn, device, args):
    """Evaluate model on dataset"""
    model.eval()
    
    metrics = {
        'mse': [],
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'l1': [],
        'kl': [],
    }
    
    print("Evaluating model...")
    for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        
        # Forward pass
        reconstruction, mu, logvar = model(images)
        
        # Apply sigmoid to get reconstruction in [0, 1] range
        reconstruction_sigmoid = torch.sigmoid(reconstruction)
        
        # Calculate metrics
        batch_mse = F.mse_loss(reconstruction_sigmoid, images, reduction='none').mean(dim=[1,2,3])
        metrics['mse'].extend(batch_mse.cpu().numpy().tolist())
        
        # PSNR for each image in batch
        for i in range(images.size(0)):
            psnr = calculate_psnr(reconstruction_sigmoid[i:i+1], images[i:i+1])
            metrics['psnr'].append(psnr)
            
            ssim = calculate_ssim(reconstruction_sigmoid[i:i+1], images[i:i+1])
            metrics['ssim'].append(ssim)
        
        # Calculate loss components
        losses = loss_fn(
            reconstruction.float(),
            images.float(),
            mu.float(),
            logvar.float(),
            discriminator=None,
            dinov2=None,
        )
        
        metrics['lpips'].append(losses['lpips'].item())
        metrics['l1'].append(losses['l1'].item())
        metrics['kl'].append(losses['kl'].item())
    
    # Calculate average metrics
    avg_metrics = {
        'mse': np.mean(metrics['mse']),
        'psnr': np.mean(metrics['psnr']),
        'ssim': np.mean(metrics['ssim']),
        'lpips': np.mean(metrics['lpips']),
        'l1': np.mean(metrics['l1']),
        'kl': np.mean(metrics['kl']),
    }
    
    # Calculate std
    std_metrics = {
        'mse_std': np.std(metrics['mse']),
        'psnr_std': np.std(metrics['psnr']),
        'ssim_std': np.std(metrics['ssim']),
    }
    
    return avg_metrics, std_metrics


@torch.no_grad()
def generate_visual_comparisons(model, dataloader, device, output_dir, num_samples=16):
    """Generate visual comparison between original and reconstructed images"""
    model.eval()
    
    images_list = []
    reconstructions_list = []
    
    print("Generating visual comparisons...")
    for images, _ in dataloader:
        images = images.to(device)
        reconstruction, _, _ = model(images)
        reconstruction = torch.sigmoid(reconstruction)
        
        images_list.append(images.cpu())
        reconstructions_list.append(reconstruction.cpu())
        
        if sum(len(x) for x in images_list) >= num_samples:
            break
    
    # Concatenate and take exactly num_samples
    all_images = torch.cat(images_list, dim=0)[:num_samples]
    all_reconstructions = torch.cat(reconstructions_list, dim=0)[:num_samples]
    
    # Create comparison grid
    comparison = torch.stack([all_images, all_reconstructions], dim=1)
    comparison = comparison.view(-1, *all_images.shape[1:])
    
    grid = make_grid(comparison, nrow=8, normalize=False, padding=2)
    save_image(grid, os.path.join(output_dir, "reconstruction_comparison.png"))
    
    # Save individual samples
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    for i in range(min(8, num_samples)):
        # Save original
        save_image(all_images[i], os.path.join(samples_dir, f"original_{i}.png"))
        # Save reconstruction
        save_image(all_reconstructions[i], os.path.join(samples_dir, f"reconstructed_{i}.png"))
        # Save side-by-side
        comparison_single = torch.cat([all_images[i], all_reconstructions[i]], dim=2)
        save_image(comparison_single, os.path.join(samples_dir, f"comparison_{i}.png"))
    
    print(f"Visual comparisons saved to {output_dir}")


@torch.no_grad()
def generate_random_samples(model, device, output_dir, num_samples=16, latent_dim=32):
    """Generate random samples from the latent space"""
    model.eval()
    
    print("Generating random samples from latent space...")
    
    # Sample from standard normal distribution
    z = torch.randn(num_samples, latent_dim, device=device)
    
    # Decode
    if hasattr(model, 'module'):
        decoder = model.module.decoder
    else:
        decoder = model.decoder
    
    samples = decoder(z)
    samples = torch.sigmoid(samples)
    
    # Save grid
    grid = make_grid(samples.cpu(), nrow=4, normalize=False, padding=2)
    save_image(grid, os.path.join(output_dir, "random_samples.png"))
    
    print(f"Random samples saved to {output_dir}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    
    print(f"Loading model from {args.checkpoint}...")
    
    # Create model
    model = TransVAE(
        config=model_cfg,
        variant=model_cfg.get("variant"),
        compression_ratio=model_cfg.get("compression_ratio"),
        latent_dim=model_cfg.get("latent_dim"),
        use_rope=model_cfg.get("use_rope", True),
        use_conv_ffn=model_cfg.get("use_conv_ffn", True),
        use_dc_path=model_cfg.get("use_dc_path", True),
    ).to(args.device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Create loss function for LPIPS calculation
    loss_fn = TransVAELoss(
        l1_weight=1.0,
        lpips_weight=1.0,
        kl_weight=1e-8,
        vf_weight=0.0,
        gan_weight=0.0,
        use_gan=False,
    ).to(args.device)
    
    # Create dataloader
    dataloader = create_evaluation_dataloader(args)
    
    # Evaluate
    avg_metrics, std_metrics = evaluate_model(model, dataloader, loss_fn, args.device, args)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of samples evaluated: {args.num_samples}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print("-"*50)
    print(f"MSE:   {avg_metrics['mse']:.6f} ± {std_metrics['mse_std']:.6f}")
    print(f"PSNR:  {avg_metrics['psnr']:.2f} dB ± {std_metrics['psnr_std']:.2f}")
    print(f"SSIM:  {avg_metrics['ssim']:.4f} ± {std_metrics['ssim_std']:.4f}")
    print(f"LPIPS: {avg_metrics['lpips']:.4f}")
    print(f"L1:    {avg_metrics['l1']:.4f}")
    print(f"KL:    {avg_metrics['kl']:.6f}")
    print("="*50)
    
    # Save metrics to JSON
    results = {
        'checkpoint': args.checkpoint,
        'epoch': checkpoint['epoch'],
        'num_samples': args.num_samples,
        'metrics': avg_metrics,
        'std': std_metrics,
    }
    
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nMetrics saved to {os.path.join(args.output_dir, 'metrics.json')}")
    
    # Generate visual comparisons
    generate_visual_comparisons(
        model, dataloader, args.device, args.output_dir, 
        num_samples=args.num_visual_samples
    )
    
    # Generate random samples
    generate_random_samples(
        model, args.device, args.output_dir,
        num_samples=16,
        latent_dim=model_cfg.get("latent_dim", 32)
    )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()