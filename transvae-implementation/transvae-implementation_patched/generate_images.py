"""
Generate images from TransVAE latent space
Supports: random generation, interpolation, and reconstruction
"""

import os
import argparse
import torch
import yaml
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from PIL import Image
import numpy as np

from transvae import TransVAE


def parse_args():
    p = argparse.ArgumentParser("Generate images with TransVAE")
    
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--config", type=str, required=True, help="Path to model config")
    p.add_argument("--output_dir", type=str, default="generated_images", help="Output directory")
    p.add_argument("--mode", type=str, default="random", 
                   choices=["random", "interpolate", "reconstruct"],
                   help="Generation mode")
    
    # Random generation
    p.add_argument("--num_samples", type=int, default=16, help="Number of random samples")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    # Interpolation
    p.add_argument("--num_interpolations", type=int, default=10, help="Steps in interpolation")
    
    # Reconstruction
    p.add_argument("--input_image", type=str, default=None, help="Path to input image for reconstruction")
    p.add_argument("--resolution", type=int, default=256, help="Image resolution")
    
    p.add_argument("--device", type=str, default="cuda")
    
    return p.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, config_path, device):
    """Load trained TransVAE model"""
    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})
    
    print(f"Loading model from {checkpoint_path}...")
    
    model = TransVAE(
        config=model_cfg,
        variant=model_cfg.get("variant"),
        compression_ratio=model_cfg.get("compression_ratio"),
        latent_dim=model_cfg.get("latent_dim"),
        use_rope=model_cfg.get("use_rope", True),
        use_conv_ffn=model_cfg.get("use_conv_ffn", True),
        use_dc_path=model_cfg.get("use_dc_path", True),
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    return model, model_cfg


@torch.no_grad()
def generate_random_samples(model, latent_dim, num_samples, device, seed=None):
    """Generate random samples from latent space"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    print(f"Generating {num_samples} random samples...")
    
    # Get spatial dimensions from model config
    # For f16 compression, latent spatial size should be H/16 x W/16
    # Assuming 256x256 input -> 16x16 latent spatial
    if hasattr(model, 'module'):
        encoder = model.module.encoder
    else:
        encoder = model.encoder
    
    # Infer spatial size from compression ratio
    # f16 means 256/16 = 16x16 spatial latent
    spatial_size = 16  # For f16 compression on 256x256 images
    
    # Sample from standard normal with spatial dimensions
    z = torch.randn(num_samples, latent_dim, spatial_size, spatial_size, device=device)
    
    # Decode
    if hasattr(model, 'module'):
        decoder = model.module.decoder
    else:
        decoder = model.decoder
    
    samples = decoder(z)
    samples = torch.sigmoid(samples)  # Convert to [0, 1]
    
    return samples


@torch.no_grad()
def interpolate_latents(model, latent_dim, num_steps, device, seed=None):
    """Interpolate between two random latent vectors"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    print(f"Generating interpolation with {num_steps} steps...")
    
    # Infer spatial size from compression ratio
    spatial_size = 16  # For f16 compression on 256x256 images
    
    # Sample two random latent vectors with spatial dimensions
    z1 = torch.randn(1, latent_dim, spatial_size, spatial_size, device=device)
    z2 = torch.randn(1, latent_dim, spatial_size, spatial_size, device=device)
    
    # Create interpolation
    alphas = torch.linspace(0, 1, num_steps, device=device)
    interpolated_samples = []
    
    if hasattr(model, 'module'):
        decoder = model.module.decoder
    else:
        decoder = model.decoder
    
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        sample = decoder(z_interp)
        sample = torch.sigmoid(sample)
        interpolated_samples.append(sample)
    
    interpolated_samples = torch.cat(interpolated_samples, dim=0)
    return interpolated_samples


@torch.no_grad()
def reconstruct_image(model, image_path, resolution, device):
    """Reconstruct an input image"""
    print(f"Reconstructing image from {image_path}...")
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Encode and decode
    reconstruction, mu, logvar = model(image_tensor)
    reconstruction = torch.sigmoid(reconstruction)
    
    # Return original and reconstruction side by side
    comparison = torch.cat([image_tensor, reconstruction], dim=3)  # Concatenate horizontally
    
    return comparison, image_tensor, reconstruction


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, model_cfg = load_model(args.checkpoint, args.config, args.device)
    latent_dim = model_cfg.get("latent_dim", 32)
    
    if args.mode == "random":
        # Generate random samples
        samples = generate_random_samples(
            model, latent_dim, args.num_samples, args.device, args.seed
        )
        
        # Save as grid
        grid = make_grid(samples, nrow=4, normalize=False, padding=2)
        output_path = os.path.join(args.output_dir, "random_samples.png")
        save_image(grid, output_path)
        print(f"✅ Saved random samples to {output_path}")
        
        # Save individual images
        for i, sample in enumerate(samples):
            individual_path = os.path.join(args.output_dir, f"sample_{i:03d}.png")
            save_image(sample, individual_path)
        
        print(f"✅ Saved {args.num_samples} individual samples to {args.output_dir}")
    
    elif args.mode == "interpolate":
        # Generate interpolation
        interpolated = interpolate_latents(
            model, latent_dim, args.num_interpolations, args.device, args.seed
        )
        
        # Save as grid
        grid = make_grid(interpolated, nrow=args.num_interpolations, normalize=False, padding=2)
        output_path = os.path.join(args.output_dir, "interpolation.png")
        save_image(grid, output_path)
        print(f"✅ Saved interpolation to {output_path}")
        
        # Save as individual frames
        for i, sample in enumerate(interpolated):
            frame_path = os.path.join(args.output_dir, f"interp_frame_{i:03d}.png")
            save_image(sample, frame_path)
        
        print(f"✅ Saved {args.num_interpolations} interpolation frames to {args.output_dir}")
    
    elif args.mode == "reconstruct":
        if args.input_image is None:
            raise ValueError("--input_image is required for reconstruct mode")
        
        # Reconstruct image
        comparison, original, reconstruction = reconstruct_image(
            model, args.input_image, args.resolution, args.device
        )
        
        # Save comparison
        comparison_path = os.path.join(args.output_dir, "reconstruction_comparison.png")
        save_image(comparison, comparison_path)
        print(f"✅ Saved comparison to {comparison_path}")
        
        # Save original and reconstruction separately
        save_image(original, os.path.join(args.output_dir, "original.png"))
        save_image(reconstruction, os.path.join(args.output_dir, "reconstructed.png"))
        print(f"✅ Saved original and reconstructed images to {args.output_dir}")
    
    print("\n🎉 Generation complete!")


if __name__ == "__main__":
    main()