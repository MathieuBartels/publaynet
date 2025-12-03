import torch
import argparse
import os
from PIL import Image
import numpy as np

from unet_model import UNet
from segformer_model import SegFormerB1
from mamba_model import MambaSegmentation
from utils import predict_image, visualize_prediction, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained UNet")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save output mask"
    )
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Image size [height width]",
    )
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "segformer-b1", "mamba"],
        help="Model architecture to use (unet, segformer-b1, or mamba)",
    )

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if args.model == "segformer-b1":
        print("Using SegFormer-B1 model")
        model = SegFormerB1(
            n_channels=3, n_classes=args.num_classes, pretrained=False
        ).to(device)
    elif args.model == "mamba":
        print("Using Mamba segmentation model")
        model = MambaSegmentation(
            n_channels=3,
            n_classes=args.num_classes,
            img_size=tuple(args.image_size),
            patch_size=4,
            embed_dim=128,
            depth=4,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to(device)
    else:
        print("Using UNet model")
        model = UNet(n_channels=3, n_classes=args.num_classes, bilinear=True).to(device)
    
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    print(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')

    # Predict
    pred_mask = predict_image(
        model, args.image, device, args.num_classes, tuple(args.image_size)
    )

    # Save output
    if args.output:
        pred_image = Image.fromarray((pred_mask * 255).astype(np.uint8))
        pred_image.save(args.output)
        print(f"Saved prediction to {args.output}")

    # Visualize
    if args.visualize:
        image = Image.open(args.image).convert("RGB")
        image_array = np.array(
            image.resize(tuple(args.image_size), Image.Resampling.BILINEAR)
        )
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

        # Create dummy ground truth (zeros) for visualization
        dummy_mask = np.zeros((args.image_size[0], args.image_size[1]), dtype=np.uint8)

        visualize_prediction(model, image_tensor, dummy_mask, device, args.num_classes)


if __name__ == "__main__":
    main()
