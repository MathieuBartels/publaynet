import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

matplotlib.use("Agg")  # Use non-interactive backend

from dataset import BinaryPublayNetDataset, PublayNetDataset
from segformer_model import SegFormerB1
from unet_model import UNet
from mamba_model import MambaSegmentation


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Apply softmax for multi-class, sigmoid for binary
        if predictions.size(1) > 1:
            predictions = torch.softmax(predictions, dim=1)
        else:
            predictions = torch.sigmoid(predictions)

        # One-hot encode targets if needed
        if predictions.size(1) > 1:
            targets_one_hot = torch.zeros_like(predictions)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            targets = targets_one_hot

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Dice Loss"""

    def __init__(self, ce_weight=0.5, dice_weight=0.5, num_classes=2):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes

        # Use CrossEntropyLoss for multi-class segmentation (including binary with 2 channels)
        # BCEWithLogitsLoss would be for single-channel binary output
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, predictions, targets):
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.ce_weight * ce + self.dice_weight * dice


def calculate_iou(predictions, targets, num_classes):
    """Calculate IoU for each class"""
    ious = []
    predictions = torch.argmax(predictions, dim=1)

    for cls in range(num_classes):
        pred_cls = predictions == cls
        target_cls = targets == cls

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append((intersection / union).item())

    return ious


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    max_batches=None,
    log_interval=10,
    global_step=0,
    use_wandb=False,
    val_check_interval=None,
    val_loader=None,
    num_classes=2,
    max_val_batches=None,
):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, masks) in enumerate(pbar):
        # Limit batches if specified
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        global_step += 1

        pbar.set_postfix({"loss": loss.item()})

        # Log more frequently during training
        if use_wandb and (batch_idx + 1) % log_interval == 0:
            wandb.log(
                {
                    "train_loss_batch": loss.item(),
                    "global_step": global_step,
                },
                step=global_step,
            )

        # Validate during training if val_check_interval is set
        if (
            val_check_interval is not None
            and val_loader is not None
            and (batch_idx + 1) % val_check_interval == 0
        ):
            model.eval()
            val_loss, mean_iou, _ = validate(
                model,
                val_loader,
                criterion,
                device,
                num_classes,
                return_samples=False,
                max_batches=max_val_batches,
            )
            model.train()
            if use_wandb:
                wandb.log(
                    {
                        "val_loss_mid_epoch": val_loss,
                        "mean_iou_mid_epoch": mean_iou,
                        "global_step": global_step,
                    },
                    step=global_step,
                )
            print(
                f"  Validation (batch {batch_idx + 1}, step {global_step}): Val Loss: {val_loss:.4f}, Mean IoU: {mean_iou:.4f}"
            )

    return running_loss / num_batches, global_step


def validate(
    model,
    dataloader,
    criterion,
    device,
    num_classes,
    return_samples=False,
    max_batches=None,
):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_ious = []
    sample_images = None
    sample_masks = None
    sample_predictions = None

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(
            tqdm(dataloader, desc="Validating")
        ):
            # Limit batches if specified
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()
            num_batches += 1

            # Calculate IoU
            ious = calculate_iou(outputs, masks, num_classes)
            all_ious.append(ious)

            # Store first batch samples for visualization
            if return_samples and batch_idx == 0:
                sample_images = images[:4].cpu()  # Take first 4 samples
                sample_masks = masks[:4].cpu()
                sample_predictions = torch.argmax(outputs[:4], dim=1).cpu()

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_ious = (
        np.nanmean(all_ious, axis=0) if len(all_ious) > 0 else [0.0] * num_classes
    )
    mean_iou = np.nanmean(avg_ious)

    if return_samples:
        return (
            avg_loss,
            mean_iou,
            avg_ious,
            sample_images,
            sample_masks,
            sample_predictions,
        )
    return avg_loss, mean_iou, avg_ious


def main():
    parser = argparse.ArgumentParser(
        description="Train UNet for PublayNet Segmentation"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes (2 for binary, 6 for multi-class)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Image size [height width]",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs", help="Directory for tensorboard logs"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--binary", action="store_true", help="Use binary segmentation")
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "segformer-b1", "mamba"],
        help="Model architecture to use (unet, segformer-b1, or mamba)",
    )
    parser.add_argument(
        "--mamba-variant",
        type=str,
        default="enc",
        choices=["enc", "bot"],
        help="U-Mamba variant: 'enc' for encoder variant, 'bot' for bottleneck variant",
    )
    parser.add_argument(
        "--mamba-d-state",
        type=int,
        default=8,
        help="Mamba state dimension (default: 8)",
    )
    parser.add_argument(
        "--mamba-d-conv",
        type=int,
        default=4,
        help="Mamba convolution dimension (default: 4)",
    )
    parser.add_argument(
        "--mamba-expand",
        type=float,
        default=2,
        help="Mamba expansion factor (default: 2.0, reduce to 1.5 or 1.0 for smaller model)",
    )
    parser.add_argument(
        "--mamba-width-multiplier",
        type=float,
        default=1,
        help="Width multiplier for all channels (default: 1.0, use 0.5 for 4x smaller, 0.25 for 16x smaller)",
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="publaynet", help="W&B project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity/team name"
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit number of training batches per epoch (for faster feedback loop)",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit number of validation batches (for faster feedback loop)",
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        default=None,
        help="Validate every N batches (instead of every epoch). Useful for quick feedback.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log training metrics every N batches (default: 10)",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Quick validation run: 10 train batches, 5 val batches, validate every 5 batches",
    )

    args = parser.parse_args()

    # Handle fast_dev_run flag
    if args.fast_dev_run:
        args.max_train_batches = 10
        args.max_val_batches = 5
        args.val_check_interval = 5
        print("Fast dev run enabled: limiting batches for quick validation")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Dataset
    if args.binary:
        train_dataset = BinaryPublayNetDataset(
            split="train", image_size=tuple(args.image_size)
        )
        val_dataset = BinaryPublayNetDataset(
            split="validation", image_size=tuple(args.image_size)
        )
        num_classes = 2
    else:
        # For multi-class, ensure num_classes matches PublayNet (6 classes)
        # PublayNet has: background, text, title, list, table, figure
        if args.num_classes != 6:
            print(
                f"Warning: PublayNet has 6 classes, but num_classes={args.num_classes}. "
                "Using 6 classes."
            )
        train_dataset = PublayNetDataset(
            split="train", image_size=tuple(args.image_size), num_classes=6
        )
        val_dataset = PublayNetDataset(
            split="validation", image_size=tuple(args.image_size), num_classes=6
        )
        num_classes = 6

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Model
    if args.model == "segformer-b1":
        print("Using SegFormer-B1 model")
        model = SegFormerB1(n_channels=3, n_classes=num_classes, pretrained=False).to(
            device
        )
    elif args.model == "mamba":
        print(f"Using U-Mamba segmentation model (variant: {args.mamba_variant})")
        print(f"  Width multiplier: {args.mamba_width_multiplier}")
        print(f"  d_state: {args.mamba_d_state}, d_conv: {args.mamba_d_conv}, expand: {args.mamba_expand}")
        model = MambaSegmentation(
            n_channels=3,
            n_classes=num_classes,
            bilinear=True,
            variant=args.mamba_variant,
            d_state=args.mamba_d_state,
            d_conv=args.mamba_d_conv,
            expand=args.mamba_expand,
            width_multiplier=args.mamba_width_multiplier,
        ).to(device)
    else:
        print("Using UNet model")
        model = UNet(n_channels=3, n_classes=num_classes, bilinear=True).to(device)

    # Loss function
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, num_classes=num_classes)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Tensorboard
    writer = SummaryWriter(args.log_dir)

    # Initialize Weights & Biases
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "num_classes": num_classes,
                "image_size": args.image_size,
                "binary": args.binary,
                "model": args.model,
                "mamba_variant": args.mamba_variant if args.model == "mamba" else None,
                "mamba_d_state": args.mamba_d_state if args.model == "mamba" else None,
                "mamba_d_conv": args.mamba_d_conv if args.model == "mamba" else None,
                "mamba_expand": args.mamba_expand if args.model == "mamba" else None,
                "mamba_width_multiplier": args.mamba_width_multiplier if args.model == "mamba" else None,
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
            },
        )
        # Log model architecture
        wandb.watch(model, log="all", log_freq=100)
        
        # Log number of parameters once
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
        })

    # Resume from checkpoint
    start_epoch = 0
    best_iou = 0.0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_iou = checkpoint.get("best_iou", 0.0)
        global_step = checkpoint.get("global_step", 0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train (with optional mid-epoch validation)
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            max_batches=args.max_train_batches,
            log_interval=args.log_interval,
            global_step=global_step,
            use_wandb=args.use_wandb,
            val_check_interval=args.val_check_interval,
            val_loader=val_loader if args.val_check_interval is not None else None,
            num_classes=num_classes,
            max_val_batches=args.max_val_batches,
        )

        # Validate at end of epoch (with samples for visualization every 5 epochs or last epoch)
        return_samples = (epoch % 5 == 0) or (epoch == args.epochs - 1)
        if return_samples:
            (
                val_loss,
                mean_iou,
                class_ious,
                sample_images,
                sample_masks,
                sample_predictions,
            ) = validate(
                model,
                val_loader,
                criterion,
                device,
                num_classes,
                return_samples=True,
                max_batches=args.max_val_batches,
            )
        else:
            val_loss, mean_iou, class_ious = validate(
                model,
                val_loader,
                criterion,
                device,
                num_classes,
                return_samples=False,
                max_batches=args.max_val_batches,
            )

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Enhanced TensorBoard logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Metrics/Mean_IoU", mean_iou, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # Log per-class IoU
        class_names = (
            ["background", "foreground"]
            if num_classes == 2
            else ["background", "text", "title", "list", "table", "figure"]
        )
        for class_name, iou in zip(class_names, class_ious):
            if not np.isnan(iou):
                writer.add_scalar(f"IoU/Class_{class_name}", iou, epoch)

        # Log sample images to TensorBoard
        if return_samples:
            # Create visualization grid
            fig, axes = plt.subplots(4, 3, figsize=(12, 16))
            for idx in range(min(4, sample_images.size(0))):
                # Original image
                img = sample_images[idx].permute(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
                axes[idx, 0].imshow(img.numpy())
                axes[idx, 0].set_title("Input Image")
                axes[idx, 0].axis("off")

                # Ground truth mask
                gt_mask = sample_masks[idx].numpy()
                axes[idx, 1].imshow(gt_mask, cmap="tab10", vmin=0, vmax=num_classes - 1)
                axes[idx, 1].set_title("Ground Truth")
                axes[idx, 1].axis("off")

                # Prediction
                pred_mask = sample_predictions[idx].numpy()
                axes[idx, 2].imshow(
                    pred_mask, cmap="tab10", vmin=0, vmax=num_classes - 1
                )
                axes[idx, 2].set_title("Prediction")
                axes[idx, 2].axis("off")

            plt.tight_layout()
            writer.add_figure("Samples/Validation", fig, epoch)
            plt.close(fig)

            # Log model histograms (every 10 epochs to avoid clutter)
            if epoch % 10 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
                    writer.add_histogram(f"Parameters/{name}", param, epoch)

        # Weights & Biases logging
        if args.use_wandb:
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mean_iou": mean_iou,
                "learning_rate": current_lr,
            }
            # Add per-class IoU
            for class_name, iou in zip(class_names, class_ious):
                if not np.isnan(iou):
                    log_dict[f"iou/{class_name}"] = iou

            wandb.log(log_dict, step=epoch)

            # Log sample images to wandb
            if return_samples:
                wandb_images = []
                for idx in range(min(4, sample_images.size(0))):
                    img = sample_images[idx].permute(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min())
                    img_np = (img.numpy() * 255).astype(np.uint8)

                    gt_mask = sample_masks[idx].numpy()
                    pred_mask = sample_predictions[idx].numpy()

                    # Create overlay visualizations
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(img_np)
                    axes[0].set_title("Input")
                    axes[0].axis("off")

                    axes[1].imshow(img_np)
                    axes[1].imshow(
                        gt_mask, alpha=0.5, cmap="tab10", vmin=0, vmax=num_classes - 1
                    )
                    axes[1].set_title("Ground Truth Overlay")
                    axes[1].axis("off")

                    axes[2].imshow(img_np)
                    axes[2].imshow(
                        pred_mask, alpha=0.5, cmap="tab10", vmin=0, vmax=num_classes - 1
                    )
                    axes[2].set_title("Prediction Overlay")
                    axes[2].axis("off")

                    plt.tight_layout()
                    wandb_images.append(wandb.Image(fig, caption=f"Sample {idx}"))
                    plt.close(fig)

                wandb.log({"validation_samples": wandb_images}, step=epoch)

        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Mean IoU: {mean_iou:.4f}, LR: {current_lr:.6f}"
        )

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "mean_iou": mean_iou,
            "best_iou": best_iou,
            "global_step": global_step,
        }

        # Save latest
        torch.save(checkpoint, os.path.join(args.save_dir, "latest.pth"))

        # Save best
        if mean_iou > best_iou:
            best_iou = mean_iou
            checkpoint["best_iou"] = best_iou
            torch.save(checkpoint, os.path.join(args.save_dir, "best.pth"))
            print(f"New best model saved with IoU: {best_iou:.4f}")

    writer.close()
    if args.use_wandb:
        wandb.finish()
    print("Training completed!")


if __name__ == "__main__":
    main()
