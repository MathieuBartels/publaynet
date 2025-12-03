import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def visualize_prediction(model, image, mask, device, num_classes=2, save_path=None):
    """
    Visualize model prediction alongside ground truth

    Args:
        model: Trained UNet model
        image: Input image tensor (C, H, W) or numpy array
        mask: Ground truth mask tensor (H, W) or numpy array
        device: torch device
        num_classes: Number of classes
        save_path: Optional path to save visualization
    """
    model.eval()

    # Prepare image
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image).float()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    # Get prediction
    with torch.no_grad():
        output = model(image)
        if num_classes > 2:
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        else:
            pred = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy()

    # Prepare mask
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Denormalize image for visualization
    if image.shape[1] == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        image_vis = image.squeeze() * std + mean
        image_vis = image_vis.cpu().permute(1, 2, 0).numpy()
        image_vis = np.clip(image_vis, 0, 1)
    else:
        image_vis = image.squeeze().cpu().permute(1, 2, 0).numpy()
        if image_vis.max() > 1:
            image_vis = image_vis / 255.0

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_vis)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="jet")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="jet")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def predict_image(model, image_path, device, num_classes=2, image_size=(512, 512)):
    """
    Predict segmentation for a single image

    Args:
        model: Trained UNet model
        image_path: Path to image file
        device: torch device
        num_classes: Number of classes
        image_size: Target image size

    Returns:
        Prediction mask as numpy array
    """
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size, Image.Resampling.BILINEAR)
    image_array = np.array(image).astype(np.float32) / 255.0

    # Normalize
    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    image_tensor = transform(image_tensor).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        if num_classes > 2:
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        else:
            pred = (
                (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
            )

    return pred


def calculate_metrics(predictions, targets, num_classes):
    """
    Calculate segmentation metrics

    Returns:
        Dictionary with accuracy, IoU per class, mean IoU, precision, recall
    """
    predictions = (
        torch.argmax(predictions, dim=1) if predictions.dim() > 2 else predictions
    )
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Accuracy
    accuracy = (predictions == targets).mean()

    # IoU per class
    ious = []
    for cls in range(num_classes):
        pred_cls = predictions == cls
        target_cls = targets == cls

        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    mean_iou = np.nanmean(ious)

    # Precision and Recall (for binary case)
    if num_classes == 2:
        tp = ((predictions == 1) & (targets == 1)).sum()
        fp = ((predictions == 1) & (targets == 0)).sum()
        fn = ((predictions == 0) & (targets == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
    else:
        precision = recall = f1 = None

    return {
        "accuracy": accuracy,
        "iou_per_class": ious,
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
