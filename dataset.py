import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from datasets import load_dataset


class PublayNetDataset(Dataset):
    """Dataset class for PublayNet semantic segmentation"""

    def __init__(
        self,
        split="train",
        transform=None,
        target_transform=None,
        num_classes=6,
        image_size=(512, 512),
    ):
        """
        Args:
            split: 'train' or 'test'
            transform: Optional transform to be applied on image
            target_transform: Optional transform to be applied on mask
            num_classes: Number of segmentation classes
                (default 6 for PublayNet)
            image_size: Target image size (height, width)
        """
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = num_classes
        self.image_size = image_size

        # Load dataset
        self.dataset = load_dataset("jordanparker6/publaynet", split=split)

        # PublayNet category mapping
        self.category_map = {
            0: 0,  # background
            1: 1,  # text
            2: 2,  # title
            3: 3,  # list
            4: 4,  # table
            5: 5,  # figure
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get data
        sample = self.dataset[idx]
        image = sample["image"]
        annotations = sample["annotations"]

        # Convert image to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get original image size for coordinate scaling
        original_size = image.size  # (width, height)

        # Resize image
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0

        # Create segmentation mask
        mask = self._create_mask(annotations, self.image_size, original_size)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )(image)

        # Clamp mask values to valid range [0, num_classes-1]
        mask = np.clip(mask, 0, self.num_classes - 1)

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(mask).long()

        return image, mask

    def _create_mask(self, annotations, target_size, original_size):
        """Create segmentation mask from annotations

        Args:
            annotations: List of annotation dictionaries
            target_size: Target image size (height, width)
            original_size: Original image size (width, height)
        """
        mask = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)

        # Scale factors for x and y coordinates
        scale_x = target_size[1] / original_size[0]  # width
        scale_y = target_size[0] / original_size[1]  # height

        for ann in annotations:
            category_id = ann["category_id"]
            segmentation = ann["segmentation"]

            # Map category_id to mask value
            if category_id in self.category_map:
                mask_value = self.category_map[category_id]
            else:
                continue  # Skip unknown categories

            # Handle segmentation format
            if isinstance(segmentation, list) and len(segmentation) > 0:
                # COCO format: list of polygons
                for seg in segmentation:
                    if isinstance(seg, list) and len(seg) >= 6:
                        # Convert polygon to mask
                        # seg is a flat list: [x1, y1, x2, y2, x3, y3, ...]
                        # Shape: (n_points, 2)
                        polygon = np.array(seg).reshape(-1, 2)

                        from PIL import ImageDraw

                        img = Image.new("L", target_size, 0)
                        draw = ImageDraw.Draw(img)

                        # Scale coordinates from original to target size
                        # Iterate over rows of polygon array
                        scaled_polygon = [
                            (int(x * scale_x), int(y * scale_y)) for x, y in polygon
                        ]

                        if len(scaled_polygon) >= 3:
                            draw.polygon(scaled_polygon, fill=mask_value)
                            mask = np.maximum(mask, np.array(img))

        return mask


class BinaryPublayNetDataset(PublayNetDataset):
    """Binary segmentation version (foreground/background)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_classes=2, **kwargs)
        self.category_map = {
            0: 0,  # background
            1: 1,  # foreground (all document elements)
            2: 1,
            3: 1,
            4: 1,
            5: 1,
        }

    def _create_mask(self, annotations, target_size, original_size):
        """Create binary segmentation mask

        Args:
            annotations: List of annotation dictionaries
            target_size: Target image size (height, width)
            original_size: Original image size (width, height)
        """
        mask = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)

        # Scale factors for x and y coordinates
        scale_x = target_size[1] / original_size[0]  # width
        scale_y = target_size[0] / original_size[1]  # height

        for ann in annotations:
            category_id = ann["category_id"]
            segmentation = ann["segmentation"]

            # Map category_id to binary mask value
            # Background (0) stays 0, all others become foreground (1)
            if category_id == 0:
                mask_value = 0
            else:
                mask_value = 1

            # Handle segmentation format (same as parent)
            if isinstance(segmentation, list) and len(segmentation) > 0:
                for seg in segmentation:
                    if isinstance(seg, list) and len(seg) >= 6:
                        # seg is a flat list: [x1, y1, x2, y2, x3, y3, ...]
                        # Shape: (n_points, 2)
                        polygon = np.array(seg).reshape(-1, 2)
                        from PIL import ImageDraw

                        img = Image.new("L", target_size, 0)
                        draw = ImageDraw.Draw(img)

                        # Scale coordinates from original to target size
                        scaled_polygon = [
                            (int(x * scale_x), int(y * scale_y)) for x, y in polygon
                        ]

                        if len(scaled_polygon) >= 3:
                            draw.polygon(scaled_polygon, fill=mask_value)
                            mask = np.maximum(mask, np.array(img))

        return mask
