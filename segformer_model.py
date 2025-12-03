import torch.nn as nn
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
)


class SegFormerB1(nn.Module):
    """
    SegFormer-B1 model wrapper for semantic segmentation
    Uses the SegFormer-B1 architecture from Hugging Face transformers
    """

    def __init__(self, n_channels=3, n_classes=2, pretrained=True):
        """
        Args:
            n_channels: Number of input channels (3 for RGB)
            n_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(SegFormerB1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Load SegFormer-B1 configuration
        model_name = "nvidia/segformer-b1-finetuned-ade-512-512"
        config = SegformerConfig.from_pretrained(model_name)

        # Update number of classes
        config.num_labels = n_classes

        # Load model (with or without pretrained weights)
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            logits: Output tensor of shape (B, n_classes, H, W)
        """
        # SegFormer expects inputs in range [0, 1] or normalized
        # The model will handle normalization internally if needed
        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        # Upsample to match input size if needed
        # SegFormer outputs are typically at 1/4 resolution
        if logits.shape[2:] != x.shape[2:]:
            logits = nn.functional.interpolate(
                logits,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return logits
