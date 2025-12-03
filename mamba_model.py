import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Install with: pip install mamba-ssm")


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class MambaBlock(nn.Module):
    """
    Mamba block for 2D feature maps.
    Converts 2D features to 1D sequence, applies Mamba, then converts back.
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm")
        
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.dim)
        
        # Mamba layer
        self.mamba = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            out: (B, C, H, W) feature map
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Apply layer norm
        x_seq = self.norm(x_seq)
        
        # Apply Mamba
        out_seq = self.mamba(x_seq)  # (B, H*W, C)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        out = out_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Residual connection
        out = out + x
        
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DownWithMamba(nn.Module):
    """Downscaling with maxpool, double conv, and Mamba block"""

    def __init__(self, in_channels, out_channels, d_state=16, d_conv=4, expand=2.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )
        self.mamba = MambaBlock(out_channels, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        x = self.maxpool_conv(x)
        x = self.mamba(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Dataset/blob/master/unet/unet_parts.py
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MambaSegmentation(nn.Module):
    """
    U-Mamba model for semantic segmentation.
    
    Supports two variants:
    - 'bot': Mamba blocks at the bottleneck (U-Mamba_Bot)
    - 'enc': Mamba blocks in the encoder (U-Mamba_Enc)
    """
    
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        bilinear: bool = False,
        variant: Literal["bot", "enc"] = "enc",
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        width_multiplier: float = 1.0,
        img_size: Optional[tuple] = None,
        patch_size: Optional[int] = None,
        embed_dim: Optional[int] = None,
        depth: Optional[int] = None,
    ):
        """
        Args:
            n_channels: Number of input channels (3 for RGB, 1 for grayscale)
            n_classes: Number of output classes (1 for binary segmentation)
            bilinear: Use bilinear upsampling instead of transpose convolution
            variant: 'bot' for Mamba at bottleneck, 'enc' for Mamba in encoder
            d_state: State dimension for Mamba blocks (reduce to 8 or 4 for smaller model)
            d_conv: Convolution dimension for Mamba blocks (reduce to 3 or 2 for smaller model)
            expand: Expansion factor for Mamba blocks (reduce to 1.5 or 1 for smaller model)
            width_multiplier: Multiplier for all channel dimensions (0.5 = half size, 0.25 = quarter size)
            img_size: Image size (for compatibility, not used in this implementation)
            patch_size: Patch size (for compatibility, not used in this implementation)
            embed_dim: Embedding dimension (for compatibility, not used in this implementation)
            depth: Depth (for compatibility, not used in this implementation)
            
        Model size reduction options (in order of impact):
        1. width_multiplier: 1.0 (default) -> 0.5 (4x smaller) -> 0.25 (16x smaller)
        2. d_state: 16 (default) -> 8 (smaller) -> 4 (much smaller)
        3. expand: 2 (default) -> 1.5 -> 1 (smaller)
        4. d_conv: 4 (default) -> 3 -> 2 (slightly smaller)
        """
        super(MambaSegmentation, self).__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required. Install with: "
                "pip install causal-conv1d>=1.2.0 && pip install mamba-ssm --no-cache-dir"
            )
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.variant = variant
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.width_multiplier = width_multiplier

        # Calculate channel dimensions with width multiplier
        base_channels = [64, 128, 256, 512, 1024]
        channels = [int(c * width_multiplier) for c in base_channels]
        # Ensure minimum channel size of 16
        channels = [max(16, c) for c in channels]
        
        c1, c2, c3, c4, c5 = channels

        # Initial convolution
        self.inc = DoubleConv(n_channels, c1)
        
        # Encoder (downsampling path)
        if variant == "enc":
            # U-Mamba_Enc: Mamba blocks in encoder
            self.down1 = DownWithMamba(
                c1, c2, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.down2 = DownWithMamba(
                c2, c3, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.down3 = DownWithMamba(
                c3, c4, d_state=d_state, d_conv=d_conv, expand=expand
            )
        else:
            # Standard downsampling for U-Mamba_Bot
            self.down1 = Down(c1, c2)
            self.down2 = Down(c2, c3)
            self.down3 = Down(c3, c4)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(c4, c5 // factor)
        
        # Bottleneck with Mamba
        if variant == "bot":
            # U-Mamba_Bot: Mamba blocks at bottleneck
            self.bottleneck_mamba = MambaBlock(
                c5 // factor, d_state=d_state, d_conv=d_conv, expand=expand
            )
        
        # Decoder (upsampling path)
        self.up1 = Up(c5, c4 // factor, bilinear)
        self.up2 = Up(c4, c3 // factor, bilinear)
        self.up3 = Up(c3, c2 // factor, bilinear)
        self.up4 = Up(c2, c1, bilinear)
        self.outc = OutConv(c1, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply Mamba at bottleneck if variant is 'bot'
        if self.variant == "bot":
            x5 = self.bottleneck_mamba(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        if self.variant == "bot":
            self.bottleneck_mamba = torch.utils.checkpoint(self.bottleneck_mamba)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

