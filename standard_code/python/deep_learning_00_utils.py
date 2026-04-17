"""
Utilities for centroid U-Net training:
  - CentroidDataset : reads paired .tif from input/target dirs
  - UNet            : 4-level encoder-decoder (32-32-64-128-256), mirrors centroid-unet
  - get_train_transform / get_val_transform : homographic augmentation
"""

from __future__ import annotations

import os
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import tifffile
import kornia.augmentation as K


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def contrast(arr, low, high):
    arr = np.clip(arr, np.percentile(arr, low), np.percentile(arr, high))
    return arr

def scale(arr):
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

class CentroidDataset(Dataset):
    """Reads paired .tif files from *input_dir* and *target_dir*.

    Files are matched by name (0.tif, 1.tif, …).  Both images are returned
    as float32 tensors with shape (1, H, W).
    """

    def __init__(
        self,
        input_dir: str | Path,
        target_dir: str | Path,
        transform: nn.Module | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)

        self.filenames = sorted(
            os.listdir(self.input_dir),
            key=lambda x: int(Path(x).stem),
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        img = tifffile.imread(self.input_dir / fname).astype(np.float32)
        tgt = tifffile.imread(self.target_dir / fname).astype(np.float32)

        img = scale(contrast(img, 0.1, 99.9))  # optional contrast + scaling

        # (H, W) -> (1, H, W)
        img = torch.from_numpy(img).unsqueeze(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0)

        if self.transform is not None:
            img, tgt = self.transform(img, tgt)
            img = img.squeeze(0)  # remove extra batch dim Kornia adds: (1,1,H,W) -> (1,H,W)
            tgt = tgt.squeeze(0)

        return img, tgt


# ---------------------------------------------------------------------------
# Augmentation (homographic / perspective + flips, NO scaling)
# ---------------------------------------------------------------------------
def get_train_transform() -> nn.Module:
    """Homographic augmentation applied jointly to input+target."""
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        # K.RandomPerspective(distortion_scale=0.3, p=0.5),
        K.RandomAffine(
            degrees=90,
            translate=(0.1, 0.1),
            scale=None,       # skip scaling
            # shear=(-10, 10),
            p=0.5,
        ),
        data_keys=["input", "mask"],
    )


def get_val_transform() -> nn.Module | None:
    return None


# ---------------------------------------------------------------------------
# U-Net  (mirrors centroid-unet Model.py: 32-32-64-128 → 256 → 128-64-32-32)
# ---------------------------------------------------------------------------
class _ConvBlock(nn.Module):
    """Two 3×3 convolutions + BN + ReLU + Dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """4-level U-Net for single-channel centroid prediction."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        # Encoder
        self.enc1 = _ConvBlock(in_channels, 32)
        self.enc2 = _ConvBlock(32, 32)
        self.enc3 = _ConvBlock(32, 64)
        self.enc4 = _ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = _ConvBlock(256 + 128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = _ConvBlock(128 + 64, 64)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = _ConvBlock(64 + 32, 32)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = _ConvBlock(32 + 32, 32)

        # Head – two 1×1 convs + sigmoid
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bottleneck
        b = self.bottleneck(self.pool(s4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return self.head(d1)
