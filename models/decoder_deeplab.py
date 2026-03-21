from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPPPooling3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x = self.pool(x)
        x = self.conv(x)
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)


class ASPP3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        dilations: Sequence[int] = (1, 6, 12, 18),
    ) -> None:
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )
        self.branches = nn.ModuleList(
            [ASPPConv3D(in_channels, out_channels, d) for d in dilations]
        )
        self.pool = ASPPPooling3D(in_channels, out_channels)

        merged_channels = out_channels * (2 + len(dilations))
        self.project = nn.Sequential(
            nn.Conv3d(merged_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Dropout3d(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.conv1x1(x)]
        feats.extend(branch(x) for branch in self.branches)
        feats.append(self.pool(x))
        x = torch.cat(feats, dim=1)
        return self.project(x)


class DeepLabV3Plus3D(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        num_classes: int = 3,
        aspp_channels: int = 128,
    ) -> None:
        super().__init__()
        c0, _, c2, c3 = encoder_channels

        self.aspp = ASPP3D(c3, out_channels=aspp_channels)
        self.low_proj = nn.Sequential(
            nn.Conv3d(c0, 48, kernel_size=1, bias=False),
            nn.InstanceNorm3d(48),
            nn.GELU(),
        )
        self.mid_proj = nn.Sequential(
            nn.Conv3d(c2, 64, kernel_size=1, bias=False),
            nn.InstanceNorm3d(64),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(aspp_channels + 64 + 48, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(128),
            nn.GELU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(128),
            nn.GELU(),
        )
        self.classifier = nn.Conv3d(128, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        low, _, mid, high = features

        aspp_out = self.aspp(high)
        aspp_up = F.interpolate(aspp_out, size=low.shape[2:], mode="trilinear", align_corners=False)

        mid_out = self.mid_proj(mid)
        mid_up = F.interpolate(mid_out, size=low.shape[2:], mode="trilinear", align_corners=False)

        low_out = self.low_proj(low)

        x = torch.cat([aspp_up, mid_up, low_out], dim=1)
        x = self.fuse(x)
        return self.classifier(x)
