from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)


class UNetPlusPlus3D(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        num_classes: int = 3,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = encoder_channels
        self.deep_supervision = deep_supervision

        self.conv0_1 = ConvBlock3D(c0 + c1, c0)
        self.conv1_1 = ConvBlock3D(c1 + c2, c1)
        self.conv2_1 = ConvBlock3D(c2 + c3, c2)

        self.conv0_2 = ConvBlock3D(c0 * 2 + c1, c0)
        self.conv1_2 = ConvBlock3D(c1 * 2 + c2, c1)

        self.conv0_3 = ConvBlock3D(c0 * 3 + c1, c0)

        self.head1 = nn.Conv3d(c0, num_classes, kernel_size=1)
        self.head2 = nn.Conv3d(c0, num_classes, kernel_size=1)
        self.head3 = nn.Conv3d(c0, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor | List[torch.Tensor]:
        x0_0, x1_0, x2_0, x3_0 = features

        x0_1 = self.conv0_1(torch.cat([x0_0, _up_to(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, _up_to(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, _up_to(x3_0, x2_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, _up_to(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, _up_to(x2_1, x1_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, _up_to(x1_2, x0_0)], dim=1))

        out1 = self.head1(x0_1)
        out2 = self.head2(x0_2)
        out3 = self.head3(x0_3)

        if self.deep_supervision:
            return [out1, out2, out3]
        return out3
