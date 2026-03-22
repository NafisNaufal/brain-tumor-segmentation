from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_att = self.mlp(self.avg_pool(x))
        max_att = self.mlp(self.max_pool(x))
        att = self.act(avg_att + max_att)
        return x * att


class SpatialAxisAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv_hw = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv_hd = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv_wd = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    @staticmethod
    def _reduce_channel(x2d: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x2d, dim=1, keepdim=True)
        mx = torch.amax(x2d, dim=1, keepdim=True)
        return torch.cat([avg, mx], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.mean(dim=4)
        att_hw = self.act(self.conv_hw(self._reduce_channel(hw))).unsqueeze(-1)

        hd = x.mean(dim=3)
        att_hd = self.act(self.conv_hd(self._reduce_channel(hd))).unsqueeze(3)

        wd = x.mean(dim=2)
        att_wd = self.act(self.conv_wd(self._reduce_channel(wd))).unsqueeze(2)

        att = (att_hw + att_hd + att_wd) / 3.0
        return x * att


class TriAxisAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttention3D(channels=channels, reduction=reduction)
        self.spatial = SpatialAxisAttention3D(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x
