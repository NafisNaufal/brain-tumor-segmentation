from __future__ import annotations

from typing import List

import timm
import torch
import torch.nn as nn


class TimmEncoder2p5D(nn.Module):
    """Extract 2D pretrained features slice-wise and reshape back to 3D feature maps.

    Input: (B, C, H, W, D)
    Output list of feature tensors: each (B, C_i, H_i, W_i, D)
    """

    def __init__(self, model_name: str, in_chans: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=in_chans,
        )
        self.out_channels = list(self.backbone.feature_info.channels())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, c, h, w, d = x.shape

        x_2d = x.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
        feats_2d = self.backbone(x_2d)

        feats_3d = []
        for feat in feats_2d:
            _, c_i, h_i, w_i = feat.shape
            feat_3d = feat.view(b, d, c_i, h_i, w_i).permute(0, 2, 3, 4, 1).contiguous()
            feats_3d.append(feat_3d)

        return feats_3d
