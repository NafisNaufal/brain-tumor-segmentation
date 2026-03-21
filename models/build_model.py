from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import TriAxisAttention
from .decoder_deeplab import DeepLabV3Plus3D
from .decoder_unetpp import UNetPlusPlus3D
from .encoder import TimmEncoder2p5D


class SegmentationModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        decoder_name: str,
        num_classes: int,
        in_channels: int = 4,
        pretrained: bool = True,
        use_attention: bool = False,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = TimmEncoder2p5D(
            model_name=encoder_name,
            in_chans=in_channels,
            pretrained=pretrained,
        )

        self.use_attention = use_attention
        if self.use_attention:
            self.attention = nn.ModuleList(
                [TriAxisAttention(channels=c) for c in self.encoder.out_channels]
            )
        else:
            self.attention = None

        if decoder_name == "unetpp":
            self.decoder: nn.Module = UNetPlusPlus3D(
                encoder_channels=self.encoder.out_channels,
                num_classes=num_classes,
                deep_supervision=deep_supervision,
            )
        elif decoder_name == "deeplabv3p":
            self.decoder = DeepLabV3Plus3D(
                encoder_channels=self.encoder.out_channels,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unsupported decoder: {decoder_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        input_size = x.shape[2:]
        feats = self.encoder(x)

        if self.attention is not None:
            feats = [att(feat) for att, feat in zip(self.attention, feats)]

        out = self.decoder(feats)
        if isinstance(out, list):
            return [F.interpolate(o, size=input_size, mode="trilinear", align_corners=False) for o in out]
        return F.interpolate(out, size=input_size, mode="trilinear", align_corners=False)

    def encoder_parameters(self):
        return self.encoder.parameters()

    def non_encoder_parameters(self):
        encoder_params = {id(p) for p in self.encoder.parameters()}
        for param in self.parameters():
            if id(param) not in encoder_params:
                yield param


def build_model(cfg: Dict) -> SegmentationModel:
    model_cfg = cfg["model"]
    return SegmentationModel(
        encoder_name=model_cfg["encoder_name"],
        decoder_name=model_cfg["decoder_name"],
        num_classes=model_cfg["num_classes"],
        in_channels=model_cfg["in_channels"],
        pretrained=model_cfg["pretrained"],
        use_attention=model_cfg["use_attention"],
        deep_supervision=model_cfg.get("deep_supervision", True),
    )
