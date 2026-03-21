from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from utils.metrics import batch_metrics


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool = True,
    compute_hd95: bool = False,
) -> Dict[str, float]:
    model.eval()

    val_loss = 0.0
    dice_et = []
    dice_tc = []
    dice_wt = []
    dice_mean = []
    hd95_vals = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast(enabled=amp_enabled):
                logits = model(images)
                loss = loss_fn(logits, labels)

            if isinstance(logits, (list, tuple)):
                logits = logits[-1]

            metrics = batch_metrics(logits=logits, targets=labels, compute_hd95=compute_hd95)
            val_loss += float(loss.detach().item())
            dice_et.append(metrics["dice_et"])
            dice_tc.append(metrics["dice_tc"])
            dice_wt.append(metrics["dice_wt"])
            dice_mean.append(metrics["dice_mean"])
            if not np.isnan(metrics["hd95"]):
                hd95_vals.append(metrics["hd95"])

    return {
        "val_loss": val_loss / max(1, len(loader)),
        "val_dice": float(np.mean(dice_mean)) if dice_mean else 0.0,
        "val_dice_et": float(np.mean(dice_et)) if dice_et else 0.0,
        "val_dice_tc": float(np.mean(dice_tc)) if dice_tc else 0.0,
        "val_dice_wt": float(np.mean(dice_wt)) if dice_wt else 0.0,
        "val_hd95": float(np.mean(hd95_vals)) if hd95_vals else float("nan"),
    }
