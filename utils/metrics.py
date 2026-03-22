from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from monai.metrics import compute_hausdorff_distance
except Exception:
    compute_hausdorff_distance = None


def dice_per_class_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum(dim=(0, 2, 3, 4))
    cardinality = preds.sum(dim=(0, 2, 3, 4)) + targets.sum(dim=(0, 2, 3, 4))
    return (2.0 * intersection + eps) / (cardinality + eps)


def batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    compute_hd95: bool = False,
) -> dict[str, Any]:
    dice_pc = dice_per_class_from_logits(logits, targets)
    metrics: dict[str, Any] = {
        "dice_et": float(dice_pc[0].item()),
        "dice_tc": float(dice_pc[1].item()),
        "dice_wt": float(dice_pc[2].item()),
        "dice_mean": float(dice_pc.mean().item()),
    }

    if compute_hd95 and compute_hausdorff_distance is not None:
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            hd = compute_hausdorff_distance(
                y_pred=preds,
                y=targets,
                include_background=False,
                percentile=95.0,
            )
            hd_np = hd.detach().cpu().numpy().astype(np.float32)
            metrics["hd95"] = float(np.nanmean(hd_np))
    else:
        metrics["hd95"] = float("nan")

    return metrics
