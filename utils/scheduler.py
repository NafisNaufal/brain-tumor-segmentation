from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
    eta_min: float = 1e-6,
) -> SequentialLR:
    warmup_epochs = max(1, warmup_epochs)
    total_epochs = max(warmup_epochs + 1, total_epochs)

    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / float(warmup_epochs),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=eta_min,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
