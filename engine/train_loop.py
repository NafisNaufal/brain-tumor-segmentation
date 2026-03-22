from __future__ import annotations

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool = True,
    grad_clip: float | None = None,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.detach().item())

    return {"train_loss": running_loss / max(1, len(loader))}
