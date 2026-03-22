from __future__ import annotations

"""Training entrypoint for BraTS-style 3D segmentation experiments."""

import argparse
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
import yaml
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from data.dataset import BraTSDataset
from data.transforms import build_train_transforms, build_val_transforms
from engine.train_loop import train_one_epoch
from engine.validate import validate
from models.build_model import build_model
from utils.loss import DiceBCELoss
from utils.scheduler import build_warmup_cosine_scheduler


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="BraTS 3D Segmentation Training")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    """Set deterministic seeds for each DataLoader worker process."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge update mapping into base mapping."""
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config with optional parent inheritance."""
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inherit_from = cfg.get("inherit")
    if inherit_from:
        parent_path = (cfg_path.parent / inherit_from).resolve()
        with parent_path.open("r", encoding="utf-8") as pf:
            parent_cfg = yaml.safe_load(pf)
        cfg = deep_update(parent_cfg, {k: v for k, v in cfg.items() if k != "inherit"})
    return cfg


def create_optimizer(cfg: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build AdamW optimizer with separate encoder/decoder learning rates."""
    enc_lr = cfg["optim"]["encoder_lr"]
    dec_lr = cfg["optim"]["decoder_lr"]
    weight_decay = cfg["optim"]["weight_decay"]

    return torch.optim.AdamW(
        [
            {"params": model.encoder_parameters(), "lr": enc_lr},
            {"params": model.non_encoder_parameters(), "lr": dec_lr},
        ],
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )


def main() -> None:
    """Run training and validation loop and save best checkpoints."""
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["seed"])
    set_seed(seed)

    reproducibility_cfg = cfg.get("reproducibility", {})
    deterministic = bool(reproducibility_cfg.get("deterministic", False))
    use_deterministic_algorithms = bool(
        reproducibility_cfg.get("use_deterministic_algorithms", False)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(use_deterministic_algorithms, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(seed)

    num_workers = int(cfg["data"]["num_workers"])
    persistent_workers = bool(cfg["data"].get("persistent_workers", num_workers > 0))
    prefetch_factor = int(cfg["data"].get("prefetch_factor", 2))

    train_ds = BraTSDataset(
        manifest_path=cfg["data"]["train_manifest"],
        transform=build_train_transforms(cfg),
    )
    val_ds = BraTSDataset(
        manifest_path=cfg["data"]["val_manifest"],
        transform=build_val_transforms(cfg),
    )

    train_loader_kwargs: dict[str, Any] = {
        "dataset": train_ds,
        "batch_size": cfg["training"]["batch_size"],
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": True,
        "worker_init_fn": seed_worker,
        "generator": data_loader_generator,
    }
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        train_loader_kwargs["prefetch_factor"] = prefetch_factor

    val_loader_kwargs: dict[str, Any] = {
        "dataset": val_ds,
        "batch_size": cfg["training"]["batch_size"],
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "generator": data_loader_generator,
    }
    if num_workers > 0:
        val_loader_kwargs["persistent_workers"] = persistent_workers
        val_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)

    model = build_model(cfg).to(device)
    loss_fn = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)

    optimizer = create_optimizer(cfg, model)
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=cfg["training"]["epochs"],
        warmup_epochs=cfg["scheduler"]["warmup_epochs"],
        eta_min=cfg["scheduler"]["eta_min"],
    )

    scaler = GradScaler(enabled=cfg["training"]["amp"])

    output_dir = Path(cfg["output"]["dir"]) / cfg["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = cfg["logging"]["use_wandb"]
    if use_wandb:
        wandb.init(
            project=cfg["logging"]["project"],
            name=cfg["experiment_name"],
            config=cfg,
            mode=cfg["logging"].get("mode", "online"),
        )

    best_dice = -1.0

    for epoch in range(cfg["training"]["epochs"]):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=cfg["training"]["amp"],
            grad_clip=cfg["training"].get("grad_clip"),
        )

        val_stats = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=cfg["training"]["amp"],
            compute_hd95=cfg["metrics"].get("compute_hd95", False),
        )

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        log_payload = {
            "epoch": epoch + 1,
            "learning_rate": lr,
            **train_stats,
            **val_stats,
        }

        print(
            f"Epoch [{epoch+1}/{cfg['training']['epochs']}] "
            f"train_loss={train_stats['train_loss']:.4f} "
            f"val_loss={val_stats['val_loss']:.4f} "
            f"val_dice={val_stats['val_dice']:.4f}"
        )

        if use_wandb:
            wandb.log(log_payload)

        latest_path = output_dir / "latest.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice": best_dice,
                "config": cfg,
            },
            latest_path,
        )

        if val_stats["val_dice"] > best_dice:
            best_dice = val_stats["val_dice"]
            best_path = output_dir / "best_dice.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_dice": best_dice,
                    "config": cfg,
                },
                best_path,
            )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    main()
