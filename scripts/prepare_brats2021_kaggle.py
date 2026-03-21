from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BraTS2021 manifests from Kaggle dataset")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help="Path to extracted BraTS root. If empty and --download is set, uses kagglehub download path.",
    )
    parser.add_argument("--output-dir", type=str, default="./manifests", help="Directory for train/val manifests")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--download", action="store_true", help="Download dataset using kagglehub")
    return parser.parse_args()


def maybe_download_with_kagglehub(download: bool) -> Path | None:
    if not download:
        return None

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is not installed. Install it with: pip install kagglehub") from exc

    path = kagglehub.dataset_download("dschettler8845/brats-2021-task1")
    return Path(path)


def discover_samples(root: Path) -> List[Dict[str, str]]:
    seg_files = sorted(root.rglob("*_seg.nii.gz"))
    if not seg_files:
        raise FileNotFoundError(f"No '*_seg.nii.gz' files found under: {root}")

    samples: List[Dict[str, str]] = []
    for seg_path in seg_files:
        stem = seg_path.name.replace("_seg.nii.gz", "")
        case_dir = seg_path.parent

        flair = case_dir / f"{stem}_flair.nii.gz"
        t1 = case_dir / f"{stem}_t1.nii.gz"
        t1ce = case_dir / f"{stem}_t1ce.nii.gz"
        t2 = case_dir / f"{stem}_t2.nii.gz"

        if not (flair.exists() and t1.exists() and t1ce.exists() and t2.exists()):
            continue

        samples.append(
            {
                "flair": str(flair.resolve()),
                "t1": str(t1.resolve()),
                "t1ce": str(t1ce.resolve()),
                "t2": str(t2.resolve()),
                "label": str(seg_path.resolve()),
            }
        )

    if not samples:
        raise RuntimeError("No complete cases found with flair,t1,t1ce,t2 and seg files")
    return samples


def split_samples(samples: List[Dict[str, str]], val_ratio: float, seed: int):
    rng = random.Random(seed)
    samples_copy = list(samples)
    rng.shuffle(samples_copy)

    n_val = max(1, int(len(samples_copy) * val_ratio))
    val_samples = samples_copy[:n_val]
    train_samples = samples_copy[n_val:]

    if not train_samples:
        raise RuntimeError("Train split is empty. Decrease --val-ratio or use more data")

    return train_samples, val_samples


def write_json(path: Path, data: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    args = parse_args()

    root: Path | None = None
    downloaded_root = maybe_download_with_kagglehub(args.download)
    if downloaded_root is not None:
        root = downloaded_root

    if args.dataset_root:
        root = Path(args.dataset_root)

    if root is None:
        raise ValueError("Provide --dataset-root or use --download")

    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    samples = discover_samples(root)
    train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)

    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"

    write_json(train_path, train_samples)
    write_json(val_path, val_samples)

    print(f"Dataset root: {root}")
    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)} -> {train_path}")
    print(f"Val samples: {len(val_samples)} -> {val_path}")


if __name__ == "__main__":
    main()
