from __future__ import annotations

"""Dataset utilities for loading BraTS-style multi-modal 3D data."""

import csv
import json
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


Sample = dict[str, Any]


def _load_manifest(path: str) -> list[Sample]:
    """Load dataset manifest from JSON or CSV."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    suffix = manifest_path.suffix.lower()
    if suffix in {".json"}:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON manifest must be a list of {'image':..., 'label':...}")
        return data

    if suffix in {".csv"}:
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [row for row in reader]

    raise ValueError("Unsupported manifest format. Use .json or .csv")


def _load_array(path: str) -> np.ndarray:
    """Load a NumPy or NIfTI array from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    if p.suffix.lower() == ".npy":
        return np.load(p)

    if p.suffix.lower() == ".npz":
        npz = np.load(p)
        if "arr_0" in npz:
            return npz["arr_0"]
        first_key = list(npz.keys())[0]
        return npz[first_key]

    if p.suffix.lower() in {".nii", ".gz"}:
        return nib.load(str(p)).get_fdata(dtype=np.float32)

    raise ValueError(f"Unsupported file format: {p}")


def _ensure_image_channels_first(image: np.ndarray) -> np.ndarray:
    """Ensure image has channel-first shape (4, H, W, D)."""
    if image.ndim != 4:
        raise ValueError(f"Expected image with 4 dims, got {image.shape}")

    if image.shape[0] == 4:
        return image.astype(np.float32, copy=False)

    if image.shape[-1] == 4:
        return np.transpose(image, (3, 0, 1, 2)).astype(np.float32, copy=False)

    raise ValueError(f"Image must have 4 modalities, got {image.shape}")


def _load_image_from_sample(sample: Sample) -> np.ndarray:
    """Load image tensor from manifest entry using supported sample schemas."""
    if "image" in sample:
        image_value = sample["image"]
        if isinstance(image_value, str):
            return _ensure_image_channels_first(_load_array(image_value))

        if isinstance(image_value, list):
            if len(image_value) != 4:
                raise ValueError("If 'image' is a list, it must contain 4 modality file paths")
            channels = [_load_array(path).astype(np.float32, copy=False) for path in image_value]
            return np.stack(channels, axis=0)

        if isinstance(image_value, dict):
            ordered_keys = ["flair", "t1", "t1ce", "t2"]
            if not all(key in image_value for key in ordered_keys):
                raise ValueError("If 'image' is a dict, required keys are flair, t1, t1ce, t2")
            channels = [_load_array(image_value[key]).astype(np.float32, copy=False) for key in ordered_keys]
            return np.stack(channels, axis=0)

    ordered_keys = ["flair", "t1", "t1ce", "t2"]
    if all(key in sample for key in ordered_keys):
        channels = [_load_array(sample[key]).astype(np.float32, copy=False) for key in ordered_keys]
        return np.stack(channels, axis=0)

    raise ValueError(
        "Sample must provide either 'image' (path/list/dict) or modality keys flair,t1,t1ce,t2"
    )


def _to_multilabel_brats(label: np.ndarray) -> np.ndarray:
    """Convert BraTS label map to ET, TC, WT channels.

    BraTS labels are commonly encoded as:
    - 1: NCR/NET
    - 2: ED
    - 4: ET

    Outputs:
    - ET: 4
    - TC: 1 or 4
    - WT: 1 or 2 or 4
    """
    if label.ndim == 4 and label.shape[0] == 3:
        return label.astype(np.float32, copy=False)

    if label.ndim == 4 and label.shape[-1] == 3:
        return np.transpose(label, (3, 0, 1, 2)).astype(np.float32, copy=False)

    if label.ndim != 3:
        raise ValueError(f"Expected label shape (H,W,D) or 3-channel mask, got {label.shape}")

    et = (label == 4).astype(np.float32)
    tc = np.logical_or(label == 1, label == 4).astype(np.float32)
    wt = (label > 0).astype(np.float32)
    return np.stack([et, tc, wt], axis=0)


def _validate_label_values(label: np.ndarray, label_path: str) -> None:
    """Validate expected BraTS label codes before conversion."""
    if label.ndim != 3:
        return

    unique_vals = np.unique(label)
    allowed = {0.0, 1.0, 2.0, 4.0}
    unique_set = {float(v) for v in unique_vals.tolist()}
    if not unique_set.issubset(allowed):
        raise ValueError(
            f"Unexpected label values in '{label_path}'. "
            f"Expected subset of {{0,1,2,4}}, got {sorted(unique_set)}"
        )


def _assert_expected_shapes(image: np.ndarray, label: np.ndarray, sample: Sample) -> None:
    """Enforce strict channel-first shapes.

    Expected:
    - image: (4, H, W, D)
    - label: (3, H, W, D)
    """
    if image.ndim != 4 or image.shape[0] != 4:
        image_path_hint = sample.get("image", "modalities: flair/t1/t1ce/t2")
        raise ValueError(
            f"Invalid image shape {image.shape}. Expected (4,H,W,D). Source={image_path_hint}"
        )

    if label.ndim != 4 or label.shape[0] != 3:
        label_path_hint = sample.get("label", "<missing>")
        raise ValueError(
            f"Invalid label shape {label.shape}. Expected (3,H,W,D). Source={label_path_hint}"
        )

    if image.shape[1:] != label.shape[1:]:
        raise ValueError(
            f"Image/label spatial mismatch: image={image.shape}, label={label.shape}, "
            f"label_source={sample.get('label', '<missing>')}"
        )


class BraTSDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset for BraTS-style segmentation samples."""

    def __init__(
        self,
        manifest_path: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        debug: bool = False,
        cache_enabled: bool = False,
        cache_max_items: int = 128,
    ) -> None:
        self.samples = _load_manifest(manifest_path)
        self.transform = transform
        self.debug = debug
        self.cache_enabled = cache_enabled and len(self.samples) <= cache_max_items
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if self.cache_enabled and idx in self._cache:
            image, label = self._cache[idx]
            image = image.copy()
            label = label.copy()
        else:
            try:
                image = _load_image_from_sample(sample)
            except Exception as exc:
                raise RuntimeError(f"Failed to load image for sample index {idx}: {sample}") from exc

            label_path = sample.get("label", "<missing>")
            try:
                raw_label = _load_array(label_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load label for sample index {idx}. label_path={label_path}"
                ) from exc

            _validate_label_values(raw_label, str(label_path))
            label = _to_multilabel_brats(raw_label)
            _assert_expected_shapes(image, label, sample)

            if self.cache_enabled:
                self._cache[idx] = (image.copy(), label.copy())

        item = {"image": image, "label": label}
        if self.transform is not None:
            item = self.transform(item)

        _assert_expected_shapes(item["image"], item["label"], sample)

        if self.debug:
            label_unique = np.unique(item["label"])
            print(
                "[BraTSDataset][debug] "
                f"idx={idx} image_shape={item['image'].shape} label_shape={item['label'].shape} "
                f"image_minmax=({float(np.min(item['image'])):.4f}, {float(np.max(item['image'])):.4f}) "
                f"label_unique={label_unique.tolist()}"
            )

        image_np = np.ascontiguousarray(item["image"], dtype=np.float32)
        label_np = np.ascontiguousarray(item["label"], dtype=np.float32)

        image_t = torch.from_numpy(image_np).contiguous()
        label_t = torch.from_numpy(label_np).contiguous()

        return {
            "image": image_t,
            "label": label_t,
        }
