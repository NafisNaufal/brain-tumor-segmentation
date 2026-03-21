from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


Sample = Dict[str, Any]


def _load_manifest(path: str) -> List[Sample]:
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
        return np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)

    raise ValueError(f"Unsupported file format: {p}")


def _ensure_image_channels_first(image: np.ndarray) -> np.ndarray:
    if image.ndim != 4:
        raise ValueError(f"Expected image with 4 dims, got {image.shape}")

    if image.shape[0] == 4:
        return image.astype(np.float32, copy=False)

    if image.shape[-1] == 4:
        return np.transpose(image, (3, 0, 1, 2)).astype(np.float32, copy=False)

    raise ValueError(f"Image must have 4 modalities, got {image.shape}")


def _load_image_from_sample(sample: Sample) -> np.ndarray:
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


class BraTSDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: str,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        self.samples = _load_manifest(manifest_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = _load_image_from_sample(sample)
        label = _to_multilabel_brats(_load_array(sample["label"]))

        item = {"image": image, "label": label}
        if self.transform is not None:
            item = self.transform(item)

        image_t = torch.as_tensor(item["image"], dtype=torch.float32)
        label_t = torch.as_tensor(item["label"], dtype=torch.float32)

        return {
            "image": image_t,
            "label": label_t,
        }
