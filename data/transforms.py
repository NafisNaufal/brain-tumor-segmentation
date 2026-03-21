from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from monai.transforms import (
    Compose,
    MapTransform,
    RandFlipd,
    Randomizable,
    RandomizableTransform,
)


class NonZeroPercentileClipd(MapTransform):
    def __init__(
        self,
        keys: Sequence[str],
        lower: float = 1.0,
        upper: float = 99.0,
    ) -> None:
        super().__init__(keys)
        self.lower = lower
        self.upper = upper

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            image = d[key].astype(np.float32, copy=False)
            out = image.copy()
            for channel in range(image.shape[0]):
                vox = image[channel]
                nz = vox[vox != 0]
                if nz.size < 10:
                    continue
                lo = np.percentile(nz, self.lower)
                hi = np.percentile(nz, self.upper)
                out[channel] = np.clip(vox, lo, hi)
            d[key] = out
        return d


class MinMaxNormalizeNonZerod(MapTransform):
    def __init__(self, keys: Sequence[str], eps: float = 1e-8) -> None:
        super().__init__(keys)
        self.eps = eps

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            image = d[key].astype(np.float32, copy=False)
            out = image.copy()
            for channel in range(image.shape[0]):
                vox = image[channel]
                nz_mask = vox != 0
                if not nz_mask.any():
                    continue
                vals = vox[nz_mask]
                v_min = vals.min()
                v_max = vals.max()
                out_channel = np.zeros_like(vox, dtype=np.float32)
                out_channel[nz_mask] = (vals - v_min) / (v_max - v_min + self.eps)
                out[channel] = out_channel
            d[key] = out
        return d


class CropToBrainBBoxd(MapTransform):
    def __init__(self, keys: Sequence[str], source_key: str = "image") -> None:
        super().__init__(keys)
        self.source_key = source_key

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        source = d[self.source_key]
        mask = np.any(source != 0, axis=0)

        if not mask.any():
            return d

        coords = np.where(mask)
        h0, h1 = int(coords[0].min()), int(coords[0].max()) + 1
        w0, w1 = int(coords[1].min()), int(coords[1].max()) + 1
        z0, z1 = int(coords[2].min()), int(coords[2].max()) + 1

        for key in self.keys:
            arr = d[key]
            if arr.ndim == 4:
                d[key] = arr[:, h0:h1, w0:w1, z0:z1]
            else:
                raise ValueError(f"Expected 4D tensor (C,H,W,D) for key={key}, got shape={arr.shape}")
        return d


class RandIntensityScaled(RandomizableTransform):
    def __init__(
        self,
        keys: Sequence[str],
        factor_range: tuple[float, float] = (0.9, 1.1),
        prob: float = 0.8,
    ) -> None:
        super().__init__(prob=prob)
        self.keys = keys
        self.factor_range = factor_range
        self._factor = 1.0

    def randomize(self, data: dict | None = None) -> None:
        super().randomize(None)
        self._factor = self.R.uniform(self.factor_range[0], self.factor_range[1])

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        self.randomize(d)
        if not self._do_transform:
            return d

        for key in self.keys:
            d[key] = (d[key].astype(np.float32, copy=False) * self._factor).astype(np.float32)
        return d


class StdChannelGaussianNoised(MapTransform):
    def __init__(self, keys: Sequence[str], scale: float = 0.1, prob: float = 0.5) -> None:
        super().__init__(keys)
        self.scale = scale
        self.prob = prob
        self._rng = np.random.default_rng()

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        if self._rng.random() > self.prob:
            return d

        for key in self.keys:
            image = d[key].astype(np.float32, copy=False)
            out = image.copy()
            for channel in range(image.shape[0]):
                vox = image[channel]
                nz = vox[vox != 0]
                if nz.size < 10:
                    continue
                std_c = float(nz.std())
                noise = self._rng.normal(loc=0.0, scale=std_c * self.scale, size=vox.shape).astype(np.float32)
                out[channel] = vox + noise
            d[key] = out
        return d


class RandChannelDropoutd(RandomizableTransform):
    def __init__(self, keys: Sequence[str], prob: float = 0.18) -> None:
        super().__init__(prob=prob)
        self.keys = keys
        self._drop_channel: int = -1

    def randomize(self, data: dict | None = None) -> None:
        super().randomize(None)
        if not self._do_transform or data is None:
            return
        n_channels = int(data[self.keys[0]].shape[0])
        self._drop_channel = int(self.R.randint(0, n_channels))

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        self.randomize(d)
        if not self._do_transform:
            return d

        for key in self.keys:
            arr = d[key].astype(np.float32, copy=False).copy()
            arr[self._drop_channel] = 0.0
            d[key] = arr
        return d


class RandAxisPermutationd(RandomizableTransform):
    """Random axis permutation over spatial dims, applied to image and label consistently."""

    def __init__(self, keys: Sequence[str], prob: float = 0.5) -> None:
        super().__init__(prob=prob)
        self.keys = keys
        self._perm = (0, 1, 2)

    def randomize(self, data: dict | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            self._perm = (0, 1, 2)
            return
        all_perms = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ]
        self._perm = all_perms[int(self.R.randint(0, len(all_perms)))]

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        self.randomize(d)
        if not self._do_transform or self._perm == (0, 1, 2):
            return d

        for key in self.keys:
            arr = d[key]
            d[key] = np.transpose(arr, axes=(0, *(axis + 1 for axis in self._perm))).copy()
        return d


def _safe_crop_to_patch(arr: np.ndarray, patch_size: Sequence[int], center: bool = False) -> np.ndarray:
    c, h, w, z = arr.shape
    ph, pw, pz = patch_size

    out = arr
    if h < ph or w < pw or z < pz:
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        pad_z = max(0, pz - z)
        pad = ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (pad_z // 2, pad_z - pad_z // 2))
        out = np.pad(out, pad, mode="constant", constant_values=0)

    _, hh, ww, zz = out.shape
    if center:
        sh = (hh - ph) // 2
        sw = (ww - pw) // 2
        sz = (zz - pz) // 2
    else:
        rng = np.random.default_rng()
        sh = int(rng.integers(0, max(1, hh - ph + 1)))
        sw = int(rng.integers(0, max(1, ww - pw + 1)))
        sz = int(rng.integers(0, max(1, zz - pz + 1)))

    return out[:, sh : sh + ph, sw : sw + pw, sz : sz + pz]


class RandomCropPatchd(MapTransform):
    def __init__(self, keys: Sequence[str], patch_size: Sequence[int]) -> None:
        super().__init__(keys)
        self.patch_size = patch_size

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        image = d[self.keys[0]]
        c, h, w, z = image.shape
        ph, pw, pz = self.patch_size

        if h < ph or w < pw or z < pz:
            for key in self.keys:
                d[key] = _safe_crop_to_patch(d[key], self.patch_size, center=False)
            return d

        rng = np.random.default_rng()
        sh = int(rng.integers(0, h - ph + 1))
        sw = int(rng.integers(0, w - pw + 1))
        sz = int(rng.integers(0, z - pz + 1))

        for key in self.keys:
            arr = d[key]
            d[key] = arr[:, sh : sh + ph, sw : sw + pw, sz : sz + pz]
        return d


class CenterCropPatchd(MapTransform):
    def __init__(self, keys: Sequence[str], patch_size: Sequence[int]) -> None:
        super().__init__(keys)
        self.patch_size = patch_size

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            d[key] = _safe_crop_to_patch(d[key], self.patch_size, center=True)
        return d


def build_train_transforms(cfg: dict) -> Compose:
    patch_size = cfg["data"]["patch_size"]
    aug_cfg = cfg["augment"]

    return Compose(
        [
            NonZeroPercentileClipd(keys=["image"], lower=1.0, upper=99.0),
            MinMaxNormalizeNonZerod(keys=["image"]),
            CropToBrainBBoxd(keys=["image", "label"], source_key="image"),
            RandomCropPatchd(keys=["image", "label"], patch_size=patch_size),
            RandIntensityScaled(keys=["image"], factor_range=(0.9, 1.1), prob=aug_cfg["intensity_prob"]),
            StdChannelGaussianNoised(keys=["image"], scale=0.1, prob=aug_cfg["gaussian_noise_prob"]),
            RandChannelDropoutd(keys=["image"], prob=aug_cfg["channel_dropout_prob"]),
            RandFlipd(keys=["image", "label"], prob=aug_cfg["flip_prob"], spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=aug_cfg["flip_prob"], spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=aug_cfg["flip_prob"], spatial_axis=2),
            RandAxisPermutationd(keys=["image", "label"], prob=aug_cfg["axis_permutation_prob"]),
        ]
    )


def build_val_transforms(cfg: dict) -> Compose:
    patch_size = cfg["data"]["patch_size"]
    return Compose(
        [
            NonZeroPercentileClipd(keys=["image"], lower=1.0, upper=99.0),
            MinMaxNormalizeNonZerod(keys=["image"]),
            CropToBrainBBoxd(keys=["image", "label"], source_key="image"),
            CenterCropPatchd(keys=["image", "label"], patch_size=patch_size),
        ]
    )
