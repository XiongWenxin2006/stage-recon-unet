"""Factories that build reconstruction / segmentation transform pipelines."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from stagerecon.data.transforms.common import Identity, Normalize, ToTensor
from stagerecon.data.transforms.paired_segmentation_transforms import (
    PairedCompose,
    PairedRandomHorizontalFlip,
    PairedRandomRotate90,
    PairedRandomVerticalFlip,
    RandomBrightnessContrast,
)
from stagerecon.data.transforms.reconstruction_corruptions import (
    CorruptionComposer,
    GaussianNoise,
    LocalPixelShuffle,
    RandomPatchMask,
)


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    """Convert OmegaConf / Mapping configs to a plain dict."""
    if cfg is None:
        return {}
    if hasattr(cfg, "items") and not isinstance(cfg, dict):
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(cfg):
                return dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
        except Exception:
            pass
        return {str(k): v for k, v in cfg.items()}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Unsupported config type: {type(cfg)!r}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _build_one_corruption(spec: Any) -> Callable[..., Any]:
    """Build a single corruption from a name string or mapping."""
    if callable(spec) and not isinstance(spec, type):
        return spec  # type: ignore[return-value]
    if isinstance(spec, str):
        name, params = spec.lower(), {}
    elif isinstance(spec, Mapping):
        plain = _to_plain_dict(spec)
        name = str(plain.get("name", plain.get("type", ""))).lower()
        params = {k: v for k, v in plain.items() if k not in {"name", "type"}}
    else:
        raise TypeError(f"Unsupported corruption spec: {type(spec)!r}")

    if name in {"gaussian_noise", "noise", "gaussian"}:
        return GaussianNoise(**params)
    if name in {"random_patch_mask", "patch_mask", "mask"}:
        return RandomPatchMask(**params)
    if name in {"local_pixel_shuffle", "pixel_shuffle", "shuffle"}:
        return LocalPixelShuffle(**params)
    if name in {"identity", "none"}:
        return Identity()
    raise KeyError(f"Unknown corruption '{name}'")


def build_reconstruction_transforms(cfg: Any) -> Callable[..., Any]:
    """Build a reconstruction transform pipeline from config.

    Supported keys::

        corruptions:
          - {name: gaussian_noise, std: 0.1}
          - {name: random_patch_mask, num_patches: 2, patch_size: 16}
          - {name: local_pixel_shuffle, patch_size: 8}
        p: 0.5
        min_corruptions: 0
        max_corruptions: null
        normalize: {mean: 0.5, std: 0.5}   # optional, applied to image after corruption
        to_tensor: false                   # usually images are already tensors

    Returns a callable that accepts a reconstruction sample dict (or tensor via
    :class:`CorruptionComposer`).
    """
    plain = _to_plain_dict(cfg)
    if not plain or plain.get("enabled", True) is False:
        return Identity()

    corruption_specs = _as_list(plain.get("corruptions", plain.get("corruption")))
    if not corruption_specs and plain.get("gaussian_noise") is not None:
        corruption_specs.append({"name": "gaussian_noise", "std": plain["gaussian_noise"]})
    if plain.get("patch_mask") is not None:
        patch_cfg = _to_plain_dict(plain["patch_mask"])
        patch_cfg.setdefault("name", "random_patch_mask")
        corruption_specs.append(patch_cfg)
    if plain.get("pixel_shuffle") is not None:
        shuffle_cfg = _to_plain_dict(plain["pixel_shuffle"])
        if not shuffle_cfg:
            shuffle_cfg = {"patch_size": 8}
        shuffle_cfg.setdefault("name", "local_pixel_shuffle")
        corruption_specs.append(shuffle_cfg)

    corruptions = [_build_one_corruption(spec) for spec in corruption_specs]
    composer = CorruptionComposer(
        corruptions=corruptions,
        p=float(plain.get("p", 0.5 if corruptions else 1.0)),
        min_corruptions=int(plain.get("min_corruptions", 0)),
        max_corruptions=plain.get("max_corruptions"),
    )

    normalize_cfg = plain.get("normalize")
    to_tensor = bool(plain.get("to_tensor", False))
    normalize = Normalize(**_to_plain_dict(normalize_cfg)) if normalize_cfg else None
    tensorize = ToTensor() if to_tensor else None

    def _pipeline(sample: Any) -> Any:
        out = composer(sample)
        if isinstance(out, dict):
            if tensorize is not None:
                out["image"] = tensorize(out["image"])
                if "target" in out:
                    out["target"] = tensorize(out["target"])
            if normalize is not None and "image" in out:
                out["image"] = normalize(out["image"])
            return out
        # bare tensor path
        if tensorize is not None:
            out = tensorize(out)
        if normalize is not None:
            out = normalize(out)
        return out

    return _pipeline


def build_segmentation_transforms(cfg: Any) -> Callable[..., Any]:
    """Build a paired segmentation transform pipeline from config.

    Supported keys::

        horizontal_flip: true | {p: 0.5}
        vertical_flip: true | {p: 0.5}
        rotate90: true | {p: 0.5}
        brightness_contrast: {brightness: 0.2, contrast: 0.2, p: 0.5}
        normalize: {mean: 0.5, std: 0.5}
    """
    plain = _to_plain_dict(cfg)
    if not plain or plain.get("enabled", True) is False:
        return Identity()

    spatial: list[Any] = []
    intensity: list[Any] = []

    def _flag_params(key: str, default_p: float = 0.5) -> dict[str, Any] | None:
        value = plain.get(key)
        if value is None or value is False:
            return None
        if value is True:
            return {"p": default_p}
        return _to_plain_dict(value)

    hf = _flag_params("horizontal_flip")
    if hf is not None:
        spatial.append(PairedRandomHorizontalFlip(**hf))
    vf = _flag_params("vertical_flip")
    if vf is not None:
        spatial.append(PairedRandomVerticalFlip(**vf))
    rot = _flag_params("rotate90")
    if rot is not None:
        spatial.append(PairedRandomRotate90(**rot))

    bc = plain.get("brightness_contrast", plain.get("intensity"))
    if bc not in (None, False):
        bc_cfg = {} if bc is True else _to_plain_dict(bc)
        intensity.append(RandomBrightnessContrast(**bc_cfg))

    compose = PairedCompose(spatial=spatial, intensity=intensity)

    normalize_cfg = plain.get("normalize")
    normalize = Normalize(**_to_plain_dict(normalize_cfg)) if normalize_cfg else None

    def _pipeline(sample: dict[str, Any]) -> dict[str, Any]:
        out = compose(sample) if (spatial or intensity) else dict(sample)
        if normalize is not None and "image" in out:
            out["image"] = normalize(out["image"])
        return out

    return _pipeline
