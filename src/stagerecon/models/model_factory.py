"""Factory helpers to build ModularUNet instances from config dicts."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from stagerecon.models.bottlenecks import build_bottleneck
from stagerecon.models.composed.modular_unet import ModularUNet
from stagerecon.models.decoders import build_decoder
from stagerecon.models.encoders import build_encoder
from stagerecon.models.heads import build_head


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    """Convert OmegaConf / Mapping configs to a plain dict."""
    if cfg is None:
        return {}
    if hasattr(cfg, "items") and not isinstance(cfg, dict):
        # OmegaConf DictConfig supports .items() and container conversion
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


def _pop_name(section: MutableMapping[str, Any], default: str | None = None) -> str:
    """Extract and remove the ``name`` key from a section dict."""
    name = section.pop("name", default)
    if name is None:
        raise KeyError("Component config must include a 'name' field.")
    return str(name)


def build_model(cfg: Any) -> ModularUNet:
    """Build a :class:`ModularUNet` from an OmegaConf/dict configuration.

    Expected structure::

        model:
          name: unet
          in_channels: 1
          out_channels: 1
          num_classes: 1
          spatial_dims: 2
          return_features: false
          encoder: {name: unet, channels: [32, 64, 128, 256], ...}
          bottleneck: {name: conv, ...}
          decoder: {name: unet, ...}
          heads:
            bottleneck_reconstruction: {...}
            reconstruction: {...}
            segmentation: {...}

    The top-level key may be the model section itself or a parent dict that
    contains a ``model`` key.

    Args:
        cfg: Model configuration (dict or OmegaConf).

    Returns:
        An initialized :class:`ModularUNet`.
    """
    root = _to_plain_dict(cfg)
    if "model" in root and isinstance(root["model"], (Mapping, dict)):
        model_cfg = _to_plain_dict(root["model"])
    else:
        model_cfg = root

    in_channels = int(model_cfg.get("in_channels", 1))
    out_channels = int(model_cfg.get("out_channels", 1))
    num_classes = int(model_cfg.get("num_classes", 1))
    spatial_dims = int(model_cfg.get("spatial_dims", model_cfg.get("dim", 2)))
    return_features = bool(model_cfg.get("return_features", False))
    norm = model_cfg.get("norm", "batch")
    activation = model_cfg.get("activation", "relu")
    num_groups = int(model_cfg.get("num_groups", 8))
    dropout = float(model_cfg.get("dropout", 0.0))

    # ---- Encoder ----
    enc_cfg = _to_plain_dict(model_cfg.get("encoder", {"name": "unet"}))
    enc_name = _pop_name(enc_cfg, default="unet")
    enc_cfg.setdefault("in_channels", in_channels)
    enc_cfg.setdefault("dim", spatial_dims)
    enc_cfg.setdefault("norm", norm)
    enc_cfg.setdefault("activation", activation)
    enc_cfg.setdefault("num_groups", num_groups)
    enc_cfg.setdefault("dropout", dropout)
    if "channels" not in enc_cfg:
        enc_cfg["channels"] = [32, 64, 128, 256]
    encoder = build_encoder(enc_name, **enc_cfg)
    skip_channels = list(encoder.out_channels)

    # ---- Bottleneck ----
    btn_cfg = _to_plain_dict(model_cfg.get("bottleneck", {"name": "conv"}))
    btn_name = _pop_name(btn_cfg, default="conv")
    btn_cfg.setdefault("in_channels", skip_channels[-1])
    btn_cfg.setdefault("out_channels", skip_channels[-1])
    btn_cfg.setdefault("dim", spatial_dims)
    btn_cfg.setdefault("norm", norm)
    btn_cfg.setdefault("activation", activation)
    btn_cfg.setdefault("num_groups", num_groups)
    btn_cfg.setdefault("dropout", dropout)
    bottleneck = build_bottleneck(btn_name, **btn_cfg)
    bottleneck_channels = int(bottleneck.out_channels)

    # ---- Decoder ----
    dec_cfg = _to_plain_dict(model_cfg.get("decoder", {"name": "unet"}))
    dec_name = _pop_name(dec_cfg, default="unet")
    dec_cfg.setdefault("in_channels", bottleneck_channels)
    dec_cfg.setdefault("skip_channels", skip_channels)
    dec_cfg.setdefault("dim", spatial_dims)
    dec_cfg.setdefault("norm", norm)
    dec_cfg.setdefault("activation", activation)
    dec_cfg.setdefault("num_groups", num_groups)
    dec_cfg.setdefault("dropout", dropout)
    decoder = build_decoder(dec_name, **dec_cfg)
    decoded_channels = int(decoder.out_channels)

    # ---- Heads ----
    heads_cfg = _to_plain_dict(model_cfg.get("heads", {}))

    # Bottleneck reconstruction head
    btn_head = None
    btn_head_cfg = heads_cfg.get("bottleneck_reconstruction")
    if btn_head_cfg is False:
        btn_head = None
    else:
        btn_head_cfg = _to_plain_dict(
            btn_head_cfg if btn_head_cfg is not None else {}
        )
        if btn_head_cfg.pop("enabled", True) is not False:
            btn_head_name = _pop_name(btn_head_cfg, default="bottleneck_reconstruction")
            btn_head_cfg.setdefault("in_channels", bottleneck_channels)
            btn_head_cfg.setdefault("out_channels", out_channels)
            btn_head_cfg.setdefault("num_upsamples", max(len(skip_channels) - 1, 0))
            btn_head_cfg.setdefault("dim", spatial_dims)
            btn_head_cfg.setdefault("norm", norm)
            btn_head_cfg.setdefault("activation", activation)
            btn_head_cfg.setdefault("num_groups", num_groups)
            btn_head = build_head(btn_head_name, **btn_head_cfg)

    # Image reconstruction head
    recon_head = None
    recon_head_cfg = heads_cfg.get("reconstruction", heads_cfg.get("image_reconstruction"))
    if recon_head_cfg is False:
        recon_head = None
    else:
        recon_head_cfg = _to_plain_dict(
            recon_head_cfg if recon_head_cfg is not None else {}
        )
        if recon_head_cfg.pop("enabled", True) is not False:
            recon_head_name = _pop_name(recon_head_cfg, default="image_reconstruction")
            recon_head_cfg.setdefault("in_channels", decoded_channels)
            recon_head_cfg.setdefault("out_channels", out_channels)
            recon_head_cfg.setdefault("dim", spatial_dims)
            recon_head = build_head(recon_head_name, **recon_head_cfg)

    # Segmentation head
    seg_head = None
    seg_head_cfg = heads_cfg.get("segmentation")
    if seg_head_cfg is False:
        seg_head = None
    else:
        seg_head_cfg = _to_plain_dict(
            seg_head_cfg if seg_head_cfg is not None else {}
        )
        if seg_head_cfg.pop("enabled", True) is not False:
            seg_head_name = _pop_name(seg_head_cfg, default="segmentation")
            seg_head_cfg.setdefault("in_channels", decoded_channels)
            seg_head_cfg.setdefault("num_classes", num_classes)
            seg_head_cfg.setdefault("dim", spatial_dims)
            seg_head = build_head(seg_head_name, **seg_head_cfg)

    return ModularUNet(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        bottleneck_reconstruction_head=btn_head,
        reconstruction_head=recon_head,
        segmentation_head=seg_head,
        return_features=return_features,
    )
