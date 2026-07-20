"""Configuration consistency checks for StageRecon experiments."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf


def _to_plain(cfg: Any) -> dict[str, Any]:
    """Convert OmegaConf / Mapping to a plain dict."""
    if cfg is None:
        return {}
    if OmegaConf.is_config(cfg):
        container = OmegaConf.to_container(cfg, resolve=True)
        return dict(container) if isinstance(container, dict) else {}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Unsupported config type: {type(cfg)!r}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _get_section(root: Mapping[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        if name in root and root[name] is not None:
            section = root[name]
            if isinstance(section, Mapping):
                return dict(section)
    return {}


def _collect_checkpoint_deps(node: Any, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    """Find nested dicts that declare ``source: checkpoint``."""
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, Mapping):
        plain = dict(node)
        source = plain.get("source")
        if isinstance(source, str) and source.lower() == "checkpoint":
            found.append((path or "<root>", plain))
        for key, value in plain.items():
            child_path = f"{path}.{key}" if path else str(key)
            found.extend(_collect_checkpoint_deps(value, child_path))
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for idx, value in enumerate(node):
            child_path = f"{path}[{idx}]"
            found.extend(_collect_checkpoint_deps(value, child_path))
    return found


def validate_config(cfg: Any) -> None:
    """Validate cross-field consistency of a StageRecon config.

    Checks performed:

    * ``data`` channel count matches ``model.in_channels``
    * ``data`` / ``model`` ``num_classes`` matches segmentation head out channels
    * ``spatial_dims`` (or ``dim``) is consistent across data and model sections
    * ``trainable_modules`` and ``frozen_modules`` do not overlap
    * Any component with ``source: checkpoint`` must provide a non-empty path

    Args:
        cfg: Full experiment config (dict or OmegaConf).

    Raises:
        ValueError: If any consistency check fails.
        TypeError: If ``cfg`` is not a mapping / OmegaConf config.
    """
    root = _to_plain(cfg)
    errors: list[str] = []

    data_cfg = _get_section(root, "data")
    model_cfg = _get_section(root, "model")
    training_cfg = _get_section(root, "training", "train")
    modules_cfg = _get_section(root, "modules", "module_sources", "init")

    # ---- channels ----
    data_channels = data_cfg.get("channels", data_cfg.get("in_channels", data_cfg.get("num_channels")))
    model_in_channels = model_cfg.get("in_channels", model_cfg.get("channels"))
    if data_channels is not None and model_in_channels is not None:
        if int(data_channels) != int(model_in_channels):
            errors.append(
                f"data channels ({data_channels}) != model.in_channels ({model_in_channels})"
            )

    # ---- spatial_dims ----
    data_dims = data_cfg.get("spatial_dims", data_cfg.get("dim"))
    model_dims = model_cfg.get("spatial_dims", model_cfg.get("dim"))
    if data_dims is not None and model_dims is not None:
        if int(data_dims) != int(model_dims):
            errors.append(
                f"data spatial_dims ({data_dims}) != model spatial_dims ({model_dims})"
            )

    # Also check encoder/decoder/bottleneck/head dim fields when present
    for section_name in ("encoder", "bottleneck", "decoder"):
        section = model_cfg.get(section_name)
        if isinstance(section, Mapping) and model_dims is not None:
            section_dims = section.get("spatial_dims", section.get("dim"))
            if section_dims is not None and int(section_dims) != int(model_dims):
                errors.append(
                    f"model.{section_name} spatial_dims/dim ({section_dims}) "
                    f"!= model spatial_dims ({model_dims})"
                )

    # ---- num_classes vs segmentation out channels ----
    num_classes = model_cfg.get("num_classes", data_cfg.get("num_classes"))
    heads = model_cfg.get("heads", {})
    seg_head: dict[str, Any] = {}
    if isinstance(heads, Mapping):
        seg_raw = heads.get("segmentation")
        if isinstance(seg_raw, Mapping):
            seg_head = dict(seg_raw)

    if seg_head and seg_head.get("enabled", True) is not False:
        seg_out = seg_head.get(
            "num_classes",
            seg_head.get("out_channels", model_cfg.get("num_classes")),
        )
        if num_classes is not None and seg_out is not None:
            if int(num_classes) != int(seg_out):
                errors.append(
                    f"num_classes ({num_classes}) != segmentation out channels "
                    f"/ num_classes ({seg_out})"
                )
        data_classes = data_cfg.get("num_classes")
        if data_classes is not None and seg_out is not None:
            if int(data_classes) != int(seg_out):
                errors.append(
                    f"data.num_classes ({data_classes}) != segmentation out channels "
                    f"/ num_classes ({seg_out})"
                )

    # ---- trainable / frozen overlap ----
    trainable = {
        str(x)
        for x in _as_list(
            training_cfg.get(
                "trainable_modules",
                root.get("trainable_modules", modules_cfg.get("trainable_modules")),
            )
        )
    }
    frozen = {
        str(x)
        for x in _as_list(
            training_cfg.get(
                "frozen_modules",
                root.get("frozen_modules", modules_cfg.get("frozen_modules")),
            )
        )
    }
    overlap = sorted(trainable & frozen)
    if overlap:
        errors.append(
            "trainable_modules and frozen_modules overlap: "
            + ", ".join(overlap)
        )

    # ---- checkpoint dependencies ----
    search_roots: list[tuple[str, Any]] = [
        ("modules", modules_cfg if modules_cfg else root.get("modules")),
        ("model", model_cfg),
        ("training", training_cfg),
        ("checkpoint", root.get("checkpoint")),
        ("init", root.get("init")),
    ]
    for base_name, node in search_roots:
        if node is None:
            continue
        for dep_path, dep in _collect_checkpoint_deps(node, base_name):
            ckpt_path = (
                dep.get("path")
                or dep.get("checkpoint")
                or dep.get("checkpoint_path")
                or dep.get("ckpt")
                or dep.get("ckpt_path")
            )
            if ckpt_path is None or (isinstance(ckpt_path, str) and not ckpt_path.strip()):
                errors.append(
                    f"checkpoint dependency at '{dep_path}' has source='checkpoint' "
                    "but no path/checkpoint_path is set"
                )

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"Invalid configuration:\n  - {joined}")
