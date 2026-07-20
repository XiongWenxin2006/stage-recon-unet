"""Flexible OmegaConf / dict accessors shared by experiment runners."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

_INTERP_RE = re.compile(r"\$\{([^}]+)\}")


def to_plain(cfg: Any, *, resolve: bool = True) -> dict[str, Any]:
    """Convert OmegaConf / Mapping to a plain dict."""
    if cfg is None:
        return {}
    if hasattr(cfg, "items") and not isinstance(cfg, dict):
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(cfg):
                container = OmegaConf.to_container(cfg, resolve=resolve)
                return dict(container) if isinstance(container, dict) else {}
        except Exception:
            pass
        try:
            return {str(k): v for k, v in cfg.items()}
        except Exception:
            return {}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {}


def get_section(root: Mapping[str, Any], *names: str) -> dict[str, Any]:
    """Return the first present mapping section among ``names``."""
    for name in names:
        if name in root and root[name] is not None:
            section = root[name]
            if isinstance(section, Mapping):
                return to_plain(section)
            # Allow OmegaConf nodes
            plain = to_plain(section)
            if plain:
                return plain
    return {}


def deep_get(root: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch a nested value via dotted path (e.g. ``paths.output_dir``)."""
    cur: Any = root
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def get_paths(cfg: Any) -> dict[str, Any]:
    root = to_plain(cfg)
    return get_section(root, "paths")


def get_checkpoint_dir(cfg: Any) -> Path:
    """Resolve the directory used for stage checkpoints."""
    root = to_plain(cfg)
    paths = get_paths(root)
    if paths.get("checkpoint_dir"):
        return Path(str(paths["checkpoint_dir"])).expanduser()
    if paths.get("checkpoints"):
        return Path(str(paths["checkpoints"])).expanduser()
    output_dir = paths.get("output_dir") or root.get("output_dir") or "outputs"
    return Path(str(output_dir)).expanduser() / "checkpoints"


def get_output_dir(cfg: Any) -> Path:
    root = to_plain(cfg)
    paths = get_paths(root)
    output_dir = paths.get("output_dir") or root.get("output_dir") or "outputs"
    return Path(str(output_dir)).expanduser()


def _lookup_path_ref(root: Mapping[str, Any], ref: str) -> str | None:
    value = deep_get(root, ref)
    if value is None:
        # Also allow bare keys under paths.*
        if not ref.startswith("paths."):
            value = deep_get(root, f"paths.{ref}")
    if value is None:
        return None
    return str(value)


def resolve_path_string(
    value: str,
    cfg: Any,
    *,
    base_dir: str | Path | None = None,
) -> str:
    """Resolve ``${paths.*}`` interpolations and relative checkpoint paths.

    Assumes OmegaConf may already have resolved interpolations. Remaining
    ``${...}`` tokens are expanded from ``cfg`` / ``cfg.paths``.

    Relative handling:

    * bare filenames (``stage1_best.pt``) → joined with ``base_dir``
      (default: checkpoint dir)
    * paths that already include ``outputs/`` or ``.../checkpoints/...``
      (typical after OmegaConf resolves ``${paths.stage1_checkpoint}``)
      → resolved against the process CWD, not re-joined under checkpoint dir
    """
    root = to_plain(cfg, resolve=False)
    # Try resolved view as well for lookups
    root_resolved = to_plain(cfg, resolve=True)
    merged = {**root, **root_resolved}
    if "paths" in root or "paths" in root_resolved:
        paths = {
            **get_section(root, "paths"),
            **get_section(root_resolved, "paths"),
        }
        merged["paths"] = paths

    text = str(value)

    def _replace(match: re.Match[str]) -> str:
        ref = match.group(1).strip()
        found = _lookup_path_ref(merged, ref)
        if found is None:
            return match.group(0)
        return found

    # Expand nested interpolations a few times
    for _ in range(5):
        new_text = _INTERP_RE.sub(_replace, text)
        if new_text == text:
            break
        text = new_text

    if _INTERP_RE.search(text):
        # Leave unresolved tokens as-is; caller may still succeed if unused
        return text

    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)

    ckpt_dir = get_checkpoint_dir(cfg)
    out_dir = get_output_dir(cfg)
    base = Path(base_dir) if base_dir is not None else ckpt_dir

    text_norm = text.replace("\\", "/").lstrip("./")
    ckpt_norm = str(ckpt_dir).replace("\\", "/").lstrip("./")
    out_norm = str(out_dir).replace("\\", "/").lstrip("./")

    already_rooted = (
        text_norm.startswith(ckpt_norm + "/")
        or text_norm == ckpt_norm
        or text_norm.startswith(out_norm + "/")
        or text_norm == out_norm
        or text_norm.startswith("outputs/")
        or "/checkpoints/" in f"/{text_norm}"
    )
    if already_rooted:
        return str(path.resolve())

    # Bare filename or short relative path → under checkpoint / base dir
    return str((base / path).resolve())


def resolve_stage_paths(stage_cfg: Mapping[str, Any], cfg: Any) -> dict[str, Any]:
    """Return a deep-copied stage config with checkpoint paths resolved."""
    data = deepcopy(dict(stage_cfg))
    ckpt_dir = get_checkpoint_dir(cfg)

    if data.get("checkpoint_output"):
        data["checkpoint_output"] = resolve_path_string(
            str(data["checkpoint_output"]), cfg, base_dir=ckpt_dir
        )
    if data.get("checkpoint_input"):
        data["checkpoint_input"] = resolve_path_string(
            str(data["checkpoint_input"]), cfg, base_dir=ckpt_dir
        )
    if data.get("save_dir"):
        data["save_dir"] = resolve_path_string(
            str(data["save_dir"]), cfg, base_dir=ckpt_dir
        )
    if data.get("checkpoint_dir"):
        data["checkpoint_dir"] = resolve_path_string(
            str(data["checkpoint_dir"]), cfg, base_dir=ckpt_dir
        )

    init = data.get("module_initialization")
    if isinstance(init, Mapping):
        resolved_init: dict[str, Any] = {}
        for module_name, mod_cfg in init.items():
            mod_plain = to_plain(mod_cfg) if not isinstance(mod_cfg, dict) else dict(mod_cfg)
            ckpt = mod_plain.get("checkpoint_path")
            if ckpt:
                mod_plain["checkpoint_path"] = resolve_path_string(
                    str(ckpt), cfg, base_dir=ckpt_dir
                )
            resolved_init[str(module_name)] = mod_plain
        data["module_initialization"] = resolved_init

    return data


_STAGE_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "stage1": ("stage1", "stage_1", "encoder_bottleneck", "pretrain_stage1"),
    "stage2": ("stage2", "stage_2", "bottleneck_decoder", "pretrain_stage2"),
    "stage3": ("stage3", "stage_3", "full_reconstruction", "pretrain_stage3"),
    "downstream": (
        "downstream",
        "downstream_segmentation",
        "segmentation",
        "finetune",
    ),
}


def normalize_stage_name(name: str) -> str:
    """Map aliases to canonical ``stage1|stage2|stage3|downstream``."""
    key = str(name).lower().strip()
    for canonical, aliases in _STAGE_NAME_ALIASES.items():
        if key == canonical or key in aliases:
            return canonical
    return key


def is_downstream_stage(stage_name: str) -> bool:
    return normalize_stage_name(stage_name) == "downstream"


def get_trainer_cfg(cfg: Any) -> dict[str, Any]:
    """Return trainer settings with aliases normalized for :class:`Trainer`."""
    root = to_plain(cfg)
    trainer = get_section(root, "trainer", "train", "training")
    # Normalize common Hydra config aliases → Trainer keys
    if "epochs" not in trainer and "max_epochs" in trainer:
        trainer["epochs"] = trainer["max_epochs"]
    if "grad_clip" not in trainer and "grad_clip_norm" in trainer:
        trainer["grad_clip"] = trainer["grad_clip_norm"]
    if "max_grad_norm" not in trainer and "grad_clip_norm" in trainer:
        trainer["max_grad_norm"] = trainer["grad_clip_norm"]
    if "accumulation_steps" not in trainer:
        for key in ("grad_accumulation_steps", "grad_accumulation", "gradient_accumulation"):
            if key in trainer:
                trainer["accumulation_steps"] = trainer[key]
                break
    if "amp" not in trainer and "amp" in root:
        trainer["amp"] = root["amp"]
    if "device" not in trainer and "device" in root:
        trainer["device"] = root["device"]
    if "seed" not in trainer and "seed" in root:
        trainer["seed"] = root["seed"]
    # Monitor from checkpoint section when present
    ckpt = get_section(root, "checkpoint")
    if "monitor" not in trainer and ckpt.get("monitor"):
        trainer["monitor"] = ckpt["monitor"]
    if "monitor_mode" not in trainer and ckpt.get("mode"):
        trainer["monitor_mode"] = ckpt["mode"]
    return trainer


def get_model_cfg(cfg: Any) -> Any:
    root = to_plain(cfg)
    if "model" in root:
        return root["model"]
    return cfg


def get_data_cfg(
    cfg: Any,
    *,
    split: str | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    """Fetch data / data_source config, optionally selecting a split + task."""
    root = to_plain(cfg)
    data = get_section(root, "data", "data_source", "dataset")
    if not data:
        return {}

    # Nested split sections: data.train / data.val
    if split is not None:
        split_key = str(split).lower()
        for key in (split_key, f"{split_key}_data"):
            if key in data and isinstance(data[key], Mapping):
                data = {**data, **to_plain(data[key])}
                break
        # Common pattern: data.splits.train
        splits = data.get("splits")
        if isinstance(splits, Mapping) and split_key in splits:
            nested = to_plain(splits[split_key])
            data = {**data, **nested}

    if task is not None:
        data = dict(data)
        data["task"] = task
        data.setdefault("mode", task)
        if task in {"segmentation", "seg"}:
            data["return_mask"] = True
            # Prefer task-specific transform subtree when present
            transforms = data.get("transforms")
            if isinstance(transforms, Mapping):
                if "segmentation" in transforms:
                    data["transforms"] = transforms["segmentation"]
                elif "reconstruction" in transforms and "segmentation" not in transforms:
                    # Avoid feeding recon corruption specs into seg factory
                    data["transforms"] = None
        elif task in {"reconstruction", "recon", "pretrain"}:
            data["return_mask"] = False
            data["task"] = "reconstruction"
            transforms = data.get("transforms")
            if isinstance(transforms, Mapping) and "reconstruction" in transforms:
                data["transforms"] = transforms["reconstruction"]

    return data


def get_dataloader_cfg(cfg: Any, *, split: str = "train") -> dict[str, Any]:
    root = to_plain(cfg)
    loader = get_section(root, "dataloader", "loader")
    trainer = get_trainer_cfg(root)
    # Prefer split-specific overrides
    for key in (f"{split}_dataloader", f"dataloader_{split}"):
        if key in root and isinstance(root[key], Mapping):
            loader = {**loader, **to_plain(root[key])}
    # Pull common keys from trainer / data if missing
    data = get_section(root, "data", "data_source")
    for key in ("batch_size", "num_workers", "pin_memory", "drop_last"):
        if key not in loader:
            if key in trainer:
                loader[key] = trainer[key]
            elif key in data:
                loader[key] = data[key]
    if split == "train":
        loader.setdefault("shuffle", True)
    else:
        loader.setdefault("shuffle", False)
    return loader


def get_optimizer_cfg(cfg: Any) -> dict[str, Any]:
    root = to_plain(cfg)
    opt = get_section(root, "optimizer")
    if opt:
        return opt
    trainer = get_trainer_cfg(root)
    nested = trainer.get("optimizer")
    if isinstance(nested, Mapping):
        return to_plain(nested)
    # Flat trainer keys
    flat = {
        k: trainer[k]
        for k in ("name", "type", "lr", "weight_decay", "betas", "momentum", "nesterov")
        if k in trainer
    }
    return flat or {"name": "adam", "lr": 1e-3}


def get_scheduler_cfg(cfg: Any) -> dict[str, Any]:
    root = to_plain(cfg)
    sch = get_section(root, "scheduler")
    if not sch:
        trainer = get_trainer_cfg(root)
        nested = trainer.get("scheduler")
        if isinstance(nested, Mapping):
            sch = to_plain(nested)
    if not sch:
        return {}
    # Fill cosine T_max from trainer epochs when omitted
    name = str(sch.get("name") or sch.get("type") or "").lower()
    if name in {"cosine", "cosineannealing", "cosine_annealing"}:
        if "T_max" not in sch and "t_max" not in sch and "epochs" not in sch:
            trainer = get_trainer_cfg(root)
            if trainer.get("epochs") is not None:
                sch = {**sch, "T_max": int(trainer["epochs"])}
    return sch


def get_loss_cfg(cfg: Any, stage_loss_name: str | None = None) -> Any:
    root = to_plain(cfg)
    if "loss" in root and root["loss"] is not None:
        return root["loss"]
    trainer = get_trainer_cfg(root)
    if "loss" in trainer and trainer["loss"] is not None:
        return trainer["loss"]
    name = stage_loss_name or "mse"
    # Map legacy / shorthand names onto build_loss vocabulary
    aliases = {
        "ce": "bce",
        "cross_entropy": "bce",
        "crossentropy": "bce",
        "l2": "mse",
    }
    return aliases.get(str(name).lower().strip(), name)


def _stage_plain_matches(plain: Mapping[str, Any], canonical: str, aliases: Sequence[str]) -> bool:
    """Return True when a stage dict's name/type matches the requested stage."""
    tokens = {canonical, *aliases}
    for key in ("type", "stage_type", "kind", "name"):
        value = plain.get(key)
        if value is None:
            continue
        normalized = normalize_stage_name(str(value))
        if normalized in tokens or str(value).lower().strip() in tokens:
            return True
    return False


def _coerce_stage_plain(cand: Any, canonical: str) -> dict[str, Any] | None:
    plain = to_plain(cand)
    if not plain:
        return None
    if "stage" in plain and isinstance(plain["stage"], Mapping):
        nested = to_plain(plain["stage"])
        if nested:
            plain = nested
    if not any(
        k in plain
        for k in (
            "forward_mode",
            "trainable_modules",
            "module_initialization",
            "type",
            "stage_type",
            "kind",
            "name",
            "checkpoint_output",
        )
    ):
        return None
    out = dict(plain)
    out.setdefault("name", canonical)
    out.setdefault("type", out.get("type", canonical))
    return out


def extract_stage_cfg(cfg: Any, stage_name: str) -> dict[str, Any]:
    """Locate a stage specification block for ``stage_name`` in the config tree."""
    root = to_plain(cfg)
    canonical = normalize_stage_name(stage_name)
    aliases = _STAGE_NAME_ALIASES.get(canonical, (canonical,))

    named_candidates: list[Any] = []
    generic_candidates: list[Any] = []

    # Prefer explicitly named stage blocks
    pretrain = get_section(root, "pretrain")
    if pretrain:
        stages = pretrain.get("stages")
        if isinstance(stages, Mapping):
            for alias in aliases:
                if alias in stages:
                    named_candidates.append(stages[alias])
        for alias in aliases:
            if alias in pretrain:
                named_candidates.append(pretrain[alias])

    for section_name in ("stages", "stage_configs"):
        section = root.get(section_name)
        if isinstance(section, Mapping):
            for alias in aliases:
                if alias in section:
                    named_candidates.append(section[alias])

    exp = get_section(root, "experiment")
    stages_cfg = exp.get("stages_config") or exp.get("stage_configs")
    if isinstance(stages_cfg, Mapping):
        for alias in aliases:
            if alias in stages_cfg:
                named_candidates.append(stages_cfg[alias])

    if canonical == "downstream" and "downstream" in root:
        named_candidates.append(root["downstream"])

    # Generic single-stage slots (used only when they match requested name/type)
    if "stage" in root:
        generic_candidates.append(root["stage"])
    if pretrain and "stage" in pretrain:
        generic_candidates.append(pretrain["stage"])

    for cand in named_candidates:
        plain = _coerce_stage_plain(cand, canonical)
        if plain is not None:
            return plain

    for cand in generic_candidates:
        plain = _coerce_stage_plain(cand, canonical)
        if plain is None:
            continue
        if _stage_plain_matches(plain, canonical, aliases):
            return plain

    # Last resort: accept generic stage block when it is the only stage config
    # (single-stage Hydra configs often use cfg.stage without an explicit name).
    if len(generic_candidates) == 1 and not named_candidates:
        plain = _coerce_stage_plain(generic_candidates[0], canonical)
        if plain is not None:
            return plain

    # Fallback: synthesize a minimal type-only stage config
    return {"name": canonical, "type": canonical}


def default_stages_for_experiment_type(exp_type: str) -> list[str]:
    key = str(exp_type).lower().strip()
    if key in {
        "pipeline",
        "full",
        "full_pipeline",
        "end_to_end",
        "staged_pipeline",
        "staged_pretrain",
    }:
        return ["stage1", "stage2", "stage3", "downstream"]
    if key in {"pretrain", "pretrain_pipeline", "pretraining"}:
        return ["stage1", "stage2", "stage3"]
    if key in {"stage1", "pretrain_stage1"}:
        return ["stage1"]
    if key in {"stage2", "pretrain_stage2"}:
        return ["stage2"]
    if key in {"stage3", "pretrain_stage3"}:
        return ["stage3"]
    if key in {"downstream", "segmentation", "finetune"}:
        return ["downstream"]
    if key in {"skip_stage2", "ablation_skip_stage2"}:
        return ["stage1", "stage3", "downstream"]
    if key in {"single", "single_stage", "stage"}:
        return []  # caller must supply stages / stage
    return ["stage1", "stage2", "stage3", "downstream"]


def resolve_stages_list(cfg: Any, stages: Sequence[str] | None = None) -> list[str]:
    """Determine which stages to run from override / experiment config."""
    if stages is not None:
        return [normalize_stage_name(s) for s in stages]

    root = to_plain(cfg)
    exp = get_section(root, "experiment")

    if exp.get("stages") is not None:
        return [normalize_stage_name(s) for s in exp["stages"]]

    exp_type = str(exp.get("type", root.get("type", "pipeline")))
    resolved = default_stages_for_experiment_type(exp_type)
    if resolved:
        return resolved

    # Single-stage configs: use cfg.stage.name / type
    stage_cfg = extract_stage_cfg(cfg, "stage1")
    name = stage_cfg.get("name") or stage_cfg.get("type") or "stage1"
    return [normalize_stage_name(str(name))]


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
