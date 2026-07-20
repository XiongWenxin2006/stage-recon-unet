"""Configuration loading and path-resolution helpers based on OmegaConf."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def get_project_root() -> Path:
    """Return the repository root (parent of ``src/``).

    Resolves from this file's location::

        <root>/src/stagerecon/utils/config.py → <root>
    """
    return Path(__file__).resolve().parents[3]


def load_config(path: str | Path, *, resolve: bool = False) -> DictConfig:
    """Load a YAML/JSON config file into an OmegaConf ``DictConfig``.

    Args:
        path: Path to a config file.
        resolve: If True, resolve interpolations immediately after loading.

    Returns:
        Loaded configuration.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create({"_root": cfg})
    if resolve:
        OmegaConf.resolve(cfg)
    return cfg


def merge_configs(*configs: Any) -> DictConfig:
    """Deep-merge multiple configs (later entries override earlier ones).

    Args:
        *configs: Dicts, OmegaConf configs, or paths to YAML/JSON files.

    Returns:
        Merged ``DictConfig``.
    """
    merged: DictConfig = OmegaConf.create({})
    for item in configs:
        if item is None:
            continue
        if isinstance(item, (str, Path)):
            cfg = load_config(item)
        elif OmegaConf.is_config(item):
            cfg = item
        elif isinstance(item, Mapping):
            cfg = OmegaConf.create(dict(item))
        else:
            raise TypeError(f"Unsupported config type for merge: {type(item)!r}")
        merged = OmegaConf.merge(merged, cfg)  # type: ignore[assignment]
    return merged


def to_container(
    cfg: Any,
    *,
    resolve: bool = True,
    throw_on_missing: bool = False,
) -> Any:
    """Convert an OmegaConf config to plain Python containers.

    Args:
        cfg: OmegaConf config or plain object.
        resolve: Resolve interpolations before conversion.
        throw_on_missing: Raise if missing values remain.

    Returns:
        Nested ``dict`` / ``list`` structure (or the original object if not
        an OmegaConf config).
    """
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(
            cfg,
            resolve=resolve,
            throw_on_missing=throw_on_missing,
        )
    return cfg


_PATH_KEY_HINTS = frozenset(
    {
        "path",
        "paths",
        "root",
        "dir",
        "directory",
        "checkpoint",
        "checkpoint_path",
        "ckpt",
        "ckpt_path",
        "data_dir",
        "data_root",
        "output_dir",
        "log_dir",
        "save_dir",
        "cache_dir",
        "file",
        "filename",
        "filepath",
    }
)


def _looks_like_path_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _PATH_KEY_HINTS:
        return True
    return any(
        lowered.endswith(suffix)
        for suffix in ("_path", "_dir", "_root", "_file", "_checkpoint", "_ckpt")
    )


def _resolve_path_value(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def resolve_paths(
    cfg: Any,
    *,
    base_dir: str | Path | None = None,
    keys: Sequence[str] | None = None,
    inplace: bool = True,
) -> Any:
    """Resolve relative path strings in a config against the project root.

    By default, keys whose names look like path fields (e.g. ``path``,
    ``data_dir``, ``checkpoint_path``) are rewritten. Pass ``keys`` to
    restrict resolution to an explicit set of key names.

    Args:
        cfg: OmegaConf config or mutable mapping.
        base_dir: Directory used as the resolution root. Defaults to the
            project repository root.
        keys: Optional explicit key names to resolve. When provided, only
            these keys are considered.
        inplace: If True (default), mutate ``cfg``; otherwise operate on a
            deep copy.

    Returns:
        Config with relative paths expanded to absolute paths.
    """
    base = Path(base_dir).expanduser().resolve() if base_dir else get_project_root()
    allowed = set(keys) if keys is not None else None

    if not inplace:
        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        elif isinstance(cfg, Mapping):
            cfg = OmegaConf.create(dict(cfg))
        else:
            raise TypeError(f"Unsupported config type: {type(cfg)!r}")

    def _walk(node: Any) -> None:
        if isinstance(node, DictConfig) or isinstance(node, MutableMapping):
            items = list(node.items())
            for key, value in items:
                key_str = str(key)
                should_resolve = (
                    allowed is not None and key_str in allowed
                ) or (allowed is None and _looks_like_path_key(key_str))
                if should_resolve and isinstance(value, str) and value.strip():
                    node[key] = _resolve_path_value(value, base)
                elif isinstance(value, (DictConfig, ListConfig, Mapping, list)):
                    _walk(value)
        elif isinstance(node, ListConfig) or isinstance(node, list):
            for item in node:
                if isinstance(item, (DictConfig, ListConfig, Mapping, list)):
                    _walk(item)

    _walk(cfg)
    return cfg


def save_config(cfg: Any, path: str | Path) -> None:
    """Serialize a config to YAML on disk.

    Args:
        cfg: OmegaConf config or mapping.
        path: Destination file path.
    """
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    OmegaConf.save(cfg, out)
