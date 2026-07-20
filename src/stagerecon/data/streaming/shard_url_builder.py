"""Helpers to expand shard URL patterns for WebDataset pipelines."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence


_BRACE_RANGE_RE = re.compile(
    r"\{(?P<start>\d+)\.\.(?P<end>\d+)(?P<step>::-?\d+)?\}"
)


def expand_brace_urls(pattern: str) -> list[str]:
    """Expand bash-like numeric brace ranges in a URL/path pattern.

    Examples:
        ``shard-{000000..000002}.tar`` ->
        ``shard-000000.tar``, ``shard-000001.tar``, ``shard-000002.tar``

        ``pipe:curl -s https://example.com/shard-{0..2}.tar`` is also supported.
    """
    match = _BRACE_RANGE_RE.search(pattern)
    if match is None:
        return [pattern]

    start_s = match.group("start")
    end_s = match.group("end")
    step_s = match.group("step")
    start = int(start_s)
    end = int(end_s)
    step = 1
    if step_s:
        step = int(step_s[2:])
        if step == 0:
            raise ValueError(f"Brace expansion step cannot be 0 in {pattern!r}")

    width = max(len(start_s), len(end_s))
    # Preserve zero-padding when either bound is zero-padded.
    zero_pad = start_s.startswith("0") or end_s.startswith("0") or (
        len(start_s) > 1 or len(end_s) > 1
    )

    if step > 0 and end < start:
        start, end, step = end, start, abs(step)
    if step < 0 and end > start:
        # Empty range under reverse step semantics.
        return []

    values: list[int] = []
    cur = start
    if step > 0:
        while cur <= end:
            values.append(cur)
            cur += step
    else:
        while cur >= end:
            values.append(cur)
            cur += step

    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    expanded: list[str] = []
    for value in values:
        token = f"{value:0{width}d}" if zero_pad else str(value)
        # Recurse to support multiple brace expressions.
        expanded.extend(expand_brace_urls(f"{prefix}{token}{suffix}"))
    return expanded


def _normalize_url(url: str) -> str:
    """Normalize local paths while preserving remote / pipe URLs."""
    if url.startswith(("s3://", "http://", "https://", "pipe:", "file:")):
        return url
    path = Path(url).expanduser()
    # Keep relative paths stable for local shards; absolute-ize existing ones.
    if path.exists():
        return str(path.resolve())
    return str(path)


def build_shard_urls(cfg: Any) -> list[str]:
    """Build a list of shard URLs/paths from config.

    Supported config keys (any of):
    - ``shards`` / ``urls`` / ``shard_urls``: str | list[str] patterns
    - ``shard_pattern``: single pattern with optional brace expansion
    - ``root`` + ``pattern``: join local root with a relative pattern

    Protocols supported in patterns:
    - local filesystem paths
    - ``s3://``
    - ``http://`` / ``https://``
    - ``pipe:`` commands

    Brace expansion like ``shard-{000000..000009}.tar`` is expanded eagerly.

    Args:
        cfg: Mapping / OmegaConf / bare string / sequence of patterns.

    Returns:
        Expanded list of shard URL strings.
    """
    if cfg is None:
        raise ValueError("Shard URL config is required")

    if isinstance(cfg, str):
        patterns: list[str] = [cfg]
    elif isinstance(cfg, Sequence) and not isinstance(cfg, (str, bytes)):
        patterns = [str(x) for x in cfg]
    elif isinstance(cfg, Mapping) or hasattr(cfg, "items"):
        plain: dict[str, Any]
        if hasattr(cfg, "items") and not isinstance(cfg, dict):
            try:
                from omegaconf import OmegaConf

                if OmegaConf.is_config(cfg):
                    plain = dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
                else:
                    plain = {str(k): v for k, v in cfg.items()}
            except Exception:
                plain = {str(k): v for k, v in cfg.items()}
        else:
            plain = dict(cfg)  # type: ignore[arg-type]

        patterns = []
        for key in ("shards", "urls", "shard_urls", "shard_pattern", "url"):
            if key in plain and plain[key] is not None:
                value = plain[key]
                if isinstance(value, (list, tuple)):
                    patterns.extend(str(v) for v in value)
                else:
                    patterns.append(str(value))

        root = plain.get("root")
        pattern = plain.get("pattern")
        if root is not None and pattern is not None:
            root_s = str(root).rstrip("/")
            pat_s = str(pattern).lstrip("/")
            if root_s.startswith(("s3://", "http://", "https://", "pipe:")):
                patterns.append(f"{root_s}/{pat_s}")
            else:
                patterns.append(str(Path(root_s) / pat_s))

        if not patterns:
            raise KeyError(
                "Shard config must include one of: shards, urls, shard_urls, "
                "shard_pattern, url, or root+pattern"
            )
    else:
        raise TypeError(f"Unsupported shard config type: {type(cfg)!r}")

    expanded: list[str] = []
    for pattern in patterns:
        expanded.extend(expand_brace_urls(pattern))

    urls = [_normalize_url(u) for u in expanded]
    if not urls:
        raise ValueError("No shard URLs produced from config")
    return urls
