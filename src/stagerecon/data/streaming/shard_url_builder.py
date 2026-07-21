"""Helpers to expand and normalize shard URL patterns for WebDataset pipelines.

Supports:
- local filesystem paths
- ``s3://`` (optionally rewritten to ``pipe:aws s3 cp ... -``)
- ``http://`` / ``https://``
- ``gs://`` (optionally rewritten to ``pipe:gsutil cat ...``)
- ``pipe:`` command streams
- brace expansion such as ``shard-{000000..000009}.tar``
"""

from __future__ import annotations

import os
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
        expanded.extend(expand_brace_urls(f"{prefix}{token}{suffix}"))
    return expanded


def _shell_quote(value: str) -> str:
    """Minimal POSIX single-quote escaping for pipe command arguments."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def rewrite_remote_url(
    url: str,
    *,
    s3_transport: str = "pipe_aws",
    gs_transport: str = "pipe_gsutil",
    http_transport: str = "raw",
) -> str:
    """Rewrite cloud object URLs into forms ``gopen`` / WebDataset can stream.

    Args:
        url: Original shard URL.
        s3_transport:
            - ``pipe_aws`` (default): ``pipe:aws s3 cp <url> -``
            - ``pipe_rclone``: ``pipe:rclone cat <url>``
            - ``raw``: leave ``s3://`` unchanged (requires a custom gopen handler)
        gs_transport:
            - ``pipe_gsutil`` (default): ``pipe:gsutil cat <url>``
            - ``raw``: leave ``gs://`` unchanged
        http_transport:
            - ``raw`` (default): leave HTTP(S) URLs as-is (WebDataset/gopen can fetch)
            - ``pipe_curl``: ``pipe:curl -fsSL <url>``

    Returns:
        Possibly rewritten URL string.

    Notes:
        Credentials must come from the environment / AWS shared config / IAM role.
        This function never embeds secrets.
    """
    if url.startswith("pipe:") or url.startswith("file:"):
        return url

    if url.startswith("s3://"):
        mode = (s3_transport or "pipe_aws").lower()
        if mode in {"pipe_aws", "aws", "pipe"}:
            return f"pipe:aws s3 cp {_shell_quote(url)} -"
        if mode in {"pipe_rclone", "rclone"}:
            return f"pipe:rclone cat {_shell_quote(url)}"
        if mode in {"raw", "none", "identity"}:
            return url
        raise ValueError(
            f"Unknown s3_transport={s3_transport!r}. "
            "Expected pipe_aws | pipe_rclone | raw."
        )

    if url.startswith("gs://"):
        mode = (gs_transport or "pipe_gsutil").lower()
        if mode in {"pipe_gsutil", "gsutil", "pipe"}:
            return f"pipe:gsutil cat {_shell_quote(url)}"
        if mode in {"raw", "none", "identity"}:
            return url
        raise ValueError(
            f"Unknown gs_transport={gs_transport!r}. Expected pipe_gsutil | raw."
        )

    if url.startswith(("http://", "https://")):
        mode = (http_transport or "raw").lower()
        if mode in {"pipe_curl", "curl"}:
            return f"pipe:curl -fsSL {_shell_quote(url)}"
        if mode in {"raw", "none", "identity"}:
            return url
        raise ValueError(
            f"Unknown http_transport={http_transport!r}. Expected raw | pipe_curl."
        )

    return url


def _normalize_local_url(url: str) -> str:
    """Normalize local paths while preserving remote / pipe URLs."""
    if url.startswith(("s3://", "gs://", "http://", "https://", "pipe:", "file:")):
        return url
    path = Path(url).expanduser()
    if path.exists():
        return str(path.resolve())
    return str(path)


def _extract_patterns(plain: Mapping[str, Any], *, split: str | None = None) -> list[str]:
    """Collect shard pattern strings from a config mapping."""
    patterns: list[str] = []

    # Split-specific overrides first.
    if split is not None:
        split_l = str(split).lower()
        split_aliases = {
            "train": ("train", "training"),
            "val": ("val", "validation", "valid"),
            "test": ("test", "testing"),
        }.get(split_l, (split_l,))

        for key in (
            f"{split_l}_shards",
            f"shards_{split_l}",
            f"{split_l}_urls",
        ):
            if key in plain and plain[key] is not None:
                value = plain[key]
                if isinstance(value, (list, tuple)):
                    patterns.extend(str(v) for v in value)
                else:
                    patterns.append(str(value))

        shards_section = plain.get("shards")
        if isinstance(shards_section, Mapping):
            for alias in split_aliases:
                if alias in shards_section and shards_section[alias] is not None:
                    value = shards_section[alias]
                    if isinstance(value, (list, tuple)):
                        patterns.extend(str(v) for v in value)
                    else:
                        patterns.append(str(value))
                    break
            # Also allow shards.urls under a split-neutral map when no split key matched.
            if not patterns and "urls" in shards_section:
                value = shards_section["urls"]
                if isinstance(value, (list, tuple)):
                    patterns.extend(str(v) for v in value)
                else:
                    patterns.append(str(value))

        splits = plain.get("splits")
        if isinstance(splits, Mapping):
            for alias in split_aliases:
                if alias in splits and isinstance(splits[alias], Mapping):
                    nested = splits[alias]
                    for nested_key in ("shards", "urls", "shard_urls", "shard_pattern"):
                        if nested_key in nested and nested[nested_key] is not None:
                            value = nested[nested_key]
                            if isinstance(value, (list, tuple)):
                                patterns.extend(str(v) for v in value)
                            else:
                                patterns.append(str(value))
                    break

    if patterns:
        return patterns

    # Neutral / global patterns.
    for key in ("shards", "urls", "shard_urls", "shard_pattern", "url"):
        if key not in plain or plain[key] is None:
            continue
        value = plain[key]
        if isinstance(value, Mapping):
            # Mapping without resolved split — try common keys only.
            for nested_key in ("urls", "pattern", "all", "default"):
                if nested_key in value and value[nested_key] is not None:
                    nested_val = value[nested_key]
                    if isinstance(nested_val, (list, tuple)):
                        patterns.extend(str(v) for v in nested_val)
                    else:
                        patterns.append(str(nested_val))
            continue
        if isinstance(value, (list, tuple)):
            patterns.extend(str(v) for v in value)
        else:
            patterns.append(str(value))

    root = plain.get("root")
    pattern = plain.get("pattern")
    if root is not None and pattern is not None:
        root_s = str(root).rstrip("/")
        pat_s = str(pattern).lstrip("/")
        if root_s.startswith(("s3://", "gs://", "http://", "https://", "pipe:")):
            patterns.append(f"{root_s}/{pat_s}")
        else:
            patterns.append(str(Path(root_s) / pat_s))

    return patterns


def build_shard_urls(cfg: Any, *, split: str | None = None) -> list[str]:
    """Build a list of shard URLs/paths from config.

    Supported config keys (any of):
    - ``shards`` / ``urls`` / ``shard_urls``: str | list[str] | mapping by split
    - ``shard_pattern``: single pattern with optional brace expansion
    - ``root`` + ``pattern``: join root with a relative pattern
    - ``train_shards`` / ``val_shards`` / ``test_shards``
    - ``splits.<split>.shards``

    Protocol options (optional):
    - ``s3_transport``: ``pipe_aws`` | ``pipe_rclone`` | ``raw``
    - ``gs_transport``: ``pipe_gsutil`` | ``raw``
    - ``http_transport``: ``raw`` | ``pipe_curl``
    - ``rewrite_remote``: bool (default True)

    Environment overrides (optional, never for secrets storage in code):
    - ``STAGERECON_S3_TRANSPORT``, ``STAGERECON_GS_TRANSPORT``,
      ``STAGERECON_HTTP_TRANSPORT``

    Args:
        cfg: Mapping / OmegaConf / bare string / sequence of patterns.
        split: Optional split name used to select split-specific shard patterns.

    Returns:
        Expanded, normalized list of shard URL strings.
    """
    if cfg is None:
        raise ValueError("Shard URL config is required")

    s3_transport = "pipe_aws"
    gs_transport = "pipe_gsutil"
    http_transport = "raw"
    rewrite_remote = True
    plain: dict[str, Any] = {}

    if isinstance(cfg, str):
        patterns: list[str] = [cfg]
    elif isinstance(cfg, Sequence) and not isinstance(cfg, (str, bytes)):
        patterns = [str(x) for x in cfg]
    elif isinstance(cfg, Mapping) or hasattr(cfg, "items"):
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

        patterns = _extract_patterns(plain, split=split)
        if not patterns:
            raise KeyError(
                "Shard config must include one of: shards, urls, shard_urls, "
                "shard_pattern, url, root+pattern, or split-specific shard keys"
                + (f" for split={split!r}" if split else "")
            )

        s3_transport = str(
            plain.get(
                "s3_transport",
                os.environ.get("STAGERECON_S3_TRANSPORT", "pipe_aws"),
            )
        )
        gs_transport = str(
            plain.get(
                "gs_transport",
                os.environ.get("STAGERECON_GS_TRANSPORT", "pipe_gsutil"),
            )
        )
        http_transport = str(
            plain.get(
                "http_transport",
                os.environ.get("STAGERECON_HTTP_TRANSPORT", "raw"),
            )
        )
        rewrite_remote = bool(plain.get("rewrite_remote", True))
    else:
        raise TypeError(f"Unsupported shard config type: {type(cfg)!r}")

    # Filter out explicit null / empty strings that Hydra may leave behind.
    patterns = [p for p in patterns if p and str(p).lower() not in {"null", "none"}]
    if not patterns:
        raise ValueError(
            "No shard URLs produced from config "
            "(shards is null/empty — set a local path, s3://, https://, or pipe: URL)."
        )

    expanded: list[str] = []
    for pattern in patterns:
        expanded.extend(expand_brace_urls(pattern))

    urls: list[str] = []
    for raw in expanded:
        url = _normalize_local_url(raw)
        if rewrite_remote:
            url = rewrite_remote_url(
                url,
                s3_transport=s3_transport,
                gs_transport=gs_transport,
                http_transport=http_transport,
            )
        urls.append(url)

    if not urls:
        raise ValueError("No shard URLs produced from config")
    return urls
