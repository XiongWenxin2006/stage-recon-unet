"""Logging setup helpers for StageRecon."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "stagerecon",
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    *,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    force: bool = False,
) -> logging.Logger:
    """Create or configure a named logger with console (and optional file) handlers.

    Args:
        name: Logger name.
        level: Logging level (int or name such as ``"INFO"``).
        log_file: Optional path to a log file. Parent directories are created.
        fmt: Log record format string.
        datefmt: Datetime format for ``asctime``.
        force: If True, clear existing handlers before attaching new ones.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    if force and logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path = str(log_path.resolve())
        has_file = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == abs_path
            for h in logger.handlers
        )
        if not has_file:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    # Avoid duplicate messages via root logger
    logger.propagate = False
    return logger
