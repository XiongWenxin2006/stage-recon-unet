"""Error handlers for resilient WebDataset iteration."""

from __future__ import annotations

import logging
import warnings
from typing import Callable


logger = logging.getLogger(__name__)


def warn_and_continue(
    exn: BaseException,
) -> bool:
    """Log a warning for ``exn`` and continue dataset iteration.

    Compatible with WebDataset's ``handler`` protocol: return ``True`` to
    ignore the exception and continue, ``False`` to re-raise.

    Args:
        exn: Exception raised while decoding or iterating a sample/shard.

    Returns:
        Always ``True`` (continue).
    """
    message = f"Skipping sample/shard due to {type(exn).__name__}: {exn}"
    logger.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return True


def reraise(exn: BaseException) -> bool:
    """Re-raise handler for strict iteration (returns ``False``)."""
    return False


def make_warn_and_continue(logger_name: str | None = None) -> Callable[[BaseException], bool]:
    """Build a ``warn_and_continue`` handler bound to a specific logger name."""
    bound_logger = logging.getLogger(logger_name or __name__)

    def _handler(exn: BaseException) -> bool:
        message = f"Skipping sample/shard due to {type(exn).__name__}: {exn}"
        bound_logger.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return True

    return _handler
