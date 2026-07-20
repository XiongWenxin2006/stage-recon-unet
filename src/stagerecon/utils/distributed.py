"""Distributed-training stubs that behave correctly for single-process runs."""

from __future__ import annotations


def is_distributed() -> bool:
    """Return whether the current process is part of a distributed group.

    Single-process stub: always ``False``. Replace with a real
    ``torch.distributed`` check when multi-GPU training is wired up.
    """
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def get_rank() -> int:
    """Return the global rank of this process (``0`` for single-process)."""
    if not is_distributed():
        return 0
    import torch.distributed as dist

    return int(dist.get_rank())


def get_world_size() -> int:
    """Return the world size (``1`` for single-process)."""
    if not is_distributed():
        return 1
    import torch.distributed as dist

    return int(dist.get_world_size())


def is_main_process() -> bool:
    """Return True if this is rank 0 (or single-process)."""
    return get_rank() == 0
