"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Integer seed applied across libraries.
        deterministic: If True, enable deterministic CuDNN / algorithm
            settings (may reduce performance). When False, CuDNN benchmark
            remains enabled for speed.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.8
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
