"""Reconstruction quality metrics operating on tensors."""

from __future__ import annotations

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between ``pred`` and ``target``.

    Args:
        pred: Predicted image tensor.
        target: Reference image tensor (broadcastable to ``pred``).

    Returns:
        Scalar MSE tensor (detached from autograd graph if inputs require grad
        only when callers detach; this function does not call ``.item()``).
    """
    pred = pred.float()
    target = target.float()
    return torch.mean((pred - target) ** 2)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between ``pred`` and ``target``.

    Args:
        pred: Predicted image tensor.
        target: Reference image tensor.

    Returns:
        Scalar MAE tensor.
    """
    pred = pred.float()
    target = target.float()
    return torch.mean(torch.abs(pred - target))


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Peak signal-to-noise ratio in decibels.

    Args:
        pred: Predicted image tensor.
        target: Reference image tensor.
        data_range: Peak signal amplitude. If ``None``, uses
            ``max(target) - min(target)`` when that span is positive, otherwise
            ``max(|target|)`` (falling back to ``1.0``).
        eps: Numerical floor for MSE in the denominator.

    Returns:
        Scalar PSNR tensor in dB.
    """
    pred = pred.float()
    target = target.float()
    err = torch.mean((pred - target) ** 2)

    if data_range is None:
        t_min = torch.min(target)
        t_max = torch.max(target)
        span = t_max - t_min
        if float(span) > 0:
            data_range_t = span
        else:
            data_range_t = torch.max(torch.abs(target))
            if float(data_range_t) <= 0:
                data_range_t = torch.tensor(1.0, device=target.device, dtype=target.dtype)
    else:
        data_range_t = torch.as_tensor(data_range, device=target.device, dtype=target.dtype)

    return 20.0 * torch.log10(data_range_t) - 10.0 * torch.log10(err + eps)
