"""Early-stopping helper for the generic Trainer loop."""

from __future__ import annotations


class EarlyStopping:
    """Track a monitored metric and signal when training should stop.

    Args:
        patience: Epochs to wait after last improvement before stopping.
        mode: ``"min"`` if lower is better, ``"max"`` if higher is better.
        min_delta: Minimum absolute change to count as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        if patience < 0:
            raise ValueError("patience must be >= 0")
        mode = str(mode).lower().strip()
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.num_bad_epochs = 0
        self.should_stop = False
        self._is_better = (
            (lambda current, best: current < best - self.min_delta)
            if mode == "min"
            else (lambda current, best: current > best + self.min_delta)
        )

    def reset(self) -> None:
        """Clear tracking state."""
        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Update with a new metric value.

        Returns:
            ``True`` if this step improved the best metric.
        """
        metric = float(metric)
        improved = False
        if self.best is None or self._is_better(metric, self.best):
            self.best = metric
            self.num_bad_epochs = 0
            improved = True
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                self.should_stop = True
        return improved

    def __call__(self, metric: float) -> bool:
        """Alias for :meth:`step`; returns whether training should stop."""
        self.step(metric)
        return self.should_stop
