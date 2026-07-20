"""Aggregate metrics across seeds / runs into mean / std summaries."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _std(values: Sequence[float], *, ddof: int = 0) -> float:
    if not values:
        return float("nan")
    if len(values) <= ddof:
        return float("nan")
    mu = _mean(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - ddof)
    return float(math.sqrt(var))


class ResultAggregator:
    """Collect per-run metric dicts and compute mean / std aggregates.

    Example::

        agg = ResultAggregator()
        agg.add("seed0", {"dice": 0.8, "loss": 0.2})
        agg.add("seed1", {"dice": 0.82, "loss": 0.18})
        summary = agg.aggregate()
        agg.save_json("results/summary.json")
        agg.save_csv("results/summary.csv")
    """

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}

    def add(self, run_id: str, metrics: Mapping[str, Any]) -> None:
        """Register metrics for a single run / seed."""
        if not run_id:
            raise ValueError("run_id must be a non-empty string")
        self.runs[str(run_id)] = dict(metrics)

    def extend(self, runs: Mapping[str, Mapping[str, Any]]) -> None:
        """Register multiple runs at once."""
        for run_id, metrics in runs.items():
            self.add(str(run_id), metrics)

    def clear(self) -> None:
        """Drop all registered runs."""
        self.runs.clear()

    def metric_names(self) -> list[str]:
        """Return sorted numeric metric names seen across runs."""
        names: set[str] = set()
        for metrics in self.runs.values():
            for key, value in metrics.items():
                if _is_number(value):
                    names.add(str(key))
        return sorted(names)

    def aggregate(self, *, ddof: int = 0) -> dict[str, Any]:
        """Aggregate registered runs into mean / std (and per-run) dicts.

        Returns:
            ``{
                "n_runs": int,
                "run_ids": [...],
                "mean": {metric: float},
                "std": {metric: float},
                "runs": {run_id: metrics},
            }``
        """
        names = self.metric_names()
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        for name in names:
            values = [
                float(metrics[name])
                for metrics in self.runs.values()
                if name in metrics and _is_number(metrics[name])
            ]
            mean[name] = _mean(values)
            std[name] = _std(values, ddof=ddof)

        return {
            "n_runs": len(self.runs),
            "run_ids": sorted(self.runs.keys()),
            "mean": mean,
            "std": std,
            "runs": {k: dict(v) for k, v in self.runs.items()},
        }

    def save_json(
        self,
        path: str | Path,
        *,
        ddof: int = 0,
        indent: int = 2,
    ) -> Path:
        """Write the aggregate summary to a JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.aggregate(ddof=ddof)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, sort_keys=True)
        return out

    def save_csv(
        self,
        path: str | Path,
        *,
        ddof: int = 0,
    ) -> Path:
        """Write mean / std (and optional per-run rows) to CSV.

        Columns: ``run_id``, then one column per metric. Aggregate rows use
        ``run_id`` values ``mean`` and ``std``.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = self.aggregate(ddof=ddof)
        names = sorted(summary["mean"].keys())
        fieldnames = ["run_id", *names]

        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for run_id in summary["run_ids"]:
                row: dict[str, Any] = {"run_id": run_id}
                metrics = summary["runs"][run_id]
                for name in names:
                    value = metrics.get(name)
                    row[name] = value if _is_number(value) else ""
                writer.writerow(row)
            mean_row: dict[str, Any] = {"run_id": "mean"}
            std_row: dict[str, Any] = {"run_id": "std"}
            for name in names:
                mean_row[name] = summary["mean"].get(name, "")
                std_row[name] = summary["std"].get(name, "")
            writer.writerow(mean_row)
            writer.writerow(std_row)
        return out


def aggregate_metrics(
    runs: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    ddof: int = 0,
) -> dict[str, Any]:
    """Convenience: aggregate a mapping or sequence of metric dicts."""
    agg = ResultAggregator()
    if isinstance(runs, Mapping):
        agg.extend(runs)
    else:
        for idx, metrics in enumerate(runs):
            agg.add(f"run_{idx}", metrics)
    return agg.aggregate(ddof=ddof)
