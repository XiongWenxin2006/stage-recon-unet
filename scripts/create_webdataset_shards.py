#!/usr/bin/env python3
"""Create WebDataset shards from a CSV manifest of ``.npy`` samples.

Manifest CSV columns:
  - sample_id
  - image_path
  - mask_path (optional)
  - split  (train / val / test)

Shard sample keys:
  - __key__
  - image.npy
  - mask.npy (optional)
  - meta.json

Splits are written to separate directories and never mixed inside a shard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _require_webdataset() -> Any:
    try:
        import webdataset as wds
    except ImportError as exc:
        raise ImportError(
            "webdataset is required. Install with: pip install webdataset"
        ) from exc
    return wds


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    return np.asarray(arr)


def _npy_bytes(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return buf.getvalue()


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read and validate the manifest CSV."""
    required = {"sample_id", "image_path", "split"}
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest {path} has no header row")
        fields = {name.strip() for name in reader.fieldnames}
        missing = required - fields
        if missing:
            raise ValueError(
                f"Manifest missing required columns {sorted(missing)}. "
                f"Found: {sorted(fields)}"
            )
        for line_no, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            split = (row.get("split") or "").strip().lower()
            mask_path = (row.get("mask_path") or "").strip()
            if not sample_id or not image_path or not split:
                raise ValueError(
                    f"Manifest line {line_no}: sample_id, image_path, and split "
                    "are required"
                )
            if split not in {"train", "val", "test", "validation"}:
                raise ValueError(
                    f"Manifest line {line_no}: unsupported split '{split}'. "
                    "Expected train/val/test."
                )
            if split == "validation":
                split = "val"
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "split": split,
                }
            )
    if not rows:
        raise ValueError(f"Manifest {path} contains no samples")
    return rows


def _check_unique_keys(rows: Iterable[dict[str, str]]) -> None:
    seen: set[str] = set()
    dupes: list[str] = []
    for row in rows:
        key = row["sample_id"]
        if key in seen:
            dupes.append(key)
        seen.add(key)
    if dupes:
        preview = ", ".join(sorted(set(dupes))[:10])
        raise ValueError(
            f"Duplicate sample_id / __key__ values in manifest: {preview}"
        )


def write_split_shards(
    rows: list[dict[str, str]],
    *,
    split: str,
    output_dir: Path,
    maxcount: int,
    maxsize: int,
    image_root: Path | None,
    compute_sha256: bool,
) -> dict[str, Any]:
    """Write one split's shards; forbids mixing other splits."""
    wds = _require_webdataset()
    split_rows = [r for r in rows if r["split"] == split]
    # Defensive: ensure no accidental mix
    for r in split_rows:
        if r["split"] != split:
            raise RuntimeError("Internal error: mixed splits in shard writer")

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(split_dir / f"{split}-%06d.tar")

    shard_files: list[Path] = []
    sample_count = 0
    keys: list[str] = []

    with wds.ShardWriter(pattern, maxcount=maxcount, maxsize=maxsize) as sink:
        for row in split_rows:
            # Forbid mixing: row.split already filtered; re-assert
            if row["split"] != split:
                raise ValueError(
                    f"Refusing to write sample {row['sample_id']} with split "
                    f"'{row['split']}' into '{split}' shards"
                )

            image_path = Path(row["image_path"])
            if not image_path.is_absolute() and image_root is not None:
                image_path = image_root / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = _load_npy(image_path)
            key = row["sample_id"]
            sample: dict[str, Any] = {
                "__key__": key,
                "image.npy": _npy_bytes(image),
            }

            mask_path_str = row.get("mask_path") or ""
            mask_path: Path | None = None
            if mask_path_str:
                mask_path = Path(mask_path_str)
                if not mask_path.is_absolute() and image_root is not None:
                    mask_path = image_root / mask_path
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                mask = _load_npy(mask_path)
                sample["mask.npy"] = _npy_bytes(mask)

            meta = {
                "sample_id": key,
                "split": split,
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else None,
                "image_shape": list(image.shape),
            }
            sample["meta.json"] = json.dumps(meta, sort_keys=True).encode("utf-8")
            sink.write(sample)
            sample_count += 1
            keys.append(key)

    # Collect written shard paths
    shard_files = sorted(split_dir.glob(f"{split}-*.tar"))
    shard_meta: list[dict[str, Any]] = []
    for shard in shard_files:
        info: dict[str, Any] = {
            "path": str(shard),
            "size_bytes": shard.stat().st_size,
        }
        if compute_sha256:
            info["sha256"] = _sha256_file(shard)
        shard_meta.append(info)

    return {
        "split": split,
        "num_samples": sample_count,
        "num_shards": len(shard_files),
        "shard_pattern": pattern,
        "shards": shard_meta,
        "keys": keys,
    }


def create_shards(
    manifest: Path,
    output_dir: Path,
    *,
    maxcount: int = 1000,
    maxsize: int = 1_000_000_000,
    image_root: Path | None = None,
    compute_sha256: bool = False,
    splits: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Create train/val/test WebDataset shards from a manifest."""
    rows = read_manifest(manifest)
    _check_unique_keys(rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    present_splits = sorted({r["split"] for r in rows})
    target_splits = list(splits) if splits is not None else present_splits

    split_summaries: dict[str, Any] = {}
    for split in target_splits:
        if split not in present_splits:
            continue
        print(f"Writing split={split} ({sum(1 for r in rows if r['split'] == split)} samples)")
        split_summaries[split] = write_split_shards(
            rows,
            split=split,
            output_dir=output_dir,
            maxcount=maxcount,
            maxsize=maxsize,
            image_root=image_root,
            compute_sha256=compute_sha256,
        )

    # Global uniqueness already checked; also ensure no key appears in 2 splits
    key_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key_to_splits[row["sample_id"]].add(row["split"])
    multi = {k: sorted(v) for k, v in key_to_splits.items() if len(v) > 1}
    if multi:
        preview = list(multi.items())[:5]
        raise ValueError(
            "sample_id appears in multiple splits (forbidden): "
            + ", ".join(f"{k}->{splits_}" for k, splits_ in preview)
        )

    summary = {
        "manifest": str(manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "maxcount": maxcount,
        "maxsize": maxsize,
        "splits": split_summaries,
        "total_samples": sum(s["num_samples"] for s in split_summaries.values()),
        "total_shards": sum(s["num_shards"] for s in split_summaries.values()),
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        # Omit full key lists from the on-disk summary for readability; keep counts
        disk_summary = json.loads(json.dumps(summary))
        for split_name, split_info in disk_summary.get("splits", {}).items():
            keys = split_info.pop("keys", [])
            split_info["num_keys"] = len(keys)
            # Optional compact key list file
            keys_path = output_dir / split_name / "keys.json"
            keys_path.parent.mkdir(parents=True, exist_ok=True)
            with keys_path.open("w", encoding="utf-8") as kf:
                json.dump(keys, kf, indent=2)
        json.dump(disk_summary, f, indent=2, sort_keys=True)

    print(f"Wrote summary to {summary_path}")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create WebDataset shards from a StageRecon CSV manifest."
    )
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV with columns: sample_id,image_path,mask_path,split",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write train/val/test shard folders into",
    )
    p.add_argument(
        "--maxcount",
        type=int,
        default=1000,
        help="Maximum samples per shard (ShardWriter maxcount)",
    )
    p.add_argument(
        "--maxsize",
        type=int,
        default=1_000_000_000,
        help="Maximum shard size in bytes (ShardWriter maxsize)",
    )
    p.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional root prepended to relative image/mask paths",
    )
    p.add_argument(
        "--sha256",
        action="store_true",
        help="Compute sha256 for each written shard and include in summary.json",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional subset of splits to write (default: all present)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    create_shards(
        manifest=args.manifest,
        output_dir=args.output_dir,
        maxcount=args.maxcount,
        maxsize=args.maxsize,
        image_root=args.image_root,
        compute_sha256=args.sha256,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()
