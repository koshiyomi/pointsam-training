#!/usr/bin/env python3
"""Merge crop folders from several prepared sites into one train/val tree for multi-site training.

  python scripts/combine_crop_sets.py \\
      --out data/multisite_wytham_ofental \\
      --train data/wytham_isprs/crops/train data/ofental_isprs/crops/train \\
      --val data/wytham_isprs/crops/val

Val defaults to the first --val path only (keep Wytham val for checkpoint picking unless you pass more).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output root (writes crops/train and crops/val)")
    parser.add_argument("--train", nargs="+", required=True, help="Train crop directories")
    parser.add_argument("--val", nargs="+", required=True, help="Val crop directories")
    args = parser.parse_args()

    out = Path(args.out)
    summary = {"train": [], "val": []}
    for split, dirs in (("train", args.train), ("val", args.val)):
        dest_dir = out / "crops" / split
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True)
        for src_dir in dirs:
            src = Path(src_dir)
            if not src.is_dir():
                raise SystemExit(f"missing crop dir: {src}")
            prefix = src.parent.parent.name if src.parent.name == "crops" else src.name
            for npz in sorted(src.glob("*.npz")):
                dest = dest_dir / f"{prefix}__{npz.name}"
                link_or_copy(npz, dest)
                summary[split].append(dest.name)
        print(f"{split}: {len(summary[split])} crops → {dest_dir}", flush=True)

    (out / "combine_summary.json").write_text(json.dumps(
        {k: {"n": len(v), "files": v[:20], "truncated": len(v) > 20} for k, v in summary.items()},
        indent=2,
    ))
    print("COMBINE_DONE", flush=True)


if __name__ == "__main__":
    main()
