#!/usr/bin/env python3
"""Build ForestCrops-style npz files from Cedar Key Florida mangrove tree LAS files.

Download the Figshare archive (doi:10.6084/m9.figshare.31324393) into the source_dir from
configs/data/cedar_key_mangrove.yaml, unzip so *.las|*.laz sit under that folder (any depth), then:

  python scripts/ingest_cedar_key.py
  python scripts/ingest_cedar_key.py configs/data/cedar_key_mangrove.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import laspy
import numpy as np
import yaml


def load_tree_las(path: Path) -> np.ndarray:
    las = laspy.read(path)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    return xyz


def voxelize_xyz(xyz: np.ndarray, size: float) -> np.ndarray:
    if len(xyz) == 0:
        return xyz
    cell = np.floor((xyz - xyz.min(axis=0)) / size).astype(np.int64)
    key = (cell[:, 0] << 42) | (cell[:, 1] << 21) | cell[:, 2]
    _, idx = np.unique(key, return_index=True)
    return xyz[idx]


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/data/cedar_key_mangrove.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    source = Path(cfg["source_dir"])
    out = Path(cfg["out_dir"])
    files = sorted(set(source.rglob("*.las")) | set(source.rglob("*.laz")))
    if not files:
        raise SystemExit(f"No .las/.laz under {source}. Unzip the Figshare Cedar Key archive there first.")

    rng = np.random.default_rng(cfg.get("seed", 0))
    records = []
    for path in files:
        xyz = load_tree_las(path)
        xyz = voxelize_xyz(xyz, cfg.get("voxel_m", 0.05))
        if len(xyz) < cfg.get("min_tree_points", 512):
            continue
        if len(xyz) > cfg.get("max_crop_points", 100000):
            xyz = xyz[rng.choice(len(xyz), cfg["max_crop_points"], replace=False)]
        ids = np.ones(len(xyz), dtype=np.int32)  # single tree instance
        records.append({"path": str(path), "xyz": xyz.astype(np.float32), "ids": ids, "n": len(xyz)})

    if len(records) < 5:
        raise SystemExit(f"Only {len(records)} usable trees; check the archive under {source}")

    order = rng.permutation(len(records))
    n_val = max(1, int(round(len(records) * cfg.get("val_fraction", 0.2))))
    val_idx = set(order[:n_val].tolist())

    for split in ("train", "val"):
        dest = out / "crops" / split
        if dest.exists():
            import shutil

            shutil.rmtree(dest)
        dest.mkdir(parents=True)

    meta = {"train": [], "val": []}
    for i, rec in enumerate(records):
        split = "val" if i in val_idx else "train"
        name = f"cedar_tree{i:03d}"
        np.savez(out / "crops" / split / f"{name}.npz", xyz=rec["xyz"], instance_ids=rec["ids"])
        meta[split].append({"crop": name, "source": rec["path"], "n_points": rec["n"]})

    (out / "summary.json").write_text(json.dumps(
        {"config": cfg, "n_train": len(meta["train"]), "n_val": len(meta["val"]), "splits": meta},
        indent=2,
    ))
    print(f"Cedar Key: {len(meta['train'])} train, {len(meta['val'])} val crops → {out / 'crops'}", flush=True)
    print(
        "Note: Blueflux Everglades TLS is unlabeled and is not ingested here "
        "(https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-tls-lidar-blueflux-mangroves-2311-1).",
        flush=True,
    )


if __name__ == "__main__":
    main()
