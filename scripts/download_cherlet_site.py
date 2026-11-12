#!/usr/bin/env python3
"""Download one Cherlet ISPRS site zip from Zenodo into data/raw/isprs_<site>.

  python scripts/download_cherlet_site.py ofental
  python scripts/download_cherlet_site.py litchfield
  python scripts/download_cherlet_site.py robsoncreek
  python scripts/download_cherlet_site.py wytham

Then: python scripts/prepare_data.py configs/data/isprs_<site>.yaml
(for wytham use configs/data/isprs_wytham_colab.yaml or isprs_wytham.yaml).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from archive_util import extract_zip  # noqa: E402

ZENODO = "https://zenodo.org/records/16875688/files/{name}.zip?download=1"
SITES = ("wytham", "ofental", "litchfield", "robsoncreek")
# Approximate lower bounds so truncated Drive downloads fail fast
MIN_BYTES = {
    "wytham": 2_500_000_000,
    "ofental": 4_000_000_000,
    "litchfield": 1_500_000_000,
    "robsoncreek": 2_000_000_000,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", choices=SITES)
    parser.add_argument("--out-root", default="data/raw", help="Parent folder for isprs_<site>")
    args = parser.parse_args()

    dest = Path(args.out_root) / f"isprs_{args.site}"
    dest.mkdir(parents=True, exist_ok=True)
    needed = [f"{args.site}_{split}.ply" for split in ("train", "val", "test")]
    if all((dest / name).is_file() for name in needed):
        print(f"PLYs already in {dest}", flush=True)
        return

    zip_path = dest / f"{args.site}.zip"
    url = ZENODO.format(name=args.site)
    min_bytes = MIN_BYTES[args.site]
    if zip_path.is_file() and zip_path.stat().st_size < min_bytes:
        print(f"Removing truncated zip ({zip_path.stat().st_size} bytes)", flush=True)
        zip_path.unlink()
    if not zip_path.is_file():
        print(f"Downloading {url} → {zip_path}", flush=True)
        urllib.request.urlretrieve(url, zip_path)

    extract_zip(zip_path, dest)

    for name in needed:
        if (dest / name).is_file():
            continue
        matches = list(dest.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} not found after unzip in {dest}")
        shutil.move(str(matches[0]), str(dest / name))
    print("Ready:", ", ".join(needed), flush=True)


if __name__ == "__main__":
    main()
