"""Extract Zenodo zips in a way that works on Colab (Deflate64 / odd methods)."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Unpack zip_path into dest.

    Prefer the system ``unzip`` binary: Python's zipfile raises
    NotImplementedError on some Zenodo compression methods (e.g. Deflate64).
    """
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)

    unzip = shutil.which("unzip")
    if unzip:
        print(f"Extracting with unzip → {dest}", flush=True)
        subprocess.check_call([unzip, "-o", "-q", str(zip_path), "-d", str(dest)])
        return

    print(f"Extracting with zipfile → {dest}", flush=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupt zip member {bad!r} in {zip_path}")
            zf.extractall(dest)
    except NotImplementedError as exc:
        raise SystemExit(
            f"Cannot extract {zip_path}: compression not supported by Python zipfile ({exc}). "
            "Install Info-ZIP `unzip` (available on Colab) or re-download the archive."
        ) from exc


def ensure_min_size(path: Path, min_bytes: int, label: str) -> None:
    size = path.stat().st_size
    if size < min_bytes:
        raise SystemExit(
            f"{label} looks truncated ({size} bytes < {min_bytes}). "
            f"Delete {path} and re-download."
        )
