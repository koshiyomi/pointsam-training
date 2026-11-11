#!/usr/bin/env python3
"""Install Point-SAM deps and (optionally) the Wytham data for Google Colab.

Run from the repo root after cloning. Keeps Colab's existing PyTorch and builds
torkit3d against it. Re-running skips finished steps.

  python scripts/setup_colab.py
  python scripts/setup_colab.py --download-data --prepare-data
  python scripts/setup_colab.py --drive  # work under Google Drive
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

POINT_SAM_COMMIT = "25f4fd9675479f3c951e830771595855ff7ed466"
WEIGHTS_URL = "https://huggingface.co/yuchen0187/Point-SAM/resolve/main/model.safetensors"
ZENODO_URL = "https://zenodo.org/records/16875688/files/wytham.zip?download=1"
# T4 / V100 / A100 / L4 / H100 — Colab free + paid GPUs
CUDA_ARCH_LIST = "7.5;8.0;8.6;8.9;9.0"


def run(cmd: list[str] | str, **kwargs) -> None:
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.check_call(cmd, shell=isinstance(cmd, str), **kwargs)


def ensure_repo_root() -> Path:
    root = Path.cwd().resolve()
    if not (root / "src" / "treeseg").is_dir():
        raise SystemExit(
            f"Run from the repo root (missing src/treeseg). Current dir: {root}"
        )
    return root


def mount_drive_if_requested(use_drive: bool) -> Path | None:
    if not use_drive:
        return None
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:
        raise SystemExit("--drive needs Google Colab") from exc
    drive.mount("/content/drive", force_remount=False)
    drive_root = Path("/content/drive/MyDrive/pointsam-training")
    drive_root.mkdir(parents=True, exist_ok=True)
    print(f"Google Drive project folder: {drive_root}", flush=True)
    return drive_root


def clone_point_sam(root: Path) -> None:
    dest = root / "thirdparty" / "Point-SAM"
    if not dest.is_dir():
        run(["git", "clone", "-q", "https://github.com/zyc00/Point-SAM.git", str(dest)])
    run(["git", "-C", str(dest), "checkout", "-q", POINT_SAM_COMMIT])
    run(["git", "-C", str(dest), "submodule", "update", "--init", "-q", "third_party/torkit3d"])
    print(f"Point-SAM commit: {POINT_SAM_COMMIT[:12]}", flush=True)


def install_python_deps(root: Path) -> None:
    # Keep Colab's torch; only fill in project packages.
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(root / "requirements.txt")])
    # ninja speeds CUDA extension builds and avoids some hang/retry loops
    run([sys.executable, "-m", "pip", "install", "-q", "ninja"])


def build_torkit3d(root: Path) -> None:
    try:
        import torkit3d  # noqa: F401
        print("torkit3d already importable; skip rebuild", flush=True)
        return
    except Exception:
        pass

    src = root / "thirdparty" / "Point-SAM" / "third_party" / "torkit3d"
    env = os.environ.copy()
    env["FORCE_CUDA"] = "1"
    env["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH_LIST
    # Newer torch headers prefer C++20; harmless on older toolchains that ignore it.
    env["NVCC_APPEND_FLAGS"] = env.get("NVCC_APPEND_FLAGS", "") + " -std=c++20"
    run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", str(src)],
        env=env,
    )


def download_weights(root: Path) -> Path:
    path = root / "thirdparty" / "checkpoints" / "point_sam" / "model.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        print(f"weights already present ({path.stat().st_size // 10**6} MB)", flush=True)
        return path
    print(f"Downloading Point-SAM weights → {path}", flush=True)
    urllib.request.urlretrieve(WEIGHTS_URL, path)
    print(f"weights MB: {path.stat().st_size // 10**6}", flush=True)
    return path


def download_wytham(source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    needed = ["wytham_train.ply", "wytham_val.ply", "wytham_test.ply"]
    if all((source_dir / name).is_file() for name in needed):
        print(f"Wytham PLYs already in {source_dir}", flush=True)
        return

    zip_path = source_dir / "wytham.zip"
    if not zip_path.is_file():
        print(f"Downloading Wytham zip (~3.2 GB) → {zip_path}", flush=True)
        urllib.request.urlretrieve(ZENODO_URL, zip_path)

    print(f"Unpacking {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(source_dir)

    # Zenodo layout can nest the PLYs one level down; flatten them.
    for name in needed:
        if (source_dir / name).is_file():
            continue
        matches = list(source_dir.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} not found after unzip in {source_dir}")
        shutil.move(str(matches[0]), str(source_dir / name))

    print("Wytham PLYs ready:", ", ".join(needed), flush=True)


def prepare_data(root: Path, config: str) -> None:
    run([sys.executable, str(root / "scripts" / "prepare_data.py"), config], cwd=root)


def verify() -> None:
    import torch

    import CSF  # noqa: F401
    import hydra  # noqa: F401
    import timm  # noqa: F401
    import torkit3d  # noqa: F401

    print(
        f"torch {torch.__version__} CUDA build {torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}",
        flush=True,
    )
    if not torch.cuda.is_available():
        print(
            "WARNING: no GPU visible. In Colab: Runtime → Change runtime type → GPU, then re-run setup.",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        action="store_true",
        help="Mount Google Drive and prefer MyDrive/pointsam-training as the work root hint",
    )
    parser.add_argument("--download-data", action="store_true", help="Download wytham.zip from Zenodo")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Build tiles/crops (needs PLYs; uses Colab data config by default)",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/isprs_wytham_colab.yaml",
        help="Data YAML for --prepare-data",
    )
    args = parser.parse_args()

    root = ensure_repo_root()
    mount_drive_if_requested(args.drive)

    (root / "thirdparty" / "checkpoints" / "point_sam").mkdir(parents=True, exist_ok=True)
    (root / "results" / "notebooks").mkdir(parents=True, exist_ok=True)

    clone_point_sam(root)
    install_python_deps(root)
    build_torkit3d(root)
    download_weights(root)

    if args.download_data:
        cfg_path = root / args.data_config
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text())
        source_dir = (root / cfg["source_dir"]).resolve()
        download_wytham(source_dir)

    if args.prepare_data:
        prepare_data(root, args.data_config)

    verify()
    print("COLAB_SETUP_DONE", flush=True)


if __name__ == "__main__":
    main()
