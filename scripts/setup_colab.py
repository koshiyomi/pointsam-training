#!/usr/bin/env python3
"""Install Point-SAM deps and (optionally) the Wytham data for Google Colab.

Designed so a *new* Colab VM can reuse work kept on Google Drive:

- repo / Point-SAM / weights / crops stay on Drive (clone once)
- a built ``torkit3d`` wheel is cached on Drive and reinstalled in seconds
- ``--download-data`` / ``--prepare-data`` no-op when crops already exist

  python scripts/setup_colab.py
  python scripts/setup_colab.py --download-data --prepare-data
"""
from __future__ import annotations

import argparse
import hashlib
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


def cache_dir(root: Path) -> Path:
    """Prefer Drive-side cache next to the repo; fall back to repo-local."""
    drive_cache = Path("/content/drive/MyDrive/pointsam-colab-cache")
    if Path("/content/drive/MyDrive").is_dir():
        drive_cache.mkdir(parents=True, exist_ok=True)
        return drive_cache
    local = root / ".colab_cache"
    local.mkdir(parents=True, exist_ok=True)
    return local


def mount_drive_if_requested(use_drive: bool) -> Path | None:
    """Prefer a Drive checkout already mounted by the notebook.

    `google.colab.drive.mount` needs the interactive IPython kernel. This script is
    usually run via `!python ...`, which has no kernel, so we never call mount here.
    """
    if not use_drive:
        return None
    drive_root = Path("/content/drive/MyDrive/pointsam-training")
    if Path("/content/drive/MyDrive").is_dir():
        drive_root.mkdir(parents=True, exist_ok=True)
        print(f"Google Drive project folder: {drive_root}", flush=True)
        return drive_root
    raise SystemExit(
        "--drive was set but /content/drive is not mounted. "
        "Mount Drive in a notebook cell first (google.colab.drive.mount), then re-run."
    )


def clone_point_sam(root: Path) -> None:
    dest = root / "thirdparty" / "Point-SAM"
    if dest.is_dir():
        head = subprocess.check_output(
            ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
        ).strip()
        if head.startswith(POINT_SAM_COMMIT) or head == POINT_SAM_COMMIT:
            torkit = dest / "third_party" / "torkit3d"
            if torkit.is_dir():
                print(f"Point-SAM already at {POINT_SAM_COMMIT[:12]}; skip clone", flush=True)
                return
    if not dest.is_dir():
        run(["git", "clone", "-q", "https://github.com/zyc00/Point-SAM.git", str(dest)])
    run(["git", "-C", str(dest), "checkout", "-q", POINT_SAM_COMMIT])
    run(["git", "-C", str(dest), "submodule", "update", "--init", "-q", "third_party/torkit3d"])
    print(f"Point-SAM commit: {POINT_SAM_COMMIT[:12]}", flush=True)


def install_python_deps(root: Path, caches: Path) -> None:
    # Keep Colab's torch; only fill in project packages. Cache wheels on local SSD
    # for this VM (Drive FUSE is slow for pip's many small files).
    pip_cache = Path("/content/.cache/pip")
    pip_cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(pip_cache)

    req = root / "requirements.txt"
    marker = caches / "requirements.sha256"
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    # Across VMs site-packages is empty, so we always pip install; the hash only
    # lets same-session re-runs skip. Cheap installs use pip's own cache.
    already = False
    try:
        import CSF  # noqa: F401
        import hydra  # noqa: F401
        import timm  # noqa: F401
        import safetensors  # noqa: F401

        already = marker.is_file() and marker.read_text().strip() == digest
    except Exception:
        already = False

    if already:
        print("Python deps already importable; skip pip install", flush=True)
    else:
        run(
            [sys.executable, "-m", "pip", "install", "-q", "--prefer-binary", "-r", str(req)],
            env=env,
        )
        run([sys.executable, "-m", "pip", "install", "-q", "ninja"], env=env)
        marker.write_text(digest + "\n")


def _torkit_wheel_stem() -> str:
    import torch

    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_v = torch.__version__.replace("+", ".")
    cuda = (torch.version.cuda or "cpu").replace(".", "")
    return f"torkit3d_{POINT_SAM_COMMIT[:8]}_{py}_torch{torch_v}_cu{cuda}"


def build_torkit3d(root: Path, caches: Path) -> None:
    try:
        import torkit3d  # noqa: F401

        print("torkit3d already importable; skip rebuild", flush=True)
        return
    except Exception:
        pass

    wheel_dir = caches / "wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    stem = _torkit_wheel_stem()
    existing = sorted(wheel_dir.glob(f"{stem}*.whl"))
    if existing:
        print(f"Installing cached torkit3d wheel: {existing[0].name}", flush=True)
        run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(existing[0])])
        return

    src = root / "thirdparty" / "Point-SAM" / "third_party" / "torkit3d"
    env = os.environ.copy()
    env["FORCE_CUDA"] = "1"
    env["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH_LIST
    env["NVCC_APPEND_FLAGS"] = env.get("NVCC_APPEND_FLAGS", "") + " -std=c++20"
    print("Building torkit3d wheel (once; cached on Drive for later VMs)...", flush=True)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "-q",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(wheel_dir),
            str(src),
        ],
        env=env,
    )
    built = sorted(wheel_dir.glob("torkit3d-*.whl"))
    if not built:
        raise SystemExit(f"torkit3d wheel build produced no files in {wheel_dir}")
    # Rename with our tag so the next Colab Python/torch combo does not reuse a bad wheel.
    tagged = wheel_dir / f"{stem}.whl"
    if built[-1].resolve() != tagged.resolve():
        if tagged.exists():
            tagged.unlink()
        built[-1].rename(tagged)
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(tagged)])
    print(f"Cached wheel → {tagged}", flush=True)


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


def crops_ready(root: Path, data_config: str) -> bool:
    import yaml

    cfg = yaml.safe_load((root / data_config).read_text())
    crop_root = root / cfg["out_dir"] / "crops"
    for split in cfg["strips"]:
        split_dir = crop_root / split
        if not split_dir.is_dir() or not any(split_dir.iterdir()):
            return False
    return True


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
        help="Expect Google Drive already mounted; print the Drive project folder",
    )
    parser.add_argument("--download-data", action="store_true", help="Download wytham.zip from Zenodo")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Build tiles/crops (needs PLYs; uses Colab data config by default)",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Rebuild/download data even when crops already exist on Disk/Drive",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/isprs_wytham_colab.yaml",
        help="Data YAML for --prepare-data",
    )
    args = parser.parse_args()

    root = ensure_repo_root()
    mount_drive_if_requested(args.drive)
    caches = cache_dir(root)
    print(f"Colab cache: {caches}", flush=True)

    (root / "thirdparty" / "checkpoints" / "point_sam").mkdir(parents=True, exist_ok=True)
    (root / "results" / "notebooks").mkdir(parents=True, exist_ok=True)

    clone_point_sam(root)
    install_python_deps(root, caches)
    build_torkit3d(root, caches)
    download_weights(root)

    need_data = args.download_data or args.prepare_data
    if need_data and crops_ready(root, args.data_config) and not args.force_data:
        print(
            f"Crops already under data/ for {args.data_config}; skip download/prepare "
            "(pass --force-data to rebuild).",
            flush=True,
        )
    else:
        if args.download_data:
            import yaml

            cfg = yaml.safe_load((root / args.data_config).read_text())
            download_wytham((root / cfg["source_dir"]).resolve())
        if args.prepare_data:
            prepare_data(root, args.data_config)

    verify()
    print("COLAB_SETUP_DONE", flush=True)


if __name__ == "__main__":
    main()
