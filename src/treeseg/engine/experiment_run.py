"""
Shared helpers for notebooks/experiments/*: train, shared val-tile benchmark, compare, shutdown.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


def ensure_repo_root() -> Path:
    root = Path.cwd().resolve()
    for _ in range(4):
        if (root / "src" / "treeseg").is_dir():
            break
        root = root.parent
    else:
        raise SystemExit(f"Could not find repo root from {Path.cwd()}")
    os.chdir(root)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def probe_resources(tag: str = "probe") -> dict[str, Any]:
    """Record GPU peak and host RAM. Call after a short forward/train step for useful peaks."""
    info: dict[str, Any] = {"tag": tag}
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_used_gb"] = round(vm.used / 1e9, 2)
        info["ram_total_gb"] = round(vm.total / 1e9, 2)
    except Exception as exc:
        info["ram_error"] = str(exc)

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        info["gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    else:
        info["gpu_name"] = None
    print(info, flush=True)
    return info


def reset_cuda_peaks() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def train_from_config(config_path: str, skip_if_best_exists: bool = False) -> dict:
    """Run the standard LoRA/DoRA training loop from an experiment YAML."""
    from torch.utils.data import Subset

    from treeseg.data.dataset import ForestCrops
    from treeseg.engine.trainer import train
    from treeseg.models.point_sam import (
        add_point_sam_to_path,
        build_point_sam,
        prepare_for_adapter,
    )

    ensure_repo_root()
    add_point_sam_to_path()
    cfg = load_yaml(config_path)
    best = Path(cfg["out_dir"]) / "best.safetensors"
    if skip_if_best_exists and best.is_file():
        print(f"skip train; found {best}", flush=True)
        return cfg

    reset_cuda_peaks()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Training needs a CUDA GPU")

    augmentation = cfg["augmentation"] if cfg.get("use_augmentation") else None
    train_set = ForestCrops(
        os.path.join(cfg["data_dir"], "train"),
        cfg["num_points"],
        cfg["num_masks"],
        cfg["min_mask_points"],
        augmentation=augmentation,
        resample_each_epoch=True,
        seed=cfg["seed"],
    )
    val_full = ForestCrops(
        os.path.join(cfg["data_dir"], "val"),
        cfg["num_points"],
        cfg["val_num_masks"],
        cfg["min_mask_points"],
        seed=cfg["seed"] + 1,
    )
    val_set = Subset(val_full, range(min(cfg["val_max_crops"], len(val_full))))

    model = build_point_sam(cfg["checkpoint"], cfg["group_number"], cfg["group_size"])
    prepare_for_adapter(
        model,
        method=cfg.get("adapter", "lora"),
        rank=cfg["lora_rank"],
        alpha=cfg["lora_alpha"],
        targets=tuple(cfg.get("lora_targets", ["q_proj", "v_proj"])),
    )
    model.to(device)
    probe_resources("after_model_to_device")
    train(model, train_set, val_set, cfg, device)
    probe_resources("after_train")
    return cfg


def build_compare_cfg(
    compare_config: str,
    run_name: str,
    run_checkpoint: str,
    out_dir: str | None = None,
    include_baseline_lora: bool = True,
) -> dict:
    """Frozen val-tile settings + this run's checkpoint (and stock / isprs_lora when present)."""
    cfg = load_yaml(compare_config)
    checkpoints: dict[str, str] = {
        "stock": "thirdparty/checkpoints/point_sam/model.safetensors",
    }
    baseline = "results/isprs_lora/best.safetensors"
    if include_baseline_lora and Path(baseline).is_file() and Path(run_checkpoint).resolve() != Path(baseline).resolve():
        checkpoints["isprs_lora"] = baseline
    checkpoints[run_name] = run_checkpoint
    # drop missing stock/baseline quietly
    checkpoints = {k: v for k, v in checkpoints.items() if Path(v).is_file()}
    if run_name not in checkpoints:
        raise FileNotFoundError(f"run checkpoint missing: {run_checkpoint}")
    cfg["checkpoints"] = checkpoints
    cfg["out_dir"] = out_dir or f"results/{run_name}/benchmark"
    return cfg


def run_shared_benchmark(compare_cfg: dict, device: str = "cuda") -> pd.DataFrame:
    from treeseg.engine.benchmark import run_benchmark, summarise
    from treeseg.models.point_sam import add_point_sam_to_path
    from treeseg.utils.curves import plot_f1_bars

    ensure_repo_root()
    add_point_sam_to_path()
    os.makedirs(compare_cfg["out_dir"], exist_ok=True)
    reset_cuda_peaks()
    results_path = run_benchmark(compare_cfg, device=device)
    probe_resources("after_benchmark")
    summary = summarise(results_path)
    summary_path = os.path.join(compare_cfg["out_dir"], "summary.csv")
    summary.to_csv(summary_path, index=False)
    figure_path = os.path.join(compare_cfg["out_dir"], "f1_bars.png")
    plot_f1_bars(summary, figure_path)
    return summary


def comparison_table(summary: pd.DataFrame, rule: str = "ignore_non_tree") -> pd.DataFrame:
    """Pivot F1 (and friends) for the scoring rule used in the study."""
    sub = summary[summary["rule"] == rule].copy()
    cols = [c for c in ("f1", "precision", "recall", "raw_recall", "click_recall", "mean_tree_iou") if c in sub.columns]
    return sub[["checkpoint", "clicks_from", *cols]].sort_values(["clicks_from", "checkpoint"]).reset_index(drop=True)


def save_probe(out_dir: str, probes: list[dict]) -> Path:
    path = Path(out_dir) / "resource_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(probes, indent=2))
    return path


def maybe_shutdown(auto_shutdown: bool) -> None:
    if not auto_shutdown:
        print("AUTO_SHUTDOWN=False; leave the runtime connected.", flush=True)
        return
    try:
        from google.colab import runtime  # type: ignore

        print("Unassigning Colab runtime to stop compute-unit use.", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Could not unassign runtime ({exc}). Disconnect manually in Colab.", flush=True)


def run_crop_validate(checkpoint: str, data_dir: str, cfg_train: dict, device: str = "cuda") -> pd.DataFrame:
    """Crop-level IoU table for single-tree sets (e.g. Cedar Key) that have no Wytham tiles."""
    from torch.utils.data import DataLoader, Subset

    from treeseg.data.dataset import ForestCrops
    from treeseg.engine.trainer import collate, validate
    from treeseg.models.point_sam import add_point_sam_to_path, build_point_sam

    add_point_sam_to_path()
    from pc_sam.model.loss import Criterion

    val_full = ForestCrops(
        os.path.join(data_dir, "val"),
        cfg_train["num_points"],
        cfg_train["val_num_masks"],
        cfg_train["min_mask_points"],
        seed=cfg_train.get("seed", 0) + 1,
    )
    val_set = Subset(val_full, range(min(cfg_train.get("val_max_crops", len(val_full)), len(val_full))))
    loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate)
    model = build_point_sam(checkpoint, cfg_train["group_number"], cfg_train["group_size"]).to(device)
    metrics = validate(model, loader, Criterion(use_soft_iou=False), device, cfg_train.get("val_click", "stem"))
    row = pd.DataFrame([{"checkpoint": Path(checkpoint).parent.name, **metrics}])
    print(row.to_string(index=False), flush=True)
    return row


def run_train_and_compare(
    train_config: str,
    compare_config: str = "configs/experiment/compare_val.yaml",
    *,
    skip_train: bool = False,
    auto_shutdown: bool = False,
    crop_only_benchmark: bool = False,
) -> pd.DataFrame:
    """Full experiment notebook body: train → shared benchmark → comparison table."""
    ensure_repo_root()
    probes: list[dict] = []
    probes.append(probe_resources("start"))

    cfg = train_from_config(train_config, skip_if_best_exists=skip_train)
    run_name = cfg["run"]
    best = os.path.join(cfg["out_dir"], "best.safetensors")

    if crop_only_benchmark:
        table = run_crop_validate(best, cfg["data_dir"], cfg)
        bench_dir = Path(cfg["out_dir"]) / "benchmark"
        bench_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(bench_dir / "comparison.csv", index=False)
    else:
        compare_cfg = build_compare_cfg(
            compare_config, run_name, best, out_dir=os.path.join(cfg["out_dir"], "benchmark")
        )
        summary = run_shared_benchmark(compare_cfg)
        table = comparison_table(summary)
        print(table.to_string(index=False), flush=True)
        table.to_csv(os.path.join(compare_cfg["out_dir"], "comparison.csv"), index=False)

    probes.append(probe_resources("end"))
    save_probe(cfg["out_dir"], probes)
    maybe_shutdown(auto_shutdown)
    return table
