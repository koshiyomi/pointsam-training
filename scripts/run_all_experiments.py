#!/usr/bin/env python3
"""Run follow-up experiments in order on one Colab/ICE session.

Skips a run when results/<run>/best.safetensors already exists (unless --force).
Shuts down the Colab runtime only after the whole queue (unless --no-shutdown).

  python scripts/run_all_experiments.py
  python scripts/run_all_experiments.py --only 01,02
  python scripts/run_all_experiments.py --skip-data-prep --force
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os_chdir_done = False

# (id, train_config or None, flags)
EXPERIMENTS = [
    {
        "id": "01",
        "name": "baseline_reproduce",
        "train_config": "configs/experiment/isprs_lora_colab.yaml",
        "crop_only": False,
        "needs": ["wytham_crops"],
    },
    {
        "id": "02",
        "name": "dora_r8",
        "train_config": "configs/experiment/isprs_dora_r8.yaml",
        "crop_only": False,
        "needs": ["wytham_crops"],
    },
    {
        "id": "03",
        "name": "lr2e4",
        "train_config": "configs/experiment/isprs_lora_lr2e4.yaml",
        "crop_only": False,
        "needs": ["wytham_crops"],
    },
    {
        "id": "04",
        "name": "lr5e5",
        "train_config": "configs/experiment/isprs_lora_lr5e5.yaml",
        "crop_only": False,
        "needs": ["wytham_crops"],
    },
    {
        "id": "05",
        "name": "batch4",
        "train_config": "configs/experiment/isprs_lora_batch4.yaml",
        "crop_only": False,
        "needs": ["wytham_crops"],
    },
    {
        "id": "06",
        "name": "multisite_wytham_ofental",
        "train_config": "configs/experiment/multisite_wytham_ofental.yaml",
        "crop_only": False,
        "needs": ["multisite_wo"],
    },
    {
        "id": "07",
        "name": "negative_clicks_infer",
        "train_config": None,  # inference-only
        "crop_only": False,
        "needs": ["wytham_crops", "some_checkpoint"],
    },
    {
        "id": "08",
        "name": "florida_cedar_key",
        "train_config": "configs/experiment/cedar_key_lora.yaml",
        "crop_only": True,
        "needs": ["cedar_key"],
    },
]


def sh(cmd: list[str] | str) -> None:
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.check_call(cmd, shell=isinstance(cmd, str), cwd=ROOT)


def ensure_root() -> None:
    global os_chdir_done
    import os

    os.chdir(ROOT)
    os_chdir_done = True


def run_name_from_config(path: str) -> str:
    import yaml

    return yaml.safe_load((ROOT / path).read_text())["run"]


def best_exists(train_config: str) -> bool:
    run = run_name_from_config(train_config)
    return (ROOT / "results" / run / "best.safetensors").is_file()


def prepare_wytham() -> None:
    crops = ROOT / "data" / "wytham_isprs" / "crops" / "train"
    if crops.is_dir() and any(crops.iterdir()):
        print("Wytham crops present", flush=True)
        return
    sh([sys.executable, "scripts/setup_colab.py", "--download-data", "--prepare-data"])


def prepare_multisite() -> None:
    out = ROOT / "data" / "multisite_wytham_ofental" / "crops" / "train"
    if out.is_dir() and any(out.iterdir()):
        print("Multisite crops present", flush=True)
        return
    prepare_wytham()
    if not (ROOT / "data" / "ofental_isprs" / "crops" / "train").is_dir():
        sh([sys.executable, "scripts/download_cherlet_site.py", "ofental"])
        sh([sys.executable, "scripts/prepare_data.py", "configs/data/isprs_ofental.yaml"])
    sh(
        [
            sys.executable,
            "scripts/combine_crop_sets.py",
            "--out",
            "data/multisite_wytham_ofental",
            "--train",
            "data/wytham_isprs/crops/train",
            "data/ofental_isprs/crops/train",
            "--val",
            "data/wytham_isprs/crops/val",
        ]
    )


def prepare_cedar_key() -> None:
    crops = ROOT / "data" / "cedar_key_mangrove" / "crops" / "train"
    if crops.is_dir() and any(crops.iterdir()):
        print("Cedar Key crops present", flush=True)
        return
    raw = ROOT / "data" / "raw" / "cedar_key_mangrove"
    las = list(raw.rglob("*.las")) + list(raw.rglob("*.laz"))
    if not las:
        raise FileNotFoundError(
            f"No LAS under {raw}. Download Figshare doi:10.6084/m9.figshare.31324393 and unzip there, "
            "then re-run. Skipping 08 until then."
        )
    sh([sys.executable, "scripts/ingest_cedar_key.py"])


def pick_checkpoint_for_neg() -> str:
    candidates = [
        ROOT / "results" / "isprs_dora_r8" / "best.safetensors",
        ROOT / "results" / "multisite_wytham_ofental" / "best.safetensors",
        ROOT / "results" / "isprs_lora_colab" / "best.safetensors",
        ROOT / "results" / "isprs_lora" / "best.safetensors",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError("No trained checkpoint for negative-click infer (run 01+ first)")


def run_negative_clicks() -> None:
    import os

    import torch
    import yaml
    from treeseg.engine.benchmark import run_benchmark, summarise
    from treeseg.engine.experiment_run import comparison_table, probe_resources
    from treeseg.models.point_sam import add_point_sam_to_path
    from treeseg.utils.curves import plot_f1_bars

    add_point_sam_to_path()
    ckpt = pick_checkpoint_for_neg()
    cfg = yaml.safe_load((ROOT / "configs/experiment/negative_clicks.yaml").read_text())
    cfg["checkpoints"] = {"run": ckpt}
    os.makedirs(cfg["out_dir"], exist_ok=True)
    run_benchmark(cfg, device="cuda")
    summary = summarise(os.path.join(cfg["out_dir"], "results.csv"))
    summary.to_csv(os.path.join(cfg["out_dir"], "summary.csv"), index=False)
    comparison_table(summary).to_csv(os.path.join(cfg["out_dir"], "comparison.csv"), index=False)
    plot_f1_bars(summary, os.path.join(cfg["out_dir"], "f1_bars.png"))
    probe_resources("neg_done")


def run_one(exp: dict, force: bool) -> str:
    from treeseg.engine.experiment_run import run_train_and_compare

    eid = exp["id"]
    if exp["train_config"] is None:
        print(f"\n===== [{eid}] negative clicks =====", flush=True)
        run_negative_clicks()
        return "ok"

    cfg_path = exp["train_config"]
    if best_exists(cfg_path) and not force:
        print(f"\n===== [{eid}] skip (best exists) {cfg_path} =====", flush=True)
        # still refresh benchmark comparison
        from treeseg.engine.experiment_run import build_compare_cfg, comparison_table, run_shared_benchmark

        run = run_name_from_config(cfg_path)
        best = str(ROOT / "results" / run / "best.safetensors")
        if exp.get("crop_only"):
            return "skipped"
        compare_cfg = build_compare_cfg(
            "configs/experiment/compare_val.yaml",
            run,
            best,
            out_dir=str(ROOT / "results" / run / "benchmark"),
        )
        summary = run_shared_benchmark(compare_cfg)
        comparison_table(summary).to_csv(
            ROOT / "results" / run / "benchmark" / "comparison.csv", index=False
        )
        return "skipped_rebench"

    print(f"\n===== [{eid}] train+bench {cfg_path} =====", flush=True)
    run_train_and_compare(
        cfg_path,
        "configs/experiment/compare_val.yaml",
        skip_train=False,
        auto_shutdown=False,
        crop_only_benchmark=bool(exp.get("crop_only")),
    )
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="Comma ids e.g. 01,02,07")
    parser.add_argument("--force", action="store_true", help="Retrain even if best.safetensors exists")
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--no-shutdown", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not abort when CUDA is missing (will fail at train)",
    )
    args = parser.parse_args()
    ensure_root()

    import torch

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "No CUDA GPU. In Colab: Runtime → Change runtime type → GPU (A100 preferred), "
            "then re-run. Or pass --allow-cpu to override."
        )
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    queue = [e for e in EXPERIMENTS if not only or e["id"] in only]

    log = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": []}
    for exp in queue:
        status = "pending"
        err = None
        try:
            if not args.skip_data_prep:
                needs = exp.get("needs", [])
                if "wytham_crops" in needs:
                    prepare_wytham()
                if "multisite_wo" in needs:
                    prepare_multisite()
                if "cedar_key" in needs:
                    try:
                        prepare_cedar_key()
                    except FileNotFoundError as exc:
                        print(f"SKIP {exp['id']}: {exc}", flush=True)
                        log["runs"].append({"id": exp["id"], "status": "skipped_no_data", "error": str(exc)})
                        continue
            status = run_one(exp, force=args.force)
        except Exception as exc:
            status = "failed"
            err = f"{exc}\n{traceback.format_exc()}"
            print(err, flush=True)
        log["runs"].append({"id": exp["id"], "name": exp["name"], "status": status, "error": err})
        (ROOT / "results" / "run_all_log.json").write_text(json.dumps(log, indent=2))

    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (ROOT / "results" / "run_all_log.json").write_text(json.dumps(log, indent=2))
    print("QUEUE_DONE", json.dumps(log, indent=2), flush=True)

    if not args.no_shutdown:
        try:
            from google.colab import runtime  # type: ignore

            print("Unassigning Colab runtime", flush=True)
            runtime.unassign()
        except Exception as exc:
            print(f"No Colab unassign ({exc})", flush=True)


if __name__ == "__main__":
    main()
