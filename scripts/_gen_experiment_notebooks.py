"""One-shot generator for notebooks/experiments/*.ipynb. Run from repo root."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def md(text, id_):
    lines = text.strip().split("\n")
    return {
        "cell_type": "markdown",
        "id": id_,
        "metadata": {},
        "source": [line + "\n" for line in lines],
    }


def code(text, id_):
    lines = text.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "id": id_,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


def write_train_notebook(path, title, hypothesis, runtime, train_config, extra_prep="", crop_only=False, extras_md=""):
    cells = [
        md(
            f"""# {title}

**Hypothesis.** {hypothesis}

**Runtime.** {runtime}

**Config.** `{train_config}`

Train → frozen shared benchmark (`configs/experiment/compare_val.yaml`) → comparison table → optional Colab `runtime.unassign()`.

{extras_md}""".strip(),
            "t0",
        ),
        md("## 0. Options", "t1"),
        code(
            f"""# Colab: mount Drive / clone repo first (see notebooks/00_colab.ipynb), then open this notebook from the repo.
TRAIN_CONFIG = "{train_config}"
COMPARE_CONFIG = "configs/experiment/compare_val.yaml"
SKIP_TRAIN = False          # True to only re-benchmark an existing results/<run>/best.safetensors
AUTO_SHUTDOWN = True        # releases the Colab VM after results are written (set False while debugging)
CROP_ONLY_BENCHMARK = {crop_only}  # True for Cedar Key (no Wytham-style tiles)""",
            "t2",
        ),
        md("## 1. Repo root + VRAM probe", "t3"),
        code(
            """import os, sys
from pathlib import Path
root = Path.cwd().resolve()
for _ in range(4):
    if (root / "src" / "treeseg").is_dir():
        break
    root = root.parent
os.chdir(root)
sys.path.insert(0, "src")

from treeseg.engine.experiment_run import ensure_repo_root, probe_resources, reset_cuda_peaks
import torch

print("cwd:", ensure_repo_root())
reset_cuda_peaks()
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
probe_resources("notebook_start")""",
            "t4",
        ),
    ]
    if extra_prep.strip():
        cells += [
            md("## 1b. Data prepare (CPU-friendly; skip if already on Drive)", "t5"),
            code(extra_prep.strip(), "t6"),
        ]
    cells += [
        md("## 2. Train + shared benchmark + compare", "t7"),
        code(
            """from treeseg.engine.experiment_run import run_train_and_compare
from IPython.display import display

table = run_train_and_compare(
    TRAIN_CONFIG,
    COMPARE_CONFIG,
    skip_train=SKIP_TRAIN,
    auto_shutdown=AUTO_SHUTDOWN,
    crop_only_benchmark=CROP_ONLY_BENCHMARK,
)
display(table)""",
            "t8",
        ),
        # path already set in section 1
        md(
            """## Compute notes

After the first successful run, inspect `results/<run>/resource_probe.json` for peak GPU GB. If peak is well under 14 GB, later LoRA twins may use a cheaper GPU than A100. Always keep checkpoints on Drive before `AUTO_SHUTDOWN`.""",
            "t9",
        ),
    ]
    path.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("wrote", path)


write_train_notebook(
    OUT / "01_baseline_reproduce.ipynb",
    "01 — Baseline reproduce (`isprs_lora`)",
    "Colab A100 reproduces the locked Wytham LoRA tile metrics.",
    "A100 (~12–16 GB VRAM est.); host RAM ≥16 GB",
    "configs/experiment/isprs_lora_colab.yaml",
)

write_train_notebook(
    OUT / "02_dora_r8.ipynb",
    "02 — DoRA r8 on Wytham",
    "DoRA r8 improves val-tile F1 vs LoRA r8 on the same Wytham recipe.",
    "A100 (~12–18 GB VRAM est.)",
    "configs/experiment/isprs_dora_r8.yaml",
)

write_train_notebook(
    OUT / "03_lr2e4.ipynb",
    "03 — Higher learning rate",
    "Doubling LR improves val-tile F1 vs baseline.",
    "A100 or cheaper GPU if baseline peak <14 GB",
    "configs/experiment/isprs_lora_lr2e4.yaml",
)

write_train_notebook(
    OUT / "04_lr5e5.ipynb",
    "04 — Lower learning rate",
    "Halving LR improves val-tile F1 vs baseline.",
    "A100 or cheaper GPU if baseline peak <14 GB",
    "configs/experiment/isprs_lora_lr5e5.yaml",
)

write_train_notebook(
    OUT / "05_batch4.ipynb",
    "05 — Batch size 4",
    "Larger effective batch improves val-tile F1.",
    "A100 required (~24–32 GB VRAM est.); not T4",
    "configs/experiment/isprs_lora_batch4.yaml",
)

write_train_notebook(
    OUT / "06_multisite_wytham_ofental.ipynb",
    "06 — Multi-site Wytham + Ofental",
    "Adding Ofental (temperate conifer) crops improves val-tile F1 / generalization.",
    "CPU for download/prepare; A100 for train",
    "configs/experiment/multisite_wytham_ofental.yaml",
    extra_prep="""# Run on a CPU runtime first if data is missing, then reconnect with A100 for section 2.
from pathlib import Path
import subprocess, sys

def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.check_call(cmd, shell=True)

if not Path("data/ofental_isprs/crops/train").is_dir():
    sh(f"{sys.executable} scripts/download_cherlet_site.py ofental")
    sh(f"{sys.executable} scripts/prepare_data.py configs/data/isprs_ofental.yaml")
if not Path("data/wytham_isprs/crops/train").is_dir():
    sh(f"{sys.executable} scripts/download_cherlet_site.py wytham")
    sh(f"{sys.executable} scripts/prepare_data.py configs/data/isprs_wytham_colab.yaml")
if not Path("data/multisite_wytham_ofental/crops/train").is_dir():
    sh(f"{sys.executable} scripts/combine_crop_sets.py --out data/multisite_wytham_ofental "
       "--train data/wytham_isprs/crops/train data/ofental_isprs/crops/train "
       "--val data/wytham_isprs/crops/val")
print("multi-site crops ready")""",
    extras_md="For tropical Robson/Litchfield, download those sites, combine into `data/multisite_tropical`, and set `TRAIN_CONFIG` to `configs/experiment/multisite_tropical.yaml`.",
)

cells07 = [
    md(
        """# 07 — Negative / correction clicks (inference only)

**Hypothesis.** Adding neighbour negative clicks at tile time raises F1 for a fixed checkpoint.

**Runtime.** T4 or L4 (~8–12 GB). No training.

Uses `configs/experiment/negative_clicks.yaml`.""",
        "n0",
    ),
    code(
        """CHECKPOINT = "results/isprs_lora/best.safetensors"  # or best from 02/06
AUTO_SHUTDOWN = True""",
        "n1",
    ),
    code(
        """import os, sys
from pathlib import Path
root = Path.cwd().resolve()
for _ in range(4):
    if (root / "src" / "treeseg").is_dir():
        break
    root = root.parent
os.chdir(root)
sys.path.insert(0, "src")

from treeseg.engine.experiment_run import ensure_repo_root, probe_resources, maybe_shutdown, comparison_table
from treeseg.engine.benchmark import run_benchmark, summarise
from treeseg.models.point_sam import add_point_sam_to_path
from treeseg.utils.curves import plot_f1_bars
from IPython.display import Image, display
import yaml, torch

ensure_repo_root()
add_point_sam_to_path()
cfg = yaml.safe_load(open("configs/experiment/negative_clicks.yaml"))
cfg["checkpoints"] = {"run": CHECKPOINT}
assert Path(CHECKPOINT).is_file(), CHECKPOINT
os.makedirs(cfg["out_dir"], exist_ok=True)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
probe_resources("neg_start")
run_benchmark(cfg)
summary = summarise(os.path.join(cfg["out_dir"], "results.csv"))
summary.to_csv(os.path.join(cfg["out_dir"], "summary.csv"), index=False)
table = comparison_table(summary)
display(table)
plot_f1_bars(summary, os.path.join(cfg["out_dir"], "f1_bars.png"))
display(Image(os.path.join(cfg["out_dir"], "f1_bars.png")))
probe_resources("neg_end")
maybe_shutdown(AUTO_SHUTDOWN)""",
        "n2",
    ),
]
(OUT / "07_negative_clicks_infer.ipynb").write_text(json.dumps(nb(cells07), indent=1), encoding="utf-8")
print("wrote 07")

write_train_notebook(
    OUT / "08_florida_mangrove_cedar_key.ipynb",
    "08 — Florida Cedar Key mangrove transfer",
    "Fine-tuning on Florida black mangrove trees improves single-tree mask IoU for subtropical mangroves.",
    "CPU for ingest; A100 for train",
    "configs/experiment/cedar_key_lora.yaml",
    crop_only=True,
    extra_prep="""# Place the Figshare LAS archive under data/raw/cedar_key_mangrove (doi:10.6084/m9.figshare.31324393), then:
from pathlib import Path
import subprocess, sys

def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.check_call(cmd, shell=True)

if not Path("data/cedar_key_mangrove/crops/train").is_dir():
    sh(f"{sys.executable} scripts/ingest_cedar_key.py configs/data/cedar_key_mangrove.yaml")
print("Cedar Key crops ready")
print("Blueflux Everglades TLS is unlabeled — not used for supervised train "
      "(https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-tls-lidar-blueflux-mangroves-2311-1)")""",
    extras_md="Benchmark is **crop IoU** (single-tree set), not Wytham tiles. For tropical plot instances in our PLY schema, prefer Robson Creek via notebook 06 + `multisite_tropical.yaml`.",
)

print("done", len(list(OUT.glob("*.ipynb"))), "notebooks")
