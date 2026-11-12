# Follow-up experiment notebooks

One notebook per experiment under [`notebooks/experiments/`](../notebooks/experiments/). Each run trains (unless skipped), scores with the frozen val-tile config [`configs/experiment/compare_val.yaml`](../configs/experiment/compare_val.yaml), writes a comparison table, and can call `google.colab.runtime.unassign()` to stop compute-unit burn.

Shared runner: [`src/treeseg/engine/experiment_run.py`](../src/treeseg/engine/experiment_run.py).

## Notebooks and recommended runtime

| Notebook | Config | GPU (est.) | Notes |
|----------|--------|------------|-------|
| `01_baseline_reproduce` | `isprs_lora_colab.yaml` | A100 ~12–16 GB | Establish baseline row |
| `02_dora_r8` | `isprs_dora_r8.yaml` | A100 ~12–18 GB | DoRA vs LoRA |
| `03_lr2e4` | `isprs_lora_lr2e4.yaml` | A100 or cheaper if probe allows | |
| `04_lr5e5` | `isprs_lora_lr5e5.yaml` | same | |
| `05_batch4` | `isprs_lora_batch4.yaml` | A100 ~24–32 GB | Not T4 |
| `06_multisite_wytham_ofental` | `multisite_wytham_ofental.yaml` | CPU prepare + A100 train | Also tropical via `multisite_tropical.yaml` |
| `07_negative_clicks_infer` | `negative_clicks.yaml` | T4/L4 ~8–12 GB | Inference only |
| `08_florida_mangrove_cedar_key` | `cedar_key_lora.yaml` | CPU ingest + A100 train | Crop IoU benchmark |

Set `AUTO_SHUTDOWN = True` at the end of each notebook (default). Set `False` while debugging.

## Data prep (CPU only)

```bash
# Cherlet sites (same Zenodo as Wytham)
python scripts/download_cherlet_site.py ofental   # or litchfield, robsoncreek, wytham
python scripts/prepare_data.py configs/data/isprs_ofental.yaml

python scripts/combine_crop_sets.py --out data/multisite_wytham_ofental \
  --train data/wytham_isprs/crops/train data/ofental_isprs/crops/train \
  --val data/wytham_isprs/crops/val

# Florida Cedar Key mangroves (Figshare doi:10.6084/m9.figshare.31324393)
# unzip LAS under data/raw/cedar_key_mangrove, then:
python scripts/ingest_cedar_key.py
```

**Blueflux Everglades** TLS is open but **unlabeled** — not used for supervised training. See NASA ORNL DAAC Blueflux mangrove product for future HAAG labeling.

## Primary metric

Mean val-tile F1 @ IoU 0.5, `ignore_non_tree`, from each notebook’s `results/<run>/benchmark/comparison.csv`. Wytham test strip stays unused for selecting winners.
