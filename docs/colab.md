# Running on Google Colab

Use a GPU runtime (**Runtime → Change runtime type → GPU**). Free Colab is usually a T4; that is enough for the smoke test and for training, but a full `isprs_lora` run is much slower than the ~20 minutes on an ICE H100, and the session can disconnect.

## Fast path

1. Open [`notebooks/00_colab.ipynb`](../notebooks/00_colab.ipynb) in Colab (GitHub → Open in Colab, or upload the file).
2. Set `REPO_URL` / `BRANCH` / `USE_DRIVE` in the first code cell if needed.
3. Run all cells. Setup clones the repo, installs Point-SAM + `torkit3d`, downloads weights and Wytham data, builds crops, then runs the smoke train notebook.

To keep data and checkpoints across sessions, set `USE_DRIVE = True` so work lives under `MyDrive/pointsam-training`.

## Speeding up new Colab VMs

Colab wipes the VM each time; only Drive (and what you cache there) survives. After the first successful setup:

| Keep on Drive | Effect on next VM |
|---|---|
| `MyDrive/pointsam-training` (repo, `thirdparty/`, `data/`) | No re-clone, no Zenodo re-download, no crop rebuild |
| `MyDrive/pointsam-colab-cache/wheels/*.whl` | `torkit3d` installs in seconds instead of a CUDA compile |
| `FORCE_REFRESH = False` | Skips `git fetch`/`reset` of the Drive checkout |

Leave `DOWNLOAD_DATA` / `PREPARE_DATA` on; `setup_colab.py` skips them when crops already exist (`--force-data` to rebuild). Pip packages still install each VM (Colab’s site-packages is ephemeral); that is usually minutes, not tens of minutes.

## What the setup does

`scripts/setup_colab.py` mirrors `scripts/slurm/install_env.sbatch`, adapted for Colab:

- keeps Colab’s PyTorch (does not force the ICE `cu126` wheel)
- clones Point-SAM at the pinned commit into `thirdparty/Point-SAM` (skipped if already checked out)
- installs `requirements.txt` and builds or **reuses a Drive-cached** `torkit3d` wheel
- downloads `thirdparty/checkpoints/point_sam/model.safetensors` once onto Drive
- with `--download-data` / `--prepare-data`, uses `configs/data/isprs_wytham_colab.yaml` (`data/raw/isprs_wytham` → `data/wytham_isprs`), skipped when crops exist


## After setup

From the repo root on Colab:

```python
import os
os.environ["TREESEG_CONFIG"] = "configs/experiment/smoke_colab.yaml"  # then isprs_lora_colab.yaml
```

Open or `%run` `notebooks/02_train.ipynb`. Benchmark and inference notebooks work the same way once crops (and a checkpoint) exist.

## Tips

- Start with `smoke_colab.yaml` (4 steps) to prove the install, then `isprs_lora_colab.yaml`.
- Until Colab support is on `main`, push this branch (or upload a zip of the repo to Drive and point the notebook at that folder).
- Colab disks are temporary unless you use Drive; copy `results/` out when a run finishes.
- TreeX (`install_treex.sbatch`) is optional and not part of the Colab setup; stem clicks and label clicks work without it.
