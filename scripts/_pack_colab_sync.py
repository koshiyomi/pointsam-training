from pathlib import Path
import base64
import json

root = Path(__file__).resolve().parents[1]
files = [
    "scripts/archive_util.py",
    "scripts/run_all_experiments.py",
    "scripts/setup_colab.py",
    "scripts/download_cherlet_site.py",
    "scripts/combine_crop_sets.py",
    "scripts/ingest_cedar_key.py",
    "src/treeseg/engine/experiment_run.py",
    "src/treeseg/models/lora.py",
    "src/treeseg/models/point_sam.py",
    "configs/experiment/compare_val.yaml",
    "configs/experiment/isprs_dora_r8.yaml",
    "configs/experiment/isprs_lora_colab.yaml",
    "configs/experiment/multisite_wytham_ofental.yaml",
    "configs/experiment/multisite_tropical.yaml",
    "configs/experiment/cedar_key_lora.yaml",
    "configs/data/isprs_ofental.yaml",
    "configs/data/isprs_litchfield.yaml",
    "configs/data/isprs_robsoncreek.yaml",
    "configs/data/cedar_key_mangrove.yaml",
    "configs/data/isprs_wytham_colab.yaml",
]
payload = {f: base64.b64encode((root / f).read_bytes()).decode("ascii") for f in files}
out = root / "scripts" / "_colab_sync_payload.json"
out.write_text(json.dumps(payload))
print(out, out.stat().st_size, len(payload))
