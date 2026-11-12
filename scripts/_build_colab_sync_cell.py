import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
payload = (root / "scripts" / "_colab_sync_payload.json").read_text(encoding="utf-8")
cell = (
    "import base64, json\n"
    "from pathlib import Path\n"
    f"PAYLOAD = json.loads({payload!r})\n"
    "root = Path('/content/pointsam-training')\n"
    "for rel, b64 in PAYLOAD.items():\n"
    "    path = root / rel\n"
    "    path.parent.mkdir(parents=True, exist_ok=True)\n"
    "    path.write_bytes(base64.b64decode(b64))\n"
    "    print('wrote', path)\n"
    "print('SYNC_DONE', len(PAYLOAD))\n"
)
out = root / "scripts" / "_colab_sync_cell.py"
out.write_text(cell, encoding="utf-8")
print(out, len(cell))
