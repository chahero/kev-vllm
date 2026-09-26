#!/usr/bin/env bash
# Run from any directory. Existing virtualenv and merged weights are preserved.
set -euo pipefail
KEV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$KEV_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python --torch-backend=cu130 \
  torch==2.13.0 vllm==0.30.0 transformers==5.17.0 peft==0.21.0
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python scripts/fetch_kev_inputs.py "$@"
if [[ ! -f artifacts/vllm-audit/merged-text/export-manifest.json ]]; then
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/prepare_kev_vllm.py
fi
.venv/bin/python - <<'PY'
import json
from pathlib import Path
audit = Path('artifacts/vllm-audit')
inputs = json.loads((audit / 'inputs.json').read_text())
out = audit / 'merged-text'
export = json.loads((out / 'export-manifest.json').read_text())
if export['base_revision'] != inputs['base_revision'] or export['adapter_sha256'] != inputs['sha256']['artifacts/vllm-audit/kev-checkpoint/adapter_model.safetensors']:
    raise ValueError('Export does not match pinned inputs')
for name in ('config.json', 'model.safetensors', 'head.pt', 'tokenizer.json', 'tokenizer_config.json'):
    if not (out / name).is_file():
        raise FileNotFoundError(out / name)
PY
echo 'Ready. Start with: bash scripts/serve_kev_vllm.sh'
