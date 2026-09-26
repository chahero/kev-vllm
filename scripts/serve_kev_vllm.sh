#!/usr/bin/env bash
set -euo pipefail
KEV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=4
exec "$KEV_ROOT/.venv-vllm/bin/vllm" serve "$KEV_ROOT/artifacts/vllm-audit/merged-text" \
  --served-model-name kev-4b-experimental \
  --host 127.0.0.1 --port 18089 \
  --runner pooling --dtype bfloat16 --enforce-eager \
  --max-model-len 2048 --max-num-seqs 2 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.18 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --pooler-config '{"task":"token_embed","use_activation":false}' \
  --attention-config '{"backend":"TRITON_ATTN"}' \
  "$@"
