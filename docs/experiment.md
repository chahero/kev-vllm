# Original Linux / GB10 experiment

This document records the original Linux BF16 experiment. See the
[README](../README.md) for current entry points and the [Windows guide](windows.md)
for the separate native Windows FP16 validation. Current commands use `.venv`;
the original workspace environment was named `.venv-vllm`.

This experiment runs the Qwen3.5 backbone and Kev pointer head **inside vLLM's
GPU worker**, using an out-of-tree model plugin. It does not proxy requests to
`kev.serve`. The native `/pooling` API returns a `[number_of_options, 1]` matrix
of calibrated probabilities. It is not a chat/generation model.

## Observed result — 2026-09-26

Both offline vLLM inference and stock `vllm serve` HTTP `/pooling` passed.
All 5 question rows matched the original Kev BF16 merged reference's argmax.
Maximum absolute probability difference was **0.0021777153** (0.2178 percentage
points). The server listed `kev-4b-experimental` at `/v1/models`; the HTTP batch
completed in 0.2926 seconds in the final rerun. The query CLI also successfully
selected the expected option through HTTP. This is a smoke test, not a latency
benchmark or proof of broad accuracy/calibration parity.

The test server was shut down after validation. vLLM logged forced worker
cleanup and a semaphore cleanup warning during shutdown. No persistent Kev
service is intentionally left running.

Expanded validation subsequently passed **27/27 question rows**, 20–2048
tokens, in full-batch, reversed-batch, and singleton execution. Every execution
matched the original reference's argmax; maximum absolute probability error
across the three modes was **0.0080635548** (0.8064 percentage points), below
the predeclared 0.02 tolerance. Batch order produced identical probabilities;
batch versus singleton differed by at most 0.0043913126. Original Kev's
multi-question versus singleton difference was at most 0.0000986361.

Coverage includes all six permutations of three options, 2/5/8 options,
reversed question order, literal control tokens, evidence at different positions
in longer states, and an exact 2048-token row. State and row overflow are
rejected by the pinned encoder. Output shape, finiteness, probability bounds,
and sum-to-one are checked. Permutations are compared separately against Kev;
this does not assert that the model itself is permutation-invariant.

Expanded evidence is in `expanded-reference.json`, `expanded-result.json`, and
`expanded-validation.log` under `artifacts/vllm-audit/`.

## Pinned inputs

- Kev: `jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101`
- Base: `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`
- Kev source: `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63`
- vLLM: `0.30.0`; PyTorch: `2.13.0+cu130`; Transformers: `5.17.0`; PEFT: `0.21.0`
- Environment: `.venv-vllm`, Python 3.12, Linux aarch64, NVIDIA GB10, driver 580.159.03
- Full package snapshot: `artifacts/vllm-audit/environment-lock.txt`

The base snapshot was already cached. Only the pinned Kev adapter/head was
downloaded for this experiment. The merged text-only checkpoint occupies about
7.83 GiB. `head.pt` has q/k weights `[256,2560]`, biases `[256]`, and fitted
temperature `2.406050072164233`.

## Reproduce from the prepared workspace

To keep the server running in a terminal (Ctrl+C stops it):

```bash
cd /path/to/kev-vllm
bash scripts/serve_kev_vllm.sh
```

In another terminal:

```bash
cd /path/to/kev-vllm
.venv/bin/python scripts/query_kev_vllm.py \
  --state "The package arrived broken. The customer requests a refund." \
  --question "What does the customer want?" \
  --options "A refund" "Tracking information" "A new password"
```

The query client only tokenizes and decodes JSON; all neural inference and
pointer scoring run inside vLLM. Use `--dry-run` to inspect the raw `/pooling`
request. The server script accepts additional vLLM flags, such as `--port 18090`;
use `--url http://127.0.0.1:18090` on the client to match.

For a new environment with compatible NVIDIA hardware/driver and `uv`, run
from this repository root (downloads the pinned source, adapter, and base):

```bash
bash scripts/setup_kev_vllm.sh

# Check prepared model inputs without downloading:
.venv/bin/python scripts/fetch_kev_inputs.py --offline

HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/test_kev_vllm.py

# Expanded reference and vLLM comparison, sequential GPU processes:
.venv/bin/python scripts/validate_kev_vllm.py

# Starts stock vllm serve on 127.0.0.1:18089, checks it, then shuts it down:
.venv/bin/python scripts/test_kev_vllm_http.py
```

The setup script resolves the base snapshot in the current HF cache and writes
`inputs.json` with its path, revisions, and source/adapter hashes. Preparation
uses the pinned Kev loader's BF16 LoRA merge; the resulting language weights
are exported to vLLM. A completed `export-manifest.json` prevents accidental
re-export. `--offline` on setup applies to model inputs; package installation
still requires a package cache or network. This CUDA 13.0 recipe was tested on
the GB10 environment above, not on every platform or on a fresh machine.
Both initial export through setup and a repeated setup that reused the export
were exercised here; see `setup-validation.log` and `setup-repeat.log`.

## What is tested

- Python/CUDA initialization, a BF16 GPU matmul, and a vLLM RMSNorm CUDA kernel.
- Direct raw adapter loading: rejected by `ModelConfig` because the adapter
  directory does not contain a full model `config.json`.
- Five short text question rows from four records, including 2/3/4 options,
  reordered options, multiple questions sharing a state, and missing information.
- Original Kev BF16 merged forward versus vLLM's custom GPU pointer readout.
- Native `vllm serve` model listing and `/pooling` probability response.

Results are recorded separately in `reference.json`, `result.json`, and
`http-result.json` under `artifacts/vllm-audit/`. A missing result file means
that phase has not completed successfully. Logs are `prepare.log`, `vllm.log`,
and `http-server.log`. The smoke-test criterion is identical argmax and maximum
absolute probability difference below 0.02; this is a test tolerance, not a
claim of general accuracy or calibration equivalence.

## Limits

- Text only; no restored vision path, quantization, training, or SGLang testing.
- Native pooling input is already encoded Kev token IDs, one question per row.
  The pinned encoder sanitizes user-supplied control tokens. Raw chat messages
  and TypeSafe's `choice/noul/score` JSON contract are not implemented here.
- This is a local experiment with trusted inputs; malformed rows must be
  rejected by a request-layer validator before production deployment.
- Fixed maximum context 2048; batch concurrency 2; one GPU; eager execution;
  prefix caching and chunked prefill disabled; GPU memory utilization 0.18.
- No throughput comparison. The reference uses PyTorch fallback GDN kernels,
  while vLLM uses optimized kernels; cold-start timings are not benchmarks.
- A virtualenv isolates packages, not GPU/host memory. Existing services share
  the device. Test processes are temporary; no existing service is stopped.

## References

- https://docs.vllm.ai/en/latest/contributing/model/registration/
- https://docs.vllm.ai/en/latest/models/pooling_models/
- https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/colqwen3_5.py
- https://huggingface.co/jaredpalmer/kev-4b
