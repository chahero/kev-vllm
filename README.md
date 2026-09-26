# kev-vllm

An unofficial, experimental vLLM plugin for [Kev](https://github.com/jaredpalmer/kev) decision models, maintained by [chahero](https://github.com/chahero).

The Qwen3.5 backbone and Kev pointer head run inside the vLLM GPU worker. The native `/pooling` endpoint returns a probability for each supplied option. This model selects among options; it does not generate chat responses.

## Quick start

Tested on Linux aarch64 / NVIDIA GB10 with Python 3.12 and CUDA 13.0 wheels. Install `uv` first and use a compatible NVIDIA driver. Other platforms and a completely fresh-machine installation have not been verified.

```bash
git clone https://github.com/chahero/kev-vllm.git
cd kev-vllm
bash scripts/setup_kev_vllm.sh
bash scripts/serve_kev_vllm.sh
```

Setup creates `.venv-vllm`, installs pinned dependencies, downloads pinned upstream inputs, and exports a BF16 merged text checkpoint under `artifacts/vllm-audit/`. Model preparation uses the GPU. Allow disk space for the base, adapter, environment, and approximately 7.83 GiB of merged weights. Existing GPU services share device memory.

In another terminal:

```bash
.venv-vllm/bin/python scripts/query_kev_vllm.py \
  --state "The package arrived broken. The customer requests a refund." \
  --question "What does the customer want?" \
  --options "A refund" "Tracking information" "A new password"
```

The server binds to `127.0.0.1:18089`. Stop it with Ctrl+C. The client uses the pinned Kev encoder; raw `/pooling` input consists of one encoded question per token-ID row, and the response contains an `[options, 1]` probability matrix.

## Validation

```bash
.venv-vllm/bin/python scripts/validate_kev_vllm.py
.venv-vllm/bin/python scripts/test_kev_vllm_http.py
```

On the original development workspace, all 27 questions (20–2,048 tokens) matched the original Kev BF16 merged reference's selected option in batch, reversed-batch, and singleton execution. Maximum absolute probability difference was 0.008064 against a predeclared tolerance of 0.02. HTTP and the query CLI also passed. This is a compatibility experiment, not a general accuracy, calibration, or performance benchmark. The reorganized repository has not yet been tested on a fresh machine.

See [experiment details](docs/experiment.md) and the [recorded numerical results](docs/validation-result.json).

## Scope and limitations

- Text only; fixed maximum row length 2,048 tokens, state limit 1,024 tokens.
- vLLM 0.30.0, PyTorch 2.13.0 (CUDA 13.0), Transformers 5.17.0, PEFT 0.21.0.
- Single GPU, eager execution, batch concurrency 2; prefix caching and chunked prefill disabled.
- Local trusted inputs only. Malformed raw rows can fail the worker; production use needs request validation before the engine.
- No `/v1/chat/completions`, original `choice/noul/score` contract, vision, quantization, or SGLang support.

## Upstream components

Kev source is downloaded at commit `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63`; model inputs are pinned in `scripts/fetch_kev_inputs.py`. Upstream source, model weights, virtual environments, and caches are not included in this repository. See [upstream attribution](THIRD_PARTY.md). This project is not an official Kev or vLLM integration.
