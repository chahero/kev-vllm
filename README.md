# kev-vllm

**English** | [한국어](README.ko.md)

An unofficial, experimental vLLM plugin for [Kev](https://github.com/jaredpalmer/kev) decision models, maintained by [chahero](https://github.com/chahero).

Given a state, a question, and options, Kev returns a probability for each option.
The Qwen3.5 backbone and Kev pointer head run inside the vLLM GPU worker;
the native `/pooling` endpoint returns an `[options, 1]` probability matrix.
This model selects an option rather than generating chat responses.

## Tested environments

| Profile | GPU | Python | vLLM | Precision | Model and result directory |
| --- | --- | --- | --- | --- | --- |
| Native Windows | TITAN RTX 24 GiB | 3.13 | 0.29.0 community Windows build | FP16 | `artifacts/windows-vllm-audit` |
| Linux aarch64 | NVIDIA GB10 | 3.12 | 0.30.0 | BF16 | `artifacts/vllm-audit` |

Both profiles use a local **`.venv`** and PyTorch 2.13.0 / CUDA 13.0,
Transformers 5.17.0, and PEFT 0.21.0. Create the environment separately on
each operating system; do not copy `.venv` between machines.

## Windows quick start

WSL and Docker are not required. Install `uv` and use a compatible NVIDIA driver.
In PowerShell at the repository root, prepare the environment and model once:

```powershell
.\scripts\setup_kev_vllm.ps1
```

Then **double-click `run_windows.bat`** in the repository root, or run:

```powershell
.\run_windows.bat
```

The launcher uses `.venv`, sets UTF-8 mode, checks that the environment and
model export exist, and starts the server. It works from any current directory.
It does not install packages or download models. Startup errors remain visible
in the console. Stop the server with Ctrl+C.

In another PowerShell window at the repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/query_kev_vllm.py --state "The package arrived broken. The customer requests a refund." --question "What does the customer want?" --options "A refund" "Tracking information" "A new password"
```

See [Windows installation, pip alternative, settings and troubleshooting](docs/windows.md).
The Windows wheel is pinned by URL and SHA256 in `requirements-windows.txt`;
plain `pip install vllm` from PyPI does not provide this tested Windows build.

## Linux quick start

With `uv` and a compatible NVIDIA driver:

```bash
git clone https://github.com/chahero/kev-vllm.git
cd kev-vllm
bash scripts/setup_kev_vllm.sh
bash scripts/serve_kev_vllm.sh
```

In another terminal:

```bash
.venv/bin/python scripts/query_kev_vllm.py \
  --state "The package arrived broken. The customer requests a refund." \
  --question "What does the customer want?" \
  --options "A refund" "Tracking information" "A new password"
```

Both servers bind to `127.0.0.1:18089`. Setup downloads pinned upstream inputs
and exports approximately 7.83 GiB of merged weights. Allow several tens of GB
for inputs, environments and caches. Preparation uses the GPU; the first
Windows startup can take several minutes to compile kernels.

## Validation

Stop an existing server before the HTTP test, which starts its own server.

Windows (27 questions plus HTTP/CLI):

```powershell
.\scripts\test_kev_vllm.ps1 -Expanded
```

Linux:

```bash
.venv/bin/python scripts/validate_kev_vllm.py
.venv/bin/python scripts/test_kev_vllm_http.py
```

| Recorded profile | Questions | Selected options match reference | Maximum probability error | HTTP / CLI |
| --- | --- | --- | --- | --- |
| Windows FP16 | 27 | 27/27 in batch, reversed batch and singleton | 0.0008902252 | Passed |
| Linux BF16 | 27 | 27/27 in batch, reversed batch and singleton | 0.0080635548 | Passed |

Each profile is compared with an original Kev reference at its own precision,
with tolerance 0.02 and inputs of 20–2048 tokens. These are compatibility tests,
not general accuracy, calibration or performance benchmarks. Windows was tested
on the local TITAN RTX; Linux results are from the original GB10 workspace.
The Linux path has not been rerun after the Windows/documentation changes.

## Documentation

| Document | Contents |
| --- | --- |
| [Windows guide](docs/windows.md) | Setup, launcher, pip installation, memory settings, troubleshooting and local results |
| [Original Linux experiment](docs/experiment.md) | Design, pinned inputs, original GB10 observations and reproduction notes |
| [Windows numerical results](docs/windows-validation-result.json) | Recorded 27-question FP16 comparison |
| [Linux numerical results](docs/validation-result.json) | Recorded 27-question BF16 comparison |
| [Third-party attribution](THIRD_PARTY.md) | Upstream components and licenses |

## Scope

- Text only; maximum row length 2048 tokens and state limit 1024 tokens.
- Single GPU, eager execution, batch concurrency 2; prefix caching and chunked prefill disabled.
- Local trusted inputs only. Malformed raw rows can fail the worker; production use needs request validation before the engine.
- No `/v1/chat/completions`, original `choice/noul/score` contract, vision, quantization or SGLang support.

Kev source is pinned to `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63`;
model revisions are in `scripts/fetch_kev_inputs.py`. Upstream source, weights,
virtual environments and caches are downloaded/generated locally and ignored by Git.
This project is not an official Kev or vLLM integration.
