# Native Windows local experiment

This profile targets the local NVIDIA TITAN RTX (24 GiB, SM 7.5).
It uses native Windows Python and CUDA, without WSL or Docker.

## Environment

- Python 3.13 in `.venv`, created locally on Windows (never copy the Linux virtualenv).
- PyTorch 2.13.0+cu130, Transformers 5.17.0, PEFT 0.21.0.
- Community vLLM 0.29.0 wheel from
  [aivrar/vllm-windows-build, v0.29.0-win-cu130-rc1](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.29.0-win-cu130-rc1).
  This is a prerelease port, not an official Windows vLLM release.
- Triton Windows 3.7.1.post27. The wheel includes SM 7.5 kernels.
- FP16 backbone (TITAN RTX has no native BF16), FP32 pointer head,
  eager execution, Triton attention, 2 concurrent sequences, 2048-token limit.
- Default GPU budget: 75% of VRAM. Override with
  `$env:KEV_GPU_MEMORY_UTILIZATION = '0.70'` if needed.

## Setup

Run in PowerShell at the repository root, with `uv` available:

```powershell
.\scripts\setup_kev_vllm.ps1
```

This installs into the dedicated environment, downloads the pinned upstream
model/source inputs, and exports the FP16 merged model. Initial setup needs
network access and several tens of GB of disk space. It does not modify the
global Python environment. `-Offline` applies to model downloads only;
package installation still requires installed or cached packages.

Official PyPI has no compatible Windows vLLM wheel. The native wheel is still
installable with pip; `requirements-windows.txt` pins its URL and SHA256.
The equivalent installation steps in an existing Python 3.13 virtualenv are:

```powershell
python -m pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-windows.txt
python -m pip install --no-deps -e .
```

The raw command `pip install vllm` cannot provision this Windows build from
PyPI. After installing the wheel, `import vllm` and the vLLM APIs are available.

## Start and query

Double-click **`run_windows.bat`** at the repository root, or use PowerShell:

```powershell
.\run_windows.bat
```

The BAT file finds the repository from its own location and uses `.venv`.
It verifies the prepared export and keeps the console open on errors.
`scripts/serve_kev_vllm.ps1` remains an equivalent PowerShell entry point.
Additional vLLM arguments are forwarded, for example:

```powershell
.\run_windows.bat --port 18090
```

Use `--url http://127.0.0.1:18090` on the query client when changing the port.
For unattended command execution, set `KEV_NO_PAUSE=1` to disable the BAT
file's pause on error. `run_windows.bat --help` shows vLLM serving options.


In another PowerShell window at the repository root:

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe scripts/query_kev_vllm.py --state "The package arrived broken. The customer requests a refund." --question "What does the customer want?" --options "A refund" "Tracking information" "A new password"
```

The service binds to `127.0.0.1:18089`. Use Ctrl+C to stop it.
UTF-8 mode avoids the Korean Windows CP949 decoding failure when PyTorch
loads its kernel templates. The BAT launcher and PowerShell wrappers set it automatically.
First startup may spend several minutes compiling Triton kernels.

## Tests and artifacts

Stop an existing server before running the HTTP test, which starts its own server.

```powershell
.\scripts\test_kev_vllm.ps1
# Expanded comparison: 27 questions, batched/reversed/singleton execution
.\scripts\test_kev_vllm.ps1 -Expanded
```

Windows models and reports are under `artifacts/windows-vllm-audit`.
The Hugging Face setup cache is under `artifacts/hf-cache`.
The original Linux profile still uses `artifacts/vllm-audit`, BF16, and vLLM 0.30.0.
Windows parity comparisons use a freshly generated FP16 Kev reference; they
do not reproduce the recorded Linux BF16 experiment or establish accuracy.

## Local validation (2026-09-26)

Windows / NVIDIA TITAN RTX 24 GiB / driver 610.74 / Python 3.13.2:

- CUDA FP16 matrix multiplication and Kev plugin imports: passed.
- `pip check`: no broken requirements.
- 5 short reference questions: passed.
- Expanded 27-question comparison: passed (20–2048 tokens).
- Batch, reversed-batch and singleton selected options all match the FP16 reference.
- Maximum probability error: 0.0008902252 (threshold 0.02).
- HTTP `/health`, `/v1/models`, `/pooling` and query CLI: passed.
- Example selected `A refund`, probability 0.99660659.
- First uncached in-process model initialization took about 345 seconds,
  including Triton compilation/autotuning. This is not a throughput benchmark.

Reports: `result.json`, `expanded-result.json`, `http-result.json` in
`artifacts/windows-vllm-audit`. The test server was stopped after validation.
No WSL or Docker was installed. Linux was not rerun during this Windows test.


## Troubleshooting

- Missing `.venv` or model export: run `scripts/setup_kev_vllm.ps1` from PowerShell.
- First startup is slow: Triton compiles and tunes kernels; the initial local run took about 345 seconds.
- `cp949` decoding error with a direct Python command: add `-X utf8` or set `PYTHONUTF8=1` before starting Python.
- GPU memory is insufficient: close other GPU workloads or lower `KEV_GPU_MEMORY_UTILIZATION`; keep enough room for approximately 8 GiB of model weights plus runtime/cache memory.
- Port 18089 is in use: stop the existing server or pass `--port` and match the query client's `--url`.

For a concise overview and Linux commands, return to the [README](../README.md).
The committed [Windows numerical report](windows-validation-result.json) records
the expanded FP16 comparison; local generated logs remain under `artifacts`.
