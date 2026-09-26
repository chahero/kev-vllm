param([switch]$Offline)
$ErrorActionPreference = 'Stop'
$KevRoot = Split-Path $PSScriptRoot -Parent
Push-Location $KevRoot
try {
    $env:PYTHONUTF8 = '1'
    $env:VLLM_NO_USAGE_STATS = '1'
    $env:HF_HOME = Join-Path $KevRoot 'artifacts/hf-cache'
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
    $Python = Join-Path $KevRoot '.venv/Scripts/python.exe'
    if (!(Test-Path $Python)) {
        & uv venv --python 3.13 --seed .venv
        if ($LASTEXITCODE) { throw 'Python 3.13 environment creation failed; install uv first.' }
    }
    & uv pip install --python $Python torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
    if ($LASTEXITCODE) { throw 'CUDA PyTorch installation failed' }
    & uv pip install --python $Python -r requirements-windows.txt --index-url https://pypi.org/simple
    if ($LASTEXITCODE) { throw 'Windows vLLM installation failed' }
    & uv pip install --python $Python --no-deps -e .
    if ($LASTEXITCODE) { throw 'Kev plugin installation failed' }
    & $Python -m pip check
    if ($LASTEXITCODE) { throw 'Dependency check failed' }
    $FetchArgs = @('scripts/fetch_kev_inputs.py')
    if ($Offline) { $FetchArgs += '--offline' }
    & $Python @FetchArgs
    if ($LASTEXITCODE) { throw 'Pinned model input preparation failed' }
    if (!(Test-Path 'artifacts/windows-vllm-audit/merged-text/export-manifest.json')) {
        & $Python scripts/prepare_kev_vllm.py
        if ($LASTEXITCODE) { throw 'FP16 export failed' }
    }
    & $Python -c "import sys; sys.path.insert(0, 'scripts'); from kev_runtime import verify_export; verify_export()"
    if ($LASTEXITCODE) { throw 'Export verification failed' }
    Write-Host 'Ready. Start with: .\scripts\serve_kev_vllm.ps1'
} finally { Pop-Location }
