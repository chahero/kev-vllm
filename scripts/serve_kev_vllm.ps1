$ErrorActionPreference = 'Stop'
$KevRoot = Split-Path $PSScriptRoot -Parent
$env:PYTHONUTF8 = '1'
$env:VLLM_NO_USAGE_STATS = '1'
& "$KevRoot/.venv/Scripts/python.exe" "$PSScriptRoot/serve_kev_vllm.py" @args
exit $LASTEXITCODE
