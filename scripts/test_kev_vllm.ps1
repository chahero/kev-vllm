param([switch]$Expanded)
$ErrorActionPreference = 'Stop'
$KevRoot = Split-Path $PSScriptRoot -Parent
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:VLLM_NO_USAGE_STATS = '1'
$env:OMP_NUM_THREADS = '4'
$Python = Join-Path $KevRoot '.venv/Scripts/python.exe'
if ($Expanded) {
    & $Python "$PSScriptRoot/validate_kev_vllm.py"
} else {
    & $Python "$PSScriptRoot/test_kev_vllm.py"
}
if ($LASTEXITCODE) { throw 'Kev/vLLM probability comparison failed' }
& $Python "$PSScriptRoot/test_kev_vllm_http.py"
if ($LASTEXITCODE) { throw 'HTTP or CLI test failed' }
