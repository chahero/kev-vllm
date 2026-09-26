@echo off
setlocal EnableExtensions
set "KEV_EXIT=1"
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHONUTF8=1"
set "VLLM_NO_USAGE_STATS=1"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment is missing. Run scripts\setup_kev_vllm.ps1 in PowerShell first.
    goto :failed
)
if /I "%~1"=="--help" goto :launch
if /I "%~1"=="-h" goto :launch
".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, 'scripts'); from kev_runtime import verify_export; verify_export()"
if errorlevel 1 (
    echo Model preparation is incomplete. Run scripts\setup_kev_vllm.ps1 in PowerShell first.
    goto :failed
)
:launch
echo Opening Kev Playground at http://127.0.0.1:18090. Press Ctrl+C to stop.
".venv\Scripts\python.exe" "scripts\playground.py" --start-model --open %*
set "KEV_EXIT=%ERRORLEVEL%"
if not "%KEV_EXIT%"=="0" goto :failed
popd
exit /b 0
:failed
if not defined KEV_EXIT set "KEV_EXIT=1"
echo.
echo Kev vLLM could not complete. See the message above.
if not "%KEV_NO_PAUSE%"=="1" pause
popd
exit /b %KEV_EXIT%
