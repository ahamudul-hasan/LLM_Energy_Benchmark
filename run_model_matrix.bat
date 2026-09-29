@echo off
setlocal enabledelayedexpansion

set LIMIT=%~1
if "%LIMIT%"=="" set LIMIT=2
set COOLDOWN=%~2
if "%COOLDOWN%"=="" set COOLDOWN=1.0

echo ===============================================================================
echo   RUNNING MULTI-MODEL BENCHMARK MATRIX (LLAMA vs QWEN 2.5)
echo   Workload: %LIMIT% prompts per dataset
echo ===============================================================================

echo.
echo [1/2] Benchmarking Meta Llama Family (Tier 1: llama3.2:1b, Tier 2: llama3.2:3b, Tier 3: llama3.1:8b)...
python -u src\evaluate_router.py --limit-per-ds %LIMIT% --cool-down %COOLDOWN% --eval-all-tiers --tier1-model llama3.2:1b --tier2-model llama3.2:3b --tier3-model llama3.1:8b --run-label llama
if errorlevel 1 (
    echo [ERROR] Llama benchmark run failed.
    exit /b 1
)

echo.
echo [2/2] Benchmarking Alibaba Qwen 2.5 Family (Tier 1: qwen2.5:1.5b, Tier 2: qwen2.5:3b, Tier 3: qwen2.5:7b)...
python -u src\evaluate_router.py --limit-per-ds %LIMIT% --cool-down %COOLDOWN% --eval-all-tiers --tier1-model qwen2.5:1.5b --tier2-model qwen2.5:3b --tier3-model qwen2.5:7b --run-label qwen
if errorlevel 1 (
    echo [ERROR] Qwen benchmark run failed.
    exit /b 1
)

echo.
echo ===============================================================================
echo   MODEL MATRIX BENCHMARK COMPLETE!
echo ===============================================================================
