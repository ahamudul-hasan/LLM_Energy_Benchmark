@echo off
setlocal enabledelayedexpansion
title LLM Energy Benchmark Matrix - RTX 3060 Ti (Platform B)

echo ===============================================================================
echo      LLM ENERGY BENCHMARK: MODEL MATRIX - PLATFORM B (RTX 3060 Ti)
echo      6 Models Across 3 Tiers (Meta Llama vs Alibaba Qwen 2.5)
echo      United International University - Dept. of CSE
echo ===============================================================================
echo.

:: 1. Check Python installation
echo [1/6] Checking Python installation...
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH!
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)
python --version

:: 2. Check NVIDIA GPU and driver
echo.
echo [2/6] Verifying NVIDIA GPU hardware...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] NVIDIA driver or nvidia-smi was not found!
    pause
    exit /b 1
)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

:: 3. Check Ollama service and auto-start
echo.
echo [3/6] Checking Ollama service...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama was not found! Install it from https://ollama.com/
    pause
    exit /b 1
)

curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Ollama server is not running. Starting Ollama in background...
    start "Ollama Server" /min cmd /c "ollama serve > ollama_server.log 2>&1"
    for /l %%i in (1,1,15) do (
        curl -s http://localhost:11434/api/tags >nul 2>&1
        if not errorlevel 1 goto OllamaReady
        timeout /t 1 /nobreak >nul
    )
    echo [ERROR] Ollama did not become ready within 15 seconds.
    pause
    exit /b 1
)
:OllamaReady

:: 4. Verify and pull all 6 models across 3 tiers
echo.
echo [4/6] Verifying candidate model pool (6 models total)...
for %%M in (llama3.2:1b llama3.2:3b llama3.1:8b qwen2.5:1.5b qwen2.5:3b qwen2.5:7b) do (
    ollama list | findstr /i "%%M" >nul 2>&1
    if errorlevel 1 (
        echo [OLLAMA] Pulling missing model: %%M...
        ollama pull %%M
    ) else (
        echo [OK] Model available: %%M
    )
)
echo All 6 candidate models are verified and ready!

:: 5. Install Python dependencies
echo.
echo [5/6] Verifying Python dependencies...
python -m pip install -r requirements.txt matplotlib pandas numpy --quiet --no-warn-script-location
if errorlevel 1 (
    echo [WARNING] Pip install had warnings. Proceeding...
)

:: 6. Select Workload Size
echo.
echo ===============================================================================
echo   SELECT WORKLOAD SIZE PER DATASET (6 datasets total):
echo   [1] Fast Validation : 2 prompts per dataset  (12 queries)  ~4-5 mins [DEFAULT]
echo   [2] Medium Matrix   : 10 prompts per dataset (60 queries)  ~20-25 mins
echo   [3] Full Benchmark  : 100 prompts per dataset (600 queries) ~2-3 hours
echo ===============================================================================
echo.

set LIMIT=%~1
set COOLDOWN=%~2
if "%COOLDOWN%"=="" set COOLDOWN=1.0

if "%LIMIT%"=="" (
    choice /c 123 /t 10 /d 1 /m "Select option (automatically starts [1] in 10s): "
    if errorlevel 3 set LIMIT=100
    if errorlevel 2 if not errorlevel 3 set LIMIT=10
    if errorlevel 1 if not errorlevel 2 set LIMIT=2
)

echo.
echo Selected limit: %LIMIT% prompts per dataset (%LIMIT% x 6 = %LIMIT%0 prompts per family)
echo Cooldown: %COOLDOWN%s
echo.

:: 7. Execute Benchmark Matrix
echo ===============================================================================
echo   RUNNING LIVE GPU BENCHMARK (RTX 3060 Ti)
echo   50 ms NVML power profiling, quiescent resting baseline active.
echo ===============================================================================

call run_model_matrix.bat %LIMIT% %COOLDOWN%
if errorlevel 1 (
    echo [ERROR] Model matrix benchmark failed!
    pause
    exit /b 1
)

:: 8. Generate Visualizations and Head-to-Head Report
echo.
echo Generating RTX 3060 Ti figures and head-to-head tier reports...
python -u src\generate_model_comparison_plots.py --gpu rtx3060ti
python -u src\compare_tier_models.py --gpu rtx3060ti
if errorlevel 1 (
    echo [WARNING] Comparison plot script had an issue.
)

echo.
echo ===============================================================================
echo   BENCHMARK COMPLETE!
echo   Results: data\results\model_comparisons\*
echo   Figures: figures\model_comparisons\rtx3060ti\
echo   Report : figures\model_comparisons\rtx3060ti\tier_comparison_report.md
echo   Table  : figures\model_comparisons\rtx3060ti\tier_model_comparison_table.tex
echo ===============================================================================
echo.

if exist figures\model_comparisons\rtx3060ti (
    start explorer figures\model_comparisons\rtx3060ti
)

pause
