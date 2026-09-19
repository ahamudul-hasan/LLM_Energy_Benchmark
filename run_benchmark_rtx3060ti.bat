@echo off
setlocal enabledelayedexpansion
title LLM Energy Benchmark Runner - RTX 3060 Ti (Platform B)

echo ===============================================================================
echo      LLM ENERGY EFFICIENCY BENCHMARK - PLATFORM B (RTX 3060 Ti)
echo      United International University - Dept. of CSE
echo ===============================================================================
echo.

:: 1. Check Python installation
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not added to your PATH!
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)
python --version

:: 2. Check and install Python dependencies
echo.
echo [2/6] Verifying required Python packages...
python -m pip install -r requirements.txt matplotlib --quiet --no-warn-script-location
if errorlevel 1 (
    echo [WARNING] Pip install had warnings or errors. Attempting to proceed...
) else (
    echo Python dependencies verified.
)

:: 3. Verify Ollama installation and service
echo.
echo [3/6] Checking Ollama service and local models...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not installed!
    echo Please install Ollama from https://ollama.com/
    pause
    exit /b 1
)

:: Check if Ollama server is responding
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
    echo See ollama_server.log for startup details.
    pause
    exit /b 1
)
:OllamaReady

:: Check and pull required models
echo Checking model tiers in Ollama...
ollama list | findstr /i "llama3.2:1b" >nul 2>&1
if errorlevel 1 (
    echo Pulling Tier 1 model: llama3.2:1b - approx 1.3 GB...
    ollama pull llama3.2:1b
)

ollama list | findstr /i "llama3.2:3b" >nul 2>&1
if errorlevel 1 (
    echo Pulling Tier 2 model: llama3.2:3b - approx 2.0 GB...
    ollama pull llama3.2:3b
)

ollama list | findstr /i "llama3.1:8b" >nul 2>&1
if errorlevel 1 (
    echo Pulling Tier 3 model: llama3.1:8b - approx 4.9 GB...
    ollama pull llama3.1:8b
)
echo All 3 model tiers are available in Ollama!

:: 4. Prompt user for benchmark size (with 10-second default timeout)
echo.
echo ===============================================================================
echo   SELECT BENCHMARK WORKLOAD SIZE:
echo   [1] Fast Validation Run  : 12 prompts (2 per dataset)  - ~4 minutes [DEFAULT]
echo   [2] Medium Benchmark Run : 60 prompts (10 per dataset) - ~15-20 minutes
echo   [3] Full Paper Benchmark : 600 prompts (all datasets)  - ~2 hours
echo ===============================================================================
echo.

set CHOICE=1
choice /c 123 /t 10 /d 1 /m "Select option (automatically starts [1] in 10s): "
if errorlevel 3 set LIMIT=100
if errorlevel 2 if not errorlevel 3 set LIMIT=10
if errorlevel 1 if not errorlevel 2 set LIMIT=2

echo.
echo Selected prompt limit per dataset: %LIMIT%
echo.

:: 5. Execute Live GPU Benchmark with NVML Power Sampling
echo [4/6] Running Benchmark with Real-Time NVML GPU Power Profiling...
echo Models: Static llama3.1:8b Baseline vs. Adaptive CPU Router
echo Quiescent resting idle power and 50ms hardware power integration active.
echo -------------------------------------------------------------------------------
python -u src\evaluate_router.py --limit-per-ds %LIMIT% --cool-down 1.0
if errorlevel 1 (
    echo [ERROR] Benchmark execution failed! Please check logs above.
    pause
    exit /b 1
)

:: 6. Generate Publication Figures & Cross-Platform Comparison
echo.
echo [5/6] Generating Publication Figures and Cross-Platform Scaling Analysis...
echo -------------------------------------------------------------------------------
python -u src\generate_paper_plots.py --gpu rtx3060ti
if errorlevel 1 (
    echo [WARNING] Plot generation script encountered an issue.
)

:: 7. Complete and show results
echo.
echo ===============================================================================
echo [6/6] BENCHMARK COMPLETE!
echo.
echo All results, CSV metrics, and figures have been generated:
echo - RTX 3060 Ti Figures : figures\rtx3060ti\*.png
echo - Comparison Figures  : figures\cross_platform\*.png
echo - RTX 3060 Ti Data    : data\results\rtx3060ti\
echo ===============================================================================
echo.

:: Automatically open figures folder in Windows Explorer
if exist figures\rtx3060ti (
    echo Opening RTX 3060 Ti figures folder...
    start explorer figures\rtx3060ti
) else (
    if exist figures start explorer figures
)

echo.
echo Press any key to exit...
pause >nul
