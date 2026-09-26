@echo off
setlocal enabledelayedexpansion
title LLM Energy Benchmark Matrix - RTX 4060

set LIMIT=%~1
if "%LIMIT%"=="" set LIMIT=2
set COOLDOWN=%~2
if "%COOLDOWN%"=="" set COOLDOWN=1.0

echo ================================================================
echo   LLM ENERGY BENCHMARK MATRIX - NVIDIA RTX 4060
echo   Validation size: %LIMIT% prompts per dataset
echo ================================================================

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    pause
    exit /b 1
)

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] NVIDIA driver or nvidia-smi was not found.
    pause
    exit /b 1
)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

where ollama >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama was not found. Install it from https://ollama.com/
    pause
    exit /b 1
)

curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Starting Ollama service...
    start "Ollama Server" /min cmd /c "ollama serve > ollama_server.log 2>&1"
    timeout /t 5 /nobreak >nul
)

echo Installing/verifying Python dependencies...
python -m pip install -r requirements.txt matplotlib --quiet --no-warn-script-location
if errorlevel 1 (
    echo [ERROR] Python dependency installation failed.
    pause
    exit /b 1
)

echo Running all three model families...
call run_model_matrix.bat %LIMIT% %COOLDOWN%
if errorlevel 1 (
    echo [ERROR] RTX 4060 benchmark matrix failed.
    pause
    exit /b 1
)

echo Generating RTX 4060 diagrams...
python -u src\generate_model_comparison_plots.py --gpu rtx4060
if errorlevel 1 (
    echo [ERROR] RTX 4060 plot generation failed.
    pause
    exit /b 1
)

echo ================================================================
echo COMPLETE
echo Results: data\results\model_comparisons\*
echo Figures: figures\model_comparisons\rtx4060\
echo ================================================================
start explorer figures\model_comparisons\rtx4060
pause