@echo off
REM Run BIPHUB Pipeline from anywhere by dragging YAML config onto this file

cd /d "C:\biphub\git\run_pipeline"

if "%~1"=="" (
    echo.
    echo ========================================
    echo BIPHUB Pipeline Runner
    echo ========================================
    echo.
    echo No YAML file provided.
    echo.
    echo Usage:
    echo   Drag your pipeline .yaml file onto this batch file
    echo.
    echo Example YAML files are in the examples/ folder
    echo.
    echo.
    cmd /k run_pipeline.exe --help
) else (
    echo.
    echo Running pipeline with: %~1
    echo.
    cmd /k run_pipeline.exe "%~1"
)
