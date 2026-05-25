@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set VERSION=0.1.0
set CUDA=0

:parse_args
if "%~1"=="" goto done_parsing
if "%~1"=="--version" (
    set VERSION=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--cuda" (
    set CUDA=1
    shift
    goto parse_args
)
echo Unknown: %~1
exit /b 1
:done_parsing

set APP_VERSION=%VERSION%

echo === Building YoloLab v%VERSION% for Windows ===

pip install --upgrade pip
if "%CUDA%"=="1" (
    pip install torch --index-url https://download.pytorch.org/whl/cu121
) else (
    pip install torch --index-url https://download.pytorch.org/whl/cpu
)
pip install ultralytics pyyaml pyinstaller PySide6

cd /d "%PROJECT_ROOT%"
pyinstaller --clean --noconfirm packaging\yolo_lab.spec

echo === PyInstaller done, now run Inno Setup to create installer ===
echo === Open packaging\windows\setup.iss in Inno Setup Compiler ===
echo === Or install Inno Setup CLI and run: ===
echo ===   iscc packaging\windows\setup.iss ===

dir packaging\dist\
echo === Build complete ===
