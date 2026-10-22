@echo off
chcp 65001 >nul
rem Fine-tunes the model on your verified pages. Close run.bat first so the GPU is free.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
python -m app.train %*
pause
