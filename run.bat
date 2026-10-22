@echo off
chcp 65001 >nul
rem Starts the app; open the printed address on your phone (same Wi-Fi).
cd /d "%~dp0"
if not exist .venv (echo Сначала запустите setup.bat & pause & exit /b 1)
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
