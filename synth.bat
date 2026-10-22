@echo off
chcp 65001 >nul
rem Draws synthetic Lithuanian handwriting lines for train.bat. Options: --count 3000 --fresh --wiki 300
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
python -m app.synth %*
pause
