@echo off
chcp 65001 >nul
rem One-time setup on Windows: creates .venv and installs PyTorch with CUDA plus the app's libraries.
rem "setup.bat auto" (used by start.bat) does not wait for a key press at the end.
cd /d "%~dp0"
where py >nul 2>nul || (echo Установите Python 3.12 с python.org и отметьте "Add python.exe to PATH". & goto fail)
if not exist .venv\Scripts\python.exe (py -3.12 -m venv .venv || py -3 -m venv .venv || goto fail)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto fail
python -m pip install -r requirements.txt || goto fail
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo Готово. Дальше запускайте start.bat
if /i not "%~1"=="auto" pause
exit /b 0

:fail
echo Установка не завершена, смотрите сообщения выше.
if /i not "%~1"=="auto" pause
exit /b 1
