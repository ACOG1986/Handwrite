@echo off
chcp 65001 >nul
rem One-time setup on Windows: creates .venv and installs PyTorch with CUDA plus the app's libraries.
cd /d "%~dp0"
where py >nul 2>nul || (echo Установите Python 3.12 с python.org и отметьте "Add python.exe to PATH". & pause & exit /b 1)
py -3.12 -m venv .venv || py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo Готово. Теперь запустите run.bat
pause
