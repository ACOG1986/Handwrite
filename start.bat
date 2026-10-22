@echo off
chcp 65001 >nul
rem One click: update the code from GitHub, install libraries if requirements.txt changed, start the app.
rem Runs from a copy in %TEMP%, because git pull may replace this very file and cmd reads .bat files line by line.
if /i not "%~1"=="--copy" (
  copy /y "%~f0" "%TEMP%\handapp_start.bat" >nul && (
    call "%TEMP%\handapp_start.bat" --copy "%~dp0"
    exit /b
  )
  set "HERE=%~dp0"
) else (
  set "HERE=%~2"
)
cd /d "%HERE%"
title Handwrite

echo === Обновление программы ===
set "WARN="
where git >nul 2>nul || (echo [!] Git не найден, обновление пропущено. & set "WARN=1" & goto deps)
if not exist .git (echo [!] Папка не является git-репозиторием, обновление пропущено. & set "WARN=1" & goto deps)
set "BRANCH="
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH=%%b"
set "OLDHEAD="
for /f "delims=" %%c in ('git rev-parse HEAD') do set "OLDHEAD=%%c"
echo Ветка: %BRANCH%
git fetch --quiet
if errorlevel 1 goto nonet
rem Fast-forward only: never merges, never discards anything. data\ is ignored by git and is not touched.
git merge --ff-only --quiet @{u}
if errorlevel 1 goto noupdate
for /f "delims=" %%c in ('git rev-parse HEAD') do if "%%c"=="%OLDHEAD%" (echo Обновлений нет.) else (echo Загружены изменения: & git --no-pager log --oneline %OLDHEAD%..HEAD)
goto deps

:nonet
echo.
echo [!] Не удалось связаться с GitHub. Возможно, нет интернета.
echo     Запускаю текущую версию без обновления.
set "WARN=1"
goto deps

:noupdate
echo.
echo [!] Обновление не применено. Ваши страницы, образцы и адаптеры в папке data не тронуты.
echo     Причина: в папке программы изменены файлы кода, есть свои коммиты,
echo     или у ветки %BRANCH% нет ветки на GitHub. Изменённые файлы:
git status --short --untracked-files=no
echo     Покажите это окно Claude, чтобы разобраться. Запускаю текущую версию.
set "WARN=1"
goto deps

:deps
if defined WARN timeout /t 20
echo.
echo === Проверка библиотек ===
set "NEWHASH="
for /f "skip=1 delims=" %%h in ('certutil -hashfile requirements.txt SHA256') do if not defined NEWHASH set "NEWHASH=%%h"
set "OLDHASH="
if exist .venv\requirements.sha set /p OLDHASH=<.venv\requirements.sha
if not exist .venv\Scripts\python.exe goto setup
if not "%NEWHASH%"=="%OLDHASH%" goto setup
echo Библиотеки в порядке.
goto run

:setup
echo requirements.txt изменился или библиотеки ещё не установлены. Устанавливаю, это может занять несколько минут...
call setup.bat auto
if errorlevel 1 goto setupfail
if defined NEWHASH (echo %NEWHASH%)>.venv\requirements.sha
goto run

:setupfail
echo.
echo [!] Установка библиотек не удалась, смотрите сообщения выше.
if not exist .venv\Scripts\python.exe (
  echo     Без библиотек программа не запустится. Проверьте интернет и запустите start.bat ещё раз.
  pause
  exit /b 1
)
echo     Пробую запустить с тем, что уже установлено. При следующем запуске установка повторится.
timeout /t 20

:run
echo.
echo === Запуск ===
call run.bat
