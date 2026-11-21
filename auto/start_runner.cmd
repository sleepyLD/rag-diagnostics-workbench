@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv-auto\Scripts\python.exe" (
  echo Creating .venv-auto ...
  py -3.11 -m venv .venv-auto || py -3 -m venv .venv-auto || python -m venv .venv-auto
)
if exist "auto\STOP" del "auto\STOP"
".venv-auto\Scripts\python.exe" auto\runner.py
echo.
echo Runner exited.
pause
