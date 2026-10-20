@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment is missing. No process was stopped.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 "%~dp0stop_demo.py" %*
pause
