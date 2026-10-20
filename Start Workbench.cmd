@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment is missing. Restore .venv according to README and try again.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 "%~dp0run_demo.py" %*
if errorlevel 1 pause
