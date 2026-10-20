@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 项目环境缺失，请按 README 恢复 .venv 后重试。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 "%~dp0run_demo.py" %*
if errorlevel 1 pause
