@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 项目环境缺失，未结束任何进程。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 "%~dp0stop_demo.py" %*
pause
