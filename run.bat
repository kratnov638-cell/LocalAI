@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv || goto :error
)
.venv\Scripts\python.exe -c "import numpy; import PySide6" >nul 2>nul
if errorlevel 1 (
  .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
)
if not exist .venv\Scripts\pythonw.exe goto :error
rem Start the GUI detached so this launcher console closes immediately.
rem Runtime diagnostics are in the in-app Console / Logs tab and logs\localai.log.
start "" .venv\Scripts\pythonw.exe main.py
exit /b 0
:error
echo LocalAI failed to start. See logs\localai.log if available.
pause
exit /b 1
