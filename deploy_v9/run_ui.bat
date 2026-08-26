@echo off
rem ============================================================
rem  deploy/run_ui.bat — Windows 启动配置管理桌面 App
rem
rem  双击即可打开。自动寻找 Python（优先项目 venv，其次系统 Python）。
rem  找不到 Python 会提示并退出。
rem ============================================================
setlocal
cd /d "%~dp0"

set "PARENT=%CD%\.."
set "VENV=%PARENT%\venv\Scripts\python.exe"
set "PY="

if exist "%VENV%" set "PY=%VENV%"
if not defined PY ( where py >nul 2>nul && set "PY=py -3" )
if not defined PY ( where python >nul 2>nul && set "PY=python" )

if not defined PY (
  echo.
  echo   [错误] 未找到 Python。
  echo   请先安装 Python 3（https://www.python.org/downloads/），
  echo   安装时勾选 "Add Python to PATH"。
  echo.
  pause
  exit /b 1
)

echo 使用 Python: %PY%
"%PY%" ui_desktop.py
if errorlevel 1 (
  echo.
  echo   [错误] 启动失败，请确认 Python 3 已正确安装。
  echo.
  pause
)
endlocal
