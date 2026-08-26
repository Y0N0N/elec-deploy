#!/usr/bin/env bash
# ============================================================
# deploy/run_ui.command — macOS 启动配置管理桌面 App
#
#   双击（需已 chmod +x）或在终端运行 ./run_ui.command
#   优先用项目 venv，否则系统 python3。
# ============================================================
cd "$(dirname "$0")"

PARENT="$(cd .. && pwd)"
PY="$PARENT/venv/bin/python"
if [ ! -x "$PY" ]; then PY="python3"; fi

echo "使用 Python: $PY"
exec "$PY" ui_desktop.py
