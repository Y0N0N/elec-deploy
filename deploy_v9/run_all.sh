#!/usr/bin/env bash
# ============================================================
# deploy/run_all.sh — 一键运行每日完整流程
#   1. 导入新 xlsx 数据            (code/common/1_ingest_xlsx.py)
#   2A. 重建因子 + 重训模型        (code/common/2A_rebuild.py)
#      重训 active_model（默认 v9）+ v9.1（ATR 过滤器主模型）
#   2B. 推理预测（默认用 v9.1，输出 csv，新命名 预测_<model>_<start>_<end>_<flag>）
#   2C. 方向核对（历史预测 vs 实际结算价，套利时机/结果，a/b/c 标记）
#   4. 归档旧模型 & 已导入的 xlsx（code/common/archive_old.py，进 archive/）
#
# 用法：
#   bash run_all.sh            # 完整流程
#   bash run_all.sh --skip-model  # 导入+重建因子+推理+核对（不重训模型，用已有模型）
# ============================================================
set -e
cd "$(dirname "$0")"

# 可通过 PYTHON 环境变量指定解释器；默认使用本机 python3。
PY="${PYTHON:-python3}"
if [ -x "../venv/bin/python" ]; then PY="../venv/bin/python"; fi

SKIP_MODEL=false
[ "$1" = "--skip-model" ] && SKIP_MODEL=true

# 重训的模型列表：active_model（config 里 v9）+ v9.1（ATR 主模型）。两者都训。
# 用 $PY 读取（与脚本执行解释器一致，避免系统 python3 缺失/版本不一致时挂，BUG-5 修复）
MODELS_TO_TRAIN="$("$PY" -c "import json; print(json.load(open('config.json'))['active_model'])") v9.1"

echo "================================================"
echo "  [1/5] 导入每日新数据"
echo "================================================"
"$PY" code/common/1_ingest_xlsx.py

if [ "$SKIP_MODEL" = "true" ]; then
  echo "================================================"
  echo "  [2/5] 重建因子（跳过模型重训）"
  echo "================================================"
  "$PY" code/common/2A_rebuild.py --factors
else
  echo "================================================"
  echo "  [2/5] 重建因子 + 重训模型"
  echo "================================================"
  "$PY" code/common/2A_rebuild.py --factors
  for MK in $MODELS_TO_TRAIN; do
    echo "── 重训模型: $MK ──"
    "$PY" code/common/2A_rebuild.py --model --model-key "$MK"
  done
fi

echo "================================================"
echo "  [3/5] 推理预测（v9.1 主模型）"
echo "================================================"
"$PY" code/common/2B_inference.py --model v9.1

echo "================================================"
echo "  [4/5] 方向核对（历史预测 vs 实际结算价）"
echo "================================================"
"$PY" code/common/2C_verify.py

echo "================================================"
echo "  [5/5] 归档旧模型 & 已导入的 xlsx"
echo "================================================"
"$PY" code/common/archive_old.py || echo "警告: 归档步骤失败（不影响主流程）"

echo
echo "完成 每日流程全部完成。预测结果见 deploy/output/ 目录（文件名带 模型+核对标记 a/b/c）。"
echo "   旧模型/已导入 xlsx 已归档到 deploy/archive/（保留 1 份最新，可手动清理）。"
