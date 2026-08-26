#!/usr/bin/env python
# ============================================================
# deploy/archive_old.py — 归档旧模型 & 已导入的 xlsx
#
# 作用（每日流程第 4 步，run_all.sh 末尾调用）：
#   1) 模型：对每个 model_key，保留其前缀下最新 1 个 joblib 在 models/，
#      其余（旧版本）移入 archive/models/。固定名（xgb_v7.joblib）不挪。
#   2) 上传：把 upload/ 下所有已导入过的 xlsx（按 manifest.json 的
#      disclosure/actual 键里的日期判断）移入 archive/upload/。
#
# 纯 shell mv（跨同盘快速）；日期目录避免文件重名。已存在于目标处则跳过。
# 只归档、不删除——archive/ 请自行定期手动清理。
#
# 用法：  python archive_old.py [--dry-run]   # --dry-run 只打印不移动
# ============================================================
import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(_HERE)
while not os.path.exists(os.path.join(_ROOT, '_cfg.py')):
    _ROOT = os.path.dirname(_ROOT)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from _cfg import cfg

# 归档目录固定在部署根（archive/ 随部署走，不随脚本子目录）
ARCHIVE = os.path.join(_ROOT, 'archive')
ARCHIVE_MODELS = os.path.join(ARCHIVE, 'models')
ARCHIVE_UPLOAD = os.path.join(ARCHIVE, 'upload')


def _safe_move(src, dst, dry):
    """移动到目标；目标已存在则跳过（内容视为已归档）。返回移动结果描述。"""
    if not os.path.isfile(src):
        return f"  [跳过] 不存在: {os.path.basename(src)}"
    if os.path.exists(dst):
        return f"  [跳过] 已归档过: {os.path.basename(src)}"
    if dry:
        return f"  → {os.path.basename(src)}  (dry-run, 未移动)"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    return f"  完成 {os.path.basename(src)} → archive/"


def archive_models(dry):
    """按 key 前缀保留最新 1 份，旧版本移入 archive/models/<date>/。"""
    os.makedirs(ARCHIVE_MODELS, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d')
    for key, reg in cfg.MODEL_REGISTRY.items():
        paths = cfg.scan_models_by_key(key)
        if not paths:
            continue
        keep = paths[-1]                       # 最新 1 份留在 models/
        olds = paths[:-1]                      # 其余归档
        for p in olds:
            name = os.path.basename(p)
            print(_safe_move(p, os.path.join(ARCHIVE_MODELS, stamp, name), dry))
        if olds:
            print(f"  [{key}] 保留最新: {os.path.basename(keep)}，归档 {len(olds)} 份")


def _imported_dates():
    """从 manifest.json 读已导入日期集合（disclosure/actual 键里的 YYYY-MM-DD）。"""
    if not os.path.exists(cfg.MANIFEST_FILE):
        return set()
    try:
        man = json.load(open(cfg.MANIFEST_FILE))
    except Exception:
        return set()
    dates = set()
    for kind in ('disclosure', 'actual'):
        for key in man.get(kind, []):
            m = re.search(r'(\d{4}-\d{2}-\d{2})', str(key))
            if m:
                dates.add(m.group(1))
    return dates


def archive_uploads(dry):
    """把 upload/ 下已导入过的 xlsx 移入 archive/upload/（按文件名里的日期匹配 manifest）。"""
    if not os.path.isdir(cfg.UPLOAD_DIR):
        return
    imported = _imported_dates()
    os.makedirs(ARCHIVE_UPLOAD, exist_ok=True)
    for f in sorted(os.listdir(cfg.UPLOAD_DIR)):
        if not f.lower().endswith('.xlsx'):
            continue
        m = re.search(r'(\d{4}-\d{2}-\d{2})', f)
        if not m or m.group(1) not in imported:
            continue
        print(_safe_move(os.path.join(cfg.UPLOAD_DIR, f),
                         os.path.join(ARCHIVE_UPLOAD, f), dry))


def main():
    ap = argparse.ArgumentParser(description='归档旧模型 & 已导入 xlsx')
    ap.add_argument('--dry-run', action='store_true', help='只打印不移动')
    args = ap.parse_args()
    dry = args.dry_run

    print("=" * 56)
    print("  [归档] 旧模型 → archive/models/，已导入 xlsx → archive/upload/")
    if dry:
        print("  （--dry-run 模式，只打印不移动）")
    print("=" * 56)
    archive_models(dry)
    archive_uploads(dry)
    print("\n归档完成（archive/ 保留 1 份最新，请自行定期清理）。")


if __name__ == '__main__':
    main()
