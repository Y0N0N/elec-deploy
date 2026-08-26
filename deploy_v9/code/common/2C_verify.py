#!/usr/bin/env python
# ============================================================
# deploy/2C_verify.py — 步骤 2C：方向核对（独立于推理）
#
# 作用：导入新实际数据（1_ingest 更新实际矩阵）后，对所有历史预测 CSV
#   进行方向核对 + 套利结果刷新，并按核对进度用新命名回写：
#     预测_<model>_<start>_<end>_<flag>.csv    flag ∈ {a 未核对, b 核对中, c 核对完}
#   旧命名文件（预测_<start>_<end>.csv）核对后自动升级为新命名。
#
# 核对口径与 2B 推理时的 add_actual_verification 完全一致（共用 _verify.py）：
#   实际价差 = 日前统一结算价 − 实时统一结算价；方向按 ±τ 阈值感知；
#   套利时机 = 出手小时的交易腿（日前买/日前卖）；套利结果 = 按实际价差结算。
#
# 幂等：重复运行安全；已核对完（c）且无新实际数据的文件跳过。
#
# 用法：
#   python 2C_verify.py                 # 核对 output/ 下所有预测文件
#   python 2C_verify.py --model v9      # 只核对指定模型的文件
#   python 2C_verify.py --force         # 强制重写（忽略已核对完的跳过）
#   python 2C_verify.py --dry-run       # 只打印计划，不写文件
# ============================================================
import argparse
import os
import shutil
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(_HERE)
while not os.path.exists(os.path.join(_ROOT, '_cfg.py')):
    _ROOT = os.path.dirname(_ROOT)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from _cfg import cfg
from _verify import (ACTUAL_COLS, CHECK_C, CHECK_LABEL, check_flag_for_dates,
                     output_filename, parse_filename, verify_rows)

# 归档目录固定在部署根（archive/ 随部署走，不随脚本子目录）
ARCHIVE = os.path.join(_ROOT, 'archive')
ARCHIVE_CSV = os.path.join(ARCHIVE, 'csv')


def is_v9_columns(header):
    """CSV 是否 v9 方向信号格式（含『模型方向』『交易建议』）。"""
    cols = list(dict.fromkeys(header))
    return '模型方向' in cols and '交易建议' in cols


def process_file(path, model_filter=None, force=False, dry=False):
    """核对单个预测文件：重算核对列 → 计算标记 → 新命名回写。

    返回 (skip, renamed_to, flag, message)。"""
    fname = os.path.basename(path)
    meta = parse_filename(fname)
    if not meta:
        return (True, None, None, f'[跳过] 无法识别命名，跳过: {fname}')
    if model_filter and meta['model'] not in (None, model_filter):
        return (True, None, None, f'[跳过] 非指定模型({meta["model"]})，跳过')

    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except Exception as e:
        return (True, None, None, f'警告: 读取失败，跳过: {fname} ({e})')
    if df.empty or '日期' not in df.columns:
        return (True, None, None, f'警告: 空文件/缺日期列，跳过: {fname}')

    # 窗口内是否已有实际数据可核对（无新数据且已核对完 → 跳过，保持幂等）
    dates = sorted({str(d)[:10] for d in df['日期'].astype(str)})
    if not dates:
        return (True, None, None, f'警告: 无日期行，跳过: {fname}')

    flag = check_flag_for_dates(dates)
    if not force and flag == 'c' and not meta['legacy'] and meta['flag'] == 'c':
        return (True, None, None, f'[跳过] 已核对完，跳过: {fname}')

    is_v9 = is_v9_columns(df.columns)
    out = verify_rows(df, is_v9)

    # 新命名（核对标记随最新实际矩阵重算）。
    #   - 新命名文件 / 历史 v9 文件 → 预测_<model>_<start>_<end>_<flag>.csv
    #   - 历史 classic 文件（无模型标记，预警等级格式）→ 保留旧名，原地刷新核对列
    #   - legacy classic 且窗口已全部核对完（flag='c'）→ 一次性归档：
    #       核对后的内容写入 archive/csv/ 并从 output/ 移除，避免每次 2C 都重写、
    #       永不升级命名（BUG-3 修复）
    if meta['legacy'] and not is_v9:
        model, new_name = None, fname
        flag_lbl = '—'
        if flag == CHECK_C and not force:
            if dry:
                return (False, fname, flag,
                        f'→ (dry-run) {fname} 已核对完，将移入 archive/csv/')
            os.makedirs(ARCHIVE_CSV, exist_ok=True)
            out.to_csv(os.path.join(ARCHIVE_CSV, fname), index=False,
                       encoding='utf-8-sig')
            os.remove(path)
            return (False, fname, flag, f'完成 {fname} 已核对完，移入 archive/csv/')
    else:
        model = meta['model'] or ('v9' if is_v9 else cfg.ACTIVE_MODEL)
        new_name = output_filename(model, meta['start'], meta['end'], flag)
        flag_lbl = CHECK_LABEL[flag]
    if dry:
        return (False, new_name, flag,
                f'→ (dry-run) {fname} → {new_name} [{flag_lbl}]')

    new_path = os.path.join(os.path.dirname(path), new_name)
    out.to_csv(new_path, index=False, encoding='utf-8-sig')
    if new_path != os.path.abspath(path):
        shutil.move(path, os.path.join(ARCHIVE_CSV, fname))
        msg = (f'完成 {fname} → {new_name} [{flag_lbl}], 旧文件移入 archive/csv/')
    else:
        msg = f'完成 {fname} 已核对 [{flag_lbl}]'
    return (False, new_name, flag, msg)


def main():
    ap = argparse.ArgumentParser(description='方向核对（历史预测 vs 实际结算价）')
    ap.add_argument('--model', type=str, default=None, help='只核对指定模型的文件')
    ap.add_argument('--force', action='store_true', help='强制重写（跳过已核对完的跳过逻辑）')
    ap.add_argument('--dry-run', action='store_true', help='只打印计划，不写文件')
    args = ap.parse_args()

    out_dir = cfg.OUTPUT_DIR
    if not os.path.isdir(out_dir):
        print(f'警告: 输出目录不存在: {out_dir}')
        return
    files = sorted(f for f in os.listdir(out_dir) if f.endswith('.csv'))
    if not files:
        print('output/ 下没有预测文件，无需核对。')
        return

    os.makedirs(ARCHIVE_CSV, exist_ok=True)
    print('=' * 64)
    print('  方向核对 | 对历史预测 CSV 按最新实际矩阵刷新 (a未核对/b核对中/c核对完)')
    if args.dry_run:
        print('  （--dry-run 模式，只打印不写文件）')
    print('=' * 64)

    n_skip = n_done = 0
    for fname in files:
        skip, new_name, flag, msg = process_file(
            os.path.join(out_dir, fname), args.model, args.force, args.dry_run)
        print(msg)
        if skip:
            n_skip += 1
        else:
            n_done += 1

    print('-' * 64)
    if args.dry_run:
        print(f'计划核对 {n_done} 个文件（{n_skip} 个跳过）。')
    else:
        print(f'已核对 {n_done} 个文件（{n_skip} 个跳过）。')


if __name__ == '__main__':
    main()
