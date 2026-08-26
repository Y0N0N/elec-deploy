#!/usr/bin/env python
# ============================================================
# deploy/code/common/0_repair_wide_dedup.py — 一次性宽表去重修复（内存安全版）
#
# 背景：披露宽表 data/披露预测数据.feather 中 2025-09-05 一天被膨胀 512 倍
#   （49152 行 = 512 组 × 96 个 15min 点，每组完全重复），导致宽表 5.4GB、
#   ingest 时 OOM。矩阵文件（事实来源）未污染。
#
# 内存安全设计：Date/Time 是表的最后两列（pandas MultiIndex 落表）。
#   → 用 pyarrow t.select(['Date','Time']) 只读两列（几十 MB）构建
#     (Date,Time) 去重掩码 → t.filter(掩码) 按行过滤全表。全程不 to_pandas
#     全量数据列，峰值内存 < 数百 MB。
#
# 通用逻辑：任何日期被重复多组都一并修掉，不只 2025-09-05。
#
# 用法：  python code/common/0_repair_wide_dedup.py [--no-backup]
#   产物：  data/披露预测数据.feather（去重后）+ 自动备份 .bak_<时间戳>
# ============================================================
import os
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(_HERE)
while not os.path.exists(os.path.join(_ROOT, '_cfg.py')):
    _ROOT = os.path.dirname(_ROOT)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pyf
from _cfg import cfg


def main():
    no_backup = '--no-backup' in sys.argv
    raw_path = cfg.DISCLOSURE_RAW
    if not os.path.exists(raw_path):
        print(f'警告: 宽表不存在: {raw_path}')
        return

    print(f'修复目标: {raw_path}')
    print(f'修复前大小: {os.path.getsize(raw_path)/1e6:.1f} MB')

    # 1) 只读 Date/Time 两列（几十 MB），构建去重掩码
    t = pyf.read_table(raw_path)
    names = t.column_names
    if 'Date' not in names or 'Time' not in names:
        print('警告: 未找到 Date/Time 列，中止（结构异常）。')
        return
    n_before = t.num_rows
    idx_tbl = t.select(['Date', 'Time'])
    dates = idx_tbl.column('Date').to_pandas().astype(str).values
    times = idx_tbl.column('Time').to_pandas().astype(str).values
    del idx_tbl, t

    idx = pd.MultiIndex.from_arrays([dates, times], names=['Date', 'Time'])
    dup_mask = idx.duplicated()
    n_dup = int(dup_mask.sum())
    del dates, times, idx
    print(f'重复行数: {n_dup} ({(n_dup/max(n_before,1))*100:.2f}%)')
    if n_dup == 0:
        print('无重复，无需修复。')
        return

    keep = ~dup_mask
    keep_arr = pa.array(keep)
    del dup_mask, keep

    # 2) 备份 + 重读全表 + 按掩码过滤
    if not no_backup:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bak = f'{raw_path}.bak_{stamp}'
        os.replace(raw_path, bak)
        print(f'已备份: {bak}')
        src = bak
    else:
        src = raw_path

    t = pyf.read_table(src)
    t2 = t.filter(keep_arr)
    del t
    n_after = t2.num_rows
    print(f'修复后: {n_after} 行 ({n_after//96} 天 × 96)')

    # 3) 写回
    pyf.write_feather(t2, raw_path)
    print(f'已写回: {raw_path} ({os.path.getsize(raw_path)/1e6:.1f} MB)')

    # 4) 校验
    t3 = pyf.read_table(raw_path)
    d = t3.column('Date').to_pandas().astype(str)
    per_day = pd.Series(d).value_counts()
    bad_days = per_day[per_day != 96]
    print(f'校验: {t3.num_rows} 行 × {t3.num_columns} 列 | 日期数 {len(per_day)}')
    if len(bad_days):
        print(f'警告: 仍有异常日期: {dict(bad_days.head())}')
    else:
        print('完成 全部日期均 96 行，修复完成。')


if __name__ == '__main__':
    main()
