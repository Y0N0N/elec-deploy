#!/usr/bin/env python
# ============================================================
# deploy/v9_add_features.py — v9 新增特征构建（幂等，写入因子库 qlib158）
#
# v9 相对 v8.1 新增两类特征（交接文档 §4.1 step 4）：
#   1) 小时 one-hot: hour_00 ~ hour_23 （24 个二进制特征，比 h_hour_sin/cos、hour_id
#      更强的「小时身份」编码，喂给方向头学习时段性偏置）
#   2) 近期实际价差 regime: sp_regime_mean7 / sp_regime_abs7
#      （过去 7 天实际价差的均值 / |价差| 均值，按小时；因果构造 shift(1)，
#       把「白天偏正、晚间偏负」的近期 regime 喂给模型）
#
# 数据红线：只用 deploy 已导入的旧数据（spread_label 到 2026-07-21），
#   不使用 新数据/ 目录的任何 8 月 xlsx。
# 幂等：可反复运行，覆盖写入同名 .fea。
# ============================================================
import os
import sys
import warnings

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(_HERE)
while not os.path.exists(os.path.join(_ROOT, '_cfg.py')):
    _ROOT = os.path.dirname(_ROOT)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from _cfg import cfg

import numpy as np
import pandas as pd


def build_hour_one_hot(cal_df):
    """在因子库日期日历上构建 24 个小时 one-hot 宽表（date×24）。

    cal_df: 已有 .fea（如 is_peak）作为日历/列名模板。每行 date：
      列 hh:00 = 1.0，其余列 = 0.0。堆叠后 (date, hh:00) 位置为该小时身份。
    """
    dates = list(cal_df.index)
    cols = [str(c) for c in cal_df.columns]          # '00:00' ... '23:00'
    assert len(cols) == 24 and cols[0] == '00:00', f"列名异常: {cols[:3]}..."
    hour_pos = {c: i for i, c in enumerate(cols)}
    for h_i in range(24):
        name = f'hour_{h_i:02d}'
        df = pd.DataFrame(0.0, index=dates, columns=cols, dtype=float)
        df[cols[h_i]] = 1.0
        df.index.name = 'date'
        _write_fea(name, df)
    print(f"  小时 one-hot: hour_00 ~ hour_23 已写入（日历 {dates[0]} ~ {dates[-1]}，"
          f"{len(dates)} 天）")


def build_regime(sp, cal_df):
    """近期实际价差 regime：过去 7 天实际价差的均值 / |价差| 均值（按小时）。

    sp: spread_label 宽表（date×24，时间戳索引）。因果构造：
       shift(1) 后 rolling(7) → 只用 D-7..D-1 的实际价差，不含当日（防泄漏）。
    输出对齐到因子库日历 cal_df 的日期（超出实际结果的日期 → NaN，XGBoost 原生处理）。
    """
    sp = sp.sort_index()
    mean7 = sp.shift(1).rolling(7, min_periods=1).mean()
    abs7 = sp.shift(1).abs().rolling(7, min_periods=1).mean()
    # 对齐到因子库日历（日期字符串，统一 date-only，避免 datetime64[ns] 的 00:00:00 差异）
    dates = [str(pd.Timestamp(d).date()) for d in cal_df.index]
    for name, df in [('sp_regime_mean7', mean7), ('sp_regime_abs7', abs7)]:
        df = df.copy()
        df.index = [str(pd.Timestamp(d).date()) for d in df.index]
        df = df.reindex(dates)
        df.columns = [str(c) for c in df.columns]
        _write_fea(name, df)
    print(f"  regime 特征: sp_regime_mean7 / sp_regime_abs7 已写入 "
          f"（实际价差至 {max(sp.index):%Y-%m-%d}，其后为 NaN）")


def _write_fea(name, df):
    df.columns = [str(c) for c in df.columns]
    out = os.path.join(cfg.FACTOR_DIR, f'{name}.fea')
    df.to_feather(out)
    return out


def main():
    os.makedirs(cfg.FACTOR_DIR, exist_ok=True)
    # 日历模板：is_peak（因子库既有，覆盖全披露因子日期）
    cal_path = os.path.join(cfg.FACTOR_DIR, 'is_peak.fea')
    if not os.path.exists(cal_path):
        raise FileNotFoundError(f"缺日历模板 {cal_path}，先跑 2A_rebuild 重建因子")
    cal = pd.read_feather(cal_path)
    cal.index = pd.to_datetime(cal.index)

    print("=" * 60)
    print("  v9 新增特征构建")
    print("=" * 60)
    build_hour_one_hot(cal)

    sp_path = os.path.join(cfg.LABEL_DIR, 'spread_label.feather')
    sp = pd.read_feather(sp_path)
    sp.index = pd.to_datetime(sp.index)
    sp = sp[~sp.index.duplicated(keep='last')]
    build_regime(sp, cal)

    print("\n完成 v9 新增特征完成（26 个 .fea 已写入因子库）")


if __name__ == '__main__':
    main()
