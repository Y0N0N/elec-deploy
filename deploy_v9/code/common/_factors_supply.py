# ============================================================
# deploy/_factors_supply.py — 供给侧因子生成（内部模块，勿直接运行）
#
# 复刻 20_features_v4_label_factors.ipynb 的"供给侧因子"生成逻辑，
# 不依赖 scutquant/notebook，直接用 pandas 实现，已实测与既有 .fea 逐值一致。
#
# 生成:
#   s_{字段}_ma{7,14,21,28,35}   滚动均值 (rolling 窗口, 按 (date,hour) 位置)
#   s_{字段}_std{...}            滚动标准差
#   s_{字段}_roc{...}            data - data.shift(窗口)
#   is_peak / is_valley          时段标志 (8-20点峰值 / 0-6点低谷)
#
# 数据源: 披露矩阵 the configured disclosure matrix/*.feather (date×96 15min)
# 输出:   写入因子库 FACTOR_DIR/*.fea (date×24)
#
# 与 notebook 完全一致的关键语义:
#   - 15min → 小时: 每小时取 4 个 15min 的均值
#   - 字段短名: field.replace('预测 | ','').replace('(MW)','').strip().replace(' ','_')
#   - 滚动: 对 (date,hour) 多索引序列做位置滚动 (rolling(n, min_periods=1/2))
# ============================================================
import os, sys, gc
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(_HERE)
while not os.path.exists(os.path.join(_ROOT, '_cfg.py')):
    _ROOT = os.path.dirname(_ROOT)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from _cfg import cfg

WINDOWS = [7, 14, 21, 28, 35]

# 与 notebook 相同的供给侧关键字段（矩阵列名的短名）
KEY_SUPPLY_COLS = [
    '统调负荷', '光伏', '风电', '水电', '火电', '正备用', '负备用',
    '一次调频备用', '发电总出力', '预测出力', 'D日', 'D+1日', 'D+2日',
    '光伏出力预测', '风电出力预测', '水电（含抽蓄）总出力',
    '西电东送电力', '省内A类电源', '省内B类电源', '地方电源出力',
    '三峡', '海南送广东总加', '粤港联络线',
]


def short_name(field):
    """矩阵列名 → 短名（与 notebook 一致）"""
    return field.replace('预测 | ', '').replace('(MW)', '').strip().replace(' ', '_')


def resample_15min_to_hourly(df):
    """96列(15min) → 24列(小时)，每小时取 4 个 15min 均值（与 notebook 一致）"""
    out = pd.DataFrame(index=df.index)
    for h in range(24):
        cols = [f"{h:02d}:{m:02d}" for m in [0, 15, 30, 45]]
        av = [c for c in cols if c in df.columns]
        if av:
            out[f"{h:02d}:00"] = df[av].mean(axis=1)
    return out


def to_hourly_series(field_wide):
    """date×24 宽表 → (date,hour) 多索引序列（已排序）"""
    s = field_wide.stack()
    s.index = s.index.rename(['date', 'hour'])
    return s.sort_index()


def _ts_mean(s, n, min_periods=1):
    """与 scutquant ts_mean(matrix=True) 一致：位置滚动均值"""
    return s.rolling(n, min_periods=min_periods).mean()


def _ts_std(s, n, min_periods=2):
    return s.rolling(n, min_periods=min_periods).std()


def _ts_delta(s, n):
    """data - shift(n)（与 scutquant ts_delta 一致）"""
    return s - s.shift(n)


def build_supply_factors(tail_dates=None):
    """
    生成供给侧因子写入因子库。
    tail_dates=None → 全量重写; 否则 list[str] 只写这些日期（覆盖既有+新增）。
    返回生成的因子名列表。
    """
    os.makedirs(cfg.FACTOR_DIR, exist_ok=True)
    factors = {}      # name -> (date×24) DataFrame
    for name in sorted(os.listdir(cfg.DISCLOSURE_MATRIX)):
        if not name.endswith('.feather'):
            continue
        field = name[:-len('.feather')]
        sname = short_name(field)
        if sname not in KEY_SUPPLY_COLS:
            continue
        f = pd.read_feather(os.path.join(cfg.DISCLOSURE_MATRIX, name))
        if len(f.columns) == 96:
            f = resample_15min_to_hourly(f)
        elif len(f.columns) != 24:
            continue
        series = to_hourly_series(f)
        short = sname.replace('（含抽蓄）', '').replace('(', '_').replace(')', '')[:15]
        for w in WINDOWS:
            factors[f's_{short}_ma{w}']  = _ts_mean(series, w, 1).unstack()
            factors[f's_{short}_std{w}'] = _ts_std(series, w, 2).unstack()
            factors[f's_{short}_roc{w}'] = _ts_delta(series, w).unstack()

    # is_peak / is_valley（时段标志，与 notebook 一致）
    hours = pd.Index([f'{h:02d}:00' for h in range(24)])
    dates = None
    if factors:
        dates = next(iter(factors.values())).index
    if dates is not None:
        is_peak = pd.DataFrame(
            np.tile([1.0 if h in range(8, 21) else 0.0 for h in range(24)], (len(dates), 1)),
            index=dates, columns=hours)
        is_valley = pd.DataFrame(
            np.tile([1.0 if h in range(0, 7) else 0.0 for h in range(24)], (len(dates), 1)),
            index=dates, columns=hours)
        factors['is_peak'] = is_peak
        factors['is_valley'] = is_valley

    # 写入（全量或尾部）
    written = 0
    for fname, df in factors.items():
        df = df.copy()
        df.index = df.index.astype(str)
        df.columns = [str(c) for c in df.columns]
        path = os.path.join(cfg.FACTOR_DIR, f'{fname}.fea')
        if tail_dates is not None:
            # 尾部模式：合并——保留旧文件未在窗口内的行，覆盖窗口内行
            if os.path.exists(path):
                old = pd.read_feather(path)
                old.index = old.index.astype(str)
                keep = old.index[~old.index.isin(tail_dates)]
                new = pd.concat([old.loc[keep], df.loc[[d for d in tail_dates if d in df.index]]])
                new = new.sort_index()
                df = new
        df.to_feather(path)
        written += 1
    print(f"  供给侧因子: 生成 {len(factors)} 个, 写入 {written} 个"
          f" (tail={len(tail_dates) if tail_dates else '全量'}天)")
    return sorted(factors.keys())


if __name__ == '__main__':
    names = build_supply_factors()
    print(f"共生成 {len(names)} 个供给侧因子")
