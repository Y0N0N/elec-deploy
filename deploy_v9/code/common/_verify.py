#!/usr/bin/env python
# ============================================================
# deploy/_verify.py — 方向核对 / 套利结果 共享逻辑
#
# 供 2B_inference.py（推理时补核对列 + 带标记命名）与 2C_verify.py
# （每日导入新实际数据后对 a/b 预测重新核对）共用，避免两处口径不一。
#
# 核对口径（§2.1）：价差 = 日前统一结算价 − 实时统一结算价（元/MWh）。
#   方向核对 = 交易方向（规则层小时先验 / 预测符号） vs 实际方向（±τ 阈值感知）
#   套利时机 = 出手小时的交易腿：日前买（押负偏差，DA<RT）/ 日前卖（押正偏差）
#   套利结果 = 按实际价差结算：盈利 / 亏损 / 中性 / 错过 / 未出手 / 待实际
#   套利盈亏 = 出手时 (日前卖?+1 : 日前买?-1) × 实际价差；未出手=0；待实际=NaN
#
# 核对标记（文件名尾缀）：
#   a = 未开始核对（窗口内无任何一天有实际结算价）
#   b = 核对中（部分日期已有实际价）
#   c = 核对完（窗口内所有日期均已核对）
# ============================================================
import os
import re
import sys

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

CHECK_A = 'a'   # 未开始核对
CHECK_B = 'b'   # 核对中
CHECK_C = 'c'   # 核对完
CHECK_LABEL = {CHECK_A: '未核对', CHECK_B: '核对中', CHECK_C: '已核对'}

ACTUAL_COLS = ['实际价差(元/MWh)', '方向核对', '套利时机', '套利结果', '套利盈亏(元/MWh)']


# ════════════════════════════════════════════════════════════
# 实际结算价
# ════════════════════════════════════════════════════════════
def actual_spread_series():
    """实际价差长表 Series[(date,hour)] = 日前统一结算价 − 实时统一结算价。

    读实际矩阵两个结算价 feather 堆叠而成；失败返回 None（不阻塞核对）。"""
    try:
        da = pd.read_feather(os.path.join(cfg.ACTUAL_MATRIX, '日前统一结算价.feather'))
        rt = pd.read_feather(os.path.join(cfg.ACTUAL_MATRIX, '实时统一结算价.feather'))
        da.index = pd.to_datetime(da.index).strftime('%Y-%m-%d')
        rt.index = pd.to_datetime(rt.index).strftime('%Y-%m-%d')
        da.columns = [str(c) for c in da.columns]
        rt.columns = [str(c) for c in rt.columns]
        actual = (da - rt).stack()
        actual.index = actual.index.rename(['date', 'hour'])
        return actual
    except Exception as e:
        print(f"  警告: 加载实际结算价失败，跳过核对: {e}")
        return None


def actual_dates():
    """已具备实际结算价的日期集合（核对标记计算用）。"""
    s = actual_spread_series()
    if s is None:
        return set()
    return set(s.index.get_level_values('date'))


def norm_date(d):
    """日期归一化 'YYYY-MM-DD'（截掉可能的时间后缀）。"""
    return str(d)[:10]


def norm_hour(h):
    """小时归一化 'HH:MM'（兼容 '0:00' / 纯数字 8 / 8.0）。"""
    s = str(h).strip()
    if ':' in s:
        hh, _, rest = s.partition(':')
        mm = rest[:2] if rest[:2].isdigit() else '00'
        try:
            return f'{int(float(hh)):02d}:{mm}'
        except (ValueError, TypeError):
            return '00:00'
    try:
        return f'{int(float(s)):02d}:00'
    except (ValueError, TypeError):
        return '00:00'


# ════════════════════════════════════════════════════════════
# 核对标记（a/b/c）与文件命名
# ════════════════════════════════════════════════════════════
def check_flag_for_dates(dates, actual=None):
    """按窗口日期内实际价覆盖情况给出核对标记。

    a=无任何实际（未开始）；c=全部有实际（核对完）；否则 b（核对中）。"""
    if actual is None:
        actual = actual_dates()
    dlist = [norm_date(d) for d in dates if d is not None and str(d).strip()]
    if not dlist:
        return CHECK_A
    covered = sum(1 for d in dlist if d in actual)
    if covered == 0:
        return CHECK_A
    if covered == len(dlist):
        return CHECK_C
    return CHECK_B


def output_filename(model_key, start, end, flag):
    """新命名：预测_<model>_<start>_<end>_<flag>.csv（无 model 时省略模型段）。"""
    model = f'{model_key}_' if model_key else ''
    return f"预测_{model}{start}_{end}_{flag}.csv"


def parse_filename(fname):
    """解析输出文件名 → dict(model, start, end, flag, legacy)。

    支持两种形态：
      新: 预测_<model>_<start>_<end>_<flag>.csv  (flag∈{a,b,c})
      旧: 预测_<start>_<end>.csv                (无模型 / 无核对标记)
    无法识别返回 None。"""
    base = fname[:-4] if fname.lower().endswith('.csv') else fname
    parts = base.split('_')
    if not parts or parts[0] != '预测':
        return None
    date_idx = [i for i, p in enumerate(parts)
                if re.fullmatch(r'\d{4}-\d{2}-\d{2}', p)]
    if len(date_idx) < 2:
        return None
    start_i, end_i = date_idx[0], date_idx[-1]
    start, end = parts[start_i], parts[end_i]
    flag, model, legacy = CHECK_A, None, True
    last = parts[-1]
    if last in (CHECK_A, CHECK_B, CHECK_C):
        flag = last
        legacy = False
        model = '_'.join(parts[1:start_i]) or None   # 日期前的段 = 模型
    return {'model': model, 'start': start, 'end': end,
            'flag': flag, 'legacy': legacy, 'base': base}


# ════════════════════════════════════════════════════════════
# 逐行核对 / 套利
# ════════════════════════════════════════════════════════════
def verify_rows(df, is_v9):
    """补全/刷新 核对相关列（幂等，按最新实际矩阵重算，覆盖旧值）。

    实际价差 / 方向核对 / 套利时机 / 套利结果 / 套利盈亏 五列。
    is_v9=True 时交易方向取自『交易建议』（规则层小时先验），
    否则（classic）取『预测价差』符号。
    """
    out = df.copy()
    for c in ACTUAL_COLS:
        if c not in out.columns:
            out[c] = np.nan if c == '实际价差(元/MWh)' else ''

    actual = actual_spread_series()
    t1 = cfg.SPREAD_THRESHOLD
    actual_map = {}
    if actual is not None:
        for (d, h), v in actual.items():
            actual_map[(norm_date(d), norm_hour(h))] = float(v)

    # 交易建议 / 是否出手（缺失列自动补齐为全空，避免 classic/残缺 CSV 报错）
    def _col(name, default):
        if name in out.columns:
            return out[name]
        return pd.Series(default, index=out.index)

    if is_v9:
        advice = _col('交易建议', '').fillna('').astype(str)
        acted = _col('是否出手', '').fillna('').astype(str).eq('是')
    else:
        pred = pd.to_numeric(_col('预测价差(元/MWh)', np.nan), errors='coerce')
        advice = pd.Series('', index=out.index)
        advice[pred < 0] = '日前买/实时卖'
        advice[pred > 0] = '日前卖/实时买'
        acted = advice.ne('')

    dates = out['日期'].map(norm_date)
    hours = out['小时'].map(norm_hour)

    for i in range(len(out)):
        d, h = dates.iloc[i], hours.iloc[i]
        a = actual_map.get((d, h), np.nan)
        adv = str(advice.iloc[i])
        is_act = bool(acted.iloc[i])
        tdir = 1 if adv.startswith('日前卖') else (-1 if adv.startswith('日前买') else 0)

        out.iat[i, out.columns.get_loc('套利时机')] = (
            '日前买' if tdir < 0 else ('日前卖' if tdir > 0 else '观望'))
        out.iat[i, out.columns.get_loc('实际价差(元/MWh)')] = (
            round(a, 1) if not np.isnan(a) else np.nan)

        if np.isnan(a):
            out.iat[i, out.columns.get_loc('方向核对')] = '待实际'
            out.iat[i, out.columns.get_loc('套利结果')] = '待实际'
            out.iat[i, out.columns.get_loc('套利盈亏(元/MWh)')] = np.nan
            continue

        # 方向核对（口径与 2B 原实现一致）
        if not is_act:
            ck = '未出手'
        elif a > t1:
            ck = '错·实际正' if tdir < 0 else '对·实际正'
        elif a < -t1:
            ck = '对·实际负' if tdir < 0 else '错·实际负'
        else:
            ck = '实际中性'
        out.iat[i, out.columns.get_loc('方向核对')] = ck

        # 套利结果 / 套利盈亏
        if ck == '未出手':
            if abs(a) >= t1:
                out.iat[i, out.columns.get_loc('套利结果')] = \
                    f'错过 {"-" if a < 0 else "+"}{abs(a):.1f}'
            else:
                out.iat[i, out.columns.get_loc('套利结果')] = '未出手'
            out.iat[i, out.columns.get_loc('套利盈亏(元/MWh)')] = 0.0
        elif ck == '实际中性':
            out.iat[i, out.columns.get_loc('套利结果')] = f'中性 {a:+.1f}'
            out.iat[i, out.columns.get_loc('套利盈亏(元/MWh)')] = round(tdir * a, 1)
        elif ck.startswith('对'):
            out.iat[i, out.columns.get_loc('套利结果')] = f'盈利 +{abs(a):.1f}'
            out.iat[i, out.columns.get_loc('套利盈亏(元/MWh)')] = round(tdir * a, 1)
        elif ck.startswith('错'):
            out.iat[i, out.columns.get_loc('套利结果')] = f'亏损 -{abs(a):.1f}'
            out.iat[i, out.columns.get_loc('套利盈亏(元/MWh)')] = round(tdir * a, 1)
        else:
            out.iat[i, out.columns.get_loc('套利结果')] = ck
            out.iat[i, out.columns.get_loc('套利盈亏(元/MWh)')] = np.nan
    return out
