#!/usr/bin/env python
# ============================================================
# Yonon/51_tool_v9_backtest.py — v9 C 策略历史回测（walk-forward 40/60/80 天）
#
# 交接文档 §5 step 5：历史 valid 段跑 §3 的 C 策略，算出净胜率、盈亏、最大回撤。
#   策略：出手 = 量级头触发(|mag|≥τ) 且 小时先验明确；方向 = 小时先验。
#   P&L 口径：正确 +|实际价差|，错 −|实际价差|，实际中性 0（单小时一单位电量）。
#
# 用法：  venv/bin/python 51_tool_v9_backtest.py [--valid-days 60 60 60 ...]
#   输出： 控制台摘要 + the local data directory/v9_backtest.json
# ============================================================
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(_HERE)
while not os.path.exists(os.path.join(_ROOT, '_cfg.py')):
    _ROOT = os.path.dirname(_ROOT)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, 'code', 'v9')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

import importlib.util
_spec = importlib.util.spec_from_file_location(
    'v9train', os.path.join(_ROOT, 'code', 'v9', '31_model_v9_direction_run.py'))
v9t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v9t)   # 复用 build_feature_list/load_X/load_y/训练/先验


def backtest_one(vd):
    """对指定 valid 天数训练 v9 并跑 C 策略回测。"""
    feats = v9t.build_feature_list()
    X = v9t.load_X(feats)
    y = v9t.load_y()
    common = X.index.intersection(y.index)
    Xc, yc = X.loc[common], y.loc[common]
    all_dates = sorted(set(Xc.index.get_level_values('date')))
    tr_d = set(all_dates[: len(all_dates) - vd])
    va_d = set(all_dates[len(all_dates) - vd:])
    tr_m = Xc.index.get_level_values('date').isin(tr_d)
    va_m = Xc.index.get_level_values('date').isin(va_d)
    X_tr, y_tr = Xc.loc[tr_m], yc.loc[tr_m]
    X_va, y_va = Xc.loc[va_m], yc.loc[va_m]

    frac, vote = v9t.build_hour_prior(y_tr)
    clf = v9t.train_dir_head(X_tr, y_tr, X_va, y_va)
    reg = v9t.train_mag_head(X_tr, y_tr)

    act = np.asarray(y_va.values, dtype=float)
    hours = np.asarray([h[:2] for h in X_va.index.get_level_values('hour')])
    dates = np.asarray(X_va.index.get_level_values('date'))
    mag = np.expm1(np.asarray(reg.predict(X_va), dtype=float))
    prior = np.array([vote.get(h, 0) for h in hours])
    as_ = np.where(act < -v9t.T1, -1, np.where(act > v9t.T1, 1, 0))
    trig = (mag >= v9t.T1) & (prior != 0)
    pred_dir = np.where(trig, prior, 0)

    n = len(act)
    pl = np.zeros(n)
    correct = (pred_dir != 0) & (as_ != 0) & (pred_dir == as_)
    wrong = (pred_dir != 0) & (as_ != 0) & (pred_dir != as_)
    pl[correct] = np.abs(act[correct])
    pl[wrong] = -np.abs(act[wrong])
    cum = np.cumsum(pl)
    daily = pd.DataFrame({'date': dates, 'pl': pl}).groupby('date')['pl'].sum()

    # 按出手样本命中
    sel = trig & (as_ != 0)
    hit = float(np.mean(pred_dir[sel] == as_[sel])) if sel.sum() else float('nan')
    res = {
        'valid_days': vd,
        'window': [all_dates[-vd], all_dates[-1]],
        'n_hours': int(n),
        'n_trigger': int(trig.sum()),
        'trigger_rate': float(trig.mean()),
        'dir_hit': hit,
        'net_win_rate': float(correct.sum() / (correct.sum() + wrong.sum())) if (correct.sum() + wrong.sum()) else float('nan'),
        'n_correct': int(correct.sum()), 'n_wrong': int(wrong.sum()),
        'pnl_total': float(pl.sum()),
        'pnl_per_trade': float(pl.sum() / (correct.sum() + wrong.sum())) if (correct.sum() + wrong.sum()) else float('nan'),
        'max_drawdown': float((np.maximum.accumulate(cum) - cum).max()),
        'best_day': float(daily.max()), 'worst_day': float(daily.min()),
        'avg_daily': float(daily.mean()),
        'n_days': int(len(daily)),
    }
    print(f"  valid={vd}天 [{res['window'][0]}~{res['window'][1]}]: 触发率 {res['trigger_rate']:.3f} | "
          f"方向命中 {res['dir_hit']:.3f} | 净胜率 {res['net_win_rate']:.3f} "
          f"({res['n_correct']}对/{res['n_wrong']}错) | P&L {res['pnl_total']:+.0f} "
          f"| 单笔 {res['pnl_per_trade']:+.1f} | 最大回撤 {res['max_drawdown']:.0f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--valid-days', type=int, nargs='+', default=[40, 60, 80])
    ap.add_argument('--out', default=os.path.join(_ROOT, 'data', 'v9_backtest.json'))
    args = ap.parse_args()

    print("=" * 72)
    print("  v9 C 策略回测（walk-forward，valid 段模型没见过）")
    print("=" * 72)
    results = [backtest_one(vd) for vd in args.valid_days]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'strategy': 'C(量级触发+小时先验方向)', 'results': results},
              open(args.out, 'w'), ensure_ascii=False, indent=2)
    print(f"\n结果已写 {args.out}")


if __name__ == '__main__':
    main()
