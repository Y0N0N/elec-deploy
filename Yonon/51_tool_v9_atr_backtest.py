#!/usr/bin/env python
# ============================================================
# Yonon/51_tool_v9_atr_backtest.py — v9.1 ATR 过滤器 + 硬风险规则 walk-forward 对比
#
# 同一批 fold 模型下对比三档 C 策略：
#   无过滤器:  出手 = 量级头触发(|mag|≥τ) 且 小时先验明确；方向 = 小时先验
#   +ATR:      上述出手 + 当日 atr_allow(D)（ratio×基线 + 绝对 ATR 下限，因果）
#   +ATR+硬停: 上述出手 + 月累计亏损硬停（当月累计 < max_monthly_loss 则当月剩余禁开仓）
#
# 用法：  venv/bin/python 51_tool_v9_atr_backtest.py \
#           [--train-days 90 --eval-days 30 --step-days 30]
#           [--n-hours 24 --baseline-days 20 --ratio 0.90 --abs-atr-floor 25.0]
#           [--max-monthly-loss -5000] [--out ...]
#   产物：  the local data directory/v9_atr_backtest.json
# ============================================================
import argparse
import importlib.util
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'deploy_v9'))

import numpy as np
import pandas as pd

from v9_atr import build_atr_gate_from_label, monthly_loss_hard_stop_mask

_spec = importlib.util.spec_from_file_location(
    'v9train', os.path.join(os.path.dirname(_HERE), 'deploy_v9',
                            '31_model_v9_direction_run.py'))
v9t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v9t)


def eval_strategy(act, as_, hours, vote, mag, allow_dates, dates,
                  max_monthly_loss=-5000.0, use_hard_stop=False):
    """C 策略评估，返回 (无过滤器, +ATR, +ATR+硬停) 三个 metrics dict。"""
    prior = np.array([vote.get(h, 0) for h in hours])
    trig = (mag >= v9t.T1) & (prior != 0)
    allow = np.array([allow_dates.get(str(d)[:10], False) for d in dates])
    trig_f = trig & allow

    def calc(sel):
        pred_dir = np.where(sel, prior, 0)
        pl = np.zeros(len(act))
        correct = (pred_dir != 0) & (as_ != 0) & (pred_dir == as_)
        wrong = (pred_dir != 0) & (as_ != 0) & (pred_dir != as_)
        pl[correct] = np.abs(act[correct])
        pl[wrong] = -np.abs(act[wrong])
        if use_hard_stop:
            exec_mask = monthly_loss_hard_stop_mask(sel, pl, dates, max_monthly_loss)
            sel = exec_mask
            pred_dir = np.where(sel, prior, 0)
            pl = np.zeros(len(act))
            correct = (pred_dir != 0) & (as_ != 0) & (pred_dir == as_)
            wrong = (pred_dir != 0) & (as_ != 0) & (pred_dir != as_)
            pl[correct] = np.abs(act[correct])
            pl[wrong] = -np.abs(act[wrong])
        cum = np.cumsum(pl)
        ssel = sel & (as_ != 0)
        hit = float(np.mean(pred_dir[ssel] == as_[ssel])) if ssel.sum() else float('nan')
        nwr = float(correct.sum() / (correct.sum() + wrong.sum())) if (correct.sum() + wrong.sum()) else float('nan')
        return dict(
            n_trigger=int(sel.sum()),
            trigger_rate=float(sel.mean()),
            dir_hit=hit,
            net_win_rate=nwr,
            n_correct=int(correct.sum()), n_wrong=int(wrong.sum()),
            pnl_total=float(pl.sum()),
            max_drawdown=float((np.maximum.accumulate(cum) - cum).max()),
        )
    return calc(trig), calc(trig_f), calc(trig_f)


def agg(rows, key):
    vals = [r[key] for r in rows if r is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return float(np.mean(vals)) if vals else float('nan')


def pack(rows, tag):
    return {
        'tag': tag,
        'n_folds': len(rows),
        'trigger_rate': agg(rows, 'trigger_rate'),
        'dir_hit': agg(rows, 'dir_hit'),
        'net_win_rate': agg(rows, 'net_win_rate'),
        'pnl_total': float(sum(r['pnl_total'] for r in rows)),
        'pnl_per_trade': float(sum(r['pnl_total'] for r in rows)
                               / max(sum(r['n_correct'] + r['n_wrong'] for r in rows), 1)),
        'max_drawdown_max': float(max(r['max_drawdown'] for r in rows)),
        'hit_min': float(min(r['dir_hit'] for r in rows)),
        'hit_max': float(max(r['dir_hit'] for r in rows)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-days', type=int, default=90)
    ap.add_argument('--eval-days', type=int, default=30)
    ap.add_argument('--step-days', type=int, default=30)
    ap.add_argument('--n-hours', type=int, default=24, help='ATR 窗口小时数')
    ap.add_argument('--baseline-days', type=int, default=20, help='ATR 基线天数')
    ap.add_argument('--ratio', type=float, default=0.90, help='ATR 相对阈值比例')
    ap.add_argument('--abs-atr-floor', type=float, default=25.0,
                    help='绝对 ATR 下限（None 关闭）')
    ap.add_argument('--max-monthly-loss', type=float, default=-5000.0,
                    help='月累计亏损硬停阈值（元/MWh）')
    ap.add_argument('--out', default=os.path.join(_HERE, 'data', 'v9_atr_backtest.json'))
    args = ap.parse_args()

    print("=" * 78)
    print(f"  v9.1 ATR 过滤器 + 硬风险规则 | ATR_{args.n_hours}h/{args.baseline_days}d "
          f"r{args.ratio:.2f} abs_floor={args.abs_atr_floor} | 月硬停 {args.max_monthly_loss}")
    print(f"  walk-forward: 训练期 {args.train_days}天 / 评估 {args.eval_days}天 / 步长 {args.step_days}天")
    print("=" * 78)

    allow_dates = build_atr_gate_from_label(
        os.path.join(v9t.cfg.LABEL_DIR, 'spread_label.feather'),
        n_hours=args.n_hours, baseline_days=args.baseline_days, ratio=args.ratio,
        abs_atr_floor=args.abs_atr_floor)
    print(f"ATR allow: 允许 {allow_dates.sum()}/{len(allow_dates)} 天 "
          f"({allow_dates.mean()*100:.0f}%)")

    feats = v9t.build_feature_list()
    X = v9t.load_X(feats)
    y = v9t.load_y()
    common = X.index.intersection(y.index)
    Xc, yc = X.loc[common], y.loc[common]
    all_dates = sorted(set(Xc.index.get_level_values('date')))
    print(f"特征 {len(feats)} | 数据 {all_dates[0]} ~ {all_dates[-1]} ({len(all_dates)} 天)")

    rows = {'无过滤器': [], 'ATR': [], 'ATR+硬停': []}
    s = 0
    fold_idx = 0
    while True:
        tr_end = s + args.train_days
        ev_end = s + args.train_days + args.eval_days
        if ev_end > len(all_dates):
            break
        tr_dates = set(all_dates[s:tr_end])
        ev_dates = set(all_dates[tr_end:ev_end])
        m_tr = Xc.index.get_level_values('date').isin(tr_dates)
        m_ev = Xc.index.get_level_values('date').isin(ev_dates)
        X_tr, y_tr = Xc.loc[m_tr], yc.loc[m_tr]
        X_ev, y_ev = Xc.loc[m_ev], yc.loc[m_ev]
        _, vote = v9t.build_hour_prior(y_tr)
        clf = v9t.train_dir_head(X_tr, y_tr, X_ev, y_ev)
        reg = v9t.train_mag_head(X_tr, y_tr)

        act = np.asarray(y_ev.values, dtype=float)
        hours = np.asarray([h[:2] for h in X_ev.index.get_level_values('hour')])
        mag = np.expm1(np.asarray(reg.predict(X_ev), dtype=float))
        as_ = np.where(act < -v9t.T1, -1, np.where(act > v9t.T1, 1, 0))
        dates = list(y_ev.index.get_level_values('date'))

        b, a, h = eval_strategy(act, as_, hours, vote, mag, allow_dates, dates,
                                args.max_monthly_loss, use_hard_stop=True)
        for name, m in (('无过滤器', b), ('ATR', a), ('ATR+硬停', h)):
            m['fold'] = fold_idx
            m['eval'] = [all_dates[tr_end], all_dates[ev_end - 1]]
            rows[name].append(m)
        print(f"  f{fold_idx}: 评估[{b['eval'][0]}~{b['eval'][1]}] "
              f"无过滤 触发率{b['trigger_rate']:.3f} 命中{b['dir_hit']:.3f} 净胜率{b['net_win_rate']:.3f} "
              f"P&L{b['pnl_total']:+.0f} | "
              f"ATR 触发率{a['trigger_rate']:.3f} 命中{a['dir_hit']:.3f} 净胜率{a['net_win_rate']:.3f} "
              f"P&L{a['pnl_total']:+.0f} 回撤{a['max_drawdown']:.0f} | "
              f"+硬停 P&L{h['pnl_total']:+.0f} 回撤{h['max_drawdown']:.0f}")
        s += args.step_days
        fold_idx += 1

    results = [pack(rows[k], k) for k in ['无过滤器', 'ATR', 'ATR+硬停']]
    print("\n===== 汇总（同批 fold 模型）=====")
    for p in results:
        print(f"  {p['tag']:10s}: 触发率 {p['trigger_rate']:.3f} | 命中 {p['dir_hit']:.3f} "
              f"(min {p['hit_min']:.3f}/max {p['hit_max']:.3f}) | 净胜率 {p['net_win_rate']:.3f} "
              f"| P&L {p['pnl_total']:+.0f} | 单笔 {p['pnl_per_trade']:+.1f} | "
              f"最大回撤 {p['max_drawdown_max']:.0f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'params': vars(args), 'summary': results,
               'folds': {k: rows[k] for k in rows},
               'atr_allow': {str(d)[:10]: bool(v) for d, v in allow_dates.items()}},
              open(args.out, 'w'), ensure_ascii=False, indent=2)
    print(f"结果已写 {args.out}")


if __name__ == '__main__':
    main()
