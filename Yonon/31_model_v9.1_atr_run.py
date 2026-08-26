#!/usr/bin/env python
# ============================================================
# Yonon/31_model_v9.1_atr_run.py — v9.1 训练：v9 方向信号模型 + ATR 波动率过滤器
#
# 相对 v9 新增 (2026-08-19 用户拍板，不引入硬停):
#   规则层加 ATR 波动率过滤器（构建策略自包含进 joblib）:
#     allow(D) = ATR_24(D-1) ≥ ratio × 20日基线(D-1)  且  ATR_24(D-1) ≥ abs_atr_floor
#   出手条件 = 量级头触发(|mag|≥τ) 且 小时先验明确 且 当日 ATR allow
#
# 部署说明（供 deploy 实装）: 模型自含 `atr_filter` 配置；推理时对预测日 D，
#   用 ≤D-1 的实际价差重算 ATR（见 v9_atr.py），不满足则不出手。
#
# 用法：  venv/bin/python 31_model_v9.1_atr_run.py [--valid-days 60]
#   产物：  the local models directory/xgb_v9.1_<ts>.joblib（自含 atr_filter，供拷入 deploy）
# ============================================================
import argparse
import importlib.util
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'deploy_v9'))

import numpy as np
import pandas as pd
import joblib

from v9_atr import build_atr_gate_from_label

# 复用 deploy_v9 的训练模块（方向头/量级头/数据加载/小时先验）
_spec = importlib.util.spec_from_file_location(
    'v9train', os.path.join(os.path.dirname(_HERE), 'deploy_v9',
                            '31_model_v9_direction_run.py'))
v9t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v9t)

# ── v9.1 ATR 过滤器参数（2026-08-19 拍板）──
ATR = dict(enabled=True, n_hours=24, baseline_days=20, ratio=0.90, abs_atr_floor=25.0)


def eval_c(reg, X_ev, y_ev, hour_vote, allow_series=None):
    """C 策略评估；allow_series 为 None 时不加 ATR 门，否则叠加。"""
    act = np.asarray(y_ev.values, dtype=float)
    hours = np.asarray([h[:2] for h in X_ev.index.get_level_values('hour')])
    mag = np.expm1(np.asarray(reg.predict(X_ev), dtype=float))
    prior = np.array([hour_vote.get(h, 0) for h in hours])
    as_ = np.where(act < -v9t.T1, -1, np.where(act > v9t.T1, 1, 0))
    trig = (mag >= v9t.T1) & (prior != 0)
    if allow_series is not None:
        dates = [str(d)[:10] for d in y_ev.index.get_level_values('date')]
        allow = np.array([allow_series.get(d, False) for d in dates])
        trig = trig & allow
    pred_dir = np.where(trig, prior, 0)
    pl = np.zeros(len(act))
    correct = (pred_dir != 0) & (as_ != 0) & (pred_dir == as_)
    wrong = (pred_dir != 0) & (as_ != 0) & (pred_dir != as_)
    pl[correct] = np.abs(act[correct])
    pl[wrong] = -np.abs(act[wrong])
    cum = np.cumsum(pl)
    sel = trig & (as_ != 0)
    return dict(
        n_trigger=int(trig.sum()), trigger_rate=float(trig.mean()),
        dir_hit=float(np.mean(pred_dir[sel] == as_[sel])) if sel.sum() else float('nan'),
        net_win_rate=float(correct.sum() / (correct.sum() + wrong.sum())) if (correct.sum() + wrong.sum()) else float('nan'),
        n_correct=int(correct.sum()), n_wrong=int(wrong.sum()),
        pnl_total=float(pl.sum()),
        max_drawdown=float((np.maximum.accumulate(cum) - cum).max()),
        trig_ge50=float(np.mean(np.abs(act[trig]) >= v9t.T1)) if trig.sum() else float('nan'),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--valid-days', type=int, default=60)
    ap.add_argument('--out-dir', default=os.path.join(_HERE, 'models'))
    args = ap.parse_args()

    print("=" * 72)
    print(f"  v9.1 训练 = v9 方向信号模型 + ATR 波动率过滤器 | valid={args.valid_days}天")
    print(f"  ATR: n_hours={ATR['n_hours']} baseline={ATR['baseline_days']}d "
          f"ratio={ATR['ratio']} abs_floor={ATR['abs_atr_floor']}")
    print("=" * 72)

    feats = v9t.build_feature_list()
    X = v9t.load_X(feats)
    y = v9t.load_y()
    common = X.index.intersection(y.index)
    Xc, yc = X.loc[common], y.loc[common]
    all_dates = sorted(set(Xc.index.get_level_values('date')))
    vd = args.valid_days
    tr_dates = set(all_dates[: len(all_dates) - vd])
    va_dates = set(all_dates[len(all_dates) - vd:])
    m_tr = Xc.index.get_level_values('date').isin(tr_dates)
    m_va = Xc.index.get_level_values('date').isin(va_dates)
    X_tr, y_tr = Xc.loc[m_tr], yc.loc[m_tr]
    X_va, y_va = Xc.loc[m_va], yc.loc[m_va]
    print(f"train {all_dates[0]} ~ {all_dates[-(vd+1)]} | valid {all_dates[-vd]} ~ {all_dates[-1]}")

    frac, vote = v9t.build_hour_prior(y_tr)
    print("\n[训练 方向头]")
    clf = v9t.train_dir_head(X_tr, y_tr, X_va, y_va)
    print("\n[训练 量级头]")
    reg = v9t.train_mag_head(X_tr, y_tr)

    # ATR 过滤器（因果，per-date allow）
    label_path = os.path.join(v9t.cfg.LABEL_DIR, 'spread_label.feather')
    allow = build_atr_gate_from_label(label_path, n_hours=ATR['n_hours'],
                                      baseline_days=ATR['baseline_days'],
                                      ratio=ATR['ratio'],
                                      abs_atr_floor=ATR['abs_atr_floor'])
    print(f"\nATR allow: 允许 {allow.sum()}/{len(allow)} 天 ({allow.mean()*100:.0f}%)")

    m_base = eval_c(reg, X_va, y_va, vote)
    m_atr = eval_c(reg, X_va, y_va, vote, allow)
    print("\n===== valid 段评估（模型没见过）=====")
    for tag, m in [('C 策略（无 ATR）', m_base), ('C 策略 + ATR 过滤器', m_atr)]:
        print(f"{tag}: 触发率 {m['trigger_rate']:.3f} | 方向命中 {m['dir_hit']:.3f} | "
              f"净胜率 {m['net_win_rate']:.3f} ({m['n_correct']}对/{m['n_wrong']}错) | "
              f"P&L {m['pnl_total']:+.0f} | 回撤 {m['max_drawdown']:.0f} | "
              f"触发后|实际|≥50 {m['trig_ge50']:.3f}")

    # 保存 v9.1 模型（自含 atr_filter，供拷入 deploy）
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f'xgb_v9.1_{ts}.joblib')
    payload = {
        'clf': clf, 'reg': reg,
        'features': feats,
        'threshold_minor': v9t.T1, 'threshold_big': v9t.T2,
        'classes': v9t.CLS_NAMES,
        'trigger_rule': 'value_driven',
        'model_type': 'v9_direction',
        'version': 'v9.1',
        'mag_transform': 'log1p',
        'direction_cut': v9t.T1,
        'hour_prior': vote, 'hour_prior_frac': frac,
        'atr_filter': {
            **ATR,
            'allow': {str(d)[:10]: bool(v) for d, v in allow.items()},
            'note': '推理时用 ≤D-1 实际价差重算 ATR（v9_atr.py）; '
                    '出手=量级头触发 且 小时先验明确 且 当日 ATR allow',
        },
        'decision_rule': (
            f"出手 = 量级头触发(|mag|≥{v9t.T1}) 且 小时先验明确 且 ATR 过滤器放行; "
            f"交易方向 = 小时先验; ATR: ATR_24 ≥ {ATR['ratio']}×20日基线 且 ≥ {ATR['abs_atr_floor']}"),
        'fixed_params': v9t.cfg.MODEL_FIXED_PARAMS,
        'train_end_date': all_dates[-(vd + 1)],
        'valid_window': [all_dates[-vd], all_dates[-1]],
        'data_end_date': all_dates[-1],
        'n_features': len(feats),
        'metrics': {'valid': {'c_base': m_base, 'c_atr': m_atr}},
        'valid_c_atr_hit': float(m_atr['dir_hit']),
        'valid_c_atr_trigger': float(m_atr['trigger_rate']),
        'valid_c_atr_pnl': float(m_atr['pnl_total']),
        'trained_at': ts,
    }
    joblib.dump(payload, save_path)
    print(f"\n  模型已保存: {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")
    print("  含 atr_filter 配置，部署实装时 2B 对预测日重算 ATR 门即可")


if __name__ == '__main__':
    main()
