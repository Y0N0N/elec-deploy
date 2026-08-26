#!/usr/bin/env python
# ============================================================
# Yonon/v9_direction_sim.py — v9 方向信号仿真
#
# 目的（回答"低命中率下能否可靠交易"）：
#   1) 用「旧数据」walk-forward 切分，训练 v8.1 同款级联模型；
#   2) 只在模型没见过的 valid 段评估：方向命中 / 翻转率 / 触发质量；
#   3) 关键实验：模型方向 + 小时先验一致时才出手，能否把方向命中拉过 0.5。
#
# 数据口径（红线）：只用 deploy 已导入的旧数据（spread_label 到 2026-07-25，
#   不含 新数据/ 目录的 8 月 xlsx）。特征因子库 qlib158 同样只到旧日期。
# 完全在 Yonon_v9_f 下运行，不改动 deploy_v9 任何文件。
#
# 用法：  python v9_direction_sim.py [--valid-days 60] [--blocks 2]
#   --valid-days 最后 N 天作 valid（默认 60）；--blocks 1 → 单段；>1 → 分块轮流评估
# ============================================================
import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.abspath(os.path.join(_HERE, '..', 'deploy_v9'))
sys.path.insert(0, DEPLOY)

import numpy as np
import pandas as pd
import xgboost as xgb
from _cfg import cfg

import joblib
import json


# ════════════════════════════════════════════════════════════
# 一、数据加载（复用 deploy 的因子库 / 标签，纯只读）
# ════════════════════════════════════════════════════════════
def load_X():
    """全因子矩阵 (date,hour)×factor，只读因子库。"""
    feats = None
    # 用 v8.1 模型的特征清单（378 个，均在因子库）
    mpath = cfg.resolve_latest_model('v8.1')
    if mpath and os.path.exists(mpath):
        m = joblib.load(mpath)
        feats = m['features']
    if not feats:
        feats = sorted(n[:-4] for n in os.listdir(cfg.FACTOR_DIR) if n.endswith('.fea'))
    fl = []
    for name in feats:
        p = os.path.join(cfg.FACTOR_DIR, f'{name}.fea')
        if not os.path.exists(p):
            continue
        df = pd.read_feather(p)
        df.index = df.index.astype(str)
        df.columns = [str(c) for c in df.columns]
        s = df.stack(); s.name = name
        fl.append(s)
    X = pd.concat(fl, axis=1)
    X.index = X.index.rename(['date', 'hour'])
    return X.sort_index(), feats


def load_y():
    """spread_label (date×24) → (date,hour) 长表。"""
    p = os.path.join(cfg.LABEL_DIR, 'spread_label.feather')
    sp = pd.read_feather(p)
    sp.index = pd.to_datetime(sp.index).strftime('%Y-%m-%d')
    sp.columns = [str(c) for c in sp.columns]
    y = sp.stack(); y.index = y.index.rename(['date', 'hour'])
    return y


def _to_class5(spr, t1, t2):
    spr = np.asarray(spr, dtype=float)
    return np.where(spr < -t2, 0,
           np.where(spr < -t1, 1,
           np.where(spr <= t1, 2,
           np.where(spr <= t2, 3, 4))))


# ════════════════════════════════════════════════════════════
# 二、训练（v8.1 同款：5类 mlogloss + Plan C' 加权回归）
# ════════════════════════════════════════════════════════════
def train_on(X_tr, y_tr, X_va, y_va, t1, t2, n_est=cfg.MODEL_N_ESTIMATORS):
    ncls = len(cfg.SPREAD_CLASSES)
    yc_tr = _to_class5(y_tr, t1, t2)
    cls_counts = np.bincount(yc_tr, minlength=ncls)
    w = len(yc_tr) / (ncls * cls_counts)
    clf = xgb.XGBClassifier(
        n_estimators=n_est, early_stopping_rounds=cfg.MODEL_EARLY_STOPPING,
        eval_metric='mlogloss', random_state=cfg.MODEL_RANDOM_STATE,
        verbosity=0, **cfg.MODEL_FIXED_PARAMS)
    clf.fit(X_tr, yc_tr, sample_weight=w[yc_tr],
            eval_set=[(X_va, _to_class5(y_va, t1, t2))], verbose=False)
    # Plan C' 加权回归：|y|>τ_minor → 1.0，|y|≤τ → 0.2（防数值塌缩，跑满 n_est 轮）
    sw_tr = np.where(np.abs(y_tr.values) > t1, 1.0, 0.2).astype(np.float32)
    reg = xgb.XGBRegressor(n_estimators=n_est, random_state=cfg.MODEL_RANDOM_STATE,
                           verbosity=0, **cfg.MODEL_FIXED_PARAMS)
    reg.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
    return clf, reg


# ════════════════════════════════════════════════════════════
# 三、评估（valid 段：只统计模型没见过的新日子）
# ════════════════════════════════════════════════════════════
# 小时先验（只由训练段统计，避免 valid 泄漏）。全局 dict: hour "14" -> 负偏差占比(0~1)
NEG_RATIO = {}


def _build_hour_prior(X_tr, y_tr, t1):
    """由训练段统计每小时「负偏差占非中性样本的比例」。"""
    yw = y_tr.unstack()
    yw.index = yw.index.astype(str)
    for h in yw.columns:
        v = yw[h].dropna().values
        n_neg = float(np.mean(v < -t1))
        n_pos = float(np.mean(v > t1))
        NEG_RATIO[str(h)[:2]] = n_neg / max(n_neg + n_pos, 1e-9)


def hour_prior_vote(hour_str):
    """小时先验投票：该小时非中性样本里更偏负 → -1；更偏正 → +1；不显著 → 0。"""
    frac_neg = NEG_RATIO[hour_str]
    if frac_neg >= 0.55:
        return -1
    if frac_neg <= 0.45:
        return +1
    return 0


def evaluate_block(clf, reg, X_va, y_va, t1, t2, name, use_prior=True):
    """在 valid 块上算指标。use_prior=True 时测试「模型方向 × 小时先验一致才出手」。"""
    yv = reg.predict(X_va)
    yc = clf.predict(X_va)
    act = np.asarray(y_va.values, dtype=float)
    hours = np.asarray([h[:2] for h in X_va.index.get_level_values('hour')])
    ps = np.where(yv < -t1, -1, np.where(yv > t1, 1, 0))       # 模型方向（回归驱动）
    as_ = np.where(act < -t1, -1, np.where(act > t1, 1, 0))    # 实际方向
    # 小时先验过滤：只有「模型方向 与 小时先验 同号」才保留（= 出手的样本）
    if use_prior:
        agree = np.array([p == hour_prior_vote(h[:2]) for p, h in zip(ps, hours)])
        kept = agree
    else:
        kept = np.ones(len(ps), dtype=bool)
    # 指标
    def metrics(sel):
        if sel.sum() == 0:
            return dict(n=0)
        p, a = ps[sel], as_[sel]
        nonneu = a != 0
        return dict(
            n=int(sel.sum()),
            hit=float(np.mean(p[nonneu] == a[nonneu])) if nonneu.sum() else float('nan'),
            trig=float(np.mean(p != 0)),
            trig_ge50=float(np.mean(np.abs(act[sel][p != 0]) >= 50)) if np.any(p != 0) else float('nan'),
            flip_neg=float(np.mean((p < 0) & (a > 50))),
            flip_pos=float(np.mean((p > 0) & (a < -50))),
            reg_mean=float(np.mean(yv[sel])),
            hit_by_hour={},
        )
    base = metrics(np.ones(len(ps), dtype=bool))
    filt = metrics(kept)
    # 按小时看命中率（诊断时段性偏置）
    for h in ['00','08','10','12','14','16','19','20','21','22']:
        hm = hours == h
        if hm.sum():
            base['hit_by_hour'][h] = float(np.mean(ps[hm][as_[hm]!=0]==as_[hm][as_[hm]!=0])) if np.any(as_[hm]!=0) else float('nan')
    return {'name': name, 'base': base, 'filtered': filt}


def evaluate_magnitude_first(reg, X_va, y_va, t1, t2):
    """量级优先实验：模型只负责挑「确实有大偏差」的小时（|reg|≥t1），
    方向不靠模型，而交给「小时先验」规则（该小时历史上更偏负→判负，更偏正→判正）。

    这是针对「模型方向弱（valid 命中 0.31）但量级识别稍强」的妥协方案：
    规避模型最弱的方向判断，只复用它的量级触发能力。"""
    yv = reg.predict(X_va)
    act = np.asarray(y_va.values, dtype=float)
    hours = np.asarray([h[:2] for h in X_va.index.get_level_values('hour')])
    as_ = np.where(act < -t1, -1, np.where(act > t1, 1, 0))

    # 出手 = 模型量级触发 + 该小时先验方向明确
    prior_dir = np.array([hour_prior_vote(h[:2]) for h in hours])
    trig = np.abs(yv) >= t1
    kept = trig & (prior_dir != 0)
    # 预测方向 = 先验方向（模型不参与方向）
    ps = np.where(kept, prior_dir, 0)

    nonneu = as_ != 0
    sel = kept
    hit = float(np.mean(ps[sel][nonneu[sel]] == as_[sel][nonneu[sel]])) if sel.sum() and np.any(nonneu[sel]) else float('nan')
    return dict(
        n=int(sel.sum()),
        hit=hit,
        trig=float(np.mean(kept)),
        trig_ge50=float(np.mean(np.abs(act[kept]) >= 50)) if kept.sum() else float('nan'),
        flip_neg=float(np.mean((ps < 0) & (act > 50))),
        flip_pos=float(np.mean((ps > 0) & (act < -50))),
        reg_mean=float(np.mean(yv[kept])) if kept.sum() else float('nan'),
    )


# ════════════════════════════════════════════════════════════
# 四、主流程
# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--valid-days', type=int, default=60)
    ap.add_argument('--blocks', type=int, default=2,
                    help='valid 天数分成几块轮流评估（1=单段）')
    ap.add_argument('--out', default=os.path.join(_HERE, 'v9_sim_results.json'))
    args = ap.parse_args()

    X, feats = load_X()
    y = load_y()
    common = X.index.intersection(y.index)
    Xc, yc = X.loc[common], y.loc[common]
    print(f"X: {Xc.shape} | 日期 {Xc.index.get_level_values('date').min()} ~ "
          f"{Xc.index.get_level_values('date').max()}")

    all_dates = sorted(set(Xc.index.get_level_values('date')))
    t1, t2 = cfg.SPREAD_THRESHOLD, cfg.SPREAD_THRESHOLD_BIG

    # ---- walk-forward：最后一次滚动切分 ----
    valid_days = args.valid_days
    n = len(all_dates)
    tr_dates = set(all_dates[: n - valid_days])
    va_dates = set(all_dates[n - valid_days:])
    print(f"train {all_dates[0]} ~ {all_dates[-(valid_days+1)]} | "
          f"valid {all_dates[n-valid_days]} ~ {all_dates[-1]} ({valid_days}天)")
    tr_m = Xc.index.get_level_values('date').isin(tr_dates)
    va_m = Xc.index.get_level_values('date').isin(va_dates)
    X_tr, y_tr = Xc.loc[tr_m], yc.loc[tr_m]
    X_va, y_va = Xc.loc[va_m], yc.loc[va_m]
    assert not set(X_tr.index.get_level_values('date')) & va_dates, "红线: valid 混入 train!"

    # ---- 小时先验：只由训练段统计 ----
    _build_hour_prior(X_tr, y_tr, t1)

    clf, reg = train_on(X_tr, y_tr, X_va, y_va, t1, t2)
    res = evaluate_block(clf, reg, X_va, y_va, t1, t2, f'valid-{valid_days}d', use_prior=True)
    res['magnitude_first'] = evaluate_magnitude_first(reg, X_va, y_va, t1, t2)

    print("\n===== valid 段评估 =====")
    for tag, m in [('基础(无过滤)', res['base']), ('小时先验过滤后', res['filtered']),
                   ('量级优先+方向给规则', res['magnitude_first'])]:
        print(f"\n{tag}:")
        if m['n'] == 0:
            print("  （无样本）"); continue
        print(f"  样本 {m['n']} | 方向命中 {m['hit']:.3f} | 触发率 {m['trig']:.3f} | "
              f"触发后|实际|≥50 {m['trig_ge50']:.3f}")
        print(f"  强翻转: 判负实际>+50 {m['flip_neg']:.3f} | 判正实际<-50 {m['flip_pos']:.3f} | "
              f"触发样本 reg均值 {m['reg_mean']:+.1f}")

    with open(args.out, 'w') as f:
        json.dump({'valid': res, 'train_end': all_dates[n-valid_days-1],
                   'data_end': all_dates[-1], 'features': len(feats)}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n结果已写 {args.out}")


if __name__ == '__main__':
    main()
