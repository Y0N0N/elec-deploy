#!/usr/bin/env python
# ============================================================
# Yonon/31_model_v9_direction_run.py — v9 方向信号模型正式训练
#
# 定位（交接文档 §4.1）：v9 = 方向信号模型，输出 (方向, 量级, 置信度) + 交易建议。
#   结构 = 方向头（3类分类, 成本矩阵 loss） + 量级头（回归 |价差|） + 规则层（小时先验）。
#   不再用 v8.1 的"回归值同时定方向和量级"——方向交给 方向头/小时先验，量级交给 量级头。
#
# 数据红线：只用 deploy 已导入的旧数据（spread_label 至 2026-07-21），
#   不使用 新数据/ 目录的任何 8 月 xlsx。
#
# 用法：  venv/bin/python 31_model_v9_direction_run.py [--valid-days 60]
#   产物： deploy_v8/models/xgb_v9_<ts>.joblib（自含 features/thresholds/trigger_rule,
#          model_type='v9_direction', 与 2B 兼容）+ 更新 latest_model.json 指针。
# ============================================================
import argparse
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = _HERE   # 本脚本随部署落地在 code/v9/（v9 专属目录），v9_wrappers 同目录
_ROOT = os.path.abspath(_HERE)
while not os.path.exists(os.path.join(_ROOT, '_cfg.py')):
    _ROOT = os.path.dirname(_ROOT)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from _cfg import cfg
from v9_wrappers import BoosterDir, BoosterMag, dir_hit_metric

# ── 阈值 ──
T1 = cfg.SPREAD_THRESHOLD          # 50.0
T2 = cfg.SPREAD_THRESHOLD_BIG      # 100.0
NCLS = 3                           # 方向头 3 类: 0=负偏差 / 1=中性 / 2=正偏差
CLS_NAMES = ['neg', 'neu', 'pos']

# ── 方向头成本矩阵（row=真实, col=预测）── 方向错（neg↔pos）重罚，中性混淆轻罚 ──
#   绝对值 = 该真实类被误分时的人均惩罚；用于 per-class 样本权重（§1.2 sample_weight 方案）
C_MAT = np.array([[0., 1., 3.],
                  [1., 0., 1.],
                  [3., 1., 0.]], dtype=float)
COST_SCALE = C_MAT.mean(axis=1)    # [2.0, 1.0, 2.0]  方向类权重翻倍


# ════════════════════════════════════════════════════════════
# 一、数据加载
# ════════════════════════════════════════════════════════════
def build_feature_list():
    """v9 特征 = v8.1 的 378 个因子 + 小时 one-hot(24) + 近期价差 regime(2)。

    优先从「既有的 v9 模型」继承特征清单——它已含全部 404 个特征
    （含小时 one-hot 与 regime），直接复用、不追加，避免列重复。
    无 v9 模型时才回退继承 v8.1 的基础特征（378）并补上 v9 新增 26 个。"""
    for key in ('v9', 'v8.1'):
        mpath = cfg.resolve_latest_model(key)
        if mpath:
            m = joblib.load(mpath)
            feats = list(m['features'])
            break
    else:
        raise FileNotFoundError('找不到 v9 或 v8.1 模型，无法继承基础特征清单')
    if key == 'v9':
        return feats   # v9 模型已含 hour one-hot + regime，直接复用
    feats += [f'hour_{i:02d}' for i in range(24)]
    feats += ['sp_regime_mean7', 'sp_regime_abs7']
    return feats


def load_X(feats):
    """全因子矩阵 (date,hour)×factor，只读因子库。

    硬保护：v9 专属特征（hour_00~23 one-hot、sp_regime_*）缺失会直接报错。
    它们由 deploy/v9_add_features.py 构建，缺了意味着"没跑因子重建第 6 步"，
    静默跳过会训出丢核心设计的废模型。"""
    v9_essential = [f'hour_{i:02d}' for i in range(24)] + ['sp_regime_mean7', 'sp_regime_abs7']
    missing_ess = [n for n in v9_essential
                   if not os.path.exists(os.path.join(cfg.FACTOR_DIR, f'{n}.fea'))]
    if missing_ess:
        raise FileNotFoundError(
            f"缺 v9 必需特征 {len(missing_ess)} 个: {missing_ess[:5]}...\n"
            f"请先运行 2A_rebuild.py（因子重建第 6 步会调 v9_add_features.py 构建），"
            f"或手动运行 python v9_add_features.py。")

    fl = []
    for name in feats:
        p = os.path.join(cfg.FACTOR_DIR, f'{name}.fea')
        if not os.path.exists(p):
            print(f"  警告: 因子缺失 {name}，置 NaN")
            continue
        df = pd.read_feather(p)
        df.index = df.index.astype(str)
        df.columns = [str(c) for c in df.columns]
        s = df.stack(); s.name = name
        fl.append(s)
    X = pd.concat(fl, axis=1)
    X.index = X.index.rename(['date', 'hour'])
    return X.sort_index()


def load_y():
    """spread_label (date×24) → (date,hour) 长表。"""
    p = os.path.join(cfg.LABEL_DIR, 'spread_label.feather')
    sp = pd.read_feather(p)
    sp.index = pd.to_datetime(sp.index).strftime('%Y-%m-%d')
    sp.columns = [str(c) for c in sp.columns]
    y = sp.stack(); y.index = y.index.rename(['date', 'hour'])
    return y


def to_class3(spr, t):
    """spread → 3 类码 {0:负偏差, 1:中性, 2:正偏差}，在 ±t 处切。"""
    spr = np.asarray(spr, dtype=float)
    return np.where(spr < -t, 0, np.where(spr <= t, 1, 2))


# ════════════════════════════════════════════════════════════
# 二、小时先验（只由训练段统计，避免 valid 泄漏）
# ════════════════════════════════════════════════════════════
def build_hour_prior(y_tr):
    """由训练段统计每小时「负偏差占非中性样本的比例」→ 先验投票 -1/0/+1。"""
    yw = y_tr.unstack()
    yw.index = yw.index.astype(str)
    frac, vote = {}, {}
    for h in yw.columns:
        v = yw[h].dropna().values
        n_neg = float(np.mean(v < -T1))
        n_pos = float(np.mean(v > T1))
        denom = n_neg + n_pos
        f = n_neg / denom if denom > 1e-9 else 0.5
        frac[str(h)[:2]] = f
        vote[str(h)[:2]] = (-1 if f >= 0.55 else (1 if f <= 0.45 else 0))
    return frac, vote


# ════════════════════════════════════════════════════════════
# 三、训练
# ════════════════════════════════════════════════════════════
def train_dir_head(X_tr, y_tr, X_va, y_va):
    """方向头：3 类 XGBClassifier，成本矩阵 sample_weight + 方向命中率早停。"""
    yc_tr = to_class3(y_tr.values, T1)
    yc_va = to_class3(y_va.values, T1)
    dist = np.bincount(yc_tr, minlength=NCLS)
    bal = len(yc_tr) / (NCLS * np.maximum(dist, 1))
    sw_tr = (bal[yc_tr] * COST_SCALE[yc_tr]).astype(np.float32)
    sw_va = (bal[yc_va] * COST_SCALE[yc_va]).astype(np.float32)

    clf = xgb.XGBClassifier(
        n_estimators=cfg.MODEL_N_ESTIMATORS,
        early_stopping_rounds=cfg.MODEL_EARLY_STOPPING,
        eval_metric=dir_hit_metric,
        random_state=cfg.MODEL_RANDOM_STATE, verbosity=0, **cfg.MODEL_FIXED_PARAMS)
    clf.fit(X_tr, yc_tr, sample_weight=sw_tr,
            eval_set=[(X_va, yc_va)], sample_weight_eval_set=[sw_va], verbose=False)
    print(f"  方向头 best_iteration={clf.best_iteration} | 类别平衡 {np.round(bal,2)} "
          f"× 成本 {COST_SCALE}")
    return clf


def train_mag_head(X_tr, y_tr):
    """量级头：XGBRegressor 学 log1p(|实际价差|)，Plan C' 加权（|y|>τ→1.0, ≤τ→0.2），
    跑满 n_estimators（无 rmse 早停，防数值塌缩）。

    警告: 经验：直接回归 |spread| 会让模型系统性锚定在高位（valid 中 mag≥50 达 91%，
    触发率 0.53 超标）；改 log1p 目标后量级分布回归合理（mag≥50≈21%），
    触发更精准（命中 0.59 / 触发率 0.13）。预测后需 expm1 还原，见 mag_transform。"""
    mag = np.abs(y_tr.values)
    sw = np.where(mag > T1, 1.0, 0.2).astype(np.float32)
    target = np.log1p(mag).astype(np.float32)
    reg = xgb.XGBRegressor(n_estimators=cfg.MODEL_N_ESTIMATORS,
                           random_state=cfg.MODEL_RANDOM_STATE, verbosity=0,
                           **cfg.MODEL_FIXED_PARAMS)
    reg.fit(X_tr, target, sample_weight=sw, verbose=False)
    return reg


# ════════════════════════════════════════════════════════════
# 四、评估（valid 段，模型没见过）+ C 策略规则层 + 回测
# ════════════════════════════════════════════════════════════
def evaluate_v9(clf, reg, X_va, y_va, hour_vote):
    act = np.asarray(y_va.values, dtype=float)
    hours = np.asarray([h[:2] for h in X_va.index.get_level_values('hour')])
    mag = np.expm1(np.asarray(reg.predict(X_va), dtype=float))   # log1p → 原量纲
    dir_codes = np.asarray(clf.predict(X_va), dtype=int)
    proba = np.asarray(clf.predict_proba(X_va))
    conf = proba.max(axis=1)
    as_ = np.where(act < -T1, -1, np.where(act > T1, 1, 0))      # 实际方向
    prior = np.array([hour_vote.get(h, 0) for h in hours])       # 小时先验
    # 规则层（C 策略）：出手 = 量级头触发(|mag|≥τ) 且 小时先验明确；方向 = 小时先验
    trig = (mag >= T1) & (prior != 0)
    pred_dir = np.where(trig, prior, 0)

    def metrics(sel, name):
        n = int(sel.sum())
        out = {'name': name, 'n': n}
        if n == 0:
            return out
        p, a = pred_dir[sel], as_[sel]
        av = np.abs(act[sel])
        nonneu = a != 0
        if nonneu.sum():
            out['hit'] = float(np.mean(p[nonneu] == a[nonneu]))
        else:
            out['hit'] = float('nan')
        out['trig'] = float(np.mean(p != 0))
        out['trig_ge50'] = float(np.mean(av[p != 0] >= T1)) if np.any(p != 0) else float('nan')
        out['flip_neg'] = float(np.mean((p < 0) & (act[sel] > T1)))   # 判负实际>+50
        out['flip_pos'] = float(np.mean((p > 0) & (act[sel] < -T1)))  # 判正实际<-50
        # 回测 P&L（C 策略）：正确 +|价差|，错 −|价差|，实际中性 0
        pl = np.zeros(n)
        correct = (p != 0) & (a != 0) & (p == a)
        wrong = (p != 0) & (a != 0) & (p != a)
        pl[correct] = av[correct]
        pl[wrong] = -av[wrong]
        cum = np.cumsum(pl)
        out['net_win_rate'] = float(correct.sum() / (correct.sum() + wrong.sum())) if (correct.sum() + wrong.sum()) else float('nan')
        out['pnl_total'] = float(pl.sum())
        out['max_drawdown'] = float((np.maximum.accumulate(cum) - cum).max())
        out['n_correct'] = int(correct.sum()); out['n_wrong'] = int(wrong.sum())
        return out

    base = metrics(np.ones(len(act), dtype=bool), '规则层C(全部小时)')
    acted = metrics(trig, '规则层C(出手小时)')
    acted['trig_global'] = base['trig']              # 全局触发率（出手小时 / 全部小时）

    # 方向头自身的命中（模型方向信号，供诊断）
    nonneu = as_ != 0
    dir_hit = float(np.mean(dir_codes[nonneu] - 1 == as_[nonneu])) if nonneu.sum() else float('nan')
    model_dir = dict(
        dir_hit=dir_hit,
        trigger_rate=float(np.mean(dir_codes != 1)),
        mean_conf=float(np.mean(conf)),
        hit_by_hour={h: float(np.mean((dir_codes[hours == h] - 1)[as_[hours == h] != 0] == as_[hours == h][as_[hours == h] != 0]))
                     if np.any(as_[hours == h] != 0) else float('nan') for h in ['00', '08', '12', '16', '20', '21', '22']},
    )
    return {'base': base, 'acted': acted, 'dir_head': model_dir}


def main():
    ap = argparse.ArgumentParser(description='v9 方向信号模型正式训练')
    ap.add_argument('--valid-days', type=int, default=60)
    args = ap.parse_args()

    print("=" * 72)
    print(f"  v9 方向信号模型训练 | τ={T1} τ_big={T2} | valid={args.valid_days}天")
    print("=" * 72)

    feats = build_feature_list()
    X = load_X(feats)
    y = load_y()
    common = X.index.intersection(y.index)
    Xc, yc = X.loc[common], y.loc[common]
    print(f"X: {Xc.shape} ({len(feats)} 特征) | 日期 {Xc.index.get_level_values('date').min()} ~ "
          f"{Xc.index.get_level_values('date').max()}")

    all_dates = sorted(set(Xc.index.get_level_values('date')))
    tr_dates = set(all_dates[: len(all_dates) - args.valid_days])
    va_dates = set(all_dates[len(all_dates) - args.valid_days:])
    tr_m = Xc.index.get_level_values('date').isin(tr_dates)
    va_m = Xc.index.get_level_values('date').isin(va_dates)
    X_tr, y_tr = Xc.loc[tr_m], yc.loc[tr_m]
    X_va, y_va = Xc.loc[va_m], yc.loc[va_m]
    assert not set(X_tr.index.get_level_values('date')) & va_dates, "红线: valid 混入 train!"
    print(f"train {all_dates[0]} ~ {all_dates[-(args.valid_days+1)]} ({len(X_tr)}h) | "
          f"valid {all_dates[-args.valid_days]} ~ {all_dates[-1]} ({len(X_va)}h)")

    # 小时先验（训练段统计）
    frac, vote = build_hour_prior(y_tr)
    print(f"小时先验投票: {vote}")

    # 训练双头
    print("\n[训练 方向头] 3类 XGBClassifier（成本矩阵权重 + 方向命中率早停）")
    clf = train_dir_head(X_tr, y_tr, X_va, y_va)
    print("\n[训练 量级头] |spread| XGBRegressor（Plan C' 加权，跑满轮数）")
    reg = train_mag_head(X_tr, y_tr)

    # 评估
    ev = evaluate_v9(clf, reg, X_va, y_va, vote)
    print("\n===== valid 段评估（模型没见过）=====")
    m = ev['acted']
    print(f"规则层C(出手): 样本 {m['n']} | 方向命中 {m['hit']:.3f} | "
          f"全局触发率 {ev['base']['trig']:.3f} | 触发后|实际|≥50 {m['trig_ge50']:.3f}")
    print(f"  回测: 净胜率 {m['net_win_rate']:.3f} (对{m['n_correct']}/错{m['n_wrong']}) | "
          f"P&L {m['pnl_total']:+.0f} | 最大回撤 {m['max_drawdown']:.0f}")
    dh = ev['dir_head']
    print(f"方向头自身: 方向命中 {dh['dir_hit']:.3f} | 触发率 {dh['trigger_rate']:.3f} | "
          f"平均置信度 {dh['mean_conf']:.3f}")
    print(f"  按小时命中: {dh['hit_by_hour']}")

    # 保存 joblib（自含 features/thresholds/trigger_rule + v9 专属字段，兼容 2B）
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    save_path = os.path.join(cfg.MODEL_DIR, f'xgb_v9_{ts}.joblib')
    payload = {
        'clf': clf,
        'reg': reg,
        'features': feats,
        'threshold_minor': T1,
        'threshold_big': T2,
        'classes': CLS_NAMES,
        'trigger_rule': 'value_driven',
        'model_type': 'v9_direction',
        'mag_transform': 'log1p',                  # 量级头 log1p 目标，predict 后需 expm1 还原
        'direction_cut': T1,                       # 方向头 3 类在 ±τ 处切
        'hour_prior': vote,                        # 小时 → 先验投票 -1/0/+1
        'hour_prior_frac': frac,                   # 小时 → 负占比（诊断）
        'decision_rule': (
            f"出手 = 量级头触发(|reg|≥{T1}) 且 小时先验方向明确; 交易方向 = 小时先验"
            f"（负→日前买/实时卖; 正→日前卖/实时买）; 模型方向/置信度来自方向头"),
        'fixed_params': cfg.MODEL_FIXED_PARAMS,
        'train_end_date': all_dates[-(args.valid_days + 1)],
        'valid_window': [all_dates[-args.valid_days], all_dates[-1]],
        'data_end_date': all_dates[-1],
        'n_features': len(feats),
        'metrics': {'valid': {'acted': ev['acted'], 'dir_head': ev['dir_head'],
                              'base': ev['base']}},
        'valid_c_hit': float(m['hit']),
        'valid_trigger_rate': float(ev['base']['trig']),   # 全局触发率（出手小时 / 全部小时）
        'valid_dir_hit': float(dh['dir_hit']),
        'trained_at': ts,
    }
    joblib.dump(payload, save_path)
    print(f"\n  模型已保存: {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

    # 更新指针
    ptr = {}
    if os.path.exists(cfg.LATEST_MODEL_FILE):
        try:
            ptr = json.load(open(cfg.LATEST_MODEL_FILE))
        except Exception:
            pass
    ptr['v9'] = save_path
    json.dump(ptr, open(cfg.LATEST_MODEL_FILE, 'w'), ensure_ascii=False, indent=2)
    print(f"  最新模型指针已更新: {cfg.LATEST_MODEL_FILE}")

    # 结果 JSON（产物固定在部署根 data/，与部署目录解耦）
    os.makedirs(os.path.join(_ROOT, 'data'), exist_ok=True)
    summary = {'model': 'v9', 'threshold': T1, 'threshold_big': T2,
               'trigger_rule': 'value_driven', 'model_type': 'v9_direction',
               'valid': ev, 'features': len(feats), 'trained_at': ts}
    out_json = os.path.join(_ROOT, 'data', 'v9_results.json')
    json.dump(summary, open(out_json, 'w'), ensure_ascii=False, indent=2)
    print(f"  结果: {out_json}")


if __name__ == '__main__':
    main()
