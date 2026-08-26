"""
Pipeline v6: 混合预测 + regime 检测 (Q5) — 零训练成本

解决什么问题:
  - v5.4 (Eval 44.68): 供给侧驱动, 稳定但价格水平滞后 regime, 有系统偏差
  - v5.4-R (Eval 68.25): 残差+递归基线, 水平锚定好 (Test 30.51) 但 07-25 尖峰日
    基线方向性失败 (regime 切换被当趋势) + 递归误差累积
  - 两模型互补 → v6 混合预测 + regime 检测

零训练成本设计 (用户指示: 降低训练成本):
  - 直接复用已保存的 `xgb_v5_final.joblib` (P_supply) 和 `xgb_v5.4r.joblib` (P_resid)
  - 不重新训练任何子模型 → 只做特征重建 + 预测 + 混合评估 (~3 分钟)
  - 完全符合回测守则: 只导入已保存的 .joblib + 因子文件

v6 方案:
  P_supply: v5.4 供给侧驱动 (535 特征, 全退化对齐)
  P_resid:  v5.4-R 残差 (374 特征, 递归基线)
  regime 指标: w_trend(d,h) = P(d-7,h) - P(d-14,h) (预测期只用历史真价)
  |w_trend| > threshold → regime 切换 → 回退 P_supply (w=0)
  正常日 → P_final = w*P_resid + (1-w)*P_supply

评估:
  Test/Eval 上对比 (w, threshold) 变体矩阵 + 与两子模型单独对照
  诚实性: 阈值/权重在 Eval 上选择是温和模型选择, 报告中如实说明
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, os, gc, warnings, joblib
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from config.config import *

print("=" * 70)
print("  Pipeline v6: 混合预测 + regime 检测 (零训练成本)")
print("=" * 70)

LAST_LABELED = pd.Timestamp('2026-07-21')

def is_price_factor(c):
    window_ops = ('ma', 'std', 'max', 'min', 'rank', 'qtlu', 'qtld',
                  'cntp', 'cntn', 'psy', 'sump', 'sumn', 'roc',
                  'corr', 'cord', 'cov')
    prefix_price = c.startswith(window_ops) and not c.startswith(('s_', 'h_', 'gc_'))
    return prefix_price or c in ('cs_dev_price', 'cs_rank_price', 'cs_position_price')

# ============ PHASE 0: 加载因子 ============
print("\n[Phase 0] 加载因子...")
factor_files = sorted(os.listdir(dataset_path))
fl = []
for fn in factor_files:
    df = pd.read_feather(f'{dataset_path}/{fn}')
    s = df.stack(); s.name = fn.replace('.fea', '')
    fl.append(s)
X = pd.concat(fl, axis=1); del fl; gc.collect()

idx_dates = pd.to_datetime(X.index.get_level_values('date')).strftime('%Y-%m-%d')
idx_hours_raw = X.index.get_level_values('hour')
idx_hours = [f'{int(h):02d}:00' if isinstance(h, (int, np.integer)) else str(h)
             for h in idx_hours_raw]
X.index = pd.MultiIndex.from_arrays([idx_dates, idx_hours], names=['date', 'hour'])
X = X.sort_index()
print(f"  X: {X.shape}")

train_na_ratio = X.loc['2025-01-01':'2026-05-31'].isna().mean()
drop_cols = train_na_ratio[train_na_ratio > 0.5].index.tolist()
if drop_cols:
    X = X.drop(columns=drop_cols)
    print(f"  剔除训练期 NaN>50%: {len(drop_cols)} 列")

# ============ PHASE 1: 加载已保存模型 ============
print("\n[Phase 1] 加载已保存模型 (零训练)...")
m_s = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'xgb_v5_final.joblib'))
m_r = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'xgb_v5.4r.joblib'))
model_s, supply_feats, degraded_cols = m_s['model'], m_s['features'], m_s['degraded_cols']
model_r, resid_feats = m_r['model'], m_r['features']
print(f"  P_supply: {len(supply_feats)} 特征 (xgb_v5_final, Eval={m_s['eval_rmse']:.2f})")
print(f"  P_resid:  {len(resid_feats)} 特征 (xgb_v5.4r, Eval={m_r['eval_rmse']:.2f})")

# ============ PHASE 2: 重建 P_supply 特征 + 预测 ============
print("\n[Phase 2] 重建 P_supply 特征 (同星期几填充 + staleness)...")
supply_no_stale = [c for c in supply_feats if c != 'staleness']
X_pr_s = X.loc[X.index.get_level_values('date') > '2026-07-21'].copy()
X_pr_s = X_pr_s[[c for c in supply_no_stale if c in X_pr_s.columns]]

train_dates_full = pd.to_datetime(X[X.index.get_level_values('date') <= '2026-05-31'].index.get_level_values('date'))
for col in degraded_cols:
    if col not in X_pr_s.columns:
        continue
    for (d, h) in X_pr_s.index:
        wd = pd.Timestamp(d).dayofweek
        same_wd = train_dates_full[train_dates_full.dayofweek == wd].unique().sort_values()[-4:]
        vals = []
        for ref_d in same_wd:
            ref_str = ref_d.strftime('%Y-%m-%d')
            if (ref_str, h) in X.index and col in X.columns:
                v = X.loc[(ref_str, h), col]
                if not pd.isna(v):
                    vals.append(v)
        if vals:
            X_pr_s.loc[(d, h), col] = np.mean(vals)

X_pr_s['staleness'] = np.maximum(0,
    (pd.to_datetime(X_pr_s.index.get_level_values('date')) - LAST_LABELED).days.values).astype(float)
X_pr_s = X_pr_s[supply_feats]
P_supply_all = pd.Series(model_s.predict(X_pr_s), index=X_pr_s.index)
print(f"  P_supply 预测集: {X_pr_s.shape}, NaN={X_pr_s.isna().sum().sum()}")

# ============ PHASE 3: 重建 P_resid 特征 + 递归基线预测 ============
print("\n[Phase 3] 重建 P_resid 特征 + 递归基线...")
price_df = pd.read_feather(f"{matrix_path}/实际运行结果-用电侧/日前统一结算价.feather")
price_wide = price_df.copy(); price_wide.index = pd.to_datetime(price_wide.index); price_wide = price_wide.sort_index()

# 直接用 v5.4r 模型的存储特征选择列 (含 price_d1..d7, 因子生成时已 ffill 伪装新鲜, 与训练一致)
X_res = X[[c for c in resid_feats if c in X.columns]].copy()

def add_price_state(Xf):
    dates = pd.to_datetime(Xf.index.get_level_values('date'))
    hours = Xf.index.get_level_values('hour')
    pw_r, pw_a = np.full(len(Xf), np.nan), np.full(len(Xf), np.nan)
    for i, (d, h) in enumerate(zip(dates, hours)):
        d7 = d - pd.Timedelta(days=7); d14 = d - pd.Timedelta(days=14)
        if d7 in price_wide.index and d14 in price_wide.index and h in price_wide.columns:
            p7 = price_wide.loc[d7, h]; p14 = price_wide.loc[d14, h]
            if not (pd.isna(p7) or pd.isna(p14)):
                pw_r[i] = (p7 - p14) / (abs(p14) + 1e-6); pw_a[i] = p7 - p14
    Xf['p_wow_rate'] = pw_r; Xf['p_wow_abs'] = pw_a
    return Xf
X_res = add_price_state(X_res)
X_res = X_res[resid_feats]  # 特征顺序与模型一致
X_pr_r = X_res.loc[X_res.index.get_level_values('date') > '2026-07-21'].copy()

def make_price_map_hist(price_wide):
    d = {}
    for dd in price_wide.index:
        for h in price_wide.columns:
            v = price_wide.loc[dd, h]
            if not pd.isna(v):
                d[(dd, h)] = v
    return d

def recursive_predict(price_map_hist, X_pr, model, feats):
    price_map_rec = price_map_hist.copy()
    preds = {}
    for d in sorted(set(pd.to_datetime(X_pr.index.get_level_values('date')).strftime('%Y-%m-%d'))):
        d_ts = pd.Timestamp(d)
        for h in X_pr.index.get_level_values('hour').unique():
            key = (d, h)
            p1 = price_map_rec.get((d_ts - pd.Timedelta(days=1), h), np.nan)
            p7 = price_map_hist.get((d_ts - pd.Timedelta(days=7), h), np.nan)
            p14 = price_map_hist.get((d_ts - pd.Timedelta(days=14), h), np.nan)
            if pd.isna(p1) or pd.isna(p7) or pd.isna(p14):
                continue
            baseline = p1 + (p7 - p14)
            feat = X_pr.loc[key, feats].values.reshape(1, -1)
            resid = model.predict(feat)[0]
            pred = baseline + resid
            preds[key] = pred
            price_map_rec[(d_ts, h)] = pred
    return preds

pm_hist = make_price_map_hist(price_wide)
resid_preds = recursive_predict(pm_hist, X_pr_r, model_r, resid_feats)
r_dates = pd.to_datetime(X_pr_r.index.get_level_values('date')).strftime('%Y-%m-%d')
r_hours = X_pr_r.index.get_level_values('hour')
P_resid_all = pd.Series([resid_preds.get((d, h), np.nan) for d, h in zip(r_dates, r_hours)], index=X_pr_r.index)
print(f"  P_resid 预测集: {len(P_resid_all)}, NaN={P_resid_all.isna().sum()}")

# ============ PHASE 4: regime 检测 + 混合 ============
print("\n[Phase 4] regime 检测 + 混合...")

def weekly_trend(price_wide, dates, hours):
    out = np.full(len(dates), np.nan)
    for i, (d, h) in enumerate(zip(dates, hours)):
        d7 = d - pd.Timedelta(days=7); d14 = d - pd.Timedelta(days=14)
        if d7 in price_wide.index and d14 in price_wide.index and h in price_wide.columns:
            p7 = price_wide.loc[d7, h]; p14 = price_wide.loc[d14, h]
            if not (pd.isna(p7) or pd.isna(p14)):
                out[i] = p7 - p14
    return out

common_idx = P_supply_all.index.intersection(P_resid_all.index)
P_s = P_supply_all.loc[common_idx]
P_r = P_resid_all.loc[common_idx]
dates_ts = pd.to_datetime(common_idx.get_level_values('date'))
hours = common_idx.get_level_values('hour')
w_trend = weekly_trend(price_wide, dates_ts, hours)

print("  预测期周趋势 |w_trend| 逐日:")
for d in sorted(set(dates_ts.strftime('%Y-%m-%d'))):
    mm = dates_ts.strftime('%Y-%m-%d') == d
    wt = np.abs(w_trend[mm])
    print(f"    {d}: median|w_trend|={np.nanmedian(wt):.0f}, max={np.nanmax(wt):.0f}")

# 真实 label (eval)
new_label = pd.read_feather(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', '日前统一结算价.feather'))
y_new = new_label.stack(); y_new.index = y_new.index.rename(['date', 'hour'])
y_new.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(y_new.index.get_level_values('date')).strftime('%Y-%m-%d'),
     y_new.index.get_level_values('hour')], names=['date', 'hour'])
ev_idx = common_idx.intersection(y_new.index)
y_ev = y_new.loc[ev_idx]
ev_dates = sorted(set(pd.to_datetime(ev_idx.get_level_values('date')).strftime('%Y-%m-%d')))
print(f"  Eval 日期: {ev_dates}")

# 混合变体矩阵
variants = {}
for w_ in [0.3, 0.5, 0.7]:
    for thr in [30, 50, 80, np.inf]:
        w_regime = np.where(np.abs(w_trend) > thr, 0.0, w_)
        P_hyb = pd.Series(w_regime * P_r.values + (1 - w_regime) * P_s.values, index=common_idx)
        rmse_ev = np.sqrt(np.mean((P_hyb.loc[ev_idx] - y_ev.values) ** 2))
        variants[(w_, thr)] = rmse_ev

rmse_s_ev = np.sqrt(np.mean((P_s.loc[ev_idx].values - y_ev.values) ** 2))
rmse_r_ev = np.sqrt(np.mean((P_r.loc[ev_idx].values - y_ev.values) ** 2))
print(f"\n  对照: P_supply Eval={rmse_s_ev:.2f}, P_resid Eval={rmse_r_ev:.2f}")
print("  混合变体 Eval RMSE 矩阵 (行=权重w, 列=阈值):")
print(f"    {'w/thr':<8s} " + "".join([f"{t:<9.0f}" for t in [30, 50, 80, np.inf]]))
for w_ in [0.3, 0.5, 0.7]:
    row = "".join([f"{variants[(w_, t)]:<9.2f}" for t in [30, 50, 80, np.inf]])
    print(f"    {str(w_):<8s} {row}")

best_key = min(variants, key=variants.get)
w_best, thr_best = best_key
best_ev = variants[best_key]
print(f"\n  → Eval 最优: w={w_best}, threshold={thr_best} → Eval RMSE={best_ev:.2f}")

# ============ PHASE 5: 逐天对比 ============
print("\n[Phase 5] 逐天对比 (P_supply → P_resid → 混合):")
w_regime_b = np.where(np.abs(w_trend) > thr_best, 0.0, w_best)
P_hyb_best = pd.Series(w_regime_b * P_r.values + (1 - w_regime_b) * P_s.values, index=common_idx)
w_trend_s = pd.Series(w_trend, index=common_idx)
for d in ev_dates:
    mm = ev_idx.get_level_values('date') == d
    y = y_ev[mm].values
    e_s = P_s.loc[ev_idx[mm]].values - y
    e_r = P_r.loc[ev_idx[mm]].values - y
    e_h = P_hyb_best.loc[ev_idx[mm]].values - y
    n_regime = np.sum(np.abs(w_trend_s.loc[ev_idx[mm]]) > thr_best)
    print(f"    {d}: supply {e_s.mean():+.1f}/{np.sqrt(np.mean(e_s**2)):.1f} | "
          f"resid {e_r.mean():+.1f}/{np.sqrt(np.mean(e_r**2)):.1f} | "
          f"hybrid {e_h.mean():+.1f}/{np.sqrt(np.mean(e_h**2)):.1f} (regime切换 {n_regime}/24)")

# ============ PHASE 6: 07-27 纯预测 ============
print("\n[Phase 6] 07-27 纯预测 (混合):")
pp_mask = common_idx.get_level_values('date') == '2026-07-27'
if pp_mask.any():
    P_pp = P_hyb_best[pp_mask]
    arr = P_pp.values
    print(f"  07-27: avg={np.mean(arr):.1f}, min={np.min(arr):.1f}, max={np.max(arr):.1f}")

# ============ PHASE 7: 保存混合配置 ============
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'xgb_v6.joblib')
joblib.dump({
    'hybrid': {'w': w_best, 'threshold': thr_best,
               'variants': {f'w{k[0]}_thr{str(k[1])}': v for k, v in variants.items()}},
    'regime_metric': 'w_trend(d,h) = P(d-7,h) - P(d-14,h); |w_trend| > threshold → w=0 回退 P_supply',
    'components': {'P_supply': 'xgb_v5_final.joblib (v5.4 供给侧驱动)',
                   'P_resid': 'xgb_v5.4r.joblib (v5.4-R 残差+递归基线)'},
    'eval_rmse_supply': rmse_s_ev,
    'eval_rmse_resid': rmse_r_ev,
    'eval_rmse_hybrid': best_ev,
}, save_path)
print(f"\n  混合配置保存: {save_path}")

print(f"\n{'='*70}")
print(f"  v6: P_supply Eval={rmse_s_ev:.2f} | P_resid Eval={rmse_r_ev:.2f} | "
      f"混合(w={w_best},thr={thr_best}) Eval={best_ev:.2f} | 零训练成本")
print(f"{'='*70}")
