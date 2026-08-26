"""
Pipeline v7.2 — Q109-B 严格 walk-forward 回测 (无 valid 泄露, 干净 sign_hit)

目的: 把 v7.1 "大窗口 sign=0.619 是否 valid 泄露假象" 彻底搞清楚 (open.md Q109-B)。
方法: 对每个预测日 d, 只用 date < d 的数据训练 (早停 valid 取 d 前最近 30 天, 严格早于 d),
      预测 d 当天 24h → 评估日 d 绝不进入 train/valid, 无任何未来信息。
      sp_wow 特征只用 d−7/d−14 真实价差 → 特征侧同样无未来泄露。

默认回测窗口: 2026-02-01 ~ 2026-07-26 (每周重训, 预测未来 7 天)。
可用环境变量覆盖:
  WF_STEP   重训间隔(天)  默认 7    WF_VALID 早停窗口(天) 默认 30
  WF_NEST   树上限         默认 400  WF_JOBS  n_jobs        默认 8
  WF_START  回测起点       默认 2026-02-01  WF_END 回测终点 默认 2026-07-26

红线: 回测验证不得混入 valid (early stopping 集) — 评估日 d 严格排除。
      单回归基线同步训练对比 (对照强制)。

产物:
  data/walkforward_v7.2_preds_{STEP}d.csv      逐日预测 (checkpoint, 可断点续跑)
  data/walkforward_v7.2_metrics_{STEP}d.json   总指标
  show/v7/report_walkforward_sign.png          月度 sign_hit 图
"""
import sys, os, time, json, gc, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.metrics import (mean_squared_error, accuracy_score, f1_score,
                             precision_score, recall_score)
from config.config import (dataset_path, spread_label_file, da_price_latest,
                           rt_price_latest, SPREAD_THRESHOLD, SPREAD_THRESHOLD_BIG,
                           SPREAD_CLASSES, YONON_PATH)

T1, T2 = SPREAD_THRESHOLD, SPREAD_THRESHOLD_BIG
NCLS = len(SPREAD_CLASSES)

# ---- 环境变量配置 ----
STEP_DAYS = int(os.environ.get('WF_STEP', 7))
VALID_DAYS = int(os.environ.get('WF_VALID', 30))
N_EST = int(os.environ.get('WF_NEST', 400))
JOBS = int(os.environ.get('WF_JOBS', 8))
BT_START = os.environ.get('WF_START', '2026-02-01')
BT_END = os.environ.get('WF_END', '2026-07-26')

print("=" * 72)
print("  v7.2 Q109-B: 严格 walk-forward 回测 (无 valid 泄露)")
print(f"  τ_minor={T1} τ_big={T2} | 回测 {BT_START} ~ {BT_END} | 重训间隔 {STEP_DAYS}d"
      f" | 早停valid {VALID_DAYS}d | n_est={N_EST} | n_jobs={JOBS}")
print("=" * 72)

# ============ PHASE 0: 加载特征 ============
print("\n[Phase 0] 加载新鲜集特征 (369)...", flush=True)
t0 = time.time()
with open(f'{YONON_PATH}/data/v7_fresh_features.json') as f:
    fresh = json.load(f)['fresh_features']
fl = []
for name in fresh:
    df = pd.read_feather(f'{dataset_path}/{name}.fea')
    s = df.stack(); s.name = name
    fl.append(s)
X = pd.concat(fl, axis=1)
del fl; gc.collect()
idx_dates = pd.to_datetime(X.index.get_level_values(0)).strftime('%Y-%m-%d')
idx_hours = [f'{int(h):02d}:00' if isinstance(h, (int, np.integer)) else str(h)
             for h in X.index.get_level_values(1)]
X.index = pd.MultiIndex.from_arrays([idx_dates, idx_hours], names=['date', 'hour'])
X = X.sort_index()
print(f"  X: {X.shape}, 加载 {time.time()-t0:.1f}s", flush=True)

# ============ PHASE 1: 标签 ============
print("\n[Phase 1] 构建 spread 标签 (到 07-26)...", flush=True)
spread = pd.read_feather(spread_label_file)
spread.index = pd.to_datetime(spread.index)
y = spread.stack(); y.index = y.index.rename(['date', 'hour'])
y.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(y.index.get_level_values('date')).strftime('%Y-%m-%d'),
     y.index.get_level_values('hour')], names=['date', 'hour'])
y.name = 'spread'
da_e = pd.read_feather(da_price_latest); rt_e = pd.read_feather(rt_price_latest)
sps = (da_e - rt_e).stack(); sps.index = sps.index.rename(['date', 'hour'])
sps.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(sps.index.get_level_values('date')).strftime('%Y-%m-%d'),
     sps.index.get_level_values('hour')], names=['date', 'hour'])
y = pd.concat([y, sps]); y = y[~y.index.duplicated(keep='first')]
print(f"  y: {len(y)}, {y.index.get_level_values('date').min()} ~ "
      f"{y.index.get_level_values('date').max()}", flush=True)

def to_class5(spr, t1=T1, t2=T2):
    spr = np.asarray(spr, dtype=float)
    return np.where(spr < -t2, 0,
           np.where(spr < -t1, 1,
           np.where(spr <= t1, 2,
           np.where(spr <= t2, 3, 4))))

def subset(mask, ys):
    Xi = X.loc[mask]
    common = Xi.index.intersection(ys.index)
    return Xi.loc[common], ys.loc[common]

xd = X.index.get_level_values('date')
FIXED = {'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8,
         'colsample_bytree': 1.0, 'tree_method': 'hist'}

# ============ 评估函数 (与 v7.1 一致) ============
def evaluate(y_true_sp, y_pred_cls, y_pred_val, tag, verbose=True):
    y_true_sp = np.asarray(y_true_sp, dtype=float)
    y_true_cls = to_class5(y_true_sp)
    acc5 = accuracy_score(y_true_cls, y_pred_cls)
    nonneu = y_true_cls != 2
    pred_dir = np.sign(y_pred_cls - 2)
    true_dir = np.sign(np.where(y_true_sp == 0, 1e-9, y_true_sp))
    sign_hit = (pred_dir[nonneu] == true_dir[nonneu]).mean() if nonneu.sum() else np.nan
    coverage = (pred_dir[nonneu] != 0).mean() if nonneu.sum() else np.nan
    n_nonneu = int(nonneu.sum())
    ci = 1.96 * np.sqrt(sign_hit * (1 - sign_hit) / max(n_nonneu, 1)) if n_nonneu else np.nan
    yt_dev = (y_true_cls != 2).astype(int); yp_dev = (y_pred_cls != 2).astype(int)
    dev_f1 = f1_score(yt_dev, yp_dev, zero_division=0)
    yt_big = ((y_true_cls == 0) | (y_true_cls == 4)).astype(int)
    yp_big = ((y_pred_cls == 0) | (y_pred_cls == 4)).astype(int)
    big_f1 = f1_score(yt_big, yp_big, zero_division=0)
    big_rec = recall_score(yt_big, yp_big, zero_division=0)
    big_prec = precision_score(yt_big, yp_big, zero_division=0)
    big_mask = yt_big == 1
    big_sign = (pred_dir[big_mask] == true_dir[big_mask]).mean() if big_mask.sum() else np.nan
    cond_rmse = np.sqrt(mean_squared_error(y_true_sp[nonneu], y_pred_val[nonneu])) if nonneu.sum() else np.nan
    full_rmse = np.sqrt(mean_squared_error(y_true_sp, y_pred_val))
    m = dict(acc5=acc5, sign_hit=sign_hit, sign_ci=ci, coverage=coverage,
             dev_f1=dev_f1, big_f1=big_f1, big_recall=big_rec, big_precision=big_prec,
             big_sign=big_sign, cond_rmse=cond_rmse, full_rmse=full_rmse,
             n=len(y_true_sp), n_nonneu=n_nonneu)
    if verbose:
        print(f"  [{tag}] sign={sign_hit:.3f}±{ci:.3f} | big方向={big_sign:.3f} | "
              f"bigF1={big_f1:.3f}(R{big_rec:.2f}/P{big_prec:.2f}) | devF1={dev_f1:.3f} | "
              f"5类acc={acc5:.3f} | 条件RMSE={cond_rmse:.1f}", flush=True)
    return m

# ============ PHASE 2: 回测折叠规划 ============
print("\n[Phase 2] 规划回测折叠...", flush=True)
dates_ts = pd.to_datetime(np.array(sorted(y.index.get_level_values('date').unique())))
bt_start_ts = pd.Timestamp(BT_START); bt_end_ts = pd.Timestamp(BT_END)

fold_starts = []
cursor = bt_start_ts
while cursor <= bt_end_ts:
    cand = dates_ts[dates_ts >= cursor]
    if len(cand) == 0:
        break
    d0 = cand[0]
    if d0 > bt_end_ts:
        break
    fold_starts.append(d0)
    cursor = d0 + pd.Timedelta(days=STEP_DAYS)
print(f"  折叠数: {len(fold_starts)} (STEP={STEP_DAYS}d)", flush=True)

# ============ PHASE 3: 逐折训练 + 预测 ============
pred_csv = f'{YONON_PATH}/data/walkforward_v7.2_preds_{STEP_DAYS}d.csv'
done = set()
if os.path.exists(pred_csv):
    ex = pd.read_csv(pred_csv, dtype={'date': str})
    if 'date' in ex.columns:
        done = set(ex['date'].astype(str).unique())
        print(f"  断点续跑: 已有 {len(done)} 天预测, 跳过已完成折叠", flush=True)

print("\n[Phase 3] 逐折 walk-forward 训练 + 预测...", flush=True)
fp = open(pred_csv, 'a') if os.path.exists(pred_csv) else open(pred_csv, 'w')
if not os.path.exists(pred_csv) or len(done) == 0:
    fp.write('date,hour,spread_true,clf_cls,reg_val,base_val\n')

log_rows = []
t_start = time.time()
for fi, D in enumerate(fold_starts):
    D_s = D.strftime('%Y-%m-%d')
    f_end = D + pd.Timedelta(days=STEP_DAYS - 1)
    fold_dates = [d for d in np.array(sorted(y.index.get_level_values('date').unique()))
                  if D_s <= d <= min(f_end.strftime('%Y-%m-%d'), BT_END)]
    fold_dates = [d for d in fold_dates if d not in done]
    if not fold_dates:
        continue
    # 训练/早停窗口 (严格早于 D)
    v_start = (D - pd.Timedelta(days=VALID_DAYS)).strftime('%Y-%m-%d')
    tr_m = xd < D_s
    va_m = (xd >= v_start) & (xd < D_s)
    Xt, yt = subset(tr_m, y)
    Xv, yv = subset(va_m, y)
    yct = to_class5(yt); ycv = to_class5(yv)
    cnt = np.bincount(yct, minlength=NCLS)
    if np.all(cnt == 0):
        continue
    w = len(yct) / (NCLS * cnt)
    t_f0 = time.time()
    clf = xgb.XGBClassifier(n_estimators=N_EST, early_stopping_rounds=50,
        eval_metric='mlogloss', random_state=42, verbosity=0, n_jobs=JOBS, **FIXED)
    clf.fit(Xt, yct, sample_weight=w[yct], eval_set=[(Xv, ycv)], verbose=False)
    t_clf = time.time() - t_f0
    reg = xgb.XGBRegressor(n_estimators=N_EST, early_stopping_rounds=50,
        eval_metric='rmse', random_state=42, verbosity=0, n_jobs=JOBS, **FIXED)
    reg.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    t_reg = time.time() - t_f0 - t_clf
    base = xgb.XGBRegressor(n_estimators=N_EST, early_stopping_rounds=50,
        eval_metric='rmse', random_state=42, verbosity=0, n_jobs=JOBS, **FIXED)
    base.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    t_base = time.time() - t_f0 - t_clf - t_reg

    # 预测本折日期
    m_fo = (xd >= fold_dates[0]) & (xd <= fold_dates[-1])
    Xf, yf = subset(m_fo, y)
    yc_f = clf.predict(Xf); yv_f = reg.predict(Xf); yb_f = base.predict(Xf)
    out = pd.DataFrame({'date': Xf.index.get_level_values('date'),
                        'hour': Xf.index.get_level_values('hour'),
                        'spread_true': yf.values, 'clf_cls': yc_f,
                        'reg_val': yv_f, 'base_val': yb_f})
    out.to_csv(fp, header=False, index=False)
    fp.flush()
    done.update(fold_dates)
    # 本折即时指标
    m_fold = evaluate(yf.values, yc_f, yv_f, f"fold {D_s} ({len(Xf)}点)",
                      verbose=(fi % 3 == 0))
    log_rows.append({'fold': D_s, 'n': m_fold['n'], 'sign_hit': m_fold['sign_hit'],
                     'n_nonneu': m_fold['n_nonneu'],
                     'clf_best': clf.best_iteration, 'reg_best': reg.best_iteration,
                     't_clf': round(t_clf, 1), 't_reg': round(t_reg, 1),
                     't_base': round(t_base, 1)})
    el = time.time() - t_start
    print(f"    [fold {D_s}] train={len(Xt)} valid={len(Xv)} 预测{fold_dates[0]}~{fold_dates[-1]} "
          f"clf{clf.best_iteration}it/{t_clf:.0f}s reg{reg.best_iteration}it/{t_reg:.0f}s "
          f"base{base.best_iteration}it/{t_base:.0f}s | 累计 {el/60:.1f}min", flush=True)
fp.close()

# ============ PHASE 4: 汇总指标 ============
print("\n[Phase 4] 汇总全部 walk-forward 预测...", flush=True)
pred = pd.read_csv(pred_csv, dtype={'date': str, 'hour': str})
pred = pred.sort_values(['date', 'hour']).reset_index(drop=True)
print(f"  总预测样本: {len(pred)}", flush=True)

m_all = evaluate(pred['spread_true'].values, pred['clf_cls'].values,
                 pred['reg_val'].values, f"级联/walk-forward总{len(pred)}样本")
m_base = evaluate(pred['spread_true'].values, to_class5(pred['base_val'].values),
                  pred['base_val'].values, f"单回归基线/walk-forward总")

# 随机 + 多数类基线 (方向)
y_tc = to_class5(pred['spread_true'].values)
nonneu = y_tc != 2
tdir = np.sign(np.where(pred['spread_true'].values == 0, 1e-9, pred['spread_true'].values))
rng = np.random.default_rng(0)
rand_hit = (rng.choice([-1, 1], size=nonneu.sum()) == tdir[nonneu]).mean()
maj_hit = (tdir[nonneu] == 1).mean()   # 全猜正
print(f"  对照: 随机方向={rand_hit:.3f} | 全猜正={maj_hit:.3f} | 级联={m_all['sign_hit']:.3f}", flush=True)
sig = "显著 >0.5 ✓" if m_all['sign_hit'] - m_all['sign_ci'] > 0.5 else "未显著 >0.5 ✗"
print(f"  → walk-forward sign_hit {m_all['sign_hit']:.3f}±{m_all['sign_ci']:.3f} ({m_all['n_nonneu']}非neu): {sig}", flush=True)

# 月度分解
pred2 = pred.copy()
pred2['ym'] = pred2['date'].str[:7]
month_agg = {}
for ym, g in pred2.groupby('ym'):
    m_g = evaluate(g['spread_true'].values, g['clf_cls'].values, g['reg_val'].values,
                   f"{ym}", verbose=False)
    month_agg[ym] = {k: (None if pd.isna(v) else round(float(v), 3))
                     for k, v in m_g.items() if k in ('sign_hit', 'sign_ci', 'n_nonneu', 'n', 'big_sign')}
print("\n  月度 sign_hit 分解:")
for ym in sorted(month_agg):
    v = month_agg[ym]
    print(f"    {ym}: sign={v['sign_hit']}±{v['sign_ci']} (n_non={v['n_nonneu']}) "
          f"big方向={v['big_sign']}", flush=True)

# 与 v7.1 对比
print("\n  与 v7.1 静态划分对比 (sign_hit):")
print(f"    v7.1 test(144点)={0.463:.3f} | eval(120点)={0.495:.3f} | 大窗口含valid(1344)={0.619:.3f}")
print(f"    v7.2 walk-forward ({m_all['n']}点)={m_all['sign_hit']:.3f}±{m_all['sign_ci']:.3f}")

# ============ PHASE 5: 保存指标 + 图 ============
metrics = {
    'method': f'walk-forward STEP={STEP_DAYS}d VALID={VALID_DAYS}d n_est={N_EST}',
    'window': f'{BT_START} ~ {BT_END}',
    'n_folds': len(fold_starts),
    'cascade': {k: (None if pd.isna(v) else float(v)) for k, v in m_all.items()},
    'baseline_reg': {k: (None if pd.isna(v) else float(v)) for k, v in m_base.items()},
    'random_dir': float(rand_hit), 'majority_pos': float(maj_hit),
    'sign_hit_significant': bool(m_all['sign_hit'] - m_all['sign_ci'] > 0.5),
    'sign_hit': float(m_all['sign_hit']), 'sign_ci': float(m_all['sign_ci']),
    'n': int(m_all['n']), 'n_nonneu': int(m_all['n_nonneu']),
    'by_month': month_agg,
    'folds': log_rows,
    'v71_reference': {'test': 0.463, 'eval': 0.495, 'big_window_incl_valid': 0.619},
}
met_path = f'{YONON_PATH}/data/walkforward_v7.2_metrics_{STEP_DAYS}d.json'
with open(met_path, 'w') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"\n  指标已保存: {met_path}", flush=True)

# 月度 sign_hit 图
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ys = sorted(month_agg)
    hits = [month_agg[k]['sign_hit'] for k in ys]
    cis = [month_agg[k]['sign_ci'] for k in ys]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(range(len(ys)), hits, yerr=cis, fmt='o-', capsize=4)
    ax.axhline(0.5, color='r', ls='--', lw=1, label='随机 0.5')
    ax.set_xticks(range(len(ys))); ax.set_xticklabels(ys, rotation=45)
    ax.set_ylabel('sign_hit'); ax.set_title(f'walk-forward 月度 sign_hit (±95%CI, {len(pred)}样本)')
    ax.legend(); ax.grid(alpha=0.3)
    out_png = f'{YONON_PATH}/show/v7/report_walkforward_sign.png'
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()
    print(f"  图已保存: {out_png}", flush=True)
except Exception as e:
    print(f"  图生成跳过: {e}", flush=True)

print("\n" + "=" * 72)
print(f"  v7.2 walk-forward 完成 | 总 {m_all['n']} 样本 | "
      f"sign_hit {m_all['sign_hit']:.3f}±{m_all['sign_ci']:.3f} ({sig})")
print(f"  大窗口 v7.1 0.619 是否真实 → 干净 walk-forward 答案见上")
print("=" * 72)
