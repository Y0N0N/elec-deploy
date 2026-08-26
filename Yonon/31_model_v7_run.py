"""
Pipeline v7.1: DA-RT 价差预测 (5 类双阈值 + 大窗口验证)

变更 (2026-08-06 用户拍板, Q108 方向 = C变种 + D):
  v7.0: 3 类 (pos/neu/neg), sign 命中 0.546 近随机
  v7.1: 5 类双阈值 + 大窗口统计验证

任务:
  label: spread(d,h) = 日前统一结算价 − 实时统一结算价 (元/MWh)
  5 类区间 (双阈值 τ_minor=5, τ_big=15, 见 config):
    0 big_neg (spread < -15) | 1 neg (-15~-5) | 2 neu (±5) | 3 pos (5~15) | 4 big_pos (>15)
  输出契约:
    |spread| <= 5  (neu)            → 无差别/可容忍误差, 不输出
    5 < |spread| <= 15 (neg/pos)    → 【有偏差预警】+ 输出具体值
    |spread| > 15 (big_neg/big_pos) → 【大偏差预警】+ 输出具体值

  主指标: sign 命中率 (非 neu 样本方向正确率)
  次指标: 大偏差(big_neg+big_pos) F1, 有偏差(非neu) F1, 条件 RMSE

  D 大窗口验证: 除 test(6天)/eval(5天) 外, 增加 valid+test+eval 合并的
               51 天回测窗口 (1224 样本), 使 sign_hit 的 95%CI 足够窄以判断 >0.5

特征: 369 新鲜集 (v7_fresh_features.json, 已剔除 price_d* 泄露)
      供给侧 s_*/日内 h_*/约束 gc_*/时间 + sp_wow, 全预测期可算, 无递归无泄露
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, json, gc, warnings, joblib
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.metrics import (mean_squared_error, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)
from config.config import (dataset_path, spread_label_file, da_price_latest,
                           rt_price_latest, SPREAD_THRESHOLD, SPREAD_THRESHOLD_BIG,
                           SPREAD_CLASSES, SPREAD_CLASS_MAP, YONON_PATH)

T1, T2 = SPREAD_THRESHOLD, SPREAD_THRESHOLD_BIG
NCLS = len(SPREAD_CLASSES)
print("=" * 70)
print("  Pipeline v7.1: DA-RT 价差 5 类双阈值 + 大窗口验证")
print(f"  τ_minor={T1} (有偏差) | τ_big={T2} (大偏差) | 类别: {SPREAD_CLASSES}")
print("=" * 70)

# ============ PHASE 0: 加载特征 ============
print("\n[Phase 0] 加载新鲜集特征...")
with open(f'{YONON_PATH}/data/v7_fresh_features.json') as f:
    fresh = json.load(f)['fresh_features']
print(f"  新鲜集特征: {len(fresh)} 列")
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
print(f"  X: {X.shape}")

# ============ PHASE 1: 标签 ============
print("\n[Phase 1] 构建标签 (spread + 5 类)...")
spread = pd.read_feather(spread_label_file)
spread.index = pd.to_datetime(spread.index)
y_spread = spread.stack(); y_spread.index = y_spread.index.rename(['date', 'hour'])
y_spread.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(y_spread.index.get_level_values('date')).strftime('%Y-%m-%d'),
     y_spread.index.get_level_values('hour')], names=['date', 'hour'])
y_spread.name = 'spread'

def to_class5(spr, t1=T1, t2=T2):
    """spread → 5 类: 0 big_neg, 1 neg, 2 neu, 3 pos, 4 big_pos"""
    spr = np.asarray(spr, dtype=float)
    return np.where(spr < -t2, 0,
           np.where(spr < -t1, 1,
           np.where(spr <= t1, 2,
           np.where(spr <= t2, 3, 4))))

# eval 期真实 spread
da_e = pd.read_feather(da_price_latest); rt_e = pd.read_feather(rt_price_latest)
sp_eval_s = (da_e - rt_e).stack()
sp_eval_s.index = sp_eval_s.index.rename(['date', 'hour'])
sp_eval_s.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(sp_eval_s.index.get_level_values('date')).strftime('%Y-%m-%d'),
     sp_eval_s.index.get_level_values('hour')], names=['date', 'hour'])

# ============ PHASE 2: 数据划分 ============
print("\n[Phase 2] 数据划分...")
xd = X.index.get_level_values('date')
tr_m = (xd >= '2025-01-01') & (xd <= '2026-05-31')
va_m = (xd >= '2026-06-01') & (xd <= '2026-07-15')
te_m = (xd >= '2026-07-16') & (xd <= '2026-07-21')
ev_m = (xd >= '2026-07-22') & (xd <= '2026-07-26')
pr_m = (xd == '2026-07-27')

def subset(mask, ys):
    Xi = X.loc[mask]
    common = Xi.index.intersection(ys.index)
    return Xi.loc[common], ys.loc[common]

X_tr, y_tr = subset(tr_m, y_spread)
X_va, y_va = subset(va_m, y_spread)
X_te, y_te = subset(te_m, y_spread)
X_ev, y_ev = subset(ev_m, sp_eval_s)
X_pr = X.loc[pr_m]
print(f"  train {len(X_tr)} | valid {len(X_va)} | test {len(X_te)} | eval {len(X_ev)} | pred {len(X_pr)}")

yc_tr = to_class5(y_tr); yc_va = to_class5(y_va)
dist = pd.Series(yc_tr).value_counts().sort_index()
print("  训练集 5 类分布: " + ", ".join(f"{SPREAD_CLASSES[i]}={dist.get(i,0)/len(yc_tr)*100:.1f}%" for i in range(NCLS)))

FIXED = {'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 1.0}

# ============ 评估函数 ============
def evaluate(y_true_sp, y_pred_cls, y_pred_val, tag, verbose=True):
    y_true_sp = np.asarray(y_true_sp, dtype=float)
    y_true_cls = to_class5(y_true_sp)
    acc5 = accuracy_score(y_true_cls, y_pred_cls)
    # sign 命中 (主): 非 neu 样本方向正确率
    nonneu = y_true_cls != 2
    pred_dir = np.sign(y_pred_cls - 2)          # 0,1→-; 2→0; 3,4→+
    true_dir = np.sign(np.where(y_true_sp == 0, 1e-9, y_true_sp))
    sign_hit = (pred_dir[nonneu] == true_dir[nonneu]).mean() if nonneu.sum() else np.nan
    coverage = (pred_dir[nonneu] != 0).mean() if nonneu.sum() else np.nan
    n_nonneu = int(nonneu.sum())
    # sign_hit 95% CI (正态近似)
    ci = 1.96 * np.sqrt(sign_hit*(1-sign_hit)/max(n_nonneu,1)) if n_nonneu else np.nan
    # 有偏差 (非 neu) F1
    yt_dev = (y_true_cls != 2).astype(int); yp_dev = (y_pred_cls != 2).astype(int)
    dev_f1 = f1_score(yt_dev, yp_dev, zero_division=0)
    # 大偏差 (big_neg+big_pos) F1
    yt_big = ((y_true_cls == 0) | (y_true_cls == 4)).astype(int)
    yp_big = ((y_pred_cls == 0) | (y_pred_cls == 4)).astype(int)
    big_f1 = f1_score(yt_big, yp_big, zero_division=0)
    big_rec = recall_score(yt_big, yp_big, zero_division=0)
    big_prec = precision_score(yt_big, yp_big, zero_division=0)
    # 大偏差方向命中 (big 样本内方向正确率 — 最关键业务指标)
    big_mask = yt_big == 1
    big_sign = (pred_dir[big_mask] == true_dir[big_mask]).mean() if big_mask.sum() else np.nan
    # 条件 RMSE (非 neu)
    cond_rmse = np.sqrt(mean_squared_error(y_true_sp[nonneu], y_pred_val[nonneu])) if nonneu.sum() else np.nan
    full_rmse = np.sqrt(mean_squared_error(y_true_sp, y_pred_val))
    m = dict(acc5=acc5, sign_hit=sign_hit, sign_ci=ci, coverage=coverage,
             dev_f1=dev_f1, big_f1=big_f1, big_recall=big_rec, big_precision=big_prec,
             big_sign=big_sign, cond_rmse=cond_rmse, full_rmse=full_rmse,
             n=len(y_true_sp), n_nonneu=n_nonneu)
    if verbose:
        print(f"  [{tag}] sign={sign_hit:.3f}±{ci:.3f} | big方向={big_sign:.3f} | "
              f"bigF1={big_f1:.3f}(R{big_rec:.2f}/P{big_prec:.2f}) | devF1={dev_f1:.3f} | "
              f"5类acc={acc5:.3f} | 条件RMSE={cond_rmse:.1f}")
    return m

# ============ PHASE 3: T104 基线 (单回归, 5 类由 ŷ 导出) ============
print("\n[Phase 3] T104 基线: 单 XGBRegressor ...")
reg_base = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50,
    eval_metric='rmse', random_state=42, verbosity=0, n_jobs=-1, **FIXED)
reg_base.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
print(f"  best_iteration: {reg_base.best_iteration}")
m_base = {}
for tag, Xs, ys in [('Test', X_te, y_te), ('Eval', X_ev, y_ev)]:
    yh = reg_base.predict(Xs)
    m_base[tag] = evaluate(ys.values, to_class5(yh), yh, f"基线/{tag}")

# ============ PHASE 4: T105 级联 (5 类分类头 + 回归头) ============
print("\n[Phase 4] T105 级联: XGBClassifier(5类) + XGBRegressor ...")
cls_counts = np.bincount(yc_tr, minlength=NCLS)
w_per_cls = len(yc_tr) / (NCLS * cls_counts)
sw_tr = w_per_cls[yc_tr]
print(f"  类别权重: {dict(zip(SPREAD_CLASSES, w_per_cls.round(2)))}")

clf = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=50,
    eval_metric='mlogloss', random_state=42, verbosity=0, n_jobs=-1, **FIXED)
clf.fit(X_tr, yc_tr, sample_weight=sw_tr, eval_set=[(X_va, yc_va)], verbose=False)
print(f"  clf best_iteration: {clf.best_iteration}")

reg_cas = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50,
    eval_metric='rmse', random_state=42, verbosity=0, n_jobs=-1, **FIXED)
reg_cas.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
print(f"  reg best_iteration: {reg_cas.best_iteration}")

m_cas = {}
for tag, Xs, ys in [('Test', X_te, y_te), ('Eval', X_ev, y_ev)]:
    m_cas[tag] = evaluate(ys.values, clf.predict(Xs), reg_cas.predict(Xs), f"级联/{tag}")

# ============ PHASE 5: D 大窗口验证 (valid+test+eval, 51 天) ============
print("\n[Phase 5] D 大窗口验证 (valid+test+eval 合并, 统计显著性) ...")
# 合并三个窗口的真实 spread (valid/test 用 y_spread, eval 用 sp_eval_s)
X_big = pd.concat([X_va, X_te, X_ev])
y_big = pd.concat([y_va, y_te, y_ev])
yc_big = clf.predict(X_big)
yv_big = reg_cas.predict(X_big)
m_big = evaluate(y_big.values, yc_big, yv_big, f"级联/大窗口{len(X_big)}样本")

# 随机基线对照 (方向瞎猜)
rng = np.random.default_rng(0)
y_true_cls_big = to_class5(y_big.values)
nonneu_big = y_true_cls_big != 2
true_dir_big = np.sign(np.where(y_big.values == 0, 1e-9, y_big.values))
rand_dir = rng.choice([-1, 1], size=nonneu_big.sum())
rand_hit = (rand_dir == true_dir_big[nonneu_big]).mean()
# 多数类基线 (全猜 pos, 训练集 pos 占多数)
maj_hit = (true_dir_big[nonneu_big] == 1).mean()
print(f"  对照: 随机方向={rand_hit:.3f} | 全猜正={maj_hit:.3f} | 级联={m_big['sign_hit']:.3f}")
sig = "显著 >0.5 ✓" if m_big['sign_hit'] - m_big['sign_ci'] > 0.5 else "未显著 >0.5 ✗"
print(f"  → sign_hit {m_big['sign_hit']:.3f}±{m_big['sign_ci']:.3f}: {sig}")

# ============ PHASE 6: 混淆矩阵 + 预测期输出 ============
print("\n[Phase 6] 级联 Eval 混淆矩阵 (行=真实, 列=预测):")
cm = confusion_matrix(to_class5(y_ev.values), clf.predict(X_ev), labels=list(range(NCLS)))
print(pd.DataFrame(cm, index=SPREAD_CLASSES, columns=SPREAD_CLASSES).to_string())

print("\n  预测期 07-27 输出 (级联):")
yc_pr = clf.predict(X_pr); yv_pr = reg_cas.predict(X_pr)
lvl = {0: '🔴大负', 1: '🟡负', 2: '⚪无', 3: '🟡正', 4: '🔴大正'}
pred_out = pd.DataFrame({'cls': yc_pr, 'value': yv_pr}, index=X_pr.index)
pred_out['预警'] = pred_out['cls'].map(lvl)
pred_out['输出值'] = np.where(pred_out['cls'] != 2, pred_out['value'].round(2), np.nan)
print(pred_out.groupby('预警')['value'].agg(['count', 'mean']).round(2).to_string())
print(f"  触发预警+输出 (非neu): {(pred_out['cls'] != 2).sum()}/{len(pred_out)} 小时")

# ============ PHASE 7: 特征重要性 ============
print("\n[Phase 7] 分类头特征重要性 Top10:")
imp = pd.Series(clf.feature_importances_, index=fresh).sort_values(ascending=False)
print(imp.head(10).round(4).to_string())
def cat_of(c):
    if c.startswith('sp_'): return 'sp_wow'
    if c.startswith('gc_'): return '电网约束'
    if c.startswith('h_'): return '日内'
    if c.startswith('s_'): return '供给侧'
    return '其他'
imp_cat = imp.groupby(imp.index.map(cat_of)).sum().sort_values(ascending=False)
print("  按类别: " + ", ".join(f"{k}={v/imp.sum()*100:.1f}%" for k, v in imp_cat.items()))

# ============ PHASE 8: 保存 ============
save_path = f'{YONON_PATH}/models/xgb_v7.joblib'
joblib.dump({
    'clf': clf, 'reg': reg_cas, 'baseline_reg': reg_base,
    'features': fresh,
    'threshold_minor': T1, 'threshold_big': T2,
    'classes': SPREAD_CLASSES, 'class_map': SPREAD_CLASS_MAP,
    'decision_rule': "clf 5类; neu(±5)不输出, neg/pos(5~15)有偏差预警+值, big(>15)大偏差预警+值",
    'fixed_params': FIXED,
    'metrics': {'baseline': m_base, 'cascade': m_cas, 'big_window': m_big},
    'sign_hit_significant': bool(m_big['sign_hit'] - m_big['sign_ci'] > 0.5),
}, save_path)
print(f"\n  模型保存: {save_path}")

print(f"\n{'='*70}")
print(f"  v7.1 完成 | sign_hit 大窗口={m_big['sign_hit']:.3f}±{m_big['sign_ci']:.3f} ({sig})")
print(f"  级联 Eval: sign={m_cas['Eval']['sign_hit']:.3f} big方向={m_cas['Eval']['big_sign']:.3f} bigF1={m_cas['Eval']['big_f1']:.3f}")
print(f"{'='*70}")
