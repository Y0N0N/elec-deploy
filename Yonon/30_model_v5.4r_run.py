"""
Pipeline v5.4-R: 残差预测 (Residual Forecasting)

背景 (用户审查 v5.4 发现的问题):
1. p=1.0 全退化价格因子的致命漏洞:
   - 退化值 = 同星期几均值 → 因子在训练集内变成星期几虚拟变量
     (max7 唯一值 5038 → 116, std 128 → 69), 近期市场情绪/连涨连跌信息全丢
   - 树模型遇常数特征分裂增益为 0, 重要性归零 — v5.4 中价格因子仅 4.3% 正是此现象
   - 若预测期用 ffill 而训练期用同星期几均值 → 新的 Covariate Shift
2. 绝对价格预测不合适 → 应转为价格偏离 (残差) 预测

v5.4-R 方案:
- 基线 (趋势外推):  baseline(d,h) = P(d-1,h) + [P(d-7,h) - P(d-14,h)]
                         昨天价格   + (上周同日 - 前周同日)
- 目标改换: 模型预测 y' = P(d,h) - baseline(d,h)  (偏离/残差)
- 特征保留: 预测期新鲜集 (供给侧 s_* + 日内 h_* + 约束 gc_* + 时间)
            + 2 个可算价格状态特征 (周同比, 只用历史真实价格, 无递归)
- 推理: P(d,h) = baseline(d,h) + 模型残差; Day2+ 的 P(d-1) 用前日预测 (递归基线)
- 供给侧: 新鲜因子主力不变 (模型学"为什么今天比基线贵/便宜")

为什么更好:
- 训练时模型学增量原因, 不死记价格水平
- 推理时基线自带趋势外推, 模型输出 0 (不偏离) 也有合理水平,
  供给侧因子负责非零的日间波动
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, os, gc, warnings, joblib
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from config.config import *

print("=" * 65)
print("  Pipeline v5.4-R: 残差预测 (基线 + 偏离)")
print("=" * 65)

LAST_LABELED = pd.Timestamp('2026-07-21')

# ============ PHASE 0: 加载因子 ============
print("\n[Phase 0] 加载因子...")
factor_files = sorted(os.listdir(dataset_path))
fl = []
for fn in factor_files:
    df = pd.read_feather(f'{dataset_path}/{fn}')
    s = df.stack(); s.name = fn.replace('.fea', '')
    fl.append(s)
X = pd.concat(fl, axis=1)
del fl; gc.collect()

# 统一索引格式
idx_dates = pd.to_datetime(X.index.get_level_values('date')).strftime('%Y-%m-%d')
idx_hours_raw = X.index.get_level_values('hour')
idx_hours = [f'{int(h):02d}:00' if isinstance(h, (int, np.integer)) else str(h)
             for h in idx_hours_raw]
X.index = pd.MultiIndex.from_arrays([idx_dates, idx_hours], names=['date', 'hour'])
X = X.sort_index()
print(f"  X: {X.shape}")

# ============ PHASE 1: 特征选择 (预测期新鲜集) ============
print("\n[Phase 1] 特征选择...")

# 步骤1: 剔除训练期 NaN>50% 的列 (假新鲜字段)
train_na_ratio = X.loc['2025-01-01':'2026-05-31'].isna().mean()
drop_cols = train_na_ratio[train_na_ratio > 0.5].index.tolist()
if drop_cols:
    X = X.drop(columns=drop_cols)
    print(f"  剔除训练期 NaN>50%: {len(drop_cols)} 列")

# 步骤2: 预测期新鲜集 = 预测期无 NaN 且非价格因子
pred_na = X.loc['2026-07-22':'2026-07-27'].isna()

def is_price_factor(c):
    window_ops = ('ma', 'std', 'max', 'min', 'rank', 'qtlu', 'qtld',
                  'cntp', 'cntn', 'psy', 'sump', 'sumn', 'roc',
                  'corr', 'cord', 'cov')
    prefix_price = c.startswith(window_ops) and not c.startswith(('s_', 'h_', 'gc_'))
    return prefix_price or c in ('cs_dev_price', 'cs_rank_price', 'cs_position_price')

fresh_cols = [c for c in X.columns if not pred_na[c].any() and not is_price_factor(c)]
price_factor_cols = [c for c in X.columns if is_price_factor(c)]
print(f"  新鲜集: {len(fresh_cols)} 列 (供给侧/日内/约束/时间)")
print(f"  剔除价格因子: {len(price_factor_cols)} 列")

X = X[fresh_cols]

# ============ PHASE 2: 价格状态特征 (预测期可算, 无递归) ============
print("\n[Phase 2] 价格状态特征...")
# 只依赖历史真实价格 (d-7, d-14), 预测期全真实可算
price_df = pd.read_feather(f"{matrix_path}/实际运行结果-用电侧/日前统一结算价.feather")
price_wide = price_df.copy()
price_wide.index = pd.to_datetime(price_wide.index)
price_wide = price_wide.sort_index()

# 宽表: date × hour → P(d-7) - P(d-14) 的周同比
def add_price_state_features(X, price_wide):
    """周同比变化率 + 绝对差 (只用 d-7, d-14 真实价格, 与基线一致)"""
    dates = pd.to_datetime(X.index.get_level_values('date'))
    hours = X.index.get_level_values('hour')
    p_wow_rate, p_wow_abs = np.full(len(X), np.nan), np.full(len(X), np.nan)
    for i, (d, h) in enumerate(zip(dates, hours)):
        d7 = d - pd.Timedelta(days=7)
        d14 = d - pd.Timedelta(days=14)
        if d7 in price_wide.index and d14 in price_wide.index and h in price_wide.columns:
            p7 = price_wide.loc[d7, h]
            p14 = price_wide.loc[d14, h]
            if not (pd.isna(p7) or pd.isna(p14)):
                p_wow_rate[i] = (p7 - p14) / (abs(p14) + 1e-6)
                p_wow_abs[i] = p7 - p14
    X['p_wow_rate'] = p_wow_rate   # 周同比变化率
    X['p_wow_abs'] = p_wow_abs     # 周同比绝对差
    return X

X = add_price_state_features(X, price_wide)
print(f"  特征: {X.shape[1]} (含 p_wow_rate, p_wow_abs)")
print(f"  预测期 p_wow_rate NaN: {X.loc['2026-07-22':'2026-07-27', 'p_wow_rate'].isna().sum()}/144")

# ============ PHASE 3: 基线 + 残差目标 ============
print("\n[Phase 3] 基线计算 + 残差目标...")

def make_price_map(price_wide):
    """(date, hour) → P 映射 (真实价格)"""
    s = price_wide.stack()
    s.index = s.index.rename(['date', 'hour'])
    return s

def calc_baseline(price_map, dates, hours):
    """
    baseline(d,h) = P(d-1) + [P(d-7) - P(d-14)]
    注: 3 周均值平滑曾尝试 (Eval 77.5 更差) — 07-09~13 高价期 vs 07-16~18 低价期
        是结构性 regime 切换而非单点尖峰, 平滑引入更多噪声。单点差分最优。
    """
    baselines = np.full(len(dates), np.nan)
    for i, (d, h) in enumerate(zip(dates, hours)):
        d1 = d - pd.Timedelta(days=1)
        d7 = d - pd.Timedelta(days=7)
        d14 = d - pd.Timedelta(days=14)
        try:
            p1 = price_map.loc[(d1, h)]
            p7 = price_map.loc[(d7, h)]
            p14 = price_map.loc[(d14, h)]
            if not (pd.isna(p1) or pd.isna(p7) or pd.isna(p14)):
                baselines[i] = p1 + (p7 - p14)
        except KeyError:
            pass
    return baselines

# 训练/验证/测试: 基线全用真实价格
label_df = pd.read_feather(f'{label_path}/label.feather')
y_true = label_df.stack(); y_true.index = y_true.index.rename(['date', 'hour']); y_true.name = 'price'

# 注意: y_true 的 hour 可能是 '00:00' 格式; 统一
price_map = make_price_map(price_wide)

# 切分 (与之前一致)
xd = X.index.get_level_values('date'); yd = y_true.index.get_level_values('date')
ld = yd.unique(); ad = xd.unique(); prd = ad.difference(ld).sort_values()

tr_m = (xd >= '2025-01-01') & (xd <= '2026-05-31')
va_m = (xd >= '2026-06-01') & (xd <= '2026-07-15')
te_m = (xd >= '2026-07-16') & (xd <= '2026-07-21')
pr_m = xd.isin(prd)

X_tr = X.loc[tr_m].copy(); X_va = X.loc[va_m].copy()
X_te = X.loc[te_m].copy(); X_pr = X.loc[pr_m].copy()

def gy(s):
    c = s.index.intersection(y_true.index)
    return y_true.loc[c].to_frame(name='price')  # DataFrame, 可加 baseline/residual 列
y_tr = gy(X_tr); y_va = gy(X_va); y_te = gy(X_te)
X_tr = X_tr.loc[y_tr.index]; X_va = X_va.loc[y_va.index]; X_te = X_te.loc[y_te.index]

# 基线 + 残差 (训练/验证/测试: 真实价格)
def to_dt(index):
    return pd.to_datetime(index.get_level_values('date')), index.get_level_values('hour')

for D, yD in [(X_tr, y_tr), (X_va, y_va), (X_te, y_te)]:
    dates, hours = to_dt(D.index)
    base = calc_baseline(price_map, dates, hours)
    yD['baseline'] = base
    yD['residual'] = yD['price'] - base

# 缺失基线的样本 (早期数据 d-14 无值) 剔除
for D, yD in [(X_tr, y_tr), (X_va, y_va), (X_te, y_te)]:
    keep = yD['residual'].notna()
    print(f"  {len(yD)} → 基线可用 {keep.sum()} (剔除 {len(yD)-keep.sum()})")
    yD.dropna(subset=['residual'], inplace=True)
    D.drop(D.index.difference(yD.index), inplace=True)

# 预测集基线: 递归 (Day1 用真实 P(07-21), Day2+ 用前日预测)
def calc_recursive_baseline(price_map_hist, X_pr, model_fn, feature_cols):
    """递归基线 + 预测: P(d,h) = baseline(d,h) + model(X[d,h])"""
    price_map_rec = price_map_hist.copy()  # 真实价格映射 (到 07-21)
    preds = {}
    for d in sorted(prd):
        d_ts = pd.Timestamp(d)
        for h in X_pr.index.get_level_values('hour').unique():
            key = (d, h)
            # 基线 (P(d-1) 可能来自递归预测)
            p1 = price_map_rec.get((d_ts - pd.Timedelta(days=1), h), np.nan)
            p7 = price_map_hist.get((d_ts - pd.Timedelta(days=7), h), np.nan)
            p14 = price_map_hist.get((d_ts - pd.Timedelta(days=14), h), np.nan)
            if pd.isna(p1) or pd.isna(p7) or pd.isna(p14):
                continue
            baseline = p1 + (p7 - p14)
            # 特征
            feat_row = X_pr.loc[key, feature_cols].values.reshape(1, -1)
            resid = model_fn(feat_row)[0]
            pred = baseline + resid
            preds[key] = pred
            price_map_rec[(d_ts, h)] = pred
    return preds

# ============ PHASE 4: 训练 (预测残差) ============
print("\n[Phase 4] 训练 XGBoost (预测残差)...")
fixed_params = {'max_depth': 8, 'learning_rate': 0.05,
                'subsample': 0.8, 'colsample_bytree': 1.0}
feature_cols = X_tr.columns.tolist()

model = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50,
    eval_metric='rmse', random_state=42, verbosity=0, n_jobs=-1, **fixed_params)
model.fit(X_tr, y_tr['residual'], eval_set=[(X_va, y_va['residual'])], verbose=False)

# Test 评估: 预测残差 → 加基线 → 绝对价格
yp_te_res = model.predict(X_te)
yp_te = y_te['baseline'].values + yp_te_res
rmse_te = np.sqrt(mean_squared_error(y_te['price'], yp_te))
r2_te = r2_score(y_te['price'], yp_te)
mape_te = np.mean(np.abs((y_te['price'].values - yp_te) / (np.abs(y_te['price'].values) + 1e-8))) * 100
print(f"  Test (绝对价格): RMSE={rmse_te:.2f}, R²={r2_te:.4f}, MAPE={mape_te:.2f}%")

# 基线单独的表现 (不加模型)
rmse_base_te = np.sqrt(mean_squared_error(y_te['price'], y_te['baseline']))
print(f"  基线单独 RMSE: {rmse_base_te:.2f} (残差目标使绝对 RMSE {rmse_te:.2f} → {rmse_base_te:.2f}, "
      f"模型改善 {100*(1-rmse_te/rmse_base_te):.0f}%)")

# ============ PHASE 5: 预测 + 独立评估 (递归基线) ============
print("\n[Phase 5] 独立评估 (递归基线 + 残差)...")

# 真实价格映射 (到 07-21) — 用宽表
price_map_hist = {}
for d in price_wide.index:
    for h in price_wide.columns:
        v = price_wide.loc[d, h]
        if not pd.isna(v):
            price_map_hist[(d, h)] = v

preds = calc_recursive_baseline(price_map_hist, X_pr, model.predict, feature_cols)

# 与真实 label 对比
new_label = pd.read_feather(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', '日前统一结算价.feather'))
y_new = new_label.stack(); y_new.index = y_new.index.rename(['date', 'hour'])
y_new.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(y_new.index.get_level_values('date')).strftime('%Y-%m-%d'),
     y_new.index.get_level_values('hour')], names=['date', 'hour'])

# 只在有真实 label 的日期上评估
y_new_keys = set(y_new.index)
ev_dates = [d for d in prd if (d, '00:00') in preds.keys() and any((d, h) in y_new_keys for h in ['00:00'])]
errs_all = []
print(f"  独立评估日期: {ev_dates}")
for d in ev_dates:
    hs = sorted(set(h for (dd, h) in preds if dd == d and (d, h) in y_new_keys))
    pred_vals = [preds[(d, h)] for h in hs]
    act_vals = [y_new.loc[(d, h)] for h in hs]
    errs = np.array(pred_vals) - np.array(act_vals)
    errs_all.extend(errs.tolist())
    rmse_d = np.sqrt(np.mean(errs ** 2))
    print(f"    {d}: Pred={np.mean(pred_vals):.1f}, Actual={np.mean(act_vals):.1f}, Bias={np.mean(errs):+.1f}, RMSE={rmse_d:.1f}")

rmse_ev = np.sqrt(np.mean(np.array(errs_all) ** 2))
print(f"\n  Eval RMSE: {rmse_ev:.2f}, Test/Eval Ratio: {rmse_ev/rmse_te:.1f}x")

# ============ PHASE 6: 跨天区分度 ============
print("\n[Phase 6] 跨天预测区分度...")
pw = pd.DataFrame({'price': [preds[(d, h)] for d, h in sorted(preds.keys())]},
                  index=pd.MultiIndex.from_tuples(sorted(preds.keys()), names=['date', 'hour']))
pw = pw['price'].unstack()
corrs = []
for i, d in enumerate(pw.index):
    r = pw.loc[d]
    print(f"    {d}: avg={r.mean():.1f}, min={r.min():.1f}, max={r.max():.1f}")
    if i > 0:
        c = r.corr(pw.iloc[i - 1])
        corrs.append(c)
        print(f"         vs {pw.index[i-1]}: corr={c:.4f}")
mean_corr = np.mean(corrs) if corrs else 1.0
print(f"  跨天平均 corr: {mean_corr:.4f}")

# ============ PHASE 7: 特征重要性 ============
print("\n[Phase 7] 特征重要性分布...")
imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"  Top 5: {imp.head(5).round(4).to_dict()}")

def cat_of(c):
    if c.startswith(('p_wow')): return '价格状态'
    if c.startswith('gc_'): return '电网约束'
    if c.startswith('h_'): return '日内'
    if c.startswith('s_'): return '供给侧纵向'
    return '其他'
imp_cat = imp.groupby(imp.index.map(cat_of)).sum().sort_values(ascending=False)
print("  按类别:")
for k, v in imp_cat.items():
    print(f"    {k}: {v:.4f} ({v/imp.sum()*100:.1f}%)")

# ============ PHASE 8: 保存 (自包含) ============
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'xgb_v5.4r.joblib')
joblib.dump({
    'model': model,
    'features': feature_cols,
    'fixed_params': fixed_params,
    'baseline_formula': "baseline(d,h) = P(d-1,h) + [P(d-7,h) - P(d-14,h)]; 预测期 P(d-1) 递归",
    'baseline_note': "3周均值平滑曾尝试 (Eval 77.5 更差): 07-09~13高价期 vs 07-16~18低价期是 regime 切换, 非单点尖峰",
    'target': "residual = P - baseline",
    'prediction': "P_pred = baseline + model_residual",
    'price_state_features': ['p_wow_rate', 'p_wow_abs'],
    'test_rmse': rmse_te,
    'test_r2': r2_te,
    'test_mape': mape_te,
    'eval_rmse': rmse_ev,
    'test_eval_ratio': rmse_ev / rmse_te,
    'cross_day_corr': mean_corr,
}, save_path)
print(f"\n  模型保存: {save_path}")

print(f"\n{'='*65}")
print(f"  v5.4-R: Test RMSE={rmse_te:.2f} | Eval RMSE={rmse_ev:.2f} | Ratio={rmse_ev/rmse_te:.1f}x | 跨天corr={mean_corr:.4f}")
print(f"{'='*65}")
