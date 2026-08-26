"""
Pipeline v5: 模拟因子退化 (Data Augmentation) + 非价格预测信号

v4 核心矛盾: 训练时价格因子永远新鲜 (days_since_label=0), 预测时全部退化
            → Test RMSE 3.88 但 Eval RMSE ~80 (20x 差距), 预测是"复读机"

v5 两个抓手:
  1. 模拟因子退化: 训练时随机"老化"退化列 (预测期有 NaN 的列),
     让模型学会处理 stale 因子, 不再过度依赖价格因子
  2. 非价格预测信号: 电网约束因子 gc_* (新增) + 日内因子 h_* + 供给侧 s_*
     (供给侧预测数据覆盖到 07-27, 预测期完全新鲜!)

退化设计:
  - 退化列 = 预测期 (07-22~27) 有 NaN 的列 (价格/负荷因子自动识别)
  - 每个训练样本以 p_degrade 概率退化, k ∈ [1,3,5,7] 天
  - 退化方式: 用 k 天前同小时的真实因子值替换 (精确模拟 staleness)
  - 新特征 staleness: 训练时=退化天数k(或0), 预测时=距最后label的天数
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, os, gc, glob, warnings, joblib
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from config.config import *

print("=" * 65)
print("  Pipeline v5: 模拟因子退化 + 非价格信号")
print("=" * 65)

RNG = np.random.RandomState(42)
P_DEGRADE = 1.0          # 退化概率 (v5.4: 全样本退化, 训练时只见过预测期因子形态)
DEGRADE_KS = [1, 3, 5, 7]  # 退化天数选项
LAST_LABELED = pd.Timestamp('2026-07-21')

# v5.3 模式: 只用"预测期新鲜集" (预测期无 NaN 的列 = 供给侧/日内/约束/时间)
# 原理: 价格因子占 78% 重要性但预测期全部退化(填充), 供给侧因子在预测期
#       有真实区分度 (统调负荷 07-22:130757 → 07-27:122387 MW) 却被遮蔽
# 等价于两阶段模型的 Stage1: 让模型被迫学习 供给侧→价格 映射
FRESH_ONLY = False       # True = 只用新鲜集 (v5.3), False = 全因子
DEGRADE_ALL = True       # v5.4: 全样本退化 — 价格因子也替换为同星期几填充值
                         #       训练时模型只见过"预测期因子形态", 训练/预测分布完全对齐

def is_price_factor(c):
    """价格因子识别 (窗口算子但非 s_/h_/gc_ 前缀 + 价格截面因子)"""
    window_ops = ('ma', 'std', 'max', 'min', 'rank', 'qtlu', 'qtld',
                  'cntp', 'cntn', 'psy', 'sump', 'sumn', 'roc',
                  'corr', 'cord', 'cov')
    prefix_price = c.startswith(window_ops) and not c.startswith(('s_', 'h_', 'gc_'))
    return prefix_price or c in ('cs_dev_price', 'cs_rank_price', 'cs_position_price')

# ============ PHASE 0: 加载因子 (包含新 gc_* 约束因子) ============
print("\n[Phase 0] 加载因子...")
factor_files = sorted(os.listdir(dataset_path))
print(f"  因子文件: {len(factor_files)}")

fl = []
for fn in factor_files:
    df = pd.read_feather(f'{dataset_path}/{fn}')
    s = df.stack()
    s.name = fn.replace('.fea', '')
    fl.append(s)
X = pd.concat(fl, axis=1)
del fl; gc.collect()

# 统一索引格式: date → str 'YYYY-MM-DD', hour → str 'HH:00'
# (新因子可能写入 datetime64/int, 旧因子是 str, label 是 'HH:00')
idx_dates = pd.to_datetime(X.index.get_level_values('date')).strftime('%Y-%m-%d')
idx_hours_raw = X.index.get_level_values('hour')
idx_hours = [f'{int(h):02d}:00' if isinstance(h, (int, np.integer)) else str(h)
             for h in idx_hours_raw]
X.index = pd.MultiIndex.from_arrays([idx_dates, idx_hours], names=['date', 'hour'])

# 因子分类统计
price_cols = [c for c in X.columns if c.startswith('price_d')]
v_feats = [c for c in X.columns if not c.startswith(('h_', 'gc_'))]
h_feats = [c for c in X.columns if c.startswith('h_')]
gc_feats = [c for c in X.columns if c.startswith('gc_')]
print(f"  X: {X.shape}")
print(f"  纵向 {len(v_feats)} + 日内 {len(h_feats)} + 电网约束 {len(gc_feats)} + price_d* {len(price_cols)}")

# 特征质量过滤: 剔除训练期 NaN 率 > 50% 的列
# (如 西电东送/三峡/海南 等 2025-06-29 后即无发布, 模型学不到信号, 预测期填充也垃圾)
train_na_ratio = X.loc['2025-01-01':'2026-05-31'].isna().mean()
drop_cols = train_na_ratio[train_na_ratio > 0.5].index.tolist()
if drop_cols:
    print(f"  剔除训练期 NaN>50% 的列: {len(drop_cols)} 个 (如 {drop_cols[:5]}...)")
    X = X.drop(columns=drop_cols)

# v5.3: 只用预测期新鲜集
# 步骤1: 剔除预测期有 NaN 的列 (load/h_tvol 等实际数据退化)
# 步骤2: 再剔除价格因子 (cell 4b 对 price 做了 ffill, 价格因子在预测期"伪装新鲜")
#        → 真正的预测期新鲜集 = 供给侧 s_* + 日内 h_*(预测字段) + 约束 gc_* + 时间
if FRESH_ONLY:
    pred_na = X.loc['2026-07-22':'2026-07-27'].isna()
    step1 = [c for c in X.columns if not pred_na[c].any()]
    fresh_cols = [c for c in step1 if not is_price_factor(c)]
    print(f"  新鲜集模式: 步骤1 预测期无NaN={len(step1)} → 步骤2 剔除价格因子 → 保留 {len(fresh_cols)} 列")
    X = X[fresh_cols]

# ============ PHASE 1: 切分 ============
print("\n[Phase 1] 切分数据集...")
label_df = pd.read_feather(f'{label_path}/label.feather')
y = label_df.stack(); y.index = y.index.rename(['date', 'hour']); y.name = 'price'

xd = X.index.get_level_values('date'); yd = y.index.get_level_values('date')
ld = yd.unique(); ad = xd.unique(); prd = ad.difference(ld).sort_values()

tr_m = (xd >= '2025-01-01') & (xd <= '2026-05-31')
va_m = (xd >= '2026-06-01') & (xd <= '2026-07-15')
te_m = (xd >= '2026-07-16') & (xd <= '2026-07-21')
pr_m = xd.isin(prd)

X_tr = X.loc[tr_m].copy(); X_va = X.loc[va_m].copy()
X_te = X.loc[te_m].copy(); X_pr = X.loc[pr_m].copy()

def gy(s):
    c = s.index.intersection(y.index)
    return y.loc[c]
y_tr = gy(X_tr); y_va = gy(X_va); y_te = gy(X_te)
X_tr = X_tr.loc[y_tr.index]; X_va = X_va.loc[y_va.index]; X_te = X_te.loc[y_te.index]

# 移除 price_d* 延迟因子
pd_cols = [c for c in X_tr.columns if c.startswith('price_d')]
print(f"  移除 price_d*: {pd_cols}")
for D in [X_tr, X_va, X_te, X_pr]:
    D.drop(columns=pd_cols, inplace=True, errors='ignore')

# ============ PHASE 2: 识别退化列 ============
print("\n[Phase 2] 识别退化列...")
nan_cnt = X_pr.isna().sum()
degraded_cols = nan_cnt[nan_cnt > 0].index.tolist()
if DEGRADE_ALL:
    # v5.4: 退化列扩展到价格因子 (cell 4b ffill 让价格因子在预测期"伪装新鲜",
    #       但真实形态是退化 — 训练时需对齐)
    price_factor_cols = [c for c in X.columns if is_price_factor(c) and c not in degraded_cols]
    degraded_cols = sorted(set(degraded_cols) | set(price_factor_cols))
    print(f"  价格因子: {len(price_factor_cols)} 个加入退化列")
print(f"  退化列: {len(degraded_cols)} 个")
print(f"  示例: {degraded_cols[:10]}")

# 预测集填充: 对退化列做同星期几填充
# (v5.4 强制: 价格因子的 ffill 伪装值也替换为同星期几填充值, 与训练退化形态一致)
print("\n  预测集同星期几填充...")
train_dates = pd.to_datetime(X_tr.index.get_level_values('date'))
fill_cols = degraded_cols if DEGRADE_ALL else [c for c in degraded_cols if X_pr[c].isna().any()]
for col in fill_cols:
    # DEGRADE_ALL: 强制填充所有退化列 (含 ffill 伪装的价格因子), 与训练退化形态对齐
    if (not DEGRADE_ALL) and (not X_pr[col].isna().any()):
        continue
    for (d, h) in X_pr.index:
        wd = pd.Timestamp(d).dayofweek
        same_wd = train_dates[train_dates.dayofweek == wd].unique().sort_values()[-4:]
        vals = []
        for ref_d in same_wd:
            ref_str = ref_d.strftime('%Y-%m-%d')
            if (ref_str, h) in X.index:
                v = X.loc[(ref_str, h), col]
                if not pd.isna(v):
                    vals.append(v)
        if vals:
            X_pr.loc[(d, h), col] = np.mean(vals)
        # 若 4 个参考日期均 NaN (如 西电东送), 保留原值 (NaN 或 ffill 值)

nan_after = X_pr[degraded_cols].isna().sum().sum()
print(f"  填充后预测集 NaN: {nan_after}")

# ============ PHASE 3: 训练集退化增强 ============
print("\n[Phase 3] 训练集退化增强...")

def degrade_train(X_tr, y_tr, degraded_cols, X_full, p_degrade=0.3, ks=(1, 3, 5, 7)):
    """
    对训练集随机注入因子退化 (v5.2: 对齐预测期分布):
      - 每个样本以概率 p 选中, 随机退化 k 天
      - 退化值 = 最近 4 个同星期几的均值 (与预测期填充逻辑完全一致!)
        → 训练时模型直接看到"预测期的因子形态", 训练/预测分布对齐
      - 若参考日期在退化列上也是 NaN (如 cord7), 则退化列该样本保持原值
      - staleness 特征: 退化样本 = k, 未退化 = 0
    """
    n = len(X_tr)
    X_tr = X_tr.copy()
    dates = pd.to_datetime(X_tr.index.get_level_values('date'))
    hours = X_tr.index.get_level_values('hour')

    if not degraded_cols:
        X_tr['staleness'] = 0.0
        print(f"  无退化列, 跳过增强")
        return X_tr

    # 随机选择退化样本和天数
    mask = RNG.rand(n) < p_degrade
    k_arr = RNG.choice(np.array(ks, dtype=int), size=n)
    print(f"  退化样本: {mask.sum()}/{n} ({mask.mean()*100:.1f}%)")

    # 预计算同星期几填充查找表: (weekday, hour) → 最近4个同星期几的退化列均值
    # 参考池: 整个训练期 (与预测期填充用的参考池一致)
    ref_pool_dates = pd.to_datetime(X_full.index.get_level_values('date'))
    ref_dates_all = ref_pool_dates.unique()
    cols_pos = X_tr.columns.get_indexer(degraded_cols)

    lookup = {}
    for wd in range(7):
        same_wd = ref_dates_all[ref_dates_all.dayofweek == wd].sort_values()[-4:]
        for h in pd.unique(hours):
            rows = []
            for ref_d in same_wd:
                ref_str = ref_d.strftime('%Y-%m-%d')
                if (ref_str, h) in X_full.index:
                    rows.append(X_full.loc[(ref_str, h), degraded_cols])
            if len(rows) >= 4:
                arr = pd.concat(rows, axis=1).T  # (4, n_cols)
                # 忽略 NaN 的逐列均值, 转 float32 与因子 dtype 一致
                lookup[(wd, h)] = arr.mean(axis=0).values.astype(np.float32)
    print(f"  填充查找表: {len(lookup)} 个 (weekday, hour) 组合")

    # 应用: 向量化替换退化样本的退化列
    applied = 0
    for i in np.where(mask)[0]:
        key = (dates[i].dayofweek, hours[i])
        if key in lookup:
            X_tr.iloc[i, cols_pos] = lookup[key]
            applied += 1
    print(f"  应用退化: {applied}/{mask.sum()}")

    # staleness 特征
    staleness = np.zeros(n)
    staleness[mask] = k_arr[mask]
    X_tr['staleness'] = staleness
    return X_tr

# 退化参考池: 只用训练期 (与预测期填充的参考池一致, 避免取到预测期值)
X_ref_pool = X.loc[(X.index.get_level_values('date') >= '2025-01-01') &
                   (X.index.get_level_values('date') <= '2026-05-31')]
X_tr_aug = degrade_train(X_tr, y_tr, degraded_cols, X_ref_pool, P_DEGRADE, DEGRADE_KS)

# 验证集/测试集: staleness = 0 (保持新鲜, 真实评估)
for D in [X_va, X_te]:
    D['staleness'] = 0.0

# 预测集: staleness = 1..6 (距最后 label 的天数)
pr_dates = pd.to_datetime(X_pr.index.get_level_values('date'))
X_pr['staleness'] = np.maximum(0, (pr_dates - LAST_LABELED).days.values).astype(float)

feature_names = X_tr_aug.columns.tolist()
print(f"\n  最终特征: {len(feature_names)}")
print(f"  Staleness: 训练 [{X_tr_aug['staleness'].min():.0f}, {X_tr_aug['staleness'].max():.0f}], "
      f"预测 [{X_pr['staleness'].min():.0f}, {X_pr['staleness'].max():.0f}]")

# ============ PHASE 4: 训练 ============
print("\n[Phase 4] 训练 XGBoost (固定参数)...")
fixed_params = {'max_depth': 8, 'learning_rate': 0.05,
                'subsample': 0.8, 'colsample_bytree': 1.0}
model = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50,
    eval_metric='rmse', random_state=42, verbosity=0, n_jobs=-1, **fixed_params)
model.fit(X_tr_aug, y_tr, eval_set=[(X_va, y_va)], verbose=False)

yp_te = model.predict(X_te)
rmse_te = np.sqrt(mean_squared_error(y_te, yp_te))
r2_te = r2_score(y_te, yp_te)
mape_te = np.mean(np.abs((y_te.values - yp_te) / (np.abs(y_te.values) + 1e-8))) * 100
print(f"  Test: RMSE={rmse_te:.2f}, R²={r2_te:.4f}, MAPE={mape_te:.2f}%")

# ============ PHASE 5: 预测 + 独立评估 ============
print("\n[Phase 5] 独立评估...")
yp_pr = model.predict(X_pr)

new_label = pd.read_feather(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', '日前统一结算价.feather'))
y_new = new_label.stack(); y_new.index = y_new.index.rename(['date', 'hour'])

ev_dates = [d for d in prd if d in y_new.index.get_level_values('date')]
print(f"  独立评估日期: {ev_dates}")

errs_all = []
for d in ev_dates:
    mask = (X_pr.index.get_level_values('date') == d)
    X_ev = X_pr.loc[mask]
    yp_ev = model.predict(X_ev)
    e = []
    for (dd, h), pred in zip(X_ev.index, yp_ev):
        act = y_new.loc[dd, h]
        e.append(pred - act)
    errs_all += e
    rmse_d = np.sqrt(np.mean(np.array(e) ** 2))
    bias_d = np.mean(e)
    print(f"    {d}: RMSE={rmse_d:.1f}, Bias={bias_d:+.1f}")

rmse_ev = np.sqrt(np.mean(np.array(errs_all) ** 2))
print(f"\n  Eval RMSE: {rmse_ev:.2f}, Test/Eval Ratio: {rmse_ev/rmse_te:.1f}x")

# ============ PHASE 6: 跨天区分度 ============
print("\n[Phase 6] 跨天预测区分度...")
pw = pd.DataFrame({'price': yp_pr}, index=X_pr.index)['price'].unstack()
corrs = []
for i, d in enumerate(pw.index):
    r = pw.loc[d]
    print(f"    {d}: avg={r.mean():.1f}, min={r.min():.1f}, max={r.max():.1f}")
    if i > 0:
        c = r.corr(pw.iloc[i - 1])
        corrs.append(c)
        print(f"         vs {pw.index[i-1]}: corr={c:.4f}")
mean_corr = np.mean(corrs) if corrs else 1.0
print(f"  跨天平均 corr: {mean_corr:.4f} (v4: >0.999)")

# ============ PHASE 7: 特征重要性 ============
print("\n[Phase 7] 特征重要性分布...")
imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

def cat_of(c):
    if c == 'staleness': return 'staleness'
    if c.startswith('gc_'): return '电网约束'
    if c.startswith('h_'): return '日内'
    if c.startswith('s_'): return '供给侧纵向'
    if c.startswith(('ma', 'std', 'max', 'min', 'rank', 'qtlu', 'qtld', 'cntp', 'cntn',
                     'psy', 'sump', 'sumn', 'roc', 'corr', 'cord', 'cov')): return '价格纵向'
    if c.startswith('cs_'): return '截面'
    return '其他'

imp_cat = imp.groupby(imp.index.map(cat_of)).sum().sort_values(ascending=False)
print(f"  Top 5: {imp.head(5).round(4).to_dict()}")
print("\n  按类别:")
for k, v in imp_cat.items():
    print(f"    {k}: {v:.4f} ({v/imp.sum()*100:.1f}%)")

# ============ PHASE 8: 保存 ============
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'xgb_v5_final.joblib')
joblib.dump({
    'model': model,
    'features': feature_names,
    'fixed_params': fixed_params,
    'p_degrade': P_DEGRADE,
    'degrade_ks': DEGRADE_KS,
    'degraded_cols': degraded_cols,
    'test_rmse': rmse_te,
    'test_r2': r2_te,
    'test_mape': mape_te,
    'eval_rmse': rmse_ev,
    'test_eval_ratio': rmse_ev / rmse_te,
    'cross_day_corr': mean_corr,
}, save_path)
print(f"\n  模型保存: {save_path}")

print(f"\n{'='*65}")
print(f"  v5: Test RMSE={rmse_te:.2f} | Eval RMSE={rmse_ev:.2f} | Ratio={rmse_ev/rmse_te:.1f}x | 跨天corr={mean_corr:.4f}")
print(f"{'='*65}")
