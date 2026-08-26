"""
Pipeline v7.2 Final: 训练最终部署模型 → /upload

特征: 369 新鲜集 + 9 gas_limit 事件因子 (最优组合, T112-C2 验证)
数据: 2025-01-01 ~ 2026-06-26 训练 + 06-27~07-26 早停valid (valid 已排除出训练, 红线) → 预测 07-27
架构: XGBoost 级联 (5类分类头 + 回归头)
输出: the local upload directory/xgb_v7_final.joblib (自包含)
"""
import sys, os, json, gc, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from config.config import (dataset_path, spread_label_file, da_price_latest,
                           rt_price_latest, SPREAD_THRESHOLD, SPREAD_THRESHOLD_BIG,
                           SPREAD_CLASSES, YONON_PATH)

T1, T2 = SPREAD_THRESHOLD, SPREAD_THRESHOLD_BIG
NCLS = len(SPREAD_CLASSES)

print("=" * 72)
print("  v7.2 Final: 训练最终部署模型")
print(f"  369 新鲜集 + 9 gas_limit | τ_minor={T1}, τ_big={T2}")
print("=" * 72)

# ============ 加载特征 ============
print("\n[1/5] 加载特征...")
with open(f'{YONON_PATH}/data/v7_fresh_features.json') as f:
    fresh = json.load(f)['fresh_features']
gas_names = sorted([n[:-4] for n in os.listdir(dataset_path)
                    if (n.startswith('ev_gas_limit') or n.startswith('ev_burst_gas'))
                    and n.endswith('.fea')])
all_features = fresh + gas_names
print(f"  特征: {len(fresh)} 新鲜 + {len(gas_names)} gas_limit = {len(all_features)}")

fl = []
for name in all_features:
    df = pd.read_feather(f'{dataset_path}/{name}.fea')
    s = df.stack(); s.name = name
    fl.append(s)
X = pd.concat(fl, axis=1); del fl; gc.collect()
idx_dates = pd.to_datetime(X.index.get_level_values(0)).strftime('%Y-%m-%d')
idx_hours = [f'{int(h):02d}:00' if isinstance(h, (int, np.integer)) else str(h)
             for h in X.index.get_level_values(1)]
X.index = pd.MultiIndex.from_arrays([idx_dates, idx_hours], names=['date', 'hour'])
X = X.sort_index()
print(f"  X: {X.shape}")

# ============ 标签 ============
print("\n[2/5] 加载标签...")
spread = pd.read_feather(spread_label_file); spread.index = pd.to_datetime(spread.index)
y = spread.stack(); y.index = y.index.rename(['date', 'hour'])
y.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(y.index.get_level_values('date')).strftime('%Y-%m-%d'),
     y.index.get_level_values('hour')], names=['date', 'hour'])
da_e = pd.read_feather(da_price_latest); rt_e = pd.read_feather(rt_price_latest)
sps = (da_e - rt_e).stack(); sps.index = sps.index.rename(['date', 'hour'])
sps.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(sps.index.get_level_values('date')).strftime('%Y-%m-%d'),
     sps.index.get_level_values('hour')], names=['date', 'hour'])
y = pd.concat([y, sps]); y = y[~y.index.duplicated(keep='first')]

def to_class5(spr):
    spr = np.asarray(spr, dtype=float)
    return np.where(spr < -T2, 0,
           np.where(spr < -T1, 1,
           np.where(spr <= T1, 2,
           np.where(spr <= T2, 3, 4))))

# ============ 数据划分 ============
print("\n[3/5] 数据划分 (train=全量到06-26, valid=06-27~07-26, predict=07-27)...")
xd = X.index.get_level_values('date')
# 红线: valid (early stopping 集) 必须排除在训练集外, 否则 best_iteration 不触发、
#       嵌入指标是记忆假象 (2026-08-11 修复: 原 tr_m<=07-26 使 valid⊂train, sign 虚高到 0.99)
tr_m = xd < '2026-06-27'
pr_m = xd == '2026-07-27'
# 用最后 30 天做验证集 (early stopping)
va_m = (xd >= '2026-06-27') & (xd <= '2026-07-26')
assert not (xd[tr_m].isin(xd[va_m]).any()), "红线: valid 窗口不得混入训练集"

def subset(mask, ys):
    Xi = X.loc[mask]
    common = Xi.index.intersection(ys.index)
    return Xi.loc[common], ys.loc[common]

X_tr, y_tr = subset(tr_m, y)
X_va, y_va = subset(va_m, y)
X_pr = X.loc[pr_m]
yc_tr = to_class5(y_tr); yc_va = to_class5(y_va)
print(f"  train {len(X_tr)} | valid {len(X_va)} | predict {len(X_pr)}")

# ============ 训练 ============
print("\n[4/5] 训练级联模型...")
FIXED = {'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8,
         'colsample_bytree': 1.0, 'tree_method': 'hist', 'n_jobs': 8}

# 类平衡权重
cls_counts = np.bincount(yc_tr, minlength=NCLS)
w_per_cls = len(yc_tr) / (NCLS * cls_counts)
sw_tr = w_per_cls[yc_tr]
print(f"  类别权重: {dict(zip(SPREAD_CLASSES, w_per_cls.round(2)))}")

clf = xgb.XGBClassifier(n_estimators=200, early_stopping_rounds=30,
    eval_metric='mlogloss', random_state=42, verbosity=0, **FIXED)
clf.fit(X_tr, yc_tr, sample_weight=sw_tr, eval_set=[(X_va, yc_va)], verbose=False)
print(f"  clf best_iteration: {clf.best_iteration}")

reg = xgb.XGBRegressor(n_estimators=200, early_stopping_rounds=30,
    eval_metric='rmse', random_state=42, verbosity=0, **FIXED)
reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
print(f"  reg best_iteration: {reg.best_iteration}")

# ============ 快速评估 ============
print("\n[5/5] 快速评估 + 保存...")
yc_va_pred = clf.predict(X_va)
nonneu = yc_va != 2
pred_dir = np.sign(yc_va_pred - 2)
true_dir = np.sign(np.where(y_va.values == 0, 1e-9, y_va.values))
sign_hit = (pred_dir[nonneu] == true_dir[nonneu]).mean()
yt_big = ((yc_va == 0) | (yc_va == 4)).astype(int)
yp_big = ((yc_va_pred == 0) | (yc_va_pred == 4)).astype(int)
big_f1 = f1_score(yt_big, yp_big, zero_division=0)
print(f"  Valid (30天): sign={sign_hit:.3f} | bigF1={big_f1:.3f} | n={len(y_va)}")

# 预测 07-27
yc_pr = clf.predict(X_pr); yv_pr = reg.predict(X_pr)
lvl = {0: '🔴大负', 1: '🟡负', 2: '⚪正常', 3: '🟡正', 4: '🔴大正'}
print(f"\n  07-27 预测 (明日):")
for i, (idx, row) in enumerate(X_pr.iterrows()):
    c = yc_pr[i]; v = yv_pr[i]
    out = f"{v:+.1f} 元/MWh" if c != 2 else "—"
    print(f"    {idx[1]}  {lvl[c]:6s}  {out}")

# ============ 保存 ============
upload_dir = f'{YONON_PATH}/upload'
os.makedirs(upload_dir, exist_ok=True)
save_path = f'{upload_dir}/xgb_v7_final.joblib'

joblib.dump({
    'clf': clf,
    'reg': reg,
    'features': all_features,
    'threshold_minor': T1,
    'threshold_big': T2,
    'classes': SPREAD_CLASSES,
    'decision_rule': (
        f"|spread| ≤ {T1} → neu (正常, 不预警); "
        f"{T1} < |spread| ≤ {T2} → neg/pos (有偏差, 预警+输出值); "
        f"|spread| > {T2} → big_neg/big_pos (大偏差, 预警+输出值)"
    ),
    'fixed_params': FIXED,
    'train_end_date': '2026-06-26',   # 实际训练截止 (valid 06-27~07-26 已排除出训练)
    'data_end_date': '2026-07-26',    # 标签数据覆盖到这天 (predict 07-27)
    'n_features': len(all_features),
    'valid_sign_hit': float(sign_hit),
    'valid_big_f1': float(big_f1),
}, save_path)
print(f"\n  模型已保存: {save_path}")
print(f"  文件大小: {os.path.getsize(save_path)/1024/1024:.1f} MB")
print(f"\n{'='*72}")
print(f"  v7.2 Final 完成 | 特征={len(all_features)} | sign={sign_hit:.3f} | bigF1={big_f1:.3f}")
print(f"{'='*72}")
