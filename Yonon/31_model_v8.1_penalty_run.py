"""
Pipeline v8.1: DA-RT 价差预测 — 触发阈值 50 / 大偏差阈值 100 / Plan C' 加权回归 / 数值驱动触发
=====================================================================================
用户拍板 (2026-08-13):
  1. 触发阈值 τ_minor=50, 大偏差阈值 τ_big=100 (改自 v8 的 50/50 双对齐)
  2. 分类头用 3 类 {big_neg/neu/big_pos}, 只在 ±50 处切 → 决定「是否触发」
  3. 回归头改「Plan C' 加权回归」: 训练所有样本 (尺度锚定, 防数值塌缩),
     |spread|>50 样本权重 1.0, |spread|≤50 样本权重 0.2 (轻量保留中性结构)
  4. 触发语义改为「数值驱动」: 分类给方向信号, 最终预警/大偏差由回归值判定
       |预测| < 50   → 正常 (不预警)
       50 ≤ |预测| < 100 → 小偏差预警 (输出值)
       |预测| ≥ 100  → 大偏差预警 (输出值)
     → 50 以内不再出现任何触发 (治 v8 里「分类触发但数值 18/-7」的病)
  5. 惩罚评分/最优权重选择按新阈值重算 (规则 A/B 在 50 处判方向, 规则 C 在 50~100 判量级)

──────────────────────────────────────────────────────────────────
惩罚评分 (硬规则, 值越接近 0 越好; 与 v8 一致但阈值感知 → 50):

  规则 A (预测错误):  触发信号 (模型输出非 neu 或真实非 neu) 时, 方向/分类判断错误 → -1
                      含 漏报 (真实有偏差但模型判 neu) 与 误报 (模型触发但真实为 neu)
  规则 B (完全反向):  预测方向与实际方向完全相反 (一正一负) → -2 (加倍重罚)
  规则 C (数值偏差):  模型触发并输出数值时, |Pred-True|/|True| > 0.2 → 额外 -1.5

平滑代理 (梯度注入):
  - 分类头 clf:  成本矩阵软目标 softmax obj = CE + λ·E_penalty(C), C 编码规则 A/B 分值
  - 回归头 reg:  Plan C' 加权回归 (全量样本, |y|>50→1.0 / |y|≤50→0.2, 尺度锚定), 规则 C 对齐

防过拟合 (继承 v8):
  数据层: 每轮特征高斯噪声注入; TimeSeriesSplit 时序交叉验证
  模型层: DART 树 dropout (rate_drop=0.3, one_drop, skip_drop=0.5); L2=reg_lambda=2.0
  训练层: EarlyStopping patience=15 (双头各自停滞); ReduceLROnPlateau
  保存层: 只存 Val_Penalty 最优迭代权重

部署契约 (deploy/2B_inference.py 已按 v8.1 改造):
  model['clf'].predict(X)  → 项目编码 {0,2,4} (neu=2), 仅方向信号
  model['reg'].predict(X)  → 连续预测值 (全量回归, 条件训练)
  触发/等级 = f(reg 值, 50/100): <50 正常 | 50~<100 小偏差 | ≥100 大偏差
"""
import sys, os, json, gc, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from v8_wrappers import BoosterClf as _BoosterClf, BoosterReg as _BoosterReg
from sklearn.metrics import (mean_squared_error, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix)
from config.config import (dataset_path, spread_label_file, da_price_latest,
                           rt_price_latest, SPREAD_THRESHOLD, SPREAD_THRESHOLD_BIG,
                           SPREAD_CLASSES, SPREAD_CLASS_MAP, YONON_PATH)

# ── v8.1 阈值 (来自 config.py, 已更新为 50/100) ──
T = SPREAD_THRESHOLD          # 50.0 触发阈值
T_BIG = SPREAD_THRESHOLD_BIG  # 100.0 大偏差阈值
assert T < T_BIG, f"v8.1 要求 τ_minor < τ_big, 实际 {T}/{T_BIG}"

NCLS3 = 3                     # 内部 3 类 {big_neg, neu, big_pos}
CODES = [0, 2, 4]             # 映射到项目 5 类编码体系的子集 {big_neg, neu, big_pos}
NCODE = SPREAD_CLASS_MAP['neu']   # 2 (部署侧 neu 判断码)
MAX_ROUNDS = int(os.environ.get('V8_MAX_ROUNDS', 800))
EARLY_STOP_PATIENCE = 15      # 早停耐心
PLATEAU_PATIENCE = 15         # ReduceLROnPlateau 耐心
MIN_LR = 5e-3
MIN_DELTA = 5e-4              # 最优权重选择最小改善 (mean penalty)
LOSS_MIN_DELTA = 2e-3         # 平滑损失早停最小改善
SMOKE = int(os.environ.get('V8_SMOKE', 0))   # 冒烟测试: 限制轮数
if SMOKE:
    MAX_ROUNDS = min(MAX_ROUNDS, 60)
    print(f"  ⚠️ 冒烟模式 (V8_SMOKE=1): MAX_ROUNDS={MAX_ROUNDS}")

# ── 回归头训练模式 (Plan C' 默认, 治条件回归数值塌缩) ──
#   anchored    (C', 默认): 全量样本加权训练 |y|>50 权重 1.0 / |y|≤50 权重 0.2 (尺度锚定)
#   conditional (原方案):   仅在 |y|>50 样本上训练 (0/1 权重) — 已证实会数值塌缩
REG_MODE = os.environ.get('V8_REG_MODE', 'anchored')
W_NEU, W_BIG = (0.2, 1.0) if REG_MODE == 'anchored' else (0.0, 1.0)
if REG_MODE == 'conditional':
    print("  ⚠️ V8_REG_MODE=conditional (仅 |y|>50 样本训练回归头, 已知会数值塌缩)")

print("=" * 72)
print("  Pipeline v8.1: DA-RT 价差 — 触发阈值 50 / 大偏差 100 / Plan C' 加权回归 / 数值驱动触发")
print(f"  τ={T} τ_big={T_BIG} | 3类分类: big_neg<-{T} / neu±{T} / big_pos>{T} | "
      f"回归头(Plan C'): 全量训练权重 {W_BIG}/{W_NEU} | 触发判定: 回归值 vs {T}/{T_BIG}")
print("=" * 72)


# ════════════════════════════════════════════════════════════════
# 0. 通用工具
# ════════════════════════════════════════════════════════════════
def to_class3(spr, t=T):
    """spread → 3 类内部索引 {0:big_neg, 1:neu, 2:big_pos}, 在 ±t 处切"""
    spr = np.asarray(spr, dtype=float)
    return np.where(spr < -t, 0, np.where(spr <= t, 1, 2))

def code_of(internal_idx):
    """3 类内部索引 → 项目 5 类编码子集 {0,2,4}"""
    return np.array([CODES[i] for i in np.asarray(internal_idx, dtype=int)])


# ── 硬惩罚评分 (规则 A/B/C) — 阈值感知 50, 值越接近 0 越好 (负数, 均值) ──
def hard_penalty(y_true, pred_code, pred_val, t=T, verbose=False):
    """规则 A/B/C 惩罚. pred_code: 项目编码 {0,2,4} (neu=2). 返回 (mean, total, n_trigger)."""
    y_true = np.asarray(y_true, dtype=float)
    pred_code = np.asarray(pred_code, dtype=int)
    pred_val = np.asarray(pred_val, dtype=float)
    n = len(y_true)
    # 阈值感知: |y|<=t → 无信号 (中性); 只有 |y|>t 才定义真实方向
    true_dir = np.sign(y_true) * (np.abs(y_true) > t)
    pred_dir = np.sign(pred_code - NCODE)          # {0,2,4}→-1/0/+1
    pen = np.zeros(n, dtype=float)

    # 规则 A/B: 至少一方有信号才计分 (双方中性 = 正确不触发, 不扣分)
    active = (pred_dir != 0) | (true_dir != 0)
    pd_, td_ = pred_dir[active], true_dir[active]
    mism = pd_ != td_
    opposite = mism & (pd_ != 0) & (td_ != 0) & (pd_ == -td_)   # 完全反向 → 规则 B
    pen[active] = np.where(opposite, -2.0, np.where(mism, -1.0, 0.0))

    # 规则 C: 模型触发并输出数值时, |Pred-True|/|True| > 0.2 → 额外 -1.5 (叠加)
    trig = pred_dir != 0
    rel = np.abs(pred_val - y_true) / np.maximum(np.abs(y_true), 1.0)
    pen[trig] += np.where(rel[trig] > 0.2, -1.5, 0.0)

    mean_pen = pen.mean()
    n_trig = int(trig.sum())
    if verbose:
        n_active = int(active.sum())
        n_opp = int((active & (pen < 0) & (np.abs(pen) >= 2.0)).sum())
        print(f"    惩罚明细: 有效样本 {n_active}/{n} | 触发 {n_trig} | 反向(规则B) ~{n_opp}")
    return mean_pen, float(pen.sum()), n_trig


# ── 分类头平滑惩罚目标 (规则 A/B: 成本矩阵软目标) ──
# 成本矩阵 C[row=真实, col=预测] (内部 3 类: 0 big_neg / 1 neu / 2 big_pos)
C_MAT = np.array([[0., -1., -2.],
                  [-1., 0., -1.],
                  [-2., -1., 0.]], dtype=float)
LAM_COST = 0.5
_EYE3 = np.eye(3)

def clf_cost_fobj(preds, dtrain):
    """softmax + 期望业务成本 (规则 A/B) 的平滑代理. 返回 (n,3) grad/hess."""
    yl = dtrain.get_label().astype(int)
    n = len(yl)
    m = preds.reshape(n, 3)
    mx = m - m.max(axis=1, keepdims=True)
    p = np.exp(mx); p /= p.sum(axis=1, keepdims=True)
    E = np.sum(p * C_MAT[yl], axis=1, keepdims=True)       # 期望成本 (≤0)
    g = (p - _EYE3[yl]) + LAM_COST * p * (E - C_MAT[yl])   # d(CE + λ·(−E))/dm
    h = np.clip(2.0 * p * (1.0 - p), 1e-3, None)
    return g, h


# ── 平滑验证损失 (用于 EarlyStopping / ReduceLROnPlateau) ──
def clf_smooth_loss(bst, dmat, label3):
    """分类头平滑损失 = CE + λ·期望业务成本 (与 clf_cost_fobj 一致)"""
    pr = bst.predict(dmat)
    n = len(label3)
    eps = 1e-9
    ce = -np.mean(np.log(pr[np.arange(n), label3] + eps))
    E = np.sum(pr * C_MAT[label3], axis=1)
    cost = -np.mean(E)
    return ce + LAM_COST * cost

def reg_smooth_loss(bst, dmat, y, w):
    """回归头平滑损失 = 加权 RMSE (与回归头训练权重一致, 原生量纲)"""
    pred = bst.predict(dmat)
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if s == 0:
        return 0.0
    return float(np.sqrt(np.sum(w * (pred - y) ** 2) / s))


# ════════════════════════════════════════════════════════════════
# 1. 加载特征 (369 新鲜集, 与 v8 一致; 不做标准化, NaN 由 XGBoost 原生处理)
# ════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════
# 2. 标签 + 数据划分
# ════════════════════════════════════════════════════════════════
print("\n[Phase 1] 标签 + 数据划分...")
spread = pd.read_feather(spread_label_file)
spread.index = pd.to_datetime(spread.index)
y_spread = spread.stack(); y_spread.index = y_spread.index.rename(['date', 'hour'])
y_spread.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(y_spread.index.get_level_values('date')).strftime('%Y-%m-%d'),
     y_spread.index.get_level_values('hour')], names=['date', 'hour'])
y_spread.name = 'spread'

da_e = pd.read_feather(da_price_latest); rt_e = pd.read_feather(rt_price_latest)
sp_eval_s = (da_e - rt_e).stack()
sp_eval_s.index = sp_eval_s.index.rename(['date', 'hour'])
sp_eval_s.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(sp_eval_s.index.get_level_values('date')).strftime('%Y-%m-%d'),
     sp_eval_s.index.get_level_values('hour')], names=['date', 'hour'])

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

yc_tr = to_class3(y_tr.values)
yc_va = to_class3(y_va.values)
dist = np.bincount(yc_tr, minlength=NCLS3)
print("  训练集 3 类分布: " + ", ".join(f"{['big_neg','neu','big_pos'][i]}={dist[i]/len(yc_tr)*100:.1f}%" for i in range(NCLS3)))

# 类别权重 (3 类平衡)
w_per_cls = len(yc_tr) / (NCLS3 * np.maximum(dist, 1))
sw_tr = w_per_cls[yc_tr]
print(f"  类别权重: {dict(zip(['big_neg','neu','big_pos'], w_per_cls.round(2)))}")

# 回归头样本权重 (Plan C' 加权回归): |spread|>50 → 1.0, 否则 → 0.2 (尺度锚定防塌缩)
reg_mask_tr = (np.abs(y_tr.values) > T)
reg_mask_va = (np.abs(y_va.values) > T)
print(f"  回归头样本 (|spread|>{T}): train {reg_mask_tr.sum()}/{len(y_tr)} "
      f"({reg_mask_tr.mean()*100:.1f}%) | valid {reg_mask_va.sum()}/{len(y_va)}")
print(f"  回归头权重 (Plan C'): |y|>{T} → {W_BIG} | |y|≤{T} → {W_NEU} (全量样本训练)")

# 特征标准差 (用于噪声注入, 只从 train 统计)
feat_std = np.nanstd(X_tr.values, axis=0)
feat_std = np.where(np.isnan(feat_std) | (feat_std == 0), 1.0, feat_std)

# ════════════════════════════════════════════════════════════════
# 3. 训练参数 (分类头 DART + 回归头 gbtree 分离配置)
# ════════════════════════════════════════════════════════════════
# 回归头量级塌缩根因: DART 树 dropout (rate_drop=0.3/one_drop) 强收缩 + 每轮噪声注入,
# 把 |y|~134 的大偏差量级压到 ~40。修复: 回归头改用无 dropout 的 gbtree + 更浅 L2 +
# 干净特征训练 (仅分类头用 DART+噪声做正则)。
FIXED = dict(booster='gbtree', tree_method='hist', max_depth=8, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.9, reg_lambda=2.0,
             rate_drop=0.3, skip_drop=0.5, one_drop=True, min_child_weight=1,
             nthread=8, seed=42)
CLF_P = {**FIXED, 'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss',
         'num_feature': len(fresh)}
# 回归头: 无 dropout 纯 gbtree, max_depth=10 (更细), reg_lambda=1.0 (更松) → 保住大偏差量级
REG_P = dict(booster='gbtree', tree_method='hist', max_depth=10, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0, min_child_weight=1,
             nthread=8, seed=42, objective='reg:squarederror', num_feature=len(fresh))

X_tr_np = X_tr.values.astype(np.float32)
X_va_np = X_va.values.astype(np.float32)
y_tr_np = y_tr.values.astype(np.float32)
y_va_np = y_va.values.astype(np.float32)
yc_tr_np = yc_tr.astype(np.int32)
yc_va_np = yc_va.astype(np.int32)
# Plan C' 加权回归: |y|>50 权重 1.0, |y|≤50 权重 0.2 (全量样本参与, 尺度锚定)
reg_w_tr = np.where(reg_mask_tr, W_BIG, W_NEU).astype(np.float32)
reg_w_va = np.where(reg_mask_va, W_BIG, W_NEU).astype(np.float32)


def build_dmat(Xn, yl, w=None):
    if w is not None:
        return xgb.DMatrix(Xn, label=yl, weight=w)
    return xgb.DMatrix(Xn, label=yl)


def inject_noise(Xn, epoch, sigma_ratio=0.01, frac=0.4, seed0=1234):
    """数据增强: 对 40% 样本注入 N(0, sigma_ratio·std) 高斯噪声 (可复现)."""
    rng = np.random.default_rng(seed0 + epoch)
    Xn = Xn.copy()
    n, d = Xn.shape
    mask = rng.random(n) < frac
    noise = rng.normal(0, sigma_ratio, size=(int(mask.sum()), d))
    Xn[mask] += (noise * feat_std).astype(np.float32)
    return Xn


# ════════════════════════════════════════════════════════════════
# 4. 时序交叉验证 (TimeSeriesSplit, 数据层防泄露) — 轻量报告
# ════════════════════════════════════════════════════════════════
print("\n[Phase 2] TimeSeriesSplit 时序交叉验证 (防数据泄露) ...")
def run_timeseries_cv(n_splits=int(os.environ.get('V8_CV_FOLDS', 4)),
                      max_rounds=int(os.environ.get('V8_CV_ROUNDS', 300)), es_patience=10):
    dates_all = sorted(set(X_tr.index.get_level_values('date')))
    chunk = int(np.ceil(len(dates_all) / (n_splits + 1)))
    folds = []
    for i in range(1, n_splits + 1):
        tr_dates = set(dates_all[: i * chunk])
        va_dates = set(dates_all[i * chunk: (i + 1) * chunk])
        if not va_dates:
            break
        m_tr = X_tr.index.get_level_values('date').isin(tr_dates)
        m_va = X_tr.index.get_level_values('date').isin(va_dates)
        Xf_tr, yf_tr = X_tr.loc[m_tr], y_tr.loc[m_tr]
        Xf_va, yf_va = X_tr.loc[m_va], y_tr.loc[m_va]
        ycf_tr = to_class3(yf_tr.values); ycf_va = to_class3(yf_va.values)
        # Plan C' 加权回归权重 (fold 内): |y|>50 → 1.0, 否则 → 0.2
        rw_tr = np.where(np.abs(yf_tr.values) > T, W_BIG, W_NEU).astype(np.float32)
        rw_va = np.where(np.abs(yf_va.values) > T, W_BIG, W_NEU).astype(np.float32)
        d_c_tr = xgb.DMatrix(Xf_tr.values, label=ycf_tr)
        d_c_va = xgb.DMatrix(Xf_va.values, label=ycf_va)
        d_r_tr = xgb.DMatrix(Xf_tr.values, label=yf_tr.values, weight=rw_tr)
        d_r_va = xgb.DMatrix(Xf_va.values, label=yf_va.values, weight=rw_va)
        bc = xgb.Booster({**CLF_P, 'learning_rate': 0.05}, [d_c_tr])
        br = xgb.Booster({**REG_P, 'learning_rate': 0.05}, [d_r_tr])
        best_p, best_c, best_r, patience = -np.inf, None, None, 0
        for it in range(max_rounds):
            bc.update(d_c_tr, it, fobj=clf_cost_fobj)
            br.update(d_r_tr, it)
            pv = hard_penalty(yf_va.values, code_of(bc.predict(d_c_va).argmax(1)),
                              br.predict(d_r_va), t=T)[0]
            pt = hard_penalty(yf_tr.values, code_of(bc.predict(d_c_tr).argmax(1)),
                              br.predict(d_r_tr), t=T)[0]
            if pv > best_p + MIN_DELTA:
                best_p, best_c, best_r, patience = pv, bc.save_raw(), br.save_raw(), 0
            else:
                patience += 1
                if patience >= es_patience:
                    break
        bcl = xgb.Booster({**CLF_P, 'learning_rate': 0.05}); bcl.load_model(best_c)
        brg = xgb.Booster({**REG_P, 'learning_rate': 0.05}); brg.load_model(best_r)
        folds.append({'fold': i, 'train_dates': [dates_all[0], sorted(tr_dates)[-1]],
                      'val_dates': [sorted(va_dates)[0], sorted(va_dates)[-1]],
                      'n_train': int(m_tr.sum()), 'n_val': int(m_va.sum()),
                      'best_epoch': None, 'val_penalty': float(best_p),
                      'train_penalty': float(pt)})
        print(f"  fold{i}: train[{folds[-1]['train_dates'][0]}~{folds[-1]['train_dates'][1]}] "
              f"val[{folds[-1]['val_dates'][0]}~{folds[-1]['val_dates'][1]}] "
              f"Val_Penalty={best_p:.4f} Train_Penalty={pt:.4f}")
    cv_pen = [f['val_penalty'] for f in folds]
    return folds, float(np.mean(cv_pen)), float(np.std(cv_pen))

cv_folds, cv_pen_mean, cv_pen_std = run_timeseries_cv()
print(f"  TS-CV 汇总: Val_Penalty = {cv_pen_mean:.4f} ± {cv_pen_std:.4f} (均值±std)")

# ════════════════════════════════════════════════════════════════
# 5. 全量训练 (锁步训练 clf+reg, 惩罚函数驱动早停/最优选择, Plan C' 加权回归)
# ════════════════════════════════════════════════════════════════
print("\n[Phase 3] 全量训练 (DART dropout + L2 + 噪声注入 + Plan C' 加权回归 + 惩罚早停) ...")
d_va_c = xgb.DMatrix(X_va_np, label=yc_va_np)
d_va_r = xgb.DMatrix(X_va_np, label=y_va_np, weight=reg_w_va)   # Plan C' 加权回归权重
d_tr_c_clean = xgb.DMatrix(X_tr_np, label=yc_tr_np)             # 干净训练集 (评估惩罚用)
d_tr_r_clean = xgb.DMatrix(X_tr_np, label=y_tr_np, weight=reg_w_tr)

_d_tr_clf0 = xgb.DMatrix(X_tr_np, label=yc_tr_np, weight=sw_tr)   # 构造缓存 (保持存活)
_d_tr_reg0 = xgb.DMatrix(X_tr_np, label=y_tr_np, weight=reg_w_tr)
clf = xgb.Booster(CLF_P, [_d_tr_clf0])
reg = xgb.Booster(REG_P, [_d_tr_reg0])

best_val_pen, best_epoch, best_raw_c, best_raw_r = -np.inf, 0, None, None
best_val_loss, best_clf_l, best_reg_l = np.inf, np.inf, np.inf
clf_patience, reg_patience, plateau_waits = 0, 0, 0
lr = FIXED['learning_rate']
pen_hist = []   # (epoch, train_pen, val_pen, lr, val_loss)
t0 = time.time()

for it in range(MAX_ROUNDS):
    # 数据增强: 噪声只注入分类头 (正则); 回归头用干净特征保住大偏差量级 (防塌缩)
    X_aug = inject_noise(X_tr_np, it)
    d_c = xgb.DMatrix(X_aug, label=yc_tr_np, weight=sw_tr)
    d_r = xgb.DMatrix(X_tr_np, label=y_tr_np, weight=reg_w_tr)   # 干净特征
    clf.update(d_c, it, fobj=clf_cost_fobj)   # 分类头: 成本矩阵软目标 (规则 A/B 梯度)
    reg.update(d_r, it)                       # 回归头: gbtree + Plan C' 加权回归 (|y|>50→1.0, 否则→0.2)

    # 平滑验证损失 (早停/LR 用; 归一化到各自 epoch0)
    clf_l = clf_smooth_loss(clf, d_va_c, yc_va_np)
    reg_l = reg_smooth_loss(reg, d_va_r, y_va_np, reg_w_va)
    if it == 0:
        clf_l0, reg_l0 = clf_l, reg_l
    val_loss = clf_l / clf_l0 + reg_l / reg_l0
    # 硬惩罚评分 (最优权重选择用, 干净数据)
    pv = hard_penalty(y_va_np, code_of(clf.predict(d_va_c).argmax(1)), reg.predict(d_va_r), t=T)
    pt = hard_penalty(y_tr_np, code_of(clf.predict(d_tr_c_clean).argmax(1)),
                      reg.predict(d_tr_r_clean), t=T)
    pen_hist.append((it, pt[0], pv[0], lr, val_loss))

    # 最优权重: 按硬 Val_Penalty 选择 (只保存惩罚分数最优迭代)
    if pv[0] > best_val_pen + MIN_DELTA:
        best_val_pen, best_epoch = pv[0], it
        best_raw_c, best_raw_r = clf.save_raw(raw_format='ubj'), reg.save_raw(raw_format='ubj')

    if it % 10 == 0 or it == 0:
        print(f"  [Epoch {it:4d}] Train_Penalty={pt[0]:+.4f} | Val_Penalty={pv[0]:+.4f} "
              f"| bestPen={best_val_pen:+.4f} | Val_Loss={val_loss:.4f} | lr={lr:.4f} "
              f"| 触发={pv[2]}/{len(y_va_np)} | {time.time()-t0:.0f}s")

    # 分类/回归双头各自的验证损失停滞计数 (早停: 双头都停滞才停)
    if clf_l < best_clf_l - LOSS_MIN_DELTA:
        best_clf_l, clf_patience = clf_l, 0
    else:
        clf_patience += 1
    if reg_l < best_reg_l - LOSS_MIN_DELTA:
        best_reg_l, reg_patience = reg_l, 0
    else:
        reg_patience += 1

    if clf_patience >= EARLY_STOP_PATIENCE and reg_patience >= EARLY_STOP_PATIENCE:
        print(f"  EarlyStopping @ epoch {it}: 分类头(停滞{clf_patience}) + "
              f"回归头(停滞{reg_patience}) 均 {EARLY_STOP_PATIENCE} 轮未改善")
        break

    # ReduceLROnPlateau: 归一化验证损失停滞 → lr 减半
    if val_loss >= best_val_loss - LOSS_MIN_DELTA:
        plateau_waits += 1
        if plateau_waits >= PLATEAU_PATIENCE and lr > MIN_LR:
            lr = max(lr / 2, MIN_LR)
            clf.set_param({'learning_rate': lr})
            reg.set_param({'learning_rate': lr})
            plateau_waits = 0
            print(f"    ↓ ReduceLROnPlateau: lr → {lr:.4f} (验证损失停滞)")
    else:
        best_val_loss = min(best_val_loss, val_loss)
        plateau_waits = 0

print(f"  训练结束: 最佳迭代={best_epoch}, 最优 Val_Penalty={best_val_pen:+.4f}, "
      f"耗时 {time.time()-t0:.0f}s")

# 载入最优迭代权重 (丢弃最终 epoch)
best_clf = xgb.Booster({**CLF_P, 'learning_rate': 0.05})
best_reg = xgb.Booster({**REG_P, 'learning_rate': 0.05})
best_clf.load_model(best_raw_c)
best_reg.load_model(best_raw_r)
print(f"  已载入最优迭代权重 (epoch {best_epoch})")

# ════════════════════════════════════════════════════════════════
# 6. 评估 (test/eval) — 触发率 / 数值驱动分级命中 / 惩罚 / 条件回归 RMSE
# ════════════════════════════════════════════════════════════════
print("\n[Phase 4] 评估 ...")
def evaluate_nn(Xs, ys, tag):
    d = xgb.DMatrix(Xs.values.astype(np.float32))
    yc_codes = code_of(best_clf.predict(d).argmax(1))
    yv = best_reg.predict(d)
    ys = np.asarray(ys.values, dtype=float)
    yt3 = to_class3(ys)
    acc = (yc_codes == code_of(yt3)).mean()
    nonneu = yt3 != 1
    pred_dir = np.sign(yc_codes - NCODE)
    true_dir = np.sign(ys) * (np.abs(ys) > T)
    sign_hit = (pred_dir[nonneu] == true_dir[nonneu]).mean() if nonneu.sum() else np.nan
    yt_big = ((yt3 == 0) | (yt3 == 2)).astype(int)
    yp_big = ((yc_codes == 0) | (yc_codes == 4)).astype(int)
    big_f1 = f1_score(yt_big, yp_big, zero_division=0)
    big_rec = recall_score(yt_big, yp_big, zero_division=0)
    big_prec = precision_score(yt_big, yp_big, zero_division=0)
    trig_rate = (yc_codes != NCODE).mean()
    pen_mean, pen_sum, n_trig = hard_penalty(ys, yc_codes, yv, t=T, verbose=True)

    # ── 数值驱动分级 (部署侧判定口径): |reg| < T 正常 | T~T_BIG 小偏差 | ≥T_BIG 大偏差 ──
    v_abs = np.abs(yv)
    num_tier = np.where(v_abs >= T_BIG, 2, np.where(v_abs >= T, 1, 0))   # 0正常/1小/2大
    yt_tier = np.where(np.abs(ys) >= T_BIG, 2, np.where(np.abs(ys) >= T, 1, 0))
    trig_num = num_tier != 0
    num_rec  = recall_score((yt_tier != 0).astype(int), trig_num.astype(int), zero_division=0)
    num_prec = precision_score((yt_tier != 0).astype(int), trig_num.astype(int), zero_division=0)
    big_acc = (num_tier[trig_num] == yt_tier[trig_num]).mean() if trig_num.sum() else np.nan
    small_f1 = f1_score((yt_tier == 1).astype(int), (num_tier == 1).astype(int), zero_division=0)
    big_f1_num = f1_score((yt_tier == 2).astype(int), (num_tier == 2).astype(int), zero_division=0)

    # 条件回归 RMSE (仅 |y|>50 样本)
    m_cond = np.abs(ys) > T
    rmse_cond = np.sqrt(mean_squared_error(ys[m_cond], yv[m_cond])) if m_cond.sum() else np.nan
    # 触发样本的数值命中: 触发中 |reg|≥T 的比例 (v8 病征的直接度量)
    trig_mask = yc_codes != NCODE
    hit50 = (v_abs[trig_mask] >= T).mean() if trig_mask.sum() else np.nan

    print(f"  [{tag}] 3类acc={acc:.3f} | sign命中={sign_hit:.3f} | bigF1={big_f1:.3f}"
          f"(R{big_rec:.2f}/P{big_prec:.2f}) | 触发率={trig_rate:.3f}")
    print(f"         数值驱动: 触发R={num_rec:.2f}/P={num_prec:.2f} | 分级acc={big_acc:.3f} "
          f"| 小F1={small_f1:.3f} 大F1={big_f1_num:.3f} | 触发中|值|≥50={hit50:.3f}")
    print(f"         条件RMSE(|y|>50)={rmse_cond:.1f} | Val_Penalty={pen_mean:+.4f} | RMSE全量={np.sqrt(mean_squared_error(ys,yv)):.1f}")
    return dict(acc=float(acc), sign_hit=float(sign_hit), big_f1=float(big_f1),
                big_recall=float(big_rec), big_precision=float(big_prec),
                trigger_rate=float(trig_rate), penalty_mean=float(pen_mean),
                penalty_sum=float(pen_sum), rmse_cond=float(rmse_cond),
                num_recall=float(num_rec), num_precision=float(num_prec),
                tier_acc=float(big_acc), small_f1=float(small_f1),
                big_f1_num=float(big_f1_num), trig_hit50=float(hit50),
                trigger_rate_num=float(trig_num.mean()), rmse=float(np.sqrt(mean_squared_error(ys, yv))),
                n=len(ys))

m_val = evaluate_nn(X_va, y_va, 'Valid')
m_test = evaluate_nn(X_te, y_te, 'Test ')
m_eval = evaluate_nn(X_ev, y_ev, 'Eval ')

# ════════════════════════════════════════════════════════════════
# 7. 旧阈值基线对照 (同一窗口, v8 阈值 50/50) — Val_Penalty 对比
# ════════════════════════════════════════════════════════════════
print("\n[Phase 5] 旧基线对照 (v8: models/xgb_v8_20260812_1528.joblib, 阈值 50/50) ...")
OLD_PATH = f'{YONON_PATH}/models/xgb_v8_20260812_1528.joblib'
baseline = {}
if os.path.exists(OLD_PATH):
    old = joblib.load(OLD_PATH)
    old_feats = old['features']
    # 加载旧模型特征 (缺失文件 → 该列 NaN)
    flo = []
    base_idx = X.index
    for name in old_feats:
        p = f'{dataset_path}/{name}.fea'
        if not os.path.exists(p):
            s = pd.Series(np.nan, index=base_idx, name=name)
        else:
            df = pd.read_feather(p); df.index = df.index.astype(str)
            s = df.stack(); s.name = name
        flo.append(s)
    Xo = pd.concat(flo, axis=1)
    Xo.index = Xo.index.rename(['date', 'hour'])
    Xo = Xo.sort_index()
    def eval_old(Xs, ys, tag):
        Xs = Xs.loc[Xs.index.intersection(Xo.index)]
        ys = ys.loc[Xs.index]
        Xok = Xo.loc[Xs.index]
        yc_old = old['clf'].predict(Xok)
        yv_old = old['reg'].predict(Xok)
        ysv = np.asarray(ys.values, float)
        p = hard_penalty(ysv, np.asarray(yc_old), np.asarray(yv_old, float), t=T)
        # 触发中 |值|≥50 命中
        trig = np.asarray(yc_old) != NCODE
        hit50 = (np.abs(np.asarray(yv_old, float))[trig] >= T).mean() if trig.sum() else np.nan
        print(f"  [旧v8/{tag}] Val_Penalty={p[0]:+.4f} 触发={p[2]}/{len(ys)} 触发中|值|≥50={hit50:.3f}")
        return p[0], hit50
    baseline['valid'] = eval_old(X_va, y_va, 'Valid')
    baseline['test'] = eval_old(X_te, y_te, 'Test ')
    baseline['eval'] = eval_old(X_ev, y_ev, 'Eval ')
else:
    print("  ⚠️ 旧模型不存在, 跳过基线对照")

# ════════════════════════════════════════════════════════════════
# 8. 保存 (只存 Val_Penalty 最优权重; joblib 自包含, 兼容 deploy 推理)
# ════════════════════════════════════════════════════════════════
print("\n[Phase 6] 保存模型 ...")
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M')
payload = {
    'clf': _BoosterClf(code_of_cls=CODES, booster=best_clf),
    'reg': _BoosterReg(booster=best_reg),
    'features': fresh,
    'threshold_minor': T, 'threshold_big': T_BIG,
    'classes': SPREAD_CLASSES, 'class_map': SPREAD_CLASS_MAP,
    'class_codes': CODES, 'effective_classes': NCLS3,
    'decision_rule': (f"3类内部头 big_neg(<-{T})/neu(±{T})/big_pos(>{T}) → 项目编码 {{0,2,4}}; "
                      f"回归头 Plan C' 加权回归 (|y|>{T}→{W_BIG}, |y|≤{T}→{W_NEU}, 全量样本); "
                      f"部署侧数值驱动: |reg|<{T} 正常 / {T}≤|reg|<{T_BIG} 小偏差 / |reg|≥{T_BIG} 大偏差"),
    'conditional_reg': {'threshold': T, 'mode': REG_MODE,
                        'note': f"Plan C' 加权回归: |y|>{T}→{W_BIG}, |y|≤{T}→{W_NEU}, 全量样本训练 (防数值塌缩)"},
    'trigger_rule': 'value_driven',
    'fixed_params': FIXED,
    'penalty_history': pen_hist,
    'best_epoch': best_epoch, 'best_val_penalty': best_val_pen,
    'cv_timeseries': {'folds': cv_folds, 'mean': cv_pen_mean, 'std': cv_pen_std},
    'metrics': {'valid': m_val, 'test': m_test, 'eval': m_eval},
    'baseline_v8_penalty': baseline,
    'augmentation': {'sigma_ratio': 0.01, 'frac': 0.4, 'per_epoch_noise': True},
    'trained_at': ts,
    'train_window': ['2025-01-01', '2026-05-31'],
    'valid_window': ['2026-06-01', '2026-07-15'],
}
save_ts = f'{YONON_PATH}/models/xgb_v8.1_{ts}.joblib'
joblib.dump(payload, save_ts)
save_fixed = f'{YONON_PATH}/models/xgb_v8.1.joblib'
joblib.dump(payload, save_fixed)
print(f"  已保存: {save_ts} ({os.path.getsize(save_ts)/1e6:.1f} MB) + xgb_v8.1.joblib")

# 结果摘要
summary = {
    'model': 'v8.1', 'threshold': T, 'threshold_big': T_BIG,
    'trigger_rule': 'value_driven',
    'best_epoch': best_epoch, 'best_val_penalty': best_val_pen,
    'cv_mean': cv_pen_mean, 'cv_std': cv_pen_std,
    'metrics': payload['metrics'], 'baseline_v8': baseline,
    'conditional_reg': {'threshold': T, 'mode': REG_MODE,
                        'note': f"回归头 Plan C' 加权回归: |y|>{T}→{W_BIG}, |y|≤{T}→{W_NEU}, 全量样本训练 (防数值塌缩)"},
    'overfit_gap_valid': m_val['penalty_mean'] - pen_hist[-1][1],
}
out_json = f'{YONON_PATH}/data/v8.1_results.json'
json.dump(summary, open(out_json, 'w'), indent=2, ensure_ascii=False)
print(f"  结果摘要: {out_json}")

# 打印惩罚曲线 (文本)
print("\n" + "=" * 72)
print("  Val_Penalty 曲线 (每 10 轮):")
for e, pt, pv, lr, vl in pen_hist:
    if e % 10 == 0 or e == best_epoch:
        mark = " ◀ best(惩罚)" if e == best_epoch else ""
        print(f"    epoch {e:4d} | Train_Penalty={pt:+.4f} | Val_Penalty={pv:+.4f} | Val_Loss={vl:.3f}{mark}")
print("=" * 72)
print(f"  v8.1 完成 | 最优 Val_Penalty={best_val_pen:+.4f} @ epoch {best_epoch}")
print(f"  Test  Val_Penalty={m_test['penalty_mean']:+.4f}")
print("=" * 72)
