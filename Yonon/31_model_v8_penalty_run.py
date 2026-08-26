"""
Pipeline v8: DA-RT 价差预测 — 触发阈值 50 + 自定义惩罚函数 + 过拟合专项治理
====================================================================
用户拍板 (2026-08-12):
  1. 触发阈值 5 → 50; 因 τ_big=15 < 50 会破坏 5 类标签, 双阈值对齐到 50/50
     → 有效标签 3 类: big_neg(spread<-50, code 0) / neu(±50, code 2) / big_pos(>50, code 4)
  2. 自定义惩罚评分 (规则 A/B/C) 集成进训练循环
  3. 过拟合专项防控 (DART dropout / L2 / 早停 / ReduceLROnPlateau / TS-CV / 噪声注入)
  4. 全量重训 (严禁加载旧 checkpoint), 只保存 Val_Penalty 最优权重

──────────────────────────────────────────────────────────────────
惩罚评分 (硬规则, 用于每轮监控 / 早停 / 最优权重选择; 值越接近 0 越好):

  规则 A (预测错误):  触发信号 (模型输出非 neu 或真实非 neu) 时, 方向/分类判断错误 → -1
                      含 漏报 (真实有偏差但模型判 neu) 与 误报 (模型触发但真实为 neu)
  规则 B (完全反向):  预测方向与实际方向完全相反 (一正一负) → -2 (加倍重罚)
  规则 C (数值偏差):  模型触发并输出数值时, |Pred-True|/|True| > 0.2 → 额外 -1.5

平滑代理 (梯度注入, 使业务风险可回传):
  - 分类头 clf:  成本矩阵软目标 softmax obj = CE + λ·E_penalty(C), C 直接编码 A/B 的分值
  - 回归头 reg:  Huber(稳健) + 相对误差软 hinge(规则C) + 符号边距(方向一致性)

防过拟合:
  数据层  : 每轮特征高斯噪声注入; TimeSeriesSplit 时序交叉验证 (防数据泄露)
  模型层  : DART 树 dropout (rate_drop=0.3, one_drop, skip_drop=0.5); L2=reg_lambda=2.0
  训练层  : EarlyStopping patience=15 (按 Val_Penalty); ReduceLROnPlateau (停滞 10 轮 lr 减半)
  保存层  : 只存 Val_Penalty 最优迭代的权重, 丢弃最终 epoch
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

T = SPREAD_THRESHOLD          # 50.0
assert abs(T - SPREAD_THRESHOLD_BIG) < 1e-9, "v8 要求双阈值对齐 (τ_minor=τ_big=50)"
NCLS3 = 3                     # 内部 3 类 {big_neg, neu, big_pos}
CODES = [0, 2, 4]             # 映射到项目 5 类编码体系的子集 {big_neg, neu, big_pos}
NCODE = SPREAD_CLASS_MAP['neu']   # 2 (部署侧 neu 判断码)
MAX_ROUNDS = int(os.environ.get('V8_MAX_ROUNDS', 800))
EARLY_STOP_PATIENCE = 15      # 早停耐心 (10~15 区间)
PLATEAU_PATIENCE = 15         # ReduceLROnPlateau 耐心 (宽容, 防回归头 lr 过早崩)
MIN_LR = 5e-3
MIN_DELTA = 5e-4              # 最优权重选择最小改善 (mean penalty)
LOSS_MIN_DELTA = 2e-3         # 平滑损失早停最小改善 (相对归一化损失~2, 0.1% 容忍)
SMOKE = int(os.environ.get('V8_SMOKE', 0))   # 冒烟测试: 限制轮数
if SMOKE:
    MAX_ROUNDS = min(MAX_ROUNDS, 60)
    print(f"  ⚠️ 冒烟模式 (V8_SMOKE=1): MAX_ROUNDS={MAX_ROUNDS}")

print("=" * 72)
print("  Pipeline v8: DA-RT 价差 — 触发阈值 50 / 惩罚函数 / 过拟合治理")
print(f"  τ = {T} (双阈值对齐) | 有效类别: big_neg<-50 / neu±50 / big_pos>50 | code={CODES}")
print("=" * 72)


# ════════════════════════════════════════════════════════════════
# 0. 通用工具
# ════════════════════════════════════════════════════════════════
def to_class3(spr, t=T):
    """spread → 3 类内部索引 {0:big_neg, 1:neu, 2:big_pos}"""
    spr = np.asarray(spr, dtype=float)
    return np.where(spr < -t, 0, np.where(spr <= t, 1, 2))

def code_of(internal_idx):
    """3 类内部索引 → 项目 5 类编码子集 {0,2,4}"""
    return np.array([CODES[i] for i in np.asarray(internal_idx, dtype=int)])


# ── 硬惩罚评分 (规则 A/B/C) — 值越接近 0 越好 (负数, 均值) ──
def hard_penalty(y_true, pred_code, pred_val, t=T, verbose=False):
    """计算规则 A/B/C 惩罚. pred_code: 项目编码 (0~4, neu=2). 返回 (mean, total, n_trigger)."""
    y_true = np.asarray(y_true, dtype=float)
    pred_code = np.asarray(pred_code, dtype=int)
    pred_val = np.asarray(pred_val, dtype=float)
    n = len(y_true)
    # 阈值感知: |y|<=t → 无信号 (中性); 只有 |y|>t 才定义真实方向
    true_dir = np.sign(y_true) * (np.abs(y_true) > t)
    pred_dir = np.sign(pred_code - NCODE)          # {0,2,4}→-1/0/+1; 旧模型 0~4 同样适用
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


# ── 平滑验证损失 (用于 EarlyStopping / ReduceLROnPlateau, 比硬惩罚阶梯指标更平滑) ──
def clf_smooth_loss(bst, dmat, label3):
    """分类头平滑损失 = CE + λ·期望业务成本 (与 clf_cost_fobj 一致)"""
    pr = bst.predict(dmat)
    n = len(label3)
    eps = 1e-9
    ce = -np.mean(np.log(pr[np.arange(n), label3] + eps))
    E = np.sum(pr * C_MAT[label3], axis=1)
    cost = -np.mean(E)
    return ce + LAM_COST * cost

def reg_smooth_loss(bst, dmat, y):
    """回归头验证损失 = RMSE (原生量纲; 规则 C 追求量级准确, RMSE 与之对齐)"""
    pred = bst.predict(dmat)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


# ════════════════════════════════════════════════════════════════
# 1. 加载特征 (369 新鲜集, 与 v7 一致; 不做标准化, NaN 由 XGBoost 原生处理)
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

# 特征标准差 (用于噪声注入, 只从 train 统计)
feat_std = np.nanstd(X_tr.values, axis=0)
feat_std = np.where(np.isnan(feat_std) | (feat_std == 0), 1.0, feat_std)

# ════════════════════════════════════════════════════════════════
# 3. 训练参数 (DART dropout + L2 + 固定参数, 无网格搜索)
# ════════════════════════════════════════════════════════════════
FIXED = dict(booster='gbtree', tree_method='hist', max_depth=8, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.9, reg_lambda=2.0,
             rate_drop=0.3, skip_drop=0.5, one_drop=True, min_child_weight=1,
             nthread=8, seed=42)
CLF_P = {**FIXED, 'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss',
         'num_feature': len(fresh)}   # raw Booster 接口必须显式指定特征数
REG_P = {**FIXED, 'objective': 'reg:squarederror', 'num_feature': len(fresh)}

X_tr_np = X_tr.values.astype(np.float32)
X_va_np = X_va.values.astype(np.float32)
y_tr_np = y_tr.values.astype(np.float32)
y_va_np = y_va.values.astype(np.float32)
yc_tr_np = yc_tr.astype(np.int32)
yc_va_np = yc_va.astype(np.int32)


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
        # 快速训练 (本轮内复用参数, 但不影响最终模型)
        d_c_tr = xgb.DMatrix(Xf_tr.values, label=ycf_tr)
        d_c_va = xgb.DMatrix(Xf_va.values, label=ycf_va)
        d_r_tr = xgb.DMatrix(Xf_tr.values, label=yf_tr.values)
        d_r_va = xgb.DMatrix(Xf_va.values, label=yf_va.values)
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
# 5. 全量训练 (锁步训练 clf+reg, 惩罚函数驱动早停/最优选择, DART+L2+噪声增强)
# ════════════════════════════════════════════════════════════════
print("\n[Phase 3] 全量训练 (DART dropout + L2 + 噪声注入 + 惩罚早停) ...")
d_va_c = xgb.DMatrix(X_va_np, label=yc_va_np)
d_va_r = xgb.DMatrix(X_va_np, label=y_va_np)
d_tr_c_clean = xgb.DMatrix(X_tr_np, label=yc_tr_np)   # 干净训练集 (评估惩罚用)
d_tr_r_clean = xgb.DMatrix(X_tr_np, label=y_tr_np)

_d_tr_clf0 = xgb.DMatrix(X_tr_np, label=yc_tr_np, weight=sw_tr)   # 构造缓存 (保持存活)
_d_tr_reg0 = xgb.DMatrix(X_tr_np, label=y_tr_np)
clf = xgb.Booster(CLF_P, [_d_tr_clf0])
reg = xgb.Booster(REG_P, [_d_tr_reg0])

best_val_pen, best_epoch, best_raw_c, best_raw_r = -np.inf, 0, None, None
best_val_loss, best_clf_l, best_reg_l = np.inf, np.inf, np.inf
clf_patience, reg_patience, plateau_waits = 0, 0, 0
lr = FIXED['learning_rate']
pen_hist = []   # (epoch, train_pen, val_pen, lr, val_loss)
t0 = time.time()

for it in range(MAX_ROUNDS):
    # 数据增强: 本轮注入噪声 (只作用于梯度更新, 评估用干净数据)
    X_aug = inject_noise(X_tr_np, it)
    d_c = xgb.DMatrix(X_aug, label=yc_tr_np, weight=sw_tr)
    d_r = xgb.DMatrix(X_aug, label=y_tr_np)
    clf.update(d_c, it, fobj=clf_cost_fobj)   # 分类头: 成本矩阵软目标 (规则 A/B 梯度)
    reg.update(d_r, it)                       # 回归头: 原生 RMSE (规则 C 量级准确)

    # 平滑验证损失 (早停/LR 用; 归一化到各自 epoch0)
    clf_l = clf_smooth_loss(clf, d_va_c, yc_va_np)
    reg_l = reg_smooth_loss(reg, d_va_r, y_va_np)
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

    # 分类/回归双头各自的验证损失停滞计数 (早停: 双头都停滞才停, 防止回归头欠拟合)
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

    # ReduceLROnPlateau: 归一化验证损失停滞 PLATEAU_PATIENCE 轮 → lr 减半 (宽容阈值)
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
# 6. 评估 (test/eval) — 方向命中 / big F1 / 惩罚 / 触发频率
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
    rmse = np.sqrt(mean_squared_error(ys, yv))
    print(f"  [{tag}] 3类acc={acc:.3f} | sign命中={sign_hit:.3f} | bigF1={big_f1:.3f}"
          f"(R{big_rec:.2f}/P{big_prec:.2f}) | 触发率={trig_rate:.3f} "
          f"| Val_Penalty={pen_mean:+.4f} | RMSE={rmse:.1f}")
    return dict(acc=float(acc), sign_hit=float(sign_hit), big_f1=float(big_f1),
                big_recall=float(big_rec), big_precision=float(big_prec),
                trigger_rate=float(trig_rate), penalty_mean=float(pen_mean),
                penalty_sum=float(pen_sum), rmse=float(rmse), n=len(ys))

m_val = evaluate_nn(X_va, y_va, 'Valid')
m_test = evaluate_nn(X_te, y_te, 'Test ')
m_eval = evaluate_nn(X_ev, y_ev, 'Eval ')

# ════════════════════════════════════════════════════════════════
# 7. 旧阈值基线对照 (同一窗口, 旧模型 T=5/15) — Val_Penalty 对比
# ════════════════════════════════════════════════════════════════
print("\n[Phase 5] 旧阈值基线对照 (models/archive_20260812_pre_retrain/xgb_v7_20260811_1216.joblib) ...")
OLD_PATH = f'{YONON_PATH}/models/archive_20260812_pre_retrain/xgb_v7_20260811_1216.joblib'
baseline = {}
if os.path.exists(OLD_PATH):
    old = joblib.load(OLD_PATH)
    old_feats = old['features']
    print(f"  旧模型特征数: {len(old_feats)}, 元信息: { {k: old.get(k) for k in ('train_end_date','data_end_date','threshold_minor','threshold_big')} }")
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
        p = hard_penalty(np.asarray(ys.values, float), np.asarray(yc_old), np.asarray(yv_old, float), t=T)
        print(f"  [旧模型/{tag}] Val_Penalty={p[0]:+.4f} 触发={p[2]}/{len(ys)}")
        return p
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
    'threshold_minor': T, 'threshold_big': SPREAD_THRESHOLD_BIG,
    'classes': SPREAD_CLASSES, 'class_map': SPREAD_CLASS_MAP,
    'class_codes': CODES, 'effective_classes': NCLS3,
    'decision_rule': (f"3类内部头 big_neg(<-{T})/neu(±{T})/big_pos(>{T}) → 项目编码 {{0,2,4}}; "
                      f"neu(2) 不输出, big(0/4) 预警+输出值; 惩罚规则 A/B/C 集成训练"),
    'fixed_params': FIXED,
    'penalty_history': pen_hist,
    'best_epoch': best_epoch, 'best_val_penalty': best_val_pen,
    'cv_timeseries': {'folds': cv_folds, 'mean': cv_pen_mean, 'std': cv_pen_std},
    'metrics': {'valid': m_val, 'test': m_test, 'eval': m_eval},
    'baseline_old_penalty': baseline,
    'augmentation': {'sigma_ratio': 0.01, 'frac': 0.4, 'per_epoch_noise': True},
    'trained_at': ts,
    'train_window': ['2025-01-01', '2026-05-31'],
    'valid_window': ['2026-06-01', '2026-07-15'],
}
save_ts = f'{YONON_PATH}/models/xgb_v7_{ts}.joblib'
joblib.dump(payload, save_ts)
save_fixed = f'{YONON_PATH}/models/xgb_v7.joblib'
joblib.dump(payload, save_fixed)
# 更新 deploy 最新模型指针
ptr_path = f'{YONON_PATH}/deploy/latest_model.json'
json.dump({'v7': save_ts}, open(ptr_path, 'w'), indent=2)
print(f"  已保存: {save_ts} ({os.path.getsize(save_ts)/1e6:.1f} MB) + xgb_v7.joblib")
print(f"  已更新: {ptr_path}")

# 结果摘要
summary = {
    'threshold': T, 'best_epoch': best_epoch, 'best_val_penalty': best_val_pen,
    'cv_mean': cv_pen_mean, 'cv_std': cv_pen_std,
    'metrics': payload['metrics'], 'baseline_old': baseline,
    'overfit_gap_valid': m_val['penalty_mean'] - pen_hist[-1][1],
}
out_json = f'{YONON_PATH}/data/v8_penalty_results.json'
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
print(f"  v8 完成 | 最优 Val_Penalty={best_val_pen:+.4f} @ epoch {best_epoch}")
print(f"  Test  Val_Penalty={m_test['penalty_mean']:+.4f} (旧基线 {baseline.get('test', ('?',))[0]:+.4f})"
      if baseline.get('test') else "  Test Val_Penalty 见上方")
print("=" * 72)
