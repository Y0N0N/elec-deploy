"""
Pipeline v7.2 — T112: 事件因子增量验证 (C1 落地)

背景: 50 个事件语义因子 (ev_*) 已生成 (51_tool_event_factors_v1.py),
      覆盖气量管控/必停/停电检修/短路控制等 828 种调度事件。
本实验: 与 scarcity 脚本 **同协议** (每周重训 + 稀缺权重 + 同种子),
  唯一改动 = 在 369 特征基础上追加 50 个 ev_* 特征,
  对比 sign_hit / 漏报率是否提升 (事件因子是否有增量价值)。

红线: 同 walk-forward — 评估日 d 绝不进 train/valid; ev_* 因子滚动基线只用过去, 无未来泄露。

产物: data/walkforward_v7.2_preds_ev.csv + data/ev_compare.json + show/v7/report_ev.png
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

W_SCALE = float(os.environ.get('W_SCALE', 3))
W_MORNING = float(os.environ.get('W_MORNING', 1.5))
STEP_DAYS = int(os.environ.get('WF_STEP', 7))
VALID_DAYS = int(os.environ.get('WF_VALID', 30))
N_EST = int(os.environ.get('WF_NEST', 400))
JOBS = int(os.environ.get('WF_JOBS', 8))
BT_START = os.environ.get('WF_START', '2026-02-01')
BT_END = os.environ.get('WF_END', '2026-07-26')

print("=" * 72)
print("  T112: 事件因子增量验证 (C1 落地)")
print(f"  369 特征 + 50 事件因子 | 稀缺权重 {W_SCALE}x/凌晨{W_MORNING}x | "
      f"回测 {BT_START}~{BT_END} | n_est={N_EST} | n_jobs={JOBS}")
print("=" * 72)

# ============ 加载特征 (369 + 50 ev) ============
print("\n[Phase 0] 加载新鲜集 369 + 事件因子 50...", flush=True)
t0 = time.time()
with open(f'{YONON_PATH}/data/v7_fresh_features.json') as f:
    fresh = json.load(f)['fresh_features']
ev_names = sorted([n[:-4] for n in os.listdir(dataset_path)
                   if n.startswith('ev_') and n.endswith('.fea')])
print(f"  事件因子: {len(ev_names)} 个", flush=True)

fl = []
for name in fresh + ev_names:
    df = pd.read_feather(f'{dataset_path}/{name}.fea')
    s = df.stack(); s.name = name
    fl.append(s)
X = pd.concat(fl, axis=1); del fl; gc.collect()
idx_dates = pd.to_datetime(X.index.get_level_values(0)).strftime('%Y-%m-%d')
idx_hours = [f'{int(h):02d}:00' if isinstance(h, (int, np.integer)) else str(h)
             for h in X.index.get_level_values(1)]
X.index = pd.MultiIndex.from_arrays([idx_dates, idx_hours], names=['date', 'hour'])
X = X.sort_index()
print(f"  X: {X.shape}, 加载 {time.time()-t0:.1f}s", flush=True)

# ============ 标签 ============
spread = pd.read_feather(spread_label_file); spread.index = pd.to_datetime(spread.index)
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

# ============ 稀缺权重 (与 scarcity 相同) ============
def scarcity_weight(y_spread, hours):
    yc = to_class5(y_spread)
    cnt = np.bincount(yc, minlength=NCLS)
    w_class = len(yc) / (NCLS * cnt)
    w_class = w_class[yc]
    mag = np.clip(np.abs(y_spread) / T2, 1.0, W_SCALE)
    morning = np.where(hours < 6, W_MORNING, 1.0)
    return w_class * mag * morning

# ============ 折叠规划 ============
dates_ts = pd.to_datetime(np.array(sorted(y.index.get_level_values('date').unique())))
fold_starts = []
cursor = pd.Timestamp(BT_START)
while cursor <= pd.Timestamp(BT_END):
    cand = dates_ts[dates_ts >= cursor]
    if len(cand) == 0: break
    d0 = cand[0]
    if d0 > pd.Timestamp(BT_END): break
    fold_starts.append(d0)
    cursor = d0 + pd.Timedelta(days=STEP_DAYS)
print(f"  折叠数: {len(fold_starts)}")

# ============ 逐折训练 ============
pred_csv = f'{YONON_PATH}/data/walkforward_v7.2_preds_ev.csv'
done = set()
if os.path.exists(pred_csv):
    ex = pd.read_csv(pred_csv, dtype={'date': str})
    done = set(ex['date'].astype(str).unique())
    print(f"  断点续跑: 已有 {len(done)} 天")

fp = open(pred_csv, 'a') if os.path.exists(pred_csv) else open(pred_csv, 'w')
if not os.path.exists(pred_csv) or len(done) == 0:
    fp.write('date,hour,spread_true,clf_cls,reg_val\n')

t_start = time.time()
for D in fold_starts:
    D_s = D.strftime('%Y-%m-%d')
    f_end = D + pd.Timedelta(days=STEP_DAYS - 1)
    fold_dates = [d for d in np.array(sorted(y.index.get_level_values('date').unique()))
                  if D_s <= d <= min(f_end.strftime('%Y-%m-%d'), BT_END)]
    fold_dates = [d for d in fold_dates if d not in done]
    if not fold_dates: continue
    v_start = (D - pd.Timedelta(days=VALID_DAYS)).strftime('%Y-%m-%d')
    tr_m = xd < D_s
    va_m = (xd >= v_start) & (xd < D_s)
    Xt, yt = subset(tr_m, y)
    Xv, yv = subset(va_m, y)
    yct = to_class5(yt); ycv = to_class5(yv)
    tr_hours = Xt.index.get_level_values('hour').str[:2].astype(int).values
    sw = scarcity_weight(yt.values, tr_hours)
    tf0 = time.time()
    clf = xgb.XGBClassifier(n_estimators=N_EST, early_stopping_rounds=50,
        eval_metric='mlogloss', random_state=42, verbosity=0, n_jobs=JOBS, **FIXED)
    clf.fit(Xt, yct, sample_weight=sw, eval_set=[(Xv, ycv)], verbose=False)
    reg = xgb.XGBRegressor(n_estimators=N_EST, early_stopping_rounds=50,
        eval_metric='rmse', random_state=42, verbosity=0, n_jobs=JOBS, **FIXED)
    reg.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)

    m_fo = (xd >= fold_dates[0]) & (xd <= fold_dates[-1])
    Xf, yf = subset(m_fo, y)
    out = pd.DataFrame({'date': Xf.index.get_level_values('date'),
                        'hour': Xf.index.get_level_values('hour'),
                        'spread_true': yf.values,
                        'clf_cls': clf.predict(Xf), 'reg_val': reg.predict(Xf)})
    out.to_csv(fp, header=False, index=False)
    fp.flush(); done.update(fold_dates)
    el = time.time() - t_start
    print(f"  [fold {D_s}] train={len(Xt)} 预测{fold_dates[0]}~{fold_dates[-1]} | 累计 {el/60:.1f}min", flush=True)
fp.close()
print(f"\n  预测已保存: {pred_csv}")

# ============ 对比: 事件因子 vs 稀缺基线 ============
print("\n[对比] 事件因子模型 vs 稀缺权重基线 (同日期):")
ev = pd.read_csv(pred_csv, dtype={'date': str, 'hour': str})
bs = pd.read_csv(f'{YONON_PATH}/data/walkforward_v7.2_preds_scarcity.csv', dtype={'date': str, 'hour': str})
ev = ev.set_index(['date', 'hour']); bs = bs.set_index(['date', 'hour'])
common_idx = ev.index.intersection(bs.index)
bs = bs.loc[common_idx]; ev = ev.loc[common_idx]
print(f"  对齐样本: {len(common_idx)} ({common_idx.get_level_values('date').min()}~{common_idx.get_level_values('date').max()})")

def miss_metrics(pred_df, tag):
    p = pred_df.copy()
    p['abs_sp'] = p['spread_true'].abs()
    p['is_big'] = p['abs_sp'] > T2
    p['clf_big'] = p['clf_cls'].isin([0, 4])
    p['h'] = p.index.get_level_values('hour').str[:2].astype(int)
    big = p[p['is_big']]
    n_big = len(big); n_miss = int((~big['clf_big']).sum())
    morning = big[big['h'] < 6]
    n_miss_morning = int((~morning['clf_big']).sum()) if len(morning) else 0
    bnd = p[(p['abs_sp'] > T2) & (p['abs_sp'] <= 25)]
    n_miss_bnd = int((~bnd['clf_big']).sum()) if len(bnd) else 0
    yt = (p['is_big']).astype(int); yp = p['clf_big'].astype(int)
    big_f1 = f1_score(yt, yp, zero_division=0)
    big_rec = recall_score(yt, yp, zero_division=0)
    big_prec = precision_score(yt, yp, zero_division=0)
    nonneu = p['abs_sp'] > T1
    dir_p = np.sign(p['clf_cls'].values - 2)
    dir_t = np.sign(np.where(p['spread_true'].values == 0, 1e-9, p['spread_true'].values))
    sign_hit = (dir_p[nonneu.values] == dir_t[nonneu.values]).mean() if nonneu.sum() else np.nan
    print(f"  [{tag}] big漏报率 {n_miss/n_big*100:.2f}% | 凌晨漏报率 {n_miss_morning/max(len(morning),1)*100:.1f}% | "
          f"边界漏报率 {n_miss_bnd/max(len(bnd),1)*100:.1f}% | bigF1 {big_f1:.3f}(R{big_rec:.2f}/P{big_prec:.2f}) | sign {sign_hit:.3f}")
    return dict(n_big=n_big, n_miss=n_miss, miss_rate=n_miss/max(n_big,1),
                n_miss_morning=n_miss_morning, n_morning=len(morning),
                morning_miss_rate=n_miss_morning/max(len(morning),1),
                n_miss_boundary=n_miss_bnd, n_boundary=len(bnd),
                boundary_miss_rate=n_miss_bnd/max(len(bnd),1),
                big_f1=big_f1, big_recall=big_rec, big_precision=big_prec, sign_hit=sign_hit)

m_sc = miss_metrics(bs, '稀缺基线')
m_ev = miss_metrics(ev, '事件因子')
print(f"\n  漏报率变化: 总体 {m_sc['miss_rate']*100:.1f}% → {m_ev['miss_rate']*100:.1f}% | "
      f"凌晨 {m_sc['morning_miss_rate']*100:.1f}% → {m_ev['morning_miss_rate']*100:.1f}% | "
      f"边界 {m_sc['boundary_miss_rate']*100:.1f}% → {m_ev['boundary_miss_rate']*100:.1f}% | "
      f"sign {m_sc['sign_hit']:.3f} → {m_ev['sign_hit']:.3f}")

res = dict(n_features=len(fresh)+len(ev_names), ev_features=len(ev_names),
           scarcity=m_sc, event_ev=m_ev,
           window=[str(common_idx.get_level_values('date').min()), str(common_idx.get_level_values('date').max())])
with open(f'{YONON_PATH}/data/ev_compare.json', 'w') as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=float)
print(f"\n  对比结果已保存: {YONON_PATH}/data/ev_compare.json")

# ============ 图 ============
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Micro Hei Mono']
    plt.rcParams['axes.unicode_minus'] = False
    cats = ['总体漏报率', '凌晨漏报率', '边界漏报率', 'bigF1', 'sign_hit']
    base_v = [m_sc['miss_rate']*100, m_sc['morning_miss_rate']*100, m_sc['boundary_miss_rate']*100, m_sc['big_f1']*100, m_sc['sign_hit']*100]
    ev_v = [m_ev['miss_rate']*100, m_ev['morning_miss_rate']*100, m_ev['boundary_miss_rate']*100, m_ev['big_f1']*100, m_ev['sign_hit']*100]
    x = np.arange(len(cats)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, base_v, w, label='稀缺基线', color='#8fa8c8')
    ax.bar(x + w/2, ev_v, w, label='事件因子', color='#e8a06a')
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel('%')
    ax.set_title('T112: 事件因子 vs 稀缺基线 (同 walk-forward 协议)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{YONON_PATH}/show/v7/report_ev.png', dpi=150); plt.close()
    print(f"  图已保存: {YONON_PATH}/show/v7/report_ev.png")
except Exception as e:
    print(f"  图跳过: {e}")

print("\n" + "=" * 72)
print("  T112 事件因子增量验证完成")
print("=" * 72)
