"""
B103 大偏差遗漏案例分析 — 漏报的大价差是"边界误差"还是"实时事件结构性不可见"?

数据: data/walkforward_v7.2_preds_7d.csv (4224 干净样本, 2026-02-01~07-26)

问题拆解:
  1. 大偏差 (|spread|>15) 的漏报率是多少? 按月度/时段分布?
  2. 漏报的大价差量级: 刚过 15 (边界误差, 可训练改善) 还是极端 (>50/100, 疑似实时事件)?
  3. 漏报样本的供给侧特征 vs 捕获样本: 统计可区分吗?
       - 无差异 → 漏报在日前供给侧不可见 → B103 结构性坐实
       - 有差异 → 模型漏用了可用信号 → 可训练改善
  4. 结论: B103 判定 + 方案 C 依据

产物: data/b103_miss_results.json + show/v7/report_b103_miss.png
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Micro Hei Mono']
plt.rcParams['axes.unicode_minus'] = False

from config.config import dataset_path, YONON_PATH

OUT = f'{YONON_PATH}/data/b103_miss_results.json'
FIG = f'{YONON_PATH}/show/v7'
os.makedirs(FIG, exist_ok=True)
TAU_BIG = 15.0

print("=" * 72)
print("  B103 大偏差遗漏案例分析")
print("=" * 72)

pred = pd.read_csv(f'{YONON_PATH}/data/walkforward_v7.2_preds_7d.csv', dtype={'date': str, 'hour': str})
pred['abs_sp'] = pred['spread_true'].abs()
pred['is_big'] = pred['abs_sp'] > TAU_BIG
pred['clf_big'] = pred['clf_cls'].isin([0, 4])
pred['dir_true'] = np.sign(np.where(pred['spread_true'] == 0, 1e-9, pred['spread_true']))
pred['dir_clf'] = np.sign(pred['clf_cls'] - 2)
pred['ym'] = pred['date'].str[:7]
pred['hour_int'] = pred['hour'].str[:2].astype(int)

# ============ 1. 漏报率总览 ============
big = pred[pred['is_big']]
caught = big[big['clf_big']]
missed = big[~big['clf_big']]
print(f"\n[1] 大偏差 (|spread|>15): {len(big)} / {len(pred)} ({len(big)/len(pred)*100:.1f}%)")
print(f"  捕获 (clf 判 big): {len(caught)} ({len(caught)/len(big)*100:.1f}%)  | 漏报: {len(missed)} ({len(missed)/len(big)*100:.1f}%)")
print(f"  大偏差 F1: 由 bigF1 (R/P) 覆盖 — 见 walk-forward 0.866 (R0.95/P0.80)")

# ============ 2. 漏报的量级分布 (边界 vs 极端) ============
print("\n[2] 漏报大价差的量级分布:")
mag_bins = [(15, 25), (25, 50), (50, 100), (100, np.inf)]
for lo, hi in mag_bins:
    nm = ((missed['abs_sp'] > lo) & (missed['abs_sp'] <= hi)).sum()
    nc = ((caught['abs_sp'] > lo) & (caught['abs_sp'] <= hi)).sum()
    print(f"  |spread| {lo:>3}-{str(int(hi)) if hi<np.inf else '∞':>3}: 漏报 {nm:>3} / 该桶 {nm+nc:>3} "
          f"(漏报率 {nm/(nm+nc)*100:5.1f}%)")
extreme_miss = (missed['abs_sp'] > 50).mean()
print(f"  漏报中 |spread|>50 占比: {extreme_miss*100:.1f}%")

# ============ 3. 月度/时段漏报率 ============
print("\n[3] 月度大偏差漏报率:")
m_month = missed.groupby('ym').size()
b_month = big.groupby('ym').size()
for ym in sorted(b_month.index):
    print(f"  {ym}: 漏报 {m_month.get(ym,0):>3}/{b_month[ym]:>3} = {m_month.get(ym,0)/b_month[ym]*100:5.1f}%")

print("\n  时段大偏差漏报率:")
big['seg'] = np.where(big['hour_int'] < 6, '凌晨', np.where(big['hour_int'] < 12, '上午',
               np.where(big['hour_int'] < 18, '午后', '晚峰')))
missed = missed.copy(); caught = caught.copy()
missed['seg'] = np.where(missed['hour_int'] < 6, '凌晨', np.where(missed['hour_int'] < 12, '上午',
                np.where(missed['hour_int'] < 18, '午后', '晚峰')))
caught['seg'] = np.where(caught['hour_int'] < 6, '凌晨', np.where(caught['hour_int'] < 12, '上午',
                np.where(caught['hour_int'] < 18, '午后', '晚峰')))
for seg in ['凌晨', '上午', '午后', '晚峰']:
    nm = (missed['seg'] == seg).sum(); nb = (big['seg'] == seg).sum()
    print(f"  {seg}: 漏报 {nm:>3}/{nb:>3} = {nm/max(nb,1)*100:5.1f}%")

# ============ 4. 供给侧特征: 捕获 vs 漏报 (结构性判定) ============
print("\n[4] 供给侧特征对比 (捕获 vs 漏报) — B103 结构性判定")
FEATS = ['s_统调负荷_ma7', 's_光伏出力预测_ma7', 's_风电出力预测_ma7', 's_D日_ma7',
         's_预测出力_ma7', 'sp_wow_abs']
feat_vals = {}
for f in FEATS:
    df = pd.read_feather(f'{dataset_path}/{f}.fea').stack()
    df.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(df.index.get_level_values(0)).strftime('%Y-%m-%d'),
         [f'{int(h):02d}:00' if isinstance(h, (int, np.integer)) else str(h) for h in df.index.get_level_values(1)]],
        names=['date', 'hour'])
    feat_vals[f] = df

p = pred.set_index(['date', 'hour'])
for f in FEATS:
    p = p.join(feat_vals[f].rename(f), how='left')
p = p.reset_index()

cmp_rows = []
for f in FEATS:
    v_c = p.loc[p['is_big'] & p['clf_big'], f].dropna()
    v_m = p.loc[p['is_big'] & ~p['clf_big'], f].dropna()
    if len(v_c) < 10 or len(v_m) < 10:
        cmp_rows.append(dict(feat=f, caught_n=len(v_c), missed_n=len(v_m), cohens_d=np.nan, p_val=np.nan))
        continue
    d = (v_c.mean() - v_m.mean()) / np.sqrt((v_c.std()**2 + v_m.std()**2) / 2)
    _, pv = stats.ttest_ind(v_c, v_m, equal_var=False)
    cmp_rows.append(dict(feat=f, caught_n=len(v_c), missed_n=len(v_m),
                         caught_mean=round(v_c.mean(), 1), missed_mean=round(v_m.mean(), 1),
                         cohens_d=round(d, 3), p_val=pv))
cmp_df = pd.DataFrame(cmp_rows)
print(cmp_df.round(3).to_string(index=False))

# ============ 5. 漏报最极端案例 ============
print("\n[5] 漏报中 |spread| 最大的 8 个案例:")
ext = missed.nlargest(8, 'abs_sp')[['date', 'hour', 'spread_true', 'clf_cls', 'reg_val']]
print(ext.round(1).to_string(index=False))

# ============ 判定 ============
sig_feats = [r for r in cmp_rows if not pd.isna(r.get('p_val')) and r['p_val'] < 0.05 and abs(r['cohens_d']) > 0.2]
extreme_miss_rate = (missed['abs_sp'] > 50).mean()
verdict = dict(
    structural_support=(len(sig_feats) == 0),
    sig_feats=[r['feat'] for r in sig_feats],
    extreme_miss_rate=float(extreme_miss_rate),
    miss_rate=float(len(missed) / len(big)),
    n_missed=int(len(missed)), n_big=int(len(big)),
)
print("\n" + "=" * 72)
print(f"  B103 判定: 结构性不可见 vs 可训练边界误差")
print(f"  - 显著区分特征 (p<0.05 & |d|>0.2): {len(sig_feats)} 个 {[r['feat'] for r in sig_feats]}")
print(f"  - 漏报中极端价差 (>50) 占比: {extreme_miss_rate*100:.1f}%")
if len(sig_feats) == 0:
    print(f"  → 漏报大价差的供给侧特征与捕获样本无显著差异 → 日前供给侧不可见, B103 结构性坐实")
else:
    print(f"  → 部分特征可区分 → 模型漏用了可用信号, 有训练改善空间")
print("=" * 72)

# ============ 保存 ============
def _native(o):
    """递归把 numpy 类型转 Python 原生类型 (JSON 可序列化)"""
    if isinstance(o, dict):
        return {k: _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if hasattr(o, 'item'):
        try: return o.item()
        except Exception: pass
    return str(o)

res = dict(miss_rate=float(len(missed) / len(big)), n_big=int(len(big)), n_missed=int(len(missed)),
           caught=int(len(caught)),
           missed_mag_dist=[dict(lo=lo, hi=('inf' if np.isinf(hi) else hi), missed=int(((missed['abs_sp']>lo)&(missed['abs_sp']<=hi)).sum()),
                                 total=int(((big['abs_sp']>lo)&(big['abs_sp']<=hi)).sum())) for lo, hi in mag_bins],
           extreme_miss_rate=float(extreme_miss_rate),
           monthly_miss=[dict(ym=ym, missed=int(m_month.get(ym, 0)), big=int(b_month[ym]),
                              rate=float(m_month.get(ym, 0) / b_month[ym])) for ym in sorted(b_month.index)],
           feature_cmp=cmp_rows,
           verdict=verdict,
           top_missed=ext.to_dict('records'))
with open(OUT, 'w') as f:
    json.dump(_native(res), f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {OUT}")

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
# 左: 漏报率按量级桶
labels = [f"{lo}-{hi}" if hi < np.inf else f">{lo}" for lo, hi in mag_bins]
miss_frac = [(int(((missed['abs_sp']>lo)&(missed['abs_sp']<=hi)).sum()) / max(int(((big['abs_sp']>lo)&(big['abs_sp']<=hi)).sum()),1) * 100) for lo, hi in mag_bins]
axes[0].bar(labels, miss_frac, color='coral')
axes[0].set_xlabel('|spread| 量级桶'); axes[0].set_ylabel('漏报率 %')
axes[0].set_title('大偏差漏报率 vs 量级'); axes[0].set_ylim(0, 50); axes[0].grid(alpha=0.3)
# 右: 捕获 vs 漏报 特征对比 (Cohen's d)
feat_names = [r['feat'].replace('s_', '').replace('_ma7', '') for r in cmp_rows]
ds = [r['cohens_d'] if not pd.isna(r['cohens_d']) else 0 for r in cmp_rows]
colors = ['green' if abs(d) < 0.2 else 'red' for d in ds]
axes[1].barh(feat_names, ds, color=colors)
axes[1].axvline(0, color='k', lw=0.8); axes[1].axvline(0.2, color='gray', ls='--', lw=0.8); axes[1].axvline(-0.2, color='gray', ls='--', lw=0.8)
axes[1].set_xlabel('Cohen d (捕获 vs 漏报)'); axes[1].set_title('供给侧特征区分度 (绿=无差异→结构性)')
plt.tight_layout()
plt.savefig(f'{FIG}/report_b103_miss.png', dpi=150); plt.close()
print(f"图已保存: {FIG}/report_b103_miss.png")
