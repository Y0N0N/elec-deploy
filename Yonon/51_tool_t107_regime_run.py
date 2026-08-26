"""
T107 阈值敏感性 + regime 分析 (Q110) — 基于 walk-forward 干净预测的后验分析

数据基础: data/walkforward_v7.2_preds_7d.csv (4224 样本, 2026-02-01~07-26, 无 valid 泄露)

Part A: T107 阈值敏感性
  - 全 567 天分布触发率 (模型无关, 精确): |spread|>τ 占比 / 正负占比 (Q107 核心)
  - walk-forward 模型 sign_hit 随 τ (后验, 多 regime 干净样本): clf 方向 & reg 方向
  → 给 τ 推荐值: "仅显著偏差时输出"的过滤意图在哪个 τ 才成立

Part B: regime 分析 (Q110 — 为什么 Feb/Jun 强 May/Jul 弱)
  - sign_hit 按: 月 / 日负荷三分位 / 时段 / 工作日 vs 周末 / |spread| 量级桶
  - 月度 sign_hit 与月度均负荷、均|spread|、spread 波动 的相关性
  → 找"方向可信"的 regime 条件

产物: data/t107_regime_results.json + show/v7/*.png
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from config.config import dataset_path, spread_label_file, YONON_PATH

OUT = f'{YONON_PATH}/data/t107_regime_results.json'
FIG = f'{YONON_PATH}/show/v7'
os.makedirs(FIG, exist_ok=True)

print("=" * 72)
print("  T107 阈值敏感性 + regime 分析 (Q110)")
print("=" * 72)

# ============ 加载 walk-forward 预测 ============
pred = pd.read_csv(f'{YONON_PATH}/data/walkforward_v7.2_preds_7d.csv', dtype={'date': str, 'hour': str})
pred['dir_true'] = np.sign(np.where(pred['spread_true'] == 0, 1e-9, pred['spread_true']))
pred['dir_clf'] = np.sign(pred['clf_cls'] - 2)
pred['dir_reg'] = np.sign(pred['reg_val'])
print(f"walk-forward 预测: {len(pred)} 样本 ({pred['date'].min()} ~ {pred['date'].max()})")

def s_metrics(mask):
    s = pred.loc[mask]
    if len(s) == 0:
        return dict(sign_clf=np.nan, ci_clf=np.nan, sign_reg=np.nan, n=0)
    hit = (s['dir_clf'] == s['dir_true']).mean()
    ci = 1.96 * np.sqrt(hit * (1 - hit) / len(s))
    hr = (s['dir_reg'] == s['dir_true']).mean()
    return dict(sign_clf=hit, ci_clf=ci, sign_reg=hr, n=len(s))

# ============ Part A: T107 阈值敏感性 ============
print("\n[Part A] T107 阈值敏感性...")
TAUS = [3, 5, 10, 15, 20, 25, 30]

# A1: 全 567 天分布触发率 (模型无关)
spread = pd.read_feather(spread_label_file); spread.index = pd.to_datetime(spread.index)
s_full = spread.stack().values
dist_rows = []
for tau in TAUS:
    dist_rows.append(dict(tau=tau, trig=float(np.mean(np.abs(s_full) > tau)),
                          pos=float(np.mean(s_full > tau)), neg=float(np.mean(s_full < -tau))))
dist_df = pd.DataFrame(dist_rows)
print("全 567 天分布触发率 (|spread|>τ 占比 — 完美分类下的输出触发率):")
print(dist_df.round(3).to_string(index=False))

# A2: walk-forward 模型 sign_hit 随 τ (后验)
tau_rows = []
for tau in TAUS:
    m = pred['spread_true'].abs() > tau
    tau_rows.append(dict(tau=tau, **s_metrics(m), trig_frac=float(m.mean())))
tau_df = pd.DataFrame(tau_rows)
print("\nwalk-forward sign_hit 随 τ (clf 方向 / reg 方向):")
print(tau_df[['tau', 'n', 'sign_clf', 'ci_clf', 'sign_reg', 'trig_frac']].round(3).to_string(index=False))

# ============ Part B: regime 分析 ============
print("\n[Part B] regime 分析 (Q110)...")

# B0: 负荷 regime 代理 (7 日均统调负荷)
load = pd.read_feather(f'{dataset_path}/s_统调负荷_ma7.fea').stack()
load.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(load.index.get_level_values(0)).strftime('%Y-%m-%d'),
     [f'{int(h):02d}:00' if isinstance(h, (int, np.integer)) else str(h) for h in load.index.get_level_values(1)]],
    names=['date', 'hour'])
load.name = 'load_ma7'
p = pred.set_index(['date', 'hour']).join(load, how='left')
p = p.reset_index()
print(f"负荷 join 缺失: {p['load_ma7'].isna().sum()} / {len(p)}")

# 日负荷 (日均 ma7 统调负荷)
p['date_dt'] = pd.to_datetime(p['date'])
p['ym'] = p['date'].str[:7]
p['weekday'] = p['date_dt'].dt.weekday
p['is_weekend'] = (p['weekday'] >= 5)
p['hour_int'] = p['hour'].str[:2].astype(int)
p['abs_sp'] = p['spread_true'].abs()

regime_results = {}

# B1: 月 (已有, 复算 + 负荷/波动)
month_rows = []
for ym, g in p.groupby('ym'):
    m = s_metrics(g.index)
    m.update(ym=ym, mean_load=float(g['load_ma7'].mean()),
             mean_abs=float(g['abs_sp'].mean()), std_sp=float(g['spread_true'].std()))
    month_rows.append(m)
month_df = pd.DataFrame(month_rows)
regime_results['by_month'] = month_rows
print("\n月度 sign_hit + regime 特征:")
print(month_df[['ym', 'n', 'sign_clf', 'ci_clf', 'mean_load', 'mean_abs', 'std_sp']].round(3).to_string(index=False))

# B2: 日负荷三分位 (低/中/高)
daily_load = p.groupby('date')['load_ma7'].mean()
terciles = daily_load.quantile([1/3, 2/3]).values
def load_bucket(d):
    dl = daily_load.get(d, np.nan)
    if pd.isna(dl): return 'NA'
    if dl < terciles[0]: return '低负荷'
    if dl < terciles[1]: return '中负荷'
    return '高负荷'
p['load_bucket'] = p['date'].map(load_bucket)
load_rows = []
for b in ['低负荷', '中负荷', '高负荷']:
    m = s_metrics(p['load_bucket'] == b)
    m.update(bucket=b, mean_load=float(p.loc[p['load_bucket'] == b, 'load_ma7'].mean()))
    load_rows.append(m)
load_df = pd.DataFrame(load_rows)
regime_results['by_load_tercile'] = load_rows
print("\nsign_hit 按日负荷三分位:")
print(load_df[['bucket', 'n', 'sign_clf', 'ci_clf', 'mean_load']].round(3).to_string(index=False))

# B3: 时段
def seg(h):
    if h < 6: return '凌晨0-5'
    if h < 12: return '上午6-11'
    if h < 18: return '午后12-17'
    return '晚峰18-23'
p['seg'] = p['hour_int'].map(seg)
seg_rows = []
for s in ['凌晨0-5', '上午6-11', '午后12-17', '晚峰18-23']:
    m = s_metrics(p['seg'] == s)
    m.update(seg=s)
    seg_rows.append(m)
seg_df = pd.DataFrame(seg_rows)
regime_results['by_segment'] = seg_rows
print("\nsign_hit 按时段:")
print(seg_df[['seg', 'n', 'sign_clf', 'ci_clf']].round(3).to_string(index=False))

# B4: 工作日 vs 周末
wd_rows = []
for lab, m in [('工作日', p['is_weekend'] == False), ('周末', p['is_weekend'] == True)]:
    mm = s_metrics(m); mm.update(group=lab)
    wd_rows.append(mm)
wd_df = pd.DataFrame(wd_rows)
regime_results['by_weekday'] = wd_rows
print("\nsign_hit 工作日 vs 周末:")
print(wd_df[['group', 'n', 'sign_clf', 'ci_clf']].round(3).to_string(index=False))

# B5: |spread| 量级桶
bins = [(5, 10), (10, 15), (15, 25), (25, 50), (50, np.inf)]
mag_rows = []
for lo, hi in bins:
    m = (p['abs_sp'] > lo) & (p['abs_sp'] <= hi)
    mm = s_metrics(m); mm.update(lo=lo, hi=('inf' if np.isinf(hi) else hi))
    mag_rows.append(mm)
mag_df = pd.DataFrame(mag_rows)
regime_results['by_spread_mag'] = mag_rows
print("\nsign_hit 按 |spread| 量级:")
print(mag_df[['lo', 'hi', 'n', 'sign_clf', 'ci_clf']].round(3).to_string(index=False))

# B6: 月度 sign_hit 与 regime 特征相关性
corr_rows = {}
for col in ['mean_load', 'mean_abs', 'std_sp']:
    corr_rows[f'corr_sign_vs_{col}'] = float(month_df['sign_clf'].corr(month_df[col]))
regime_results['month_corr'] = corr_rows
print("\n月度 sign_hit 与 regime 特征 Pearson 相关 (n=6 月):")
for k, v in corr_rows.items(): print(f"  {k} = {v:.3f}")

# ============ 保存 ============
res = {'threshold_dist': dist_df.to_dict('records'),
       'threshold_model': tau_df.to_dict('records'),
       'regime': regime_results}
with open(OUT, 'w') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {OUT}")

# ============ 图 ============
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False

    # 图1: τ-曲线 (触发率 vs sign_hit)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(tau_df['tau'], tau_df['sign_clf'], 'o-', color='steelblue', label='sign_hit (clf dir)')
    ax1.plot(tau_df['tau'], tau_df['sign_reg'], 's--', color='orange', label='sign_hit (reg dir)')
    ax1.errorbar(tau_df['tau'], tau_df['sign_clf'], yerr=tau_df['ci_clf'], fmt='none', ecolor='steelblue', alpha=0.5)
    ax1.axhline(0.5, color='r', ls='--', lw=1, label='random 0.5')
    ax1.set_xlabel('threshold tau (yuan/MWh)'); ax1.set_ylabel('sign_hit')
    ax1.set_title('T107: sign_hit vs tau (walk-forward, clean)')
    ax2 = ax1.twinx()
    ax2.plot(dist_df['tau'], dist_df['trig'] * 100, 'v-', color='crimson', label='|spread|>tau frac (%)')
    ax2.set_ylabel('trigger rate (%)'); ax2.set_ylim(0, 100)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='center right')
    plt.tight_layout(); plt.savefig(f'{FIG}/report_t107_tau.png', dpi=120); plt.close()

    # 图2: regime — 月度 sign_hit (load overlay)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ys = month_df['ym'].tolist()
    ax.errorbar(range(len(ys)), month_df['sign_clf'], yerr=month_df['ci_clf'], fmt='o-', color='steelblue')
    ax.axhline(0.5, color='r', ls='--')
    ax.set_xticks(range(len(ys))); ax.set_xticklabels(ys); ax.set_ylabel('sign_hit')
    ax.set_title('regime: monthly sign_hit + load level')
    ax2 = ax.twinx()
    ax2.bar(range(len(ys)), month_df['mean_load'], alpha=0.25, color='green')
    ax2.set_ylabel('mean load (ma7)'); ax2.set_ylim(month_df['mean_load'].min() * 0.95, month_df['mean_load'].max() * 1.05)
    plt.tight_layout(); plt.savefig(f'{FIG}/report_regime_month_load.png', dpi=120); plt.close()

    # 图3: 按量级桶 sign_hit
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"{r['lo']}-{r['hi']}" if r['hi'] != 'inf' else f">{r['lo']}" for r in mag_rows]
    ax.errorbar(range(len(labels)), mag_df['sign_clf'], yerr=mag_df['ci_clf'], fmt='o-', color='purple')
    ax.axhline(0.5, color='r', ls='--')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel('|spread| bucket'); ax.set_ylabel('sign_hit'); ax.grid(alpha=0.3)
    ax.set_title('regime: sign_hit vs |spread| magnitude')
    plt.tight_layout(); plt.savefig(f'{FIG}/report_regime_magnitude.png', dpi=120); plt.close()
    print(f"图已保存: {FIG}/report_t107_tau.png, report_regime_month_load.png, report_regime_magnitude.png")
except Exception as e:
    print(f"图生成跳过: {e}")

print("\n" + "=" * 72)
print("  T107 + regime 分析完成")
print("=" * 72)
