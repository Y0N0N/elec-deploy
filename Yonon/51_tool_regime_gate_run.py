"""
T108b 方向 regime 门控策略验证 — coverage vs sign_hit 权衡 (可部署门控)

背景: walk-forward 显示方向能力"应激 regime"强 (|spread|>50 / 极端负荷 / 周末), 过渡期随机。
本脚本验证: 部署时**只用预测期可得的信号** (负荷三分位 / 周末 / 预测价差量级) 做门控,
输出"只在不该沉默的时候说方向"的 coverage-sign_hit 权衡曲线, 给出可落地的操作点。

数据: data/walkforward_v7.2_preds_7d.csv (4224 样本, 无 valid 泄露) + s_统调负荷_ma7 (负荷 regime)

门控信号 (全部预测期可得, 无泄露):
  load_tercile : 日负荷 (7日均统调负荷) 三分位 — 低/高 = 极端供需
  weekend      : 周六日
  |reg_val|    : 回归头预测的价差量级 (模型预测的应激程度)
  clf_big      : 分类头判 big_neg/big_pos (预测大价差)

评估口径 (与 v7.1 一致):
  eligible = 模型给出方向 (clf ≠ neu)
  gated    = eligible 且 门控放行
  coverage = gated 占真实非neu比例
  sign_hit = gated 内方向正确率

产物: data/regime_gate_results.json + show/v7/report_regime_gate.png
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Micro Hei Mono']
plt.rcParams['axes.unicode_minus'] = False

from config.config import dataset_path, YONON_PATH

OUT = f'{YONON_PATH}/data/regime_gate_results.json'
FIG = f'{YONON_PATH}/show/v7'
os.makedirs(FIG, exist_ok=True)

print("=" * 72)
print("  T108b 方向 regime 门控策略验证 (coverage vs sign_hit)")
print("=" * 72)

# ============ 加载预测 + 负荷 regime ============
pred = pd.read_csv(f'{YONON_PATH}/data/walkforward_v7.2_preds_7d.csv', dtype={'date': str, 'hour': str})
pred['dir_true'] = np.sign(np.where(pred['spread_true'] == 0, 1e-9, pred['spread_true']))
pred['dir_clf'] = np.sign(pred['clf_cls'] - 2)
pred['clf_big'] = pred['clf_cls'].isin([0, 4])
pred['abs_true'] = pred['spread_true'].abs()
pred['abs_pred'] = pred['reg_val'].abs()

load = pd.read_feather(f'{dataset_path}/s_统调负荷_ma7.fea').stack()
load.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(load.index.get_level_values(0)).strftime('%Y-%m-%d'),
     [f'{int(h):02d}:00' if isinstance(h, (int, np.integer)) else str(h) for h in load.index.get_level_values(1)]],
    names=['date', 'hour'])
load.name = 'load_ma7'
p = pred.set_index(['date', 'hour']).join(load, how='left').reset_index()
p['date_dt'] = pd.to_datetime(p['date'])
p['weekend'] = (p['date_dt'].dt.weekday >= 5).astype(bool)

daily_load = p.groupby('date')['load_ma7'].mean()
terc = daily_load.quantile([1 / 3, 2 / 3]).values
p['load_extreme'] = p['date'].map(lambda d: bool(daily_load.get(d, 0) < terc[0] or daily_load.get(d, 0) > terc[1]))
print(f"样本: {len(p)} | 负荷极端日占比: {p['load_extreme'].mean():.3f} | 周末占比: {p['weekend'].mean():.3f}")

# ============ 门控策略 ============
nonneu = p['abs_true'] > 5.0          # 真实非neu (评估口径: |spread|>τ_minor)
eligible = (p['dir_clf'] != 0)        # 模型想输出方向
print(f"真实非neu: {nonneu.sum()} | 模型给方向 (eligible): {eligible.sum()} "
      f"({eligible[nonneu].mean()*100:.1f}% of nonneu)")

def eval_policy(gate_cond, label):
    """gate_cond: bool Series (预测期可得)"""
    gated = nonneu & eligible & gate_cond
    n = int(gated.sum())
    if n == 0:
        return dict(policy=label, n=0, coverage=0.0, sign_hit=np.nan, ci=np.nan, sign_out=np.nan)
    hit = (p.loc[gated, 'dir_clf'] == p.loc[gated, 'dir_true']).mean()
    ci = 1.96 * np.sqrt(hit * (1 - hit) / n)
    # 被门控挡掉的方向样本 (本来会错的比例)
    suppressed = nonneu & eligible & ~gate_cond
    s_out = (p.loc[suppressed, 'dir_clf'] == p.loc[suppressed, 'dir_true']).mean() if suppressed.sum() else np.nan
    return dict(policy=label, n=n, coverage=n / max(nonneu.sum(), 1),
                sign_hit=hit, ci=ci, sign_out=s_out, n_supp=int(suppressed.sum()))

policies = [
    ('P0 无门控 (基线)', np.ones(len(p), dtype=bool)),
    ('P1 负荷极端 (低或高)', p['load_extreme'].values),
    ('P2 周末', p['weekend'].values),
    ('P3 预测|价差|>15', (p['abs_pred'] > 15).values),
    ('P4 clf判大价差(big)', p['clf_big'].values),
    ('P5 应激时间窗 (负荷极端或周末)', (p['load_extreme'] | p['weekend']).values),
    ('P6 P5 且 预测|价差|>15', ((p['load_extreme'] | p['weekend']) & (p['abs_pred'] > 15)).values),
    ('P7 预测|价差|>50', (p['abs_pred'] > 50).values),
]
rows = [eval_policy(c, lab) for lab, c in policies]
res_df = pd.DataFrame(rows)
print("\n门控策略 coverage vs sign_hit:")
print(res_df.round(3).to_string(index=False))

# 上界参考: 用真实 |spread| (不可部署, 仅作理论)
oracle = eval_policy((p['abs_true'] > 15).values, 'ORACLE 真实|价差|>15 (上界,不可部署)')
print("\n上界参考 (用真实|spread|, 不可部署):")
print(pd.DataFrame([oracle]).round(3).to_string(index=False))

# ============ 门控阈值扫描 (P3 细化) ============
scan_rows = []
for th in [0, 5, 10, 15, 20, 30, 40, 50, 75, 100]:
    r = eval_policy((p['abs_pred'] > th).values, f'|reg_val|>{th}')
    scan_rows.append(r)
scan_df = pd.DataFrame(scan_rows)
print("\n门控 P3 扫描 (预测|价差|阈值 → coverage/sign_hit):")
print(scan_df[['policy', 'n', 'coverage', 'sign_hit', 'ci']].round(3).to_string(index=False))

# ============ 保存 ============
res = {'policies': rows, 'oracle': oracle, 'p3_scan': scan_rows,
       'baseline': rows[0], 'coverage_sign_curve': [
           dict(coverage=float(r['coverage']), sign_hit=float(r['sign_hit'])) for r in rows]}
with open(OUT, 'w') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {OUT}")

# ============ 图: coverage vs sign_hit ============
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot([r['coverage'] for r in rows], [r['sign_hit'] for r in rows], 'o-', color='steelblue',
        label='门控策略 (P0~P7)')
ax.plot([r['coverage'] for r in scan_rows], [r['sign_hit'] for r in scan_rows], 's--', color='orange',
        label='|reg_val|>τ 扫描')
ax.axhline(0.5, color='red', ls='--', lw=1, label='随机 0.5')
ax.plot([oracle['coverage']], [oracle['sign_hit']], 'D', color='green', ms=10, label='ORACLE 真实|spread|>15')
for r in rows:
    ax.annotate(r['policy'].split(' ')[0], (r['coverage'], r['sign_hit']), fontsize=8,
                textcoords='offset points', xytext=(4, 4))
ax.set_xlabel('方向输出覆盖率 (coverage of true non-neu)')
ax.set_ylabel('sign_hit (门控内方向正确率)')
ax.set_title('T108b: 方向 regime 门控 — coverage vs sign_hit 权衡')
ax.grid(alpha=0.3); ax.legend(loc='lower right')
plt.tight_layout()
fig_path = f'{FIG}/report_regime_gate.png'
plt.savefig(fig_path, dpi=150); plt.close()
print(f"图已保存: {fig_path}")

print("\n" + "=" * 72)
print("  T108b regime 门控验证完成")
print("=" * 72)
