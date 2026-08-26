"""构建 41_report_v7_walkforward.ipynb — 只读 walk-forward 产物 (preds csv + metrics json), 不重训。"""
import json, nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

md("""# 41_report_v7_walkforward | v7.2 Q109-B 严格 walk-forward 回测

**目标 (open.md Q109):** 大窗口 sign=0.619 是真实信号还是 valid (early stopping 集) 泄露的假象?
用**严格 walk-forward** (每天只用更早数据训练, 评估日 d 绝不进 train/valid) 拿干净 sign_hit。

**执行:** `31_model_v7_walkforward_run.py` (每周重训, 2026-02-01~07-26, 26 折, n_est=400, n_jobs=8)

**产物 (只读):**
- `data/walkforward_v7.2_preds_7d.csv` — 4224 条逐日预测
- `data/walkforward_v7.2_metrics_7d.json` — 汇总指标
- `show/v7/report_walkforward_sign.png` — 月度 sign_hit 图

⚠️ 本 notebook **只导入产物, 绝不重训** (红线: report 只读 joblib/csv)。
""")

code("""import json, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

M = json.load(open('data/walkforward_v7.2_metrics_7d.json'))
P = pd.read_csv('data/walkforward_v7.2_preds_7d.csv', dtype={'date': str, 'hour': str})
print(f"预测样本: {len(P)}  ({P['date'].min()} ~ {P['date'].max()})")
print(f"方法: {M['method']} | 回测窗口: {M['window']} | 折叠数: {M['n_folds']}")""")

md("## 1. 总体结果 — 方向信号真实但弱, 且**强月度分化**")

code("""c = M['cascade']; b = M['baseline_reg']
print("=== walk-forward 总指标 (4224 样本, 无 valid 泄露) ===")
print(f"级联  sign_hit = {c['sign_hit']:.3f} ± {c['sign_ci']:.3f}  (n_nonneu={M['n_nonneu']})")
print(f"      big方向  = {c['big_sign']:.3f} | bigF1={c['big_f1']:.3f}(R{c['big_recall']:.2f}/P{c['big_precision']:.2f}) | "
      f"devF1={c['dev_f1']:.3f} | 5类acc={c['acc5']:.3f}")
print(f"基线  sign_hit = {b['sign_hit']:.3f} ± {b['sign_ci']:.3f}")
print(f"对照: 随机方向={M['random_dir']:.3f} | 全猜正={M['majority_pos']:.3f}")
sig = M['sign_hit_significant']
print(f"\\n→ sign_hit 显著 >0.5?  {'✓ 是' if sig else '✗ 否'}  (CI下界 {c['sign_hit']-c['sign_ci']:.3f})")
print(f"→ 相对 v7.1 大窗口 0.619: 干净 walk-forward 实际 {c['sign_hit']:.3f} — 0.619 确有 valid 泄露虚高, 但方向并非纯噪声")""")

md("## 2. 月度分解 — Feb/Jun 强, May/Jul 近随机")

code("""rows = []
for k in sorted(M['by_month']):
    v = M['by_month'][k]
    rows.append({'月': k, 'sign_hit': v['sign_hit'], 'CI': v['sign_ci'],
                 'big方向': v['big_sign'], 'n_non': int(v['n_nonneu']),
                 '显著>0.5': '✓' if v['sign_hit'] - v['sign_ci'] > 0.5 else ''})
tb = pd.DataFrame(rows).set_index('月')
print(tb.to_string())
print("\\n解读: 2月/6月方向显著可判 (0.64~0.66), 3-4月中庸, 5月/7月近随机 (~0.50)。")
print("v7.1 的 test/eval 恰落在 7 月弱区间 → 0.463/0.495 是'运气差的窗口'而非全局水平。")""")

md("## 3. 月度 sign_hit 图 (±95%CI)")

code("""ys = sorted(M['by_month']); hits=[M['by_month'][k]['sign_hit'] for k in ys]
cis =[M['by_month'][k]['sign_ci']  for k in ys]
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.errorbar(range(len(ys)), hits, yerr=cis, fmt='o-', capsize=5, color='steelblue', lw=2)
ax.axhline(0.5, color='r', ls='--', lw=1, label='random 0.5')
ax.axhline(M['cascade']['sign_hit'], color='green', ls=':', lw=1.2, label=f"overall {M['cascade']['sign_hit']:.3f}")
ax.set_xticks(range(len(ys))); ax.set_xticklabels(ys, rotation=45)
ax.set_ylabel('sign_hit'); ax.set_ylim(0.4, 0.75); ax.grid(alpha=0.3); ax.legend()
ax.set_title('walk-forward monthly sign_hit (+-95%CI) — strong month dependence')
plt.tight_layout(); plt.savefig('show/v7/report_walkforward_sign.png', dpi=120); plt.show()""")

md("""## 4. 结论 (Q109 回答)

**方向预测不是纯噪声, 但不可靠且强月度分化:**

| 结论 | 证据 |
|------|------|
| ✅ 总体方向信号**真实存在** | 干净 walk-forward sign=**0.577±0.016** (4224 样本), CI 下界 0.561 > 0.5 |
| ✅ 优于随机/多数类基线 | 随机 0.504 / 全猜正 0.520 / 级联 0.577 |
| ⚠️ v7.1 大窗口 0.619 **虚高** | 含 valid → 干净值 0.577, 差距 ~0.04 |
| ⚠️ 能力**强月度分化** | 2月 0.642 / 6月 0.662 **显著**; 5月/7月 ~0.50 近随机 |
| ⚠️ v7.1 test/eval 近随机是窗口假象 | 恰落在 7 月弱区间 (0.463/0.495), 非全局水平 |
| ✅ 大偏差量级预警仍最可靠 | bigF1=0.866 (R0.95/P0.80), 全天候稳定 |

**v7.2 建议 (数据驱动):**
- **不要放弃方向预测** — 它在 2 月/6 月显著 (0.64~0.66), 可上线但需**分 regime/月份校准预期**。
- 优先做 **T107 阈值敏感性 + regime 识别** (月度/天气/负荷驱动), 把"什么时候方向可信"搞清楚。
- 大偏差量级预警 (bigF1 0.87) 作为稳的底座, 方向作为有条件的增强。

**遗留 (下棒):**
- 为什么 2 月/6 月方向可判而 5 月/7 月随机? (负荷/气温 regime 假设)
- 回归头值预测仍弱 (条件 RMSE 154), 方向之外的量值精度待 T108。
""")

nb.cells = cells
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
with open('41_report_v7_walkforward.ipynb', 'w') as f:
    nbf.write(nb, f)
print("已生成 41_report_v7_walkforward.ipynb, 共", len(cells), "cells")
