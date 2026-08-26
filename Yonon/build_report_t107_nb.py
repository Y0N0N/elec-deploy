"""构建 41_report_v7_t107_regime.ipynb — 读取 t107_regime_results.json 呈现 T107 + regime 分析。"""
import json
import nbformat as nbf

R = json.load(open('data/t107_regime_results.json'))
nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# 41_report_v7_t107_regime | T107 阈值敏感性 + regime 分析 (Q110)

**数据:** `data/walkforward_v7.2_preds_7d.csv` (4224 样本, 无 valid 泄露) + 全 567 天 spread 分布
**方法:** 后验分析 (不重训) — `51_tool_t107_regime_run.py`
**目的:** ① τ 怎么定 (Q107: τ=5 时 88.7% 触发, 过滤意图落空)  ② 为什么 Feb/Jun 方向强 May/Jul 弱 (Q110)
""")

code("""import json, pandas as pd, numpy as np
R = json.load(open('data/t107_regime_results.json'))
dist = pd.DataFrame(R['threshold_dist'])
mod  = pd.DataFrame(R['threshold_model'])
reg  = R['regime']
print('threshold distribution + model loaded')""")

md("## Part A — T107 阈值敏感性: 触发率 vs sign_hit")

code("""t = mod[['tau','n','sign_clf','ci_clf','sign_reg']].merge(dist[['tau','trig']], on='tau')
t['trig_%'] = (t['trig']*100).round(1)
t['sign_clf'] = t['sign_clf'].round(3); t['sign_reg'] = t['sign_reg'].round(3)
print("τ 扫描 (walk-forward 干净样本, 后验):")
print(t.to_string(index=False))
print("\\n分布触发率 = 完美分类下 |spread|>τ 会触发输出的比例 (Q107 核心)")
print("\\n读法: τ 从 5→30, sign_hit 仅 0.577→0.598 (+0.02), 但触发率从 88.7%→51.7%")
print("→ τ 主要控制预警频率, 对方向精度几乎无增益")""")

md("## Part B — regime 分析 (Q110): 方向什么时候可信")

code("""m = pd.DataFrame(reg['by_month'])
m[['sign_clf','ci_clf','mean_load','mean_abs','std_sp']] = m[['sign_clf','ci_clf','mean_load','mean_abs','std_sp']].round(3)
print("月度 sign_hit + 负荷 (load = 7日均统调负荷):")
print(m[['ym','n','sign_clf','ci_clf','mean_load','mean_abs']].to_string(index=False))
print("\\n⚠️ sign_hit 与负荷不单调: Feb 低负荷 / Jun 高负荷 都强 (0.63~0.65), May/Jul 高负荷弱 (~0.50)""")

code("""ld = pd.DataFrame(reg['by_load_tercile']); ld[['sign_clf','ci_clf']]=ld[['sign_clf','ci_clf']].round(3)
print("日负荷三分位 sign_hit (U 型):")
print(ld[['bucket','n','sign_clf','ci_clf','mean_load']].to_string(index=False))
sg = pd.DataFrame(reg['by_segment']); sg['sign_clf']=sg['sign_clf'].round(3)
print("\\n时段 sign_hit:")
print(sg[['seg','n','sign_clf','ci_clf']].to_string(index=False))
wd = pd.DataFrame(reg['by_weekday']); wd['sign_clf']=wd['sign_clf'].round(3)
print("\\n工作日/周末:")
print(wd[['group','n','sign_clf','ci_clf']].to_string(index=False))
mg = pd.DataFrame(reg['by_spread_mag']); mg['sign_clf']=mg['sign_clf'].round(3)
print("\\n|spread| 量级桶 sign_hit:")
print(mg[['lo','hi','n','sign_clf','ci_clf']].to_string(index=False))""")

md("## 图")

code("""import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, path, title in [
    (axes[0], 'show/v7/report_t107_tau.png', 'T107: sign_hit vs tau'),
    (axes[1], 'show/v7/report_regime_month_load.png', 'month sign_hit + load'),
    (axes[2], 'show/v7/report_regime_magnitude.png', 'sign_hit vs |spread|')]:
    im = Image.open(path); ax.imshow(im); ax.axis('off'); ax.set_title(title)
plt.tight_layout(); plt.show()""")

md("""## 结论与建议

### T107 — τ 推荐
| τ | 触发率 | sign_hit | 说明 |
|---|--------|----------|------|
| 5 (现状 τ_minor) | 88.7% | 0.577 | 几乎全触发, "过滤"名存实亡 |
| 15 (现状 τ_big) | 70.0% | 0.590 | 仍偏高 |
| 20 | 62.8% | 0.594 | 平衡点 |
| 25~30 | 51.7~56.8% | 0.595~0.598 | 接近"少数异动提醒" |

**建议:** sign_hit 随 τ 几乎不变 (0.577→0.598), τ 本质是**预警频率旋钮**而非精度旋钮。
- 业务若接受 ~60% 预警率 → τ_minor=15~20, τ_big=25~30 (过滤更有意义)
- 保持现状 (5/15) 亦可, 但要接受"高频输出"的现实
- ⚠️ 后验 re-threshold 只改评估不改模型; 定稿前需按新 τ 重训验证

### regime — 方向能力有"应激 regime"
| 条件 | sign_hit | 解读 |
|------|----------|------|
| \|spread\| > 50 | **0.608** | 大偏差方向最可判 (结构驱动) |
| 低负荷 / 高负荷日 | 0.595 / 0.588 | 极端供需 regime 可判 (U 型) |
| 周末 | 0.596 | 调度模式规律 |
| 中负荷日 | 0.523 | 过渡期随机 |
| \|spread\| 10~15 | 0.500 | 边界量级纯噪声 |
| 5月/7月 | ~0.50 | 过渡季近随机 |

**Q110 答案:** 不是"负荷高低"单一因素 — 是**供需紧张的极端 regime** (春节低负荷 / 迎峰度夏高负荷、
大价差、周末) 方向规律强; **过渡/平衡 regime** (中负荷、中量级价差、5月7月) 随机。

**v7.2 落地方案:** 方向输出做 **regime 门控** — 只在 \|spread\| 预测大 或 负荷极端时输出方向 (sign 0.59~0.61),
过渡期只给量级预警 (base 底座)。这比"全年无差别输出方向" (0.577) 更诚实可用。

**遗留:** 大偏差遗漏案例分析 (B103); 按 regime 重训验证门控。
""")

nb.cells = cells
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
with open('41_report_v7_t107_regime.ipynb', 'w') as f:
    nbf.write(nb, f)
print("已生成 41_report_v7_t107_regime.ipynb, 共", len(cells), "cells")
