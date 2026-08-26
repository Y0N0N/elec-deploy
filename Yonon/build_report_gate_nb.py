"""构建 41_report_v7_regime_gate.ipynb — T108b regime 门控验证呈现。"""
import json
import nbformat as nbf

R = json.load(open('data/regime_gate_results.json'))
nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# 41_report_v7_regime_gate | T108b 方向 regime 门控策略验证

**数据:** `data/walkforward_v7.2_preds_7d.csv` (4224 干净样本) + `s_统调负荷_ma7` (负荷 regime)
**方法:** 后验门控 (`51_tool_regime_gate_run.py`) — 门控信号全部**预测期可得** (负荷/周末/模型预测量级), 无泄露
**目的:** "只在应激 regime 输出方向" 是否值得? 量化 coverage vs sign_hit 权衡, 给可落地操作点
""")

code("""import json, pandas as pd, numpy as np
R = json.load(open('data/regime_gate_results.json'))
pol = pd.DataFrame(R['policies'])
scan = pd.DataFrame(R['p3_scan'])
orc = pd.DataFrame([R['oracle']])
print('门控策略表:')
print(pol[['policy','n','coverage','sign_hit','ci','sign_out']].round(3).to_string(index=False))""")

md("## 解读 — 门控给什么, 不给什么")

code("""base = R['baseline']
print(f"基线 P0 (无门控): 覆盖率 {base['coverage']*100:.1f}%, sign_hit {base['sign_hit']:.3f}")
print()
print("三档可部署门控 vs 基线:")
for lab in ['P1 负荷极端','P5 应激时间窗','P6 P5且预测大价差','P7 预测|价差|>50']:
    r = next(x for x in R['policies'] if x['policy'].startswith(lab.split(' ')[0]))
    print(f"  {lab:<18} 覆盖 {r['coverage']*100:.0f}%  sign {r['sign_hit']:.3f}  "
          f"(被挡样本 sign {r['sign_out']:.3f})")
print()
print(f"上界 ORACLE (真实|spread|>15, 不可部署): sign {orc.iloc[0]['sign_hit']:.3f}")
print(f"→ 即使门控 + 知道真实量级, 方向精度天花板 ≈ 0.60~0.62")
print(f"→ 门控把 sign 从 0.595 提到 ~0.62 (+0.025), 但覆盖率降到 27~74%")""")

md("## 结论 — T108b 战略判断")

code("""print('''
门控是"可选项"而非"银弹":
  ✅ 增益真实: P0 0.595 → 应激窗 0.615~0.624, 且被挡样本确实更不可靠 (0.53~0.56)
  ⚠️ 幅度有限: +0.025~0.03; 覆盖率牺牲大 (97% → 27~74%)
  ⚠️ 天花板: 即使真实量级已知 (ORACLE) 方向也只有 0.602
→ 方向精度受 B103 (实时事件日前不可见) 结构性限制, 门控到不了 0.65+

可落地操作点:
  P5 应激时间窗 (负荷极端或周末): sign 0.615 @ 覆盖 74%  ← 推荐默认
  P6 (再加预测大价差): sign 0.624 @ 覆盖 54%           ← 要精度时
  被挡 (0.53) 略高于随机, 放弃它不损失多少真信号

产品定位建议:
  量级预警 (bigF1 0.87, 全天候) = 主产品, 稳定可靠
  方向 (门控后 ~0.62) = 增强信息, 只在应激窗输出, 明示置信度
  要突破方向天花板 → 需实时侧/事件新数据 (方案 C), 而非更多日前供给侧特征
''')""")

md("## 图 — coverage vs sign_hit 权衡曲线")

code("""import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Micro Hei Mono']
plt.rcParams['axes.unicode_minus'] = False
im = Image.open('show/v7/report_regime_gate.png')
plt.figure(figsize=(9, 6)); plt.imshow(im); plt.axis('off'); plt.title('T108b 门控权衡曲线')
plt.tight_layout(); plt.show()""")

md("## 遗留 (下棒)")
code("print('1. 若要上线: 按 P5/P6 门控 + 明示置信度; 2. B103 大偏差遗漏案例; 3. 突破方向需实时侧新数据 (方案 C)')")

nb.cells = cells
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
with open('41_report_v7_regime_gate.ipynb', 'w') as f:
    nbf.write(nb, f)
print("已生成 41_report_v7_regime_gate.ipynb, 共", len(cells), "cells")
