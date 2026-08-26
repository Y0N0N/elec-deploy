"""构建 41_report_v7_b103.ipynb — B103 大偏差遗漏案例分析呈现。"""
import json
import nbformat as nbf

R = json.load(open('data/b103_miss_results.json'))
nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# 41_report_v7_b103 | B103 大偏差遗漏案例分析

**数据:** walk-forward 4224 干净样本 (2026-02-01~07-26)
**脚本:** `51_tool_b103_miss_analysis.py`
**目的:** 漏报的大价差是"边界误差"(可训练) 还是"实时事件结构性不可见"(B103)?
""")

code("""import json, pandas as pd, numpy as np
R = json.load(open('data/b103_miss_results.json'))
print(f"大偏差 |spread|>15: {R['n_big']} 个 | 捕获 {R['caught']} (94.7%) | 漏报 {R['n_missed']} ({R['miss_rate']*100:.1f}%)")
print("漏报按量级桶:")
for d in R['missed_mag_dist']:
    print(f"  |spread| {d['lo']:>3}-{d['hi']:>3}: 漏报 {d['missed']:>3}/{d['total']:>3} = {d['missed']/d['total']*100:5.1f}%")
print("\\n漏报按时段 (凌晨0-5点漏报率 14% → 实时稀缺尖峰):")
seg = {'凌晨': (0,5), '上午': (6,11), '午后': (12,17), '晚峰': (18,23)}""")

md("## 1. 漏报的量级与时段画像")

code("""# 时段漏报率 (从 monthly 之外直接读原始: 重新算)
import pandas as pd
pred = pd.read_csv('data/walkforward_v7.2_preds_7d.csv', dtype={'date':str,'hour':str})
pred['abs_sp']=pred['spread_true'].abs(); pred['is_big']=pred['abs_sp']>15
pred['clf_big']=pred['clf_cls'].isin([0,4])
pred['h']=pred['hour'].str[:2].astype(int)
pred['seg']=np.where(pred['h']<6,'凌晨',np.where(pred['h']<12,'上午',np.where(pred['h']<18,'午后','晚峰')))
big=pred[pred['is_big']]
rows=[]
for s,g in big.groupby('seg'):
    nm=(~g['clf_big']).sum(); rows.append((s,len(g),nm,nm/len(g)*100))
print(pd.DataFrame(rows,columns=['时段','大偏差数','漏报','漏报率%']).sort_values('漏报率%',ascending=False).round(2).to_string(index=False))
print("\\n⚠️ 漏报高度集中在凌晨 (14%) + 上午 (8.6%); 午后/晚峰几乎不漏 (0.2%/1.1%)")
print("→ 漏报的大价差 = 凌晨实时稀缺尖峰 (DA 正常, RT 暴涨)""")

md("## 2. 漏报 vs 捕获: 供给侧特征显著不同 (可训练信号)")

code("""cmp = pd.DataFrame(R['feature_cmp'])
cmp[['caught_mean','missed_mean']]=cmp[['caught_mean','missed_mean']].round(0)
cmp['cohens_d']=cmp['cohens_d'].round(3); cmp['p_val']=cmp['p_val'].round(4)
print(cmp[['feat','caught_mean','missed_mean','cohens_d','p_val']].to_string(index=False))
print("\\n漏报样本特征画像: 近零光伏预测 (131 vs 2478)、低负荷 (93k vs 110k)、负预测出力 (-2056)")
print("→ 凌晨无太阳 + 低负荷 + 备用吃紧 = 实时稀缺前兆, 模型漏用了这些信号""")

md("## 3. 漏报最极端案例 (实时稀缺尖峰)")

code("""ext=pd.DataFrame(R['top_missed'])[['date','hour','spread_true','clf_cls','reg_val']]
ext['spread_true']=ext['spread_true'].round(1); ext['reg_val']=ext['reg_val'].round(1)
print(ext.to_string(index=False))
print("\\n→ 全部是凌晨 3-7 点的极端负价差 (-250~-670, DA<RT), 模型几乎全判 neu (cls=2)")
print("→ 这是实时侧稀缺事件 (早高峰前机组爬坡不足/备用吃紧), 日前供给侧只部分可见""")

md("## 结论 — B103 部分坐实, 但有明确训练空间")

code("""print('''
1. 量级预警整体可靠: 94.7% 捕获, 极端(>100)捕获率 97.8%
2. 漏报两类:
   ① 边界漏报 (|spread|15~25, 11.8%) → 阈值/权重可改善
   ② 凌晨稀缺尖峰漏报 (0-5点 14%, 极端 -250~-670) → 关键
3. 凌晨漏报有可区分供给侧信号 (近零光伏/低负荷/负出力) → 模型漏用, 可训练改善
4. 但极端事件罕见 + 部分成因 (机组跳闸/实时阻塞) 日前确实不可见 → B103 部分成立
行动:
   A. 加"凌晨稀缺"特征/权重 (低负荷×近零光伏×负备用交互)
   B. 边界漏报: 类权重/阈值微调
   C. 方案 C: 深挖披露事件数据 (检修/出力约束/备用 6366 文件) + 外部气象/跳闸
''')""")

md("## 图 — 漏报率按量级 + 特征区分度")
code("""import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams['figure.dpi']=150; plt.rcParams['font.family']=['sans-serif']
plt.rcParams['font.sans-serif']=['WenQuanYi Micro Hei','WenQuanYi Micro Hei Mono']; plt.rcParams['axes.unicode_minus']=False
im=Image.open('show/v7/report_b103_miss.png')
plt.figure(figsize=(12,4.5)); plt.imshow(im); plt.axis('off'); plt.title('B103 漏报画像')
plt.tight_layout(); plt.show()""")

nb.cells = cells
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
with open('41_report_v7_b103.ipynb', 'w') as f:
    nbf.write(nb, f)
print("已生成 41_report_v7_b103.ipynb, 共", len(cells), "cells")
