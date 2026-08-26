"""构建 41_report_v8_penalty.ipynb — v8 惩罚模型报告。

回测守则: 本 report 只导入 models/xgb_v8_*.joblib, 绝不重训。
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# 41_report_v8_penalty | 触发阈值 50 + 自定义惩罚函数 + 过拟合治理

**模型:** `models/xgb_v8_20260812_1528.joblib` (自包含, 只读, 绝不重训)
**训练:** `31_model_v8_penalty_run.py` | 阈值 τ_minor=τ_big=50 (有效 3 类: big_neg<-50 / neu±50 / big_pos>50, 编码 {0,2,4})
**目标:** ① 触发阈值 5→50 ② 惩罚规则 A/B/C 集成训练 ③ 过拟合治理 ④ 只存 Val_Penalty 最优权重
""")

code("""import json, numpy as np, pandas as pd, joblib
from config.config import dataset_path, spread_label_file, YONON_PATH

M = joblib.load('models/xgb_v8_20260812_1528.joblib')
print('阈值:', M['threshold_minor'], '/', M['threshold_big'], '| 类别编码:', M['class_codes'])
print('best_epoch:', M['best_epoch'], '| best Val_Penalty:', round(M['best_val_penalty'], 4))
print('TS-CV 4折:', round(M['cv_timeseries']['mean'], 4), '±', round(M['cv_timeseries']['std'], 4))
print('trained_at:', M['trained_at'])
print('决策规则:', M['decision_rule'][:80], '...')""")

md("## 1. 惩罚函数 (规则 A/B/C)")
md("""
| 规则 | 触发条件 | 分值 |
|---|---|---|
| A 预测错误 | 触发信号 (模型或真实任一方非 neu) 时方向/分类错误 (含漏报/误报) | **-1** |
| B 完全反向 | 预测方向与实际完全相反 (一正一负) | **-2** (加倍) |
| C 数值偏差 | 触发并输出值时 \\|Pred-True\\|/\\|True\\| > 0.2 | **额外 -1.5** |

训练内集成: 分类头成本矩阵软目标 (规则 A/B 梯度注入) + 原生 RMSE (规则 C 对齐) + 硬惩罚监控/早停/最优选择。
""")

code("""# 每轮硬惩罚曲线 (来自训练历史)
hist = np.asarray(M['penalty_history'])
ep, tpen, vpen, lr, vl = hist[:,0], hist[:,1], hist[:,2], hist[:,3], hist[:,4]
print('Epoch  0:', f'Train_Penalty={tpen[0]:+.4f} Val_Penalty={vpen[0]:+.4f}')
print('Epoch', int(ep[-1]), f': Train_Penalty={tpen[-1]:+.4f} Val_Penalty={vpen[-1]:+.4f}')
print(f'早停 @ {len(ep)} 轮 (分类+回归双头各停滞15) | 最优惩罚 @ epoch {int(M["best_epoch"])}')
print()
print('Val_Penalty 演化 (每 25 轮):')
for e,t,p,v,l in hist[::25]:
    print(f'  epoch {int(e):4d} | Train {t:+.4f} | Val {p:+.4f} | Val_Loss {v:.3f} | lr {l:.4f}')""")

code("""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': ['WenQuanYi Micro Hei','DejaVu Sans'], 'axes.facecolor':'#fcfcfb',
    'figure.facecolor':'#fcfcfb','axes.edgecolor':'#c3c2b7','axes.labelcolor':'#0b0b0b',
    'xtick.color':'#898781','ytick.color':'#898781','axes.grid':True,'grid.color':'#e1e0d9','grid.linewidth':0.6})
C1, C2 = '#2a78d6', '#eb6834'
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(ep, tpen, color=C1, lw=2, label='Train_Penalty')
ax[0].plot(ep, vpen, color=C2, lw=2, label='Val_Penalty')
ax[0].axvline(M['best_epoch'], color='#0b0b0b', lw=1.2, ls='--', alpha=.7)
ax[0].set_xlabel('epoch'); ax[0].set_ylabel('Penalty Score (越高越好)')
ax[0].set_title('惩罚分数曲线 (规则 A/B/C)'); ax[0].legend(frameon=False, loc='lower right')
ax[1].plot(ep, vl, color=C1, lw=2)
ax[1].set_xlabel('epoch'); ax[1].set_ylabel('归一化 Val_Loss (越低越好)')
ax[1].set_title('归一化验证损失 (早停/LR 依据)')
fig.suptitle(f'v8 训练监控 | 最优 Val_Penalty={M["best_val_penalty"]:+.3f} @epoch {M["best_epoch"]}', y=1.02)
fig.tight_layout(); plt.show()""")

md("## 2. 评估 — 独立留出集 (valid/test/eval, 模型从未见过 test/eval)")
code("""# 加载特征 + spread 标签 (只读, 与训练同源)
from config.config import da_price_latest, rt_price_latest
fresh = M['features']
fl = []
for name in fresh:
    df = pd.read_feather(f'{dataset_path}/{name}.fea'); s = df.stack(); s.name = name; fl.append(s)
X = pd.concat(fl, axis=1)
X.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(X.index.get_level_values(0)).strftime('%Y-%m-%d'),
     [f'{int(h):02d}:00' if isinstance(h,(int,np.integer)) else str(h) for h in X.index.get_level_values(1)]],
    names=['date','hour'])
X = X.sort_index()
spread = pd.read_feather(spread_label_file); spread.index = pd.to_datetime(spread.index)
ys = spread.stack(); ys.index = ys.index.rename(['date','hour'])
ys.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(ys.index.get_level_values('date')).strftime('%Y-%m-%d'), ys.index.get_level_values('hour')],
    names=['date','hour'])
# eval 期真实 spread (07-22~07-26 在 spread_label 之外, 从最新 DA/RT 补)
da_e = pd.read_feather(da_price_latest); rt_e = pd.read_feather(rt_price_latest)
sp_eval = (da_e - rt_e).stack(); sp_eval.index = sp_eval.index.rename(['date','hour'])
sp_eval.index = pd.MultiIndex.from_arrays(
    [pd.to_datetime(sp_eval.index.get_level_values('date')).strftime('%Y-%m-%d'),
     sp_eval.index.get_level_values('hour')], names=['date','hour'])
ys = pd.concat([ys, sp_eval[~sp_eval.index.isin(ys.index)]]).sort_index()

def eval_win(a, b, tag):
    vd = X.index.get_level_values('date')
    Xs = X.loc[(vd>=a)&(vd<=b)]; c = Xs.index.intersection(ys.index)
    Xs = Xs.loc[c]; yv = ys.loc[c]
    yc = M['clf'].predict(Xs); yv_ = M['reg'].predict(Xs)
    yt = np.asarray(yv.values, float)
    T = M['threshold_minor']
    yt3 = np.where(yt < -T, 0, np.where(yt <= T, 1, 2))
    nonneu = yt3 != 1
    pd_ = np.sign(yc-2); td_ = np.sign(yt)*(np.abs(yt)>T)
    sign_hit = (pd_[nonneu]==td_[nonneu]).mean() if nonneu.sum() else np.nan
    trig = pd_!=0
    yt_big = ((yt3==0)|(yt3==2)).astype(int); yp_big = ((yc==0)|(yc==4)).astype(int)
    from sklearn.metrics import f1_score
    from sklearn.metrics import mean_squared_error
    big_f1 = f1_score(yt_big, yp_big, zero_division=0)
    rmse = np.sqrt(mean_squared_error(yt, yv_))
    return dict(n=len(yt), sign_hit=sign_hit, trigger=trig.mean(), big_f1=big_f1, rmse=rmse)

rows = {}
for a,b,tag in [('2026-06-01','2026-07-15','Valid'), ('2026-07-16','2026-07-21','Test'), ('2026-07-22','2026-07-26','Eval')]:
    rows[tag] = eval_win(a,b,tag)
df = pd.DataFrame(rows).T
df['Val_Penalty'] = [M['metrics'][k]['penalty_mean'] for k in ['valid','test','eval']]
print(df.round(3).to_string())
print()
print('旧基线 (τ=5/15) Val_Penalty:', {k: round(v[0],3) for k,v in M['baseline_old_penalty'].items()})""")

md("## 3. 与旧基线对比")
code("""import pandas as pd
comp = pd.DataFrame({
    'v8 Val_Penalty':  [M['metrics'][k]['penalty_mean'] for k in ['valid','test','eval']],
    '旧基线 Val_Penalty': [M['baseline_old_penalty'][k][0] for k in ['valid','test','eval']],
    'v8 触发率':  [M['metrics'][k]['trigger_rate'] for k in ['valid','test','eval']],
    '旧基线触发率': [v[2]/1080 if k=='valid' else v[2]/144 if k=='test' else v[2]/120
                    for k,v in [('valid',M['baseline_old_penalty']['valid']),
                                ('test',M['baseline_old_penalty']['test']),
                                ('eval',M['baseline_old_penalty']['eval'])]],
}, index=['Valid','Test','Eval'])
print('阈值 50 后触发率从 96~100% 收敛到 8~27%, Val_Penalty 提升 2.3~5.8 倍:')
print(comp.round(3).to_string())""")

md("## 4. 过拟合核对")
md("""
- **TS-CV (时序防泄露):** 4 折 Val_Penalty **-0.6093 ± 0.1859** (fold 顺序切分, 无未来信息)
- **训练/验证差距:** Train_Penalty -0.50 vs Val_Penalty -0.89 (惩罚提升主要来自不误触发; 差距在可控范围, 窗口分布差异亦贡献一部分)
- **防过拟合手段:** DART 树 dropout (rate_drop=0.3) + L2 (reg_lambda=2.0) + 每轮特征噪声注入 + 双头早停(15) + ReduceLROnPlateau
""")

md("## 5. 遗留问题")
code("""print('''
- sign 命中仍弱 (Valid 0.26 / Test 0.00 / Eval 0.27) — 与项目史一致 (方向预测能力存疑);
  惩罚提升主要来自"不误触发", 特征对方向/量级的预测能力有限
- 回归头量级仍偏低 (触发样本约 24% |值|>50) — 条件回归 (仅 |y|>50 样本) 是下一步候选
- deploy 侧: 新 deploy 已接入 v8 (models/xgb_v8_*.joblib, active_model=v8, 阈值 50/50),
  深度调优 (因子重建/增量训练) 留待后续
''')""")

nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python'}}
with open('41_report_v8_penalty.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('已生成 41_report_v8_penalty.ipynb')
