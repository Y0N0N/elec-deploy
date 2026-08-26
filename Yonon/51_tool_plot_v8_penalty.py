# ============================================================
# 51_tool_plot_v8_penalty.py — 绘制 v8 惩罚曲线 / 验证损失曲线
# 读取最新 v8 模型 joblib 里的 penalty_history, 输出 show/v7/ 图
# ============================================================
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

YONON = os.path.dirname(os.path.abspath(__file__))

# 最新 v8 模型指针 (v8 优先, 回退 v7)
ptr = json.load(open(f'{YONON}/deploy/latest_model.json'))
path = ptr.get('v8') or ptr.get('v7') or f'{YONON}/models/xgb_v8.joblib'
assert os.path.exists(path), f"模型不存在: {path}"
import joblib
m = joblib.load(path)
hist = np.asarray(m['penalty_history'])   # (epoch, train_pen, val_pen, lr, val_loss)
ep, train_pen, val_pen, lr, val_loss = hist[:,0], hist[:,1], hist[:,2], hist[:,3], hist[:,4]
best_ep = m['best_epoch']

# ---- dataviz 调色板 (light mode, 分类槽位 1 蓝 / 2 橙) ----
C1, C2 = '#2a78d6', '#eb6834'          # series: Train / Val
C_BEST = '#0b0b0b'                     # primary ink (标记最优 epoch)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'; MUTED = '#898781'; GRID = '#e1e0d9'

plt.rcParams.update({
    'font.family': ['WenQuanYi Micro Hei', 'DejaVu Sans'],
    'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
})

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

# ── 左: 惩罚分数曲线 (Train_Penalty / Val_Penalty) ──
ax[0].plot(ep, train_pen, color=C1, lw=2.0, label='Train_Penalty')
ax[0].plot(ep, val_pen, color=C2, lw=2.0, label='Val_Penalty')
ax[0].axvline(best_ep, color=C_BEST, lw=1.2, ls='--', alpha=0.7)
ax[0].annotate(f'best @epoch {int(best_ep)}\nVal_Penalty={val_pen[best_ep]:+.3f}',
               xy=(best_ep, val_pen[best_ep]), xytext=(best_ep + 6, val_pen[best_ep] + 0.10),
               color=INK, fontsize=9,
               arrowprops=dict(arrowstyle='-', color=MUTED, lw=1))
ax[0].set_xlabel('epoch'); ax[0].set_ylabel('Penalty Score (越高越好)')
ax[0].set_title('Penalty Score 曲线 (规则 A/B/C)', fontsize=11, color=INK)
ax[0].legend(frameon=False, fontsize=9, loc='lower right')
ax[0].set_ylim(min(val_pen.min(), train_pen.min()) - 0.05, 0.05)

# ── 右: 归一化验证损失 (早停 / LR 用) ──
ax[1].plot(ep, val_loss, color=C1, lw=2.0, label='Val_Loss (归一化)')
ax[1].axvline(best_ep, color=C_BEST, lw=1.2, ls='--', alpha=0.7)
ax[1].set_xlabel('epoch'); ax[1].set_ylabel('归一化 Val_Loss (越低越好)')
ax[1].set_title('归一化验证损失 (EarlyStopping/ReduceLROnPlateau 依据)', fontsize=11, color=INK)
ax[1].legend(frameon=False, fontsize=9, loc='upper right')

fig.suptitle(f'v8 训练监控 — 触发阈值 50 | 最优 Val_Penalty={m["best_val_penalty"]:+.3f} @epoch {int(best_ep)}',
             fontsize=12, color=INK, y=1.02)
fig.tight_layout()
outdir = f'{YONON}/show/v8'
os.makedirs(outdir, exist_ok=True)
out = f'{outdir}/v8_penalty_curves.png'
fig.savefig(out, dpi=140, bbox_inches='tight', facecolor=SURFACE)
print(f'已保存: {out}')
