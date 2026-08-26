"""
生成电网约束/机组状态聚合因子 (Grid Constraint Factors v1) — 向量化版
数据源: the configured disclosure matrix/ 中 5540+ 个机组约束文件
规则: 只用供给侧预测数据 + 约束信息，绝对不使用 price/rt_price (标签泄露)

文件格式: 573天 × 96列(15分钟), NaN 表示约束不生效, 非 NaN 表示约束生效
文件类型:
  - 最大开机台数(台): 约束机组最大可开机数量
  - 最小开机台数(台): 约束机组最小开机数量
  - 出力上限(MW):     机组出力上限
  - 出力下限(MW):     机组出力下限
  - 其他场景 (必停/停电/检修/停运/同停/必开)

聚合方案: 向量化逐文件聚合 → 日/时频指标 → 日内形态因子
关键优势: 约束数据覆盖到 2026-07-27 (预测日期), 是完全安全的非价格信号!
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import os, glob, gc, warnings
warnings.filterwarnings('ignore')
from config.config import *

print("=" * 60)
print("生成电网约束聚合因子 (Grid Constraint v1, 向量化)")
print("=" * 60)

PY = f"{matrix_path}/披露预测数据"
DATASET = dataset_path
os.makedirs(DATASET, exist_ok=True)

# ============================
# 1. 收集并分类约束文件
# ============================
print("\n[1/4] 收集约束文件...")
all_files = glob.glob(os.path.join(PY, "*.feather"))
constraint_files = []
for f in all_files:
    base = os.path.basename(f)
    if any(k in base for k in ['最大开机台数', '最小开机台数', '出力上限', '出力下限',
                                '必停', '必开', '同停', '停电', '检修', '停运', '受限']):
        constraint_files.append(f)

print(f"  约束文件: {len(constraint_files)} / {len(all_files)}")

cats = {'出力上限': [], '出力下限': [], '最大开机台数': [], '最小开机台数': [], '其他': []}
for f in constraint_files:
    base = os.path.basename(f)
    if '出力上限' in base: cats['出力上限'].append(f)
    elif '出力下限' in base: cats['出力下限'].append(f)
    elif '最大开机台数' in base: cats['最大开机台数'].append(f)
    elif '最小开机台数' in base: cats['最小开机台数'].append(f)
    else: cats['其他'].append(f)

for k, v in cats.items():
    print(f"  {k}: {len(v)}")

# ============================
# 2. 向量化聚合 → (date, hour) 面板
# ============================
print("\n[2/4] 向量化聚合约束数据...")

# 预分配 numpy 数组: [date, hour]
# 日期范围可被 deploy 通过环境变量扩展（默认截至 2026-07-27，保持原行为）
import os as _dep_os
DATES = pd.date_range(_dep_os.environ.get('DEPLOY_FACTOR_START', '2025-01-01'),
                      _dep_os.environ.get('DEPLOY_FACTOR_END', '2026-07-27'), freq='D')
N_D, N_H = len(DATES), 24
date_to_row = {d: i for i, d in enumerate(DATES)}

agg = {cat: {'active': np.zeros((N_D, N_H)), 'sum': np.zeros((N_D, N_H)),
             'count': np.zeros((N_D, N_H))} for cat in cats}

def agg_file_vec(f, cat):
    """向量化聚合单个约束文件: 96列(15min) → 24小时"""
    df = pd.read_feather(f)
    df.index = pd.to_datetime(df.index)
    # 对齐日期行
    rows = df.index.get_indexer(DATES)
    valid = rows >= 0
    rows = rows[valid]
    # 值矩阵: [n_valid_days, 96]
    vals = df.values[rows]
    vals = vals.reshape(-1, 24, 4)  # [day, hour, 15min]
    v = np.nan_to_num(vals, nan=np.nan)
    active_hr = (~np.isnan(vals)).any(axis=2)     # [day, hour] 活跃
    with np.errstate(all='ignore'):
        sum_hr = np.nansum(vals, axis=2)          # [day, hour] 和 (NaN 忽略)
        cnt_hr = (~np.isnan(vals)).sum(axis=2)    # [day, hour] 非NaN数
    sum_hr = np.where(active_hr, np.nan_to_num(sum_hr), 0.0)
    cnt_hr = np.where(active_hr, cnt_hr, 0.0)
    # 累加
    day_idx = np.where(valid)[0]
    agg[cat]['active'][day_idx] += active_hr.astype(float)
    agg[cat]['sum'][day_idx] += sum_hr
    agg[cat]['count'][day_idx] += cnt_hr

for cat, files in cats.items():
    print(f"  处理 {cat} ({len(files)} 个文件)...")
    for i, f in enumerate(files):
        agg_file_vec(f, cat)
        if (i + 1) % 400 == 0:
            print(f"    {i+1}/{len(files)}")
    gc.collect()

print("\n  聚合完成, 构建 DataFrame...")

# 组装成 (date, hour) MultiIndex 宽表格式 (与现有 .fea 一致: date=str, hour='HH:00')
idx = pd.MultiIndex.from_product(
    [DATES.strftime('%Y-%m-%d'), [f'{h:02d}:00' for h in range(24)]],
    names=['date', 'hour'])
base_factors = {}
for cat in cats.keys():
    a = agg[cat]
    active_s = pd.Series(a['active'].ravel(), index=idx, name=f'gc_{cat}_active')
    vsum_s = pd.Series(a['sum'].ravel(), index=idx, name=f'gc_{cat}_total')
    vcnt_s = pd.Series(a['count'].ravel(), index=idx, name=f'gc_{cat}_count')
    with np.errstate(all='ignore'):
        vmean_arr = np.divide(a['sum'], a['count'], out=np.full_like(a['sum'], np.nan), where=a['count'] > 0)
    vmean_s = pd.Series(vmean_arr.ravel(), index=idx, name=f'gc_{cat}_mean')
    base_factors[f'gc_{cat}_active'] = active_s
    base_factors[f'gc_{cat}_total'] = vsum_s
    base_factors[f'gc_{cat}_mean'] = vmean_s

print(f"  基础聚合指标: {len(base_factors)}")

# ============================
# 3. 生成日内形态因子
# ============================
print("\n[3/4] 生成日内形态因子...")

gc_df = pd.DataFrame(base_factors).sort_index()
print(f"  GC 面板: {gc_df.shape}")

intraday_factors = {}

def add_factor(name, series):
    if isinstance(series, pd.Series):
        series.name = name
        intraday_factors[name] = series

g_date = gc_df.groupby('date')

key_cols = ['gc_出力上限_active', 'gc_出力上限_total', 'gc_出力下限_total',
            'gc_其他_active', 'gc_最大开机台数_active', 'gc_出力上限_mean']
for col in key_cols:
    if col not in gc_df.columns:
        continue
    short = col.replace('gc_', '')
    d_mean = g_date[col].transform('mean')
    d_std = g_date[col].transform('std')
    add_factor(f'gc_{short}_zscore', (gc_df[col] - d_mean) / (d_std + 1e-6))
    add_factor(f'gc_{short}_ratio_davg', gc_df[col] / (d_mean.abs() + 1e-6))
    shifted = gc_df[col].groupby('date').shift(1)
    add_factor(f'gc_{short}_delta_1h', (gc_df[col] - shifted) / (shifted.abs() + 1e-6))

# 组合指标
if 'gc_出力上限_total' in gc_df.columns and 'gc_出力下限_total' in gc_df.columns:
    add_factor('gc_limit_band', gc_df['gc_出力上限_total'] - gc_df['gc_出力下限_total'])

total_active = gc_df[[f'gc_{c}_active' for c in cats]].sum(axis=1)
add_factor('gc_other_share', gc_df['gc_其他_active'] / (total_active + 1e-6))

if 'gc_出力上限_active' in gc_df.columns:
    d_max = g_date['gc_出力上限_active'].transform('max')
    d_min = g_date['gc_出力上限_active'].transform('min')
    add_factor('gc_active_range_pos', (gc_df['gc_出力上限_active'] - d_min) / (d_max - d_min + 1e-6))

print(f"  日内形态因子: {len(intraday_factors)}")

# ============================
# 4. 保存 + 验证
# ============================
print("\n[4/4] 保存 + 验证...")

for old in os.listdir(DATASET):
    if old.startswith('gc_') and old.endswith('.fea'):
        os.remove(os.path.join(DATASET, old))

saved = 0
for name, series in intraday_factors.items():
    if isinstance(series, pd.Series) and series.index.nlevels == 2:
        df_wide = series.unstack()
    elif isinstance(series, pd.Series):
        df_wide = series.to_frame()
    else:
        continue
    df_wide.to_feather(f"{DATASET}/{name}.fea")
    saved += 1
print(f"  保存 {saved} 个电网约束因子")

print(f"\n  预测日期 (2026-07-22~27) 空值检查:")
all_gc = sorted([f for f in os.listdir(DATASET) if f.startswith('gc_')])
for f in all_gc[:5]:
    df_test = pd.read_feather(f'{DATASET}/{f}')
    future = df_test.loc['2026-07-22':'2026-07-27']
    na_count = future.isna().sum().sum()
    total = future.size
    print(f"    {f}: {na_count}/{total} 空值 ({na_count/total*100:.0f}%)")

leaky = [f for f in all_gc if 'price' in f.lower() or 'rt_' in f.lower()]
print(f"  GC 因子总数: {len(all_gc)}, 泄露检查: {'⚠️ ' + str(leaky) if leaky else '✅ 无泄露'}")

print("\n" + "=" * 60)
print("电网约束因子生成完成!")
print("=" * 60)
