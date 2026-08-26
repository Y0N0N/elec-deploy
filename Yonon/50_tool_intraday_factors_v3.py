"""
生成安全的日内横向因子 (Intraday Factors v2)
规则: 只用供给侧预测数据 + 时间结构，绝对不使用 price/rt_price (标签泄露)

安全数据源 (预测时可用):
  - 负荷: load, trade_vol, 统调负荷
  - 发电: 光伏, 风电, 水电, 火电, 发电总出力, 预测出力
  - 备用: 正备用, 负备用, 一次调频备用
  - 联络线: 西电东送电力, 三峡, 粤港联络线 等
  - 时段结构: hour of day

不安全 (标签泄露, 已排除):
  - price, rt_price 的所有日内统计
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import os, gc, warnings
warnings.filterwarnings('ignore')
from config.config import *
import glob

print("=" * 60)
print("生成安全的日内横向因子 v2 (无价格泄露)")
print("=" * 60)

# ============================
# 1. 加载原始数据
# ============================
print("\n[1/4] 加载原始数据...")

price_df   = pd.read_feather(f"{matrix_path}/实际运行结果-用电侧/日前统一结算价.feather")
load_df    = pd.read_feather(f"{matrix_path}/实际运行结果-用电侧/实际用电量.feather")
trade_df   = pd.read_feather(f"{matrix_path}/实际运行结果-用电侧/日前成交电量.feather")
rt_price_df = pd.read_feather(f"{matrix_path}/实际运行结果-用电侧/实时统一结算价.feather")

def stack_to_matrix(df, name):
    stacked = df.stack()
    stacked.index = stacked.index.rename(['date', 'hour'])
    stacked.name = name
    return stacked

price    = stack_to_matrix(price_df, 'price')
load     = stack_to_matrix(load_df, 'load')
trade_vol = stack_to_matrix(trade_df, 'trade_vol')
rt_price = stack_to_matrix(rt_price_df, 'rt_price')

# 预测数据
py_path = f"{matrix_path}/披露预测数据"
feather_files = glob.glob(os.path.join(py_path, "*.feather"))
all_fields = [os.path.splitext(os.path.basename(f))[0] for f in feather_files]
data_field2 = [f for f in all_fields if f.startswith("预测 | ")]
data_fields = ['发电总出力(MW)','D日(MW)','D+1日(MW)','D+2日(MW)','光伏出力预测(MW)',
               '风电出力预测(MW)','水电（含抽蓄）总出力(MW)','预测出力(MW)',
               '正备用(MW)','负备用(MW)','一次调频备用(MW)','']
new_data_field = list(set(data_fields + data_field2))

datafield_dict = {}
for datafield in new_data_field:
    madata_path = f"{py_path}/{datafield}.feather"
    if os.path.exists(madata_path):
        df = pd.read_feather(madata_path)
        datafield_dict[datafield] = df

def resample_15min_to_hourly(df_15min):
    df_hourly = pd.DataFrame(index=df_15min.index)
    for h in range(24):
        cols_15min = [f"{h:02d}:{m:02d}" for m in [0, 15, 30, 45]]
        available_cols = [c for c in cols_15min if c in df_15min.columns]
        if available_cols:
            df_hourly[f"{h:02d}:00"] = df_15min[available_cols].mean(axis=1)
    return df_hourly

pred_stacked = {}
for field in new_data_field:
    if field not in datafield_dict:
        continue
    df = datafield_dict[field]
    n_cols = len(df.columns)
    short_name = field.replace('预测 | ', '').replace('(MW)', '').strip().replace(' ', '_')
    if n_cols == 96:
        df_h = resample_15min_to_hourly(df)
    elif n_cols == 24:
        df_h = df.copy()
    else:
        continue
    pred_stacked[short_name] = stack_to_matrix(df_h, short_name)

del datafield_dict; gc.collect()

matrix_parts = {
    'price': price, 'load': load, 'trade_vol': trade_vol, 'rt_price': rt_price,
}
matrix_parts.update(pred_stacked)
matrix_df = pd.DataFrame(matrix_parts)
print(f"  Matrix: {matrix_df.shape}")

hour_idx = matrix_df.index.get_level_values('hour')
date_idx = matrix_df.index.get_level_values('date')
hour_int = hour_idx.str[:2].astype(int)

# ============================
# 2. 安全的日内因子 (只用供给侧数据)
# ============================
print("\n[2/4] 生成安全的日内因子 (无价格)...")
intraday_factors = {}

def add_factor(name, series):
    if isinstance(series, pd.Series):
        series.name = name
        intraday_factors[name] = series

g_date = matrix_df.groupby('date')

# ----- A. 负荷日内形态 (load 是预测日可用的) -----
print("  A. 负荷日内形态...")
daily_load_mean = g_date['load'].transform('mean')
daily_load_std  = g_date['load'].transform('std')
daily_load_min  = g_date['load'].transform('min')
daily_load_max  = g_date['load'].transform('max')

add_factor('h_load_zscore', (matrix_df['load'] - daily_load_mean) / (daily_load_std + 1e-6))
add_factor('h_load_ratio_davg', matrix_df['load'] / (daily_load_mean + 1e-6))
add_factor('h_load_range_pos', (matrix_df['load'] - daily_load_min) / (daily_load_max - daily_load_min + 1e-6))
add_factor('h_load_diff_davg', matrix_df['load'] - daily_load_mean)

for lag in [1, 2, 3]:
    shifted = matrix_df['load'].groupby('date').shift(lag)
    add_factor(f'h_load_delta_{lag}h', (matrix_df['load'] - shifted) / (shifted.abs() + 1e-6))

ld1 = matrix_df['load'] - matrix_df['load'].groupby('date').shift(1)
ld2 = matrix_df['load'].groupby('date').shift(1) - matrix_df['load'].groupby('date').shift(2)
add_factor('h_load_accel', ld1 - ld2)

# 成交电量日内形态
daily_tvol_mean = g_date['trade_vol'].transform('mean')
daily_tvol_std  = g_date['trade_vol'].transform('std')
add_factor('h_tvol_zscore', (matrix_df['trade_vol'] - daily_tvol_mean) / (daily_tvol_std + 1e-6))
add_factor('h_tvol_ratio_davg', matrix_df['trade_vol'] / (daily_tvol_mean + 1e-6))

# ----- B. 净负荷 & 可再生能源渗透 (供给侧, 安全) -----
print("  B. 净负荷 & 可再生能源...")

solar_col = None; wind_col = None
for c in matrix_df.columns:
    if '光伏' in c and '预测' not in c and solar_col is None: solar_col = c
    if '风电' in c and '预测' not in c and wind_col is None: wind_col = c
if solar_col is None:
    for c in matrix_df.columns:
        if '光伏' in c and solar_col is None: solar_col = c
if wind_col is None:
    for c in matrix_df.columns:
        if '风电' in c and wind_col is None: wind_col = c
solar_col = solar_col or '光伏出力预测'
wind_col = wind_col or '风电出力预测'
print(f"    光伏: {solar_col}, 风电: {wind_col}")

if solar_col in matrix_df.columns and wind_col in matrix_df.columns:
    net_load = matrix_df['load'] - matrix_df[solar_col] - matrix_df[wind_col]
    add_factor('h_net_load', net_load)

    d_nl_mean = net_load.groupby('date').transform('mean')
    d_nl_std  = net_load.groupby('date').transform('std')
    d_nl_min  = net_load.groupby('date').transform('min')
    d_nl_max  = net_load.groupby('date').transform('max')

    add_factor('h_net_load_zscore', (net_load - d_nl_mean) / (d_nl_std + 1e-6))
    add_factor('h_net_load_ratio_davg', net_load / (d_nl_mean.abs() + 1e-6))
    add_factor('h_net_load_range_pos', (net_load - d_nl_min) / (d_nl_max - d_nl_min + 1e-6))

    renew = matrix_df[solar_col] + matrix_df[wind_col]
    add_factor('h_renew_ratio', renew / (matrix_df['load'] + 1e-6))
    add_factor('h_solar_ratio', matrix_df[solar_col] / (matrix_df['load'] + 1e-6))
    add_factor('h_wind_ratio', matrix_df[wind_col] / (matrix_df['load'] + 1e-6))

    solar_delta = matrix_df[solar_col] - matrix_df[solar_col].groupby('date').shift(1)
    wind_delta  = matrix_df[wind_col] - matrix_df[wind_col].groupby('date').shift(1)
    add_factor('h_solar_ramp', solar_delta)
    add_factor('h_wind_ramp', wind_delta)

    nl_delta = net_load - net_load.groupby('date').shift(1)
    add_factor('h_net_load_delta_1h', nl_delta)

    # 净负荷加速度
    nld1 = net_load - net_load.groupby('date').shift(1)
    nld2 = net_load.groupby('date').shift(1) - net_load.groupby('date').shift(2)
    add_factor('h_net_load_accel', nld1 - nld2)

# ----- C. 供给侧关键字段日内形态 -----
print("  C. 供给侧关键字段...")

key_supply = ['统调负荷', '发电总出力', '预测出力', 'D日', 'D+1日', 'D+2日',
              '正备用', '负备用', '一次调频备用', '西电东送电力',
              '省内A类电源', '省内B类电源', '地方电源出力',
              '光伏出力预测', '风电出力预测']

for col in key_supply:
    if col not in matrix_df.columns:
        continue
    short = col.replace('（含抽蓄）', '').replace('(', '_').replace(')', '')[:15]
    d_mean = g_date[col].transform('mean')
    d_std  = g_date[col].transform('std')
    add_factor(f'h_{short}_zscore', (matrix_df[col] - d_mean) / (d_std + 1e-6))
    add_factor(f'h_{short}_ratio_davg', matrix_df[col] / (d_mean.abs() + 1e-6))

    # 日内变化率
    d_shifted = matrix_df[col].groupby('date').shift(1)
    add_factor(f'h_{short}_delta_1h', (matrix_df[col] - d_shifted) / (d_shifted.abs() + 1e-6))

# 水电和火电的特殊日内因子
for col in ['水电', '火电']:
    if col in matrix_df.columns:
        short = col[:15]
        d_mean = g_date[col].transform('mean')
        d_std  = g_date[col].transform('std')
        add_factor(f'h_{short}_zscore', (matrix_df[col] - d_mean) / (d_std + 1e-6))
        add_factor(f'h_{short}_ratio_davg', matrix_df[col] / (d_mean.abs() + 1e-6))

# ----- D. 净负荷爬坡特征 (电力系统关键) -----
print("  D. 净负荷爬坡特征...")

if solar_col in matrix_df.columns and wind_col in matrix_df.columns:
    # 早晚爬坡速率
    s = matrix_df[solar_col]
    # 上午爬坡 (hours 6-10): solar increase
    morning_mask = (hour_int >= 6) & (hour_int <= 10)
    evening_mask = (hour_int >= 16) & (hour_int <= 20)

    # 光伏上午爬坡: hour 10 solar - hour 6 solar
    for h_start, h_end, label in [(6, 10, 'morning'), (7, 11, 'late_morning'), (16, 20, 'evening')]:
        s_start = s[hour_int == h_start].groupby('date').first() if h_start in hour_int.values else None
        s_end   = s[hour_int == h_end].groupby('date').first() if h_end in hour_int.values else None

    # 简化版: 用 shift 计算连续小时的变化累积
    add_factor('h_solar_ramp_3h', matrix_df[solar_col] - matrix_df[solar_col].groupby('date').shift(3))
    add_factor('h_wind_ramp_3h',  matrix_df[wind_col] - matrix_df[wind_col].groupby('date').shift(3))
    add_factor('h_load_ramp_3h',  matrix_df['load'] - matrix_df['load'].groupby('date').shift(3))

# ----- E. 时段细分 + 典型负荷模式距离 -----
print("  E. 时段结构...")

hour_s = pd.Series(hour_int.values, index=matrix_df.index)

add_factor('h_seg_overnight',  ((hour_s >= 0) & (hour_s <= 5)).astype(float))
add_factor('h_seg_morning',    ((hour_s >= 6) & (hour_s <= 9)).astype(float))
add_factor('h_seg_midday',     ((hour_s >= 10) & (hour_s <= 14)).astype(float))
add_factor('h_seg_afternoon',  ((hour_s >= 15) & (hour_s <= 17)).astype(float))
add_factor('h_seg_evening',    ((hour_s >= 18) & (hour_s <= 20)).astype(float))
add_factor('h_seg_night',      ((hour_s >= 21) & (hour_s <= 23)).astype(float))

# 到正午的距离 (中午12点通常是光伏峰值)
dist_midday = np.minimum(np.abs(hour_int.values - 12), 24 - np.abs(hour_int.values - 12))
add_factor('h_dist_to_midday', pd.Series(dist_midday.astype(float), index=matrix_df.index))

# 到典型早晚高峰的距离 (早8晚19)
dist_morning_peak = np.minimum(np.abs(hour_int.values - 8), 24 - np.abs(hour_int.values - 8))
dist_evening_peak = np.minimum(np.abs(hour_int.values - 19), 24 - np.abs(hour_int.values - 19))
add_factor('h_dist_to_morning', pd.Series(dist_morning_peak.astype(float), index=matrix_df.index))
add_factor('h_dist_to_evening', pd.Series(dist_evening_peak.astype(float), index=matrix_df.index))

# 距离日出/日落的小时数 (光伏相关, 简化: 日出6点, 日落18点)
dist_sunrise = np.minimum(np.abs(hour_int.values - 6), 24 - np.abs(hour_int.values - 6))
dist_sunset  = np.minimum(np.abs(hour_int.values - 18), 24 - np.abs(hour_int.values - 18))
add_factor('h_dist_to_sunrise', pd.Series(dist_sunrise.astype(float), index=matrix_df.index))
add_factor('h_dist_to_sunset',  pd.Series(dist_sunset.astype(float), index=matrix_df.index))

# cos/sin 编码 (周期性)
add_factor('h_hour_cos', pd.Series(np.cos(2 * np.pi * hour_int.values / 24), index=matrix_df.index))
add_factor('h_hour_sin', pd.Series(np.sin(2 * np.pi * hour_int.values / 24), index=matrix_df.index))

# ----- F. 供给侧耦合因子 -----
print("  F. 供给侧耦合...")

# 省内电源出力占比
if '省内A类电源' in matrix_df.columns and '省内B类电源' in matrix_df.columns:
    total_prov = matrix_df['省内A类电源'] + matrix_df['省内B类电源']
    add_factor('h_prov_A_ratio', matrix_df['省内A类电源'] / (total_prov + 1e-6))
    add_factor('h_prov_total_ratio', total_prov / (matrix_df['load'] + 1e-6))

# 西电东送占比
if '西电东送电力' in matrix_df.columns:
    add_factor('h_west_east_ratio', matrix_df['西电东送电力'] / (matrix_df['load'] + 1e-6))

# 备用容量利用率
if '正备用' in matrix_df.columns and '负备用' in matrix_df.columns:
    add_factor('h_reserve_ratio', matrix_df['正备用'] / (matrix_df['负备用'] + 1e-6))

# ----- H. 市场结构指标 (供需紧度, 预测期可用) -----
# 注: 直流传输线因子 (h_dc_*) 已废弃 — 数据审计发现 DC 线路数据在 2025-06-29
#     后即无发布 (整体 70.5% NaN), 预测期 100% NaN, 填充参考值也是 NaN, 无信号.
print("  H. 市场结构...")

# 供需紧度: 发电总出力 / 负荷
if '发电总出力' in matrix_df.columns:
    add_factor('h_supply_demand_ratio', matrix_df['发电总出力'] / (matrix_df['load'] + 1e-6))

# 备用相对负荷水平
if '正备用' in matrix_df.columns:
    add_factor('h_reserve_load_ratio', matrix_df['正备用'] / (matrix_df['load'] + 1e-6))

# 负荷曲线形态 (日级形态, 每小时有值)
daily_load_max = g_date['load'].transform('max')
daily_load_min = g_date['load'].transform('min')
daily_load_avg = g_date['load'].transform('mean')
add_factor('h_load_peak_valley_ratio', daily_load_max / (daily_load_min + 1e-6))
add_factor('h_load_factor', daily_load_avg / (daily_load_max + 1e-6))

# D 日预测差分: 负荷变化预期 (D+1 vs D日, D+2 vs D+1)
if 'D日' in matrix_df.columns and 'D+1日' in matrix_df.columns:
    add_factor('h_d1_d0_diff', (matrix_df['D+1日'] - matrix_df['D日']) / (matrix_df['D日'].abs() + 1e-6))
if 'D+1日' in matrix_df.columns and 'D+2日' in matrix_df.columns:
    add_factor('h_d2_d1_diff', (matrix_df['D+2日'] - matrix_df['D+1日']) / (matrix_df['D+1日'].abs() + 1e-6))

# 可再生能源 3h 波动 (太阳/风力的短时变化)
if solar_col in matrix_df.columns and wind_col in matrix_df.columns:
    renew = matrix_df[solar_col] + matrix_df[wind_col]
    renew_3h = renew - renew.groupby('date').shift(3)
    renew_6h = renew - renew.groupby('date').shift(6)
    add_factor('h_renew_vol_3h', renew_3h.abs())
    add_factor('h_renew_vol_6h', renew_6h.abs())

# 供给侧供需平衡: 发电总出力 vs 负荷 的缺口 (供需失衡信号)
if '发电总出力' in matrix_df.columns:
    balance = matrix_df['发电总出力'] - matrix_df['load']
    add_factor('h_balance_gap', balance)
    b_mean = balance.groupby('date').transform('mean')
    b_std = balance.groupby('date').transform('std')
    add_factor('h_balance_gap_zscore', (balance - b_mean) / (b_std + 1e-6))

print(f"\n  安全的日内因子总数: {len(intraday_factors)}")

# ============================
# 3. 删除所有旧的 h_* 因子 (包含泄露的) + 保存
# ============================
print("\n[3/4] 清理旧日内因子 + 保存...")
for old_file in sorted(os.listdir(dataset_path)):
    if old_file.startswith('h_') and old_file.endswith('.fea'):
        os.remove(os.path.join(dataset_path, old_file))

saved = 0
for name, series in intraday_factors.items():
    if isinstance(series, pd.Series) and series.index.nlevels == 2:
        df_wide = series.unstack()
    elif isinstance(series, pd.Series):
        df_wide = series.to_frame()
    else:
        continue
    df_wide.to_feather(f"{dataset_path}/{name}.fea")
    saved += 1
print(f"  保存 {saved} 个安全的日内因子")

# ============================
# 4. 验证
# ============================
print("\n[4/4] 验证...")
all_feas = sorted(os.listdir(dataset_path))
h_feas = [f for f in all_feas if f.startswith('h_')]
print(f"  总因子: {len(all_feas)}")
print(f"  日内因子: {len(h_feas)}")
print(f"  示例: {h_feas[:12]}")

# 验证无价格泄露
h_names = [f.replace('.fea', '') for f in h_feas]
leaky = [n for n in h_names if 'price' in n.lower() or 'rt_' in n.lower()]
if leaky:
    print(f"  ⚠️ 警告: 仍有价格泄露因子: {leaky}")
else:
    print(f"  ✅ 无价格泄露因子")

# 验证预测集可用性
print(f"\n  预测日期 (2026-07-22~27) 的空值检查:")
for f in h_feas[:5]:
    df_test = pd.read_feather(f'{dataset_path}/{f}')
    future = df_test.loc['2026-07-22':'2026-07-27']
    na_count = future.isna().sum().sum()
    total = future.size
    print(f"    {f}: {na_count}/{total} 空值 ({na_count/total*100:.0f}%)")

print("\n" + "=" * 60)
print("安全日内因子 v2 生成完成!")
print(f"新增 {len(h_feas)} 个供给侧日内因子 (无价格泄露)")
print("=" * 60)
