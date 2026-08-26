"""
生成事件语义聚合因子 (Event Factors v1) — C1 落地
数据源: the configured disclosure matrix/ 6366 个机组约束/调度事件文件
背景: 现有 gc_* 只用 4 类字段 (出力上下限/开机台数) + 一个"其他"桶,
      828 种调度/约束事件 (气量管控/必停/停电检修/短路控制等) 全被混进"其他",
      只算 active/count, 丢失事件语义。
本工具: 按业务事件类别拆分聚合 → 每类 active/total/count/mean 四件套,
      替代"其他"桶, 保留事件语义 (稀缺前兆信号)。

红线: 只用供给侧/调度约束信息, 绝不用 price/rt_price/spread (标签泄露);
      全部信号预测期可得 (披露数据覆盖到 2026-07-27 预测日)。

产物: qlib158 因子库新增 ev_* .fea (事件类别聚合因子)
"""
import os, glob, gc, warnings, re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from config.config import *

print("=" * 60)
print("生成事件语义聚合因子 (Event Factors v1, C1)")
print("=" * 60)

PY = f"{matrix_path}/披露预测数据"
DATASET = dataset_path
os.makedirs(DATASET, exist_ok=True)

# ============================
# 1. 事件类别映射 (文件名第1段关键词)
# ============================
# 设计: 业务语义类别 → 关键词列表 (顺序匹配, 命中即归类)
# 注意: 类别互斥, 用首命中; 排前面的类别优先
EVENT_CATS = [
    # 气量/燃料约束 → 稀缺前兆 (最关键)
    ('gas_limit',   ['气量管控', '缺气', '气量受限', '供气', 'LNG', '气量']),
    # 必开/必停/限开/限高 → 机组状态直接约束
    ('must_run',    ['必开', '必停']),
    ('limit_open',  ['限开', '限高', '出力受限', '受限']),
    # 停电/检修/同停/轮停 → 电网约束
    ('outage',      ['停电', '检修', '同停', '轮停', '临停', '退运', '停运']),
    # 短路电流/稳控/短路比 → 安全约束
    ('short_circuit', ['短路电流', '短路比', '稳控']),
    # 调压/电压/保供应 → 系统紧张度
    ('voltage',     ['调压', '电压', '保供应']),
    # 日前出力调用 → 调度提前干预
    ('dispatch',    ['出力调用', '日前调用', '调用']),
    # 台风/特殊时段
    ('special',     ['台风', '春节', '节日', '国庆']),
    # 频率效应
    ('freq_effect', ['频率效应']),
]

# 已覆盖的字段类型 (现有 gc_* 已聚合, 排除避免重复)
# ⚠️ 注意: 事件类型在第1段, 字段类型在后缀 —— 两者正交。
# 事件文件 (气量管控/停电检修等) 的后缀往往也是"出力下限/上限(MW)",
# 所以不能按后缀排除, 而要按第1段事件类型分类 (见 classify_event)。
EXCLUDED_FIELDS = []  # 不再按后缀排除

def classify_event(base):
    """按文件名第1段 (| 前) 分类 → 事件类别名 or None
    第1段是调度/约束事件类型 (如'广东中调气量管控'),
    后段是机组/字段明细。事件语义在第1段。"""
    seg1 = base.split('|')[0]
    for cat, kws in EVENT_CATS:
        if any(k in seg1 for k in kws):
            return cat
    return None

# ============================
# 2. 收集并分类事件文件
# ============================
print("\n[1/3] 收集事件文件...")
all_files = glob.glob(os.path.join(PY, "*.feather"))
cat_files = {cat: [] for cat, _ in EVENT_CATS}
unmatched = []
for f in all_files:
    base = os.path.basename(f)
    # 跳过已覆盖的字段类型文件
    if any(k in base for k in EXCLUDED_FIELDS):
        continue
    cat = classify_event(base)
    if cat:
        cat_files[cat].append(f)
    else:
        unmatched.append(base)

print(f"  总文件: {len(all_files)} | 事件文件: {sum(len(v) for v in cat_files.values())} | 未匹配: {len(unmatched)}")
for cat, files in cat_files.items():
    print(f"  {cat}: {len(files)}")

# ============================
# 3. 向量化聚合 → (date, hour) 面板
# ============================
print("\n[2/3] 向量化聚合事件数据...")

import os as _dep_os
DATES = pd.date_range(_dep_os.environ.get('DEPLOY_FACTOR_START', '2025-01-01'),
                      _dep_os.environ.get('DEPLOY_FACTOR_END', '2026-07-27'), freq='D')
N_D, N_H = len(DATES), 24

# 只聚合有文件的类别
active_cats = {c: fs for c, fs in cat_files.items() if len(fs) > 0}
agg = {cat: {'active': np.zeros((N_D, N_H)), 'sum': np.zeros((N_D, N_H)),
             'count': np.zeros((N_D, N_H))} for cat in active_cats}

def agg_file_vec(f, cat):
    """向量化聚合单个事件文件: 96列(15min) → 24小时"""
    try:
        df = pd.read_feather(f)
    except Exception as e:
        print(f"  ⚠️ 读取失败 {f}: {e}")
        return
    df.index = pd.to_datetime(df.index)
    rows = df.index.get_indexer(DATES)
    valid = rows >= 0
    rows = rows[valid]
    if len(rows) == 0:
        return
    vals = df.values[rows]
    vals = vals.reshape(-1, 24, 4)
    active_hr = (~np.isnan(vals)).any(axis=2)
    with np.errstate(all='ignore'):
        sum_hr = np.nansum(vals, axis=2)
        cnt_hr = (~np.isnan(vals)).sum(axis=2)
    sum_hr = np.where(active_hr, np.nan_to_num(sum_hr), 0.0)
    cnt_hr = np.where(active_hr, cnt_hr, 0.0)
    day_idx = np.where(valid)[0]
    agg[cat]['active'][day_idx] += active_hr.astype(float)
    agg[cat]['sum'][day_idx] += sum_hr
    agg[cat]['count'][day_idx] += cnt_hr

for cat, files in active_cats.items():
    print(f"  处理 {cat} ({len(files)} 个文件)...")
    for i, f in enumerate(files):
        agg_file_vec(f, cat)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(files)}")
    gc.collect()

# 组装宽表 (与现有 .fea 一致: date=str, hour='HH:00')
idx = pd.MultiIndex.from_product(
    [DATES.strftime('%Y-%m-%d'), [f'{h:02d}:00' for h in range(24)]],
    names=['date', 'hour'])
base_factors = {}
for cat in active_cats:
    a = agg[cat]
    base_factors[f'ev_{cat}_active'] = pd.Series(a['active'].ravel(), index=idx)
    base_factors[f'ev_{cat}_total'] = pd.Series(a['sum'].ravel(), index=idx)
    base_factors[f'ev_{cat}_count'] = pd.Series(a['count'].ravel(), index=idx)
    with np.errstate(all='ignore'):
        vmean = np.divide(a['sum'], a['count'], out=np.full_like(a['sum'], np.nan), where=a['count'] > 0)
    base_factors[f'ev_{cat}_mean'] = pd.Series(vmean.ravel(), index=idx)

ev_df = pd.DataFrame(base_factors).sort_index()
print(f"  基础事件聚合: {ev_df.shape}")

# ============================
# 4. 日内形态因子 + 保存
# ============================
print("\n[3/3] 生成形态因子 + 保存...")

intraday = {}
def add_factor(name, series):
    series.name = name
    intraday[name] = series

# ===== 基础信号: 原始 active/total (当日事件数/强度) 直接保存 =====
# 这是事件因子的核心: 当日生效事件数 (active) 与强度 (total) 本身就是稀缺前兆
for cat in active_cats:
    for col in [f'ev_{cat}_active', f'ev_{cat}_total']:
        if col in ev_df.columns:
            add_factor(col, ev_df[col])  # 原始值 (无未来信息: 当日披露)

# ===== 形态因子: 事件是日级常数 → 用日级变化特征 (30天滚动基线) =====
ROLL = 30  # 滚动基线天数

def rolling_base(col_series):
    """按日期滚动 30 天均值/标准差 (预测期只用到过去, 无未来泄露)"""
    df = col_series.unstack()  # date × hour
    base_mean = df.rolling(ROLL, min_periods=7).mean().stack()
    base_std = df.rolling(ROLL, min_periods=7).std().stack()
    return base_mean, base_std

for cat in active_cats:
    for col in [f'ev_{cat}_active', f'ev_{cat}_total']:
        if col not in ev_df.columns:
            continue
        short = col[3:]
        bm, bs = rolling_base(ev_df[col])
        z = (ev_df[col] - bm) / (bs + 1e-6)
        add_factor(f'ev_{short}_zbase', z.fillna(0.0))  # 滚动窗开头 NaN → 0 (与 gc 一致)
        add_factor(f'ev_{short}_ratio_base', (ev_df[col] / (bm.abs() + 1e-6)).fillna(0.0))
    # 当日事件是否新出现/活跃度 (active>0)
    ac = f'ev_{cat}_active'
    if ac in ev_df.columns:
        add_factor(f'ev_{cat}_active_d', (ev_df[ac] > 0).astype(float))

# 事件总强度 (所有事件类别的 active 之和) — 替代 gc_其他_active
total_active = ev_df[[f'ev_{c}_active' for c in active_cats]].sum(axis=1)
add_factor('ev_total_active', total_active)
bm, bs = rolling_base(total_active)
add_factor('ev_total_active_zbase', ((total_active - bm) / (bs + 1e-6)).fillna(0.0))

# 稀缺前兆组合 (事件并发度, 气量管控 × 限开/必停)
for a, b in [('ev_gas_limit_active', 'ev_limit_open_active'),
             ('ev_gas_limit_active', 'ev_must_run_active'),
             ('ev_outage_active', 'ev_must_run_active')]:
    if a in ev_df.columns and b in ev_df.columns:
        add_factor(f'ev_burst_{a[3:]}_{b[3:]}',
                   ev_df[a] * ev_df[b] / (ev_df[a] + ev_df[b] + 1e-6))

# 保存
for old in os.listdir(DATASET):
    if old.startswith('ev_') and old.endswith('.fea'):
        os.remove(os.path.join(DATASET, old))
saved = 0
for name, series in intraday.items():
    df_wide = series.unstack() if series.index.nlevels == 2 else series.to_frame()
    df_wide.to_feather(f"{DATASET}/{name}.fea")
    saved += 1
print(f"  保存 {saved} 个事件因子")

# 验证: 预测期空值 + 泄露检查
all_ev = sorted([f for f in os.listdir(DATASET) if f.startswith('ev_')])
print(f"\n  预测日期 (2026-07-22~27) 空值检查:")
for f in all_ev[:5]:
    df_test = pd.read_feather(f'{DATASET}/{f}')
    future = df_test.loc['2026-07-22':'2026-07-27']
    na_count = future.isna().sum().sum()
    print(f"    {f}: {na_count}/{future.size} 空值 ({na_count/future.size*100:.0f}%)")
# 泄露检查: 事件因子名 (英文, 如 ev_short_circuit_*) 里 'rt_' 会误匹配 short_circuit。
# 用中文价格关键词判断更准 (因子名无中文价格词); 数据源本身无价格因子。
leaky = [f for f in all_ev if any(k in f for k in ['price', 'rt_', 'spread']) and 'short_circuit' not in f]
print(f"  事件因子总数: {len(all_ev)}, 泄露检查: {'⚠️ ' + str(leaky) if leaky else '✅ 无泄露 (short_circuit 为英文名误报)'}")

print("\n" + "=" * 60)
print("事件语义聚合因子生成完成!")
print("=" * 60)
