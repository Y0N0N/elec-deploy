#!/usr/bin/env python
# ============================================================
# deploy/1_ingest_xlsx.py — 步骤 1：导入每日新收到的 xlsx
#
# 作用：
#   1) 扫描 upload 目录，找出未导入过的 披露xlsx 和 实际结果xlsx
#   2) 披露xlsx → 追加到 披露预测数据.feather + 逐通道矩阵（新一天）
#   3) 实际结果xlsx → 追加到 实际运行结果-用电侧 矩阵
#   4) 重建 spread_label.feather（日前-实时价差标签）
#   5) 更新 sp_wow 价差状态因子（d-7/d-14 真实价差）
#
# 幂等：manifest.json 记录已处理文件；重复运行自动跳过。
# 用法：  python 1_ingest_xlsx.py
#         python 1_ingest_xlsx.py --dry-run    （只检查，不写入）
# ============================================================
import argparse
import json
import os
import re
import sys
import warnings

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(_HERE)
while not os.path.exists(os.path.join(_ROOT, '_cfg.py')):
    _ROOT = os.path.dirname(_ROOT)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from _cfg import cfg

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pyf
import pyarrow.ipc as ipc


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════
def load_manifest():
    if os.path.exists(cfg.MANIFEST_FILE):
        with open(cfg.MANIFEST_FILE) as f:
            return json.load(f)
    return {"disclosure": [], "actual": [], "rebuilt_factors_dates": []}


def save_manifest(man):
    os.makedirs(os.path.dirname(cfg.MANIFEST_FILE), exist_ok=True)
    with open(cfg.MANIFEST_FILE, 'w') as f:
        json.dump(man, f, ensure_ascii=False, indent=2)


def extract_date(filename):
    """从文件名提取日期，如 '信息披露查询预测信息(2026-07-27).xlsx' → '2026-07-27'"""
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def classify_file(filename):
    """按文件名判断类型: 'disclosure' | 'actual' | None。

    严格规则 (2026-08-13 修复):
      - 含 '实际运行结果' 或 '实际信息' → actual（'信息披露查询实际信息*.xlsx' 是实际信息，不是披露预测）
      - 含 '预测信息' → disclosure（'信息披露查询预测信息*.xlsx' 等）
      - 其它无法识别 → None（跳过 + 警告，不写任何数据）
    """
    if '实际运行结果' in filename or '实际信息' in filename:
        return 'actual'
    if '预测信息' in filename or ('信息披露' in filename and '预测' in filename):
        return 'disclosure'
    return None


def sheet_base(sheet, date):
    """去掉 sheet 名里的日期后缀，如 '负荷预测信息(2026-07-27)' → '负荷预测信息'"""
    for suf in [f'({date})', f'（{date}）']:
        if sheet.endswith(suf):
            return sheet[: -len(suf)]
    return sheet


# ════════════════════════════════════════════════════════════
# 披露 xlsx 解析（规则已用 07-27 数据实测 182/182 验证）
# ════════════════════════════════════════════════════════════
def parse_disclosure_xlsx(path, date):
    """返回 {列名: np.array(96)}。只解析供给侧 + 必开必停机组（群）约束两类 sheet。"""
    xl = pd.ExcelFile(path)
    rows = {}
    for sheet in xl.sheet_names:
        base = sheet_base(sheet, date)
        sh = xl.parse(sheet)
        tc = [c for c in sh.columns if str(c).count(':') == 1]   # 时间列 00:00..23:45
        if not tc:
            continue   # 无时间列的 sheet（检修/容量等）历史上未进矩阵，跳过

        is_supply = any(k in base for k in cfg.SUPPLY_SHEET_KEYWORDS)
        is_constraint = (base == '必开必停机组（群）约束预测信息'
                         or base == '机组群约束(含电量约束)-约束详情')
        if not (is_supply or is_constraint):
            continue

        idc = [c for c in sh.columns if c not in tc]
        if is_constraint:
            # 约束 sheet: 列名 = 机组群名|台数|电厂ID|电厂名称|机组ID|机组名称|数据类型
            fields = [f for f in cfg.CONSTRAINT_COL_FIELDS if f in sh.columns]
        else:
            # 供给侧 sheet: 列名 = 所有非时间列用 ' | ' 连接
            fields = idc

        for _, row in sh.iterrows():
            colname = ' | '.join(str(row[c]) for c in fields)
            vals = pd.to_numeric(row[tc], errors='coerce').values.astype(float)
            if len(vals) == 96:
                rows[colname] = vals
    return rows


def time_index_15min(date):
    """某天的 96 个 15min 时间点"""
    return [f'{h:02d}:{m:02d}' for h in range(24) for m in (0, 15, 30, 45)]


def _merge_wide_arrow(raw_path, date, times, new_df):
    """用 pyarrow 列式把新一天并入 披露宽表（避免全量 to_pandas 5GB+ 拷贝）。

    宽表结构：长表，(Date,Time) MultiIndex 落在表末两列，其余为通道列。
    统一目标列序 = [旧数据列..., 新通道列..., Date, Time]，old 与新行都按此对齐。
    返回 (merged_table, n_new_cols, n_new_dates, n_upd_dates)。
    """
    old = pyf.read_table(raw_path)                       # 列式，零拷贝读入
    names = old.column_names
    if 'Date' not in names or 'Time' not in names:
        raise ValueError(f'披露宽表缺 Date/Time 列（结构异常: {names[:3]}...）')

    # 已有日期集合（只读索引两列，几十 MB）
    old_dates_arr = old.column('Date').to_pandas().astype(str)
    old_dates = set(old_dates_arr.unique())
    old_data_cols = [c for c in names if c not in ('Date', 'Time')]
    n_old_rows = old.num_rows

    is_new_date = date not in old_dates
    n_new_dates = 1 if is_new_date else 0
    n_upd_dates = 0 if is_new_date else 1
    if not is_new_date:
        print(f"  警告: 披露数据已有 {date}，将覆盖该天值")
        # 覆盖语义：滤掉该日期旧行（96 行）
        keep_mask = pa.array(old_dates_arr.values != date)
        old = old.filter(keep_mask)
        n_old_rows = old.num_rows
        del keep_mask
    del old_dates_arr

    # 新增通道列（历史上未出现过的通道 → 追加到数据列序末尾）
    new_cols = [c for c in new_df.columns if c not in set(old_data_cols)]
    n_new_cols = len(new_cols)
    if n_new_cols:
        print(f"  新增 新增披露通道 {n_new_cols} 个（历史上未出现过）")
    target_data_cols = old_data_cols + new_cols

    # 旧表补新通道列（全 NaN），并对齐目标列序 [数据列..., Date, Time]
    for c in new_cols:
        old = old.append_column(c, pa.nulls(n_old_rows, type=pa.float64()))
    if old.column_names != target_data_cols + ['Date', 'Time']:
        old = old.select(target_data_cols + ['Date', 'Time'])

    # 构造新一天的 96 行（列序 = target_data_cols + Date/Time，与 old 一致）
    n_new_rows = len(times)
    new_arrays = []
    for c in target_data_cols:
        col_type = old.schema.field(c).type    # 跟随目标列类型（double/float）
        if c in new_df.columns:                # 新通道（本次提供值）
            vals = new_df[c].values.astype(float)
            arr = pa.array(vals, type=col_type)
        else:                                  # 旧通道（本日 NaN）
            arr = pa.nulls(n_new_rows, type=col_type)
        new_arrays.append(arr)
    new_tbl = pa.table(dict(zip(target_data_cols, new_arrays)))
    new_tbl = new_tbl.append_column('Date', pa.array([date] * n_new_rows))
    new_tbl = new_tbl.append_column('Time', pa.array(times))

    # 合并（concat_tables 只做拼接，行数增量 = 96，内存增量小）
    merged = pa.concat_tables([old, new_tbl])

    # 排序：按 (Date, Time)。直接用 numpy lexsort 对两列数组排序（避免
    #   to_pandas 把 Date/Time 还原成 MultiIndex 索引的干扰）。返回的
    #   order 即 argsort 语义的原位置序列，传给 pyarrow take 重排全表。
    d_arr = merged.column('Date').to_pandas().astype(str).values
    t_arr = merged.column('Time').to_pandas().astype(str).values
    order = np.lexsort((t_arr, d_arr))              # 主键 Date，次键 Time
    merged = merged.take(pa.array(order.astype('int64')))
    del d_arr, t_arr, order

    return merged, n_new_cols, n_new_dates, n_upd_dates


def ingest_disclosure(date, rows, dry_run):
    """把披露数据并入 披露预测数据.feather + 逐通道矩阵。"""
    times = time_index_15min(date)
    new_idx = pd.MultiIndex.from_arrays([[date] * 96, times], names=['Date', 'Time'])
    new_df = pd.DataFrame(rows, index=new_idx).astype(np.float32)

    # ---- 1) 合并到 披露预测数据.feather（清洗宽表，pyarrow 列式） ----
    raw_path = cfg.DISCLOSURE_RAW
    if os.path.exists(raw_path):
        merged, n_new_cols, n_new_dates, n_upd_dates = _merge_wide_arrow(
            raw_path, date, times, new_df)
        if not dry_run:
            pyf.write_feather(merged, raw_path)
        print(f"  披露宽表: {merged.num_rows} 行 × {merged.num_columns} 列"
              f"（新增日期 {n_new_dates} / 更新 {n_upd_dates} / 新通道 {n_new_cols}）")
    else:
        print(f"  警告: 未找到 {raw_path}，跳过宽表更新（仅更新逐通道矩阵）")
        merged = new_df

    # ---- 2) 追加到逐通道矩阵（date×96） ----
    mx = cfg.DISCLOSURE_MATRIX
    os.makedirs(mx, exist_ok=True)
    n_new = n_upd = 0
    for colname in new_df.columns:
        path = os.path.join(mx, f"{colname}.feather")
        col_series = new_df[colname]
        if os.path.exists(path):
            f = pd.read_feather(path)
            f.index = f.index.astype(str)
            if date in f.index:
                f.loc[date] = col_series.values
                n_upd += 1
            else:
                f.loc[date] = col_series.values
                f = f.sort_index()
                n_new += 1
        else:
            # 新通道：date×96，列用时间点（与既有矩阵文件结构一致）
            f = pd.DataFrame(col_series.values.reshape(1, 96),
                             index=[date], columns=times)
            n_new += 1
        f.columns = [str(c) for c in f.columns]
        if not dry_run:
            f.to_feather(path)
    print(f"  逐通道矩阵: 新增 {n_new} 个通道/日期, 更新 {n_upd} 个")

    return list(new_df.columns)


# ════════════════════════════════════════════════════════════
# 实际结果 xlsx 解析
# ════════════════════════════════════════════════════════════
def _has_actual_items(path):
    """检查 xlsx 是否含 4 个结算价数据项（日前/实时统一结算价、日前成交电量、实际用电量）。
    用于拦截被误分类为 actual 的文件（如 '信息披露查询实际信息' 只有备用等，缺结算价）。

    2026-08-13 修复：实际运行结果 xlsx 的『数据项』列名就叫 '数据项'，但不一定在第 0 列
    （正式版第 0 列是企业名称、第 1 列才是数据项）。改为按列名定位，不再假设第 0 列。"""
    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return False
    for sn in xl.sheet_names:
        try:
            sh = xl.parse(sn, nrows=30)
        except Exception:
            continue
        if '数据项' not in sh.columns:
            continue
        vals = set(str(v) for v in sh['数据项'].dropna().tolist())
        if cfg.ACTUAL_ITEM_MAP['日前统一结算价'] in vals:
            return True
    return False


def ingest_actual(date, dry_run):
    """实际结果 → 追加到 实际运行结果-用电侧 矩阵 (4个文件) + 重建 spread_label + sp_wow。"""
    # 找到对应日期的实际结果文件
    cand = [f for f in os.listdir(cfg.UPLOAD_DIR) if '实际运行结果' in f and extract_date(f) == date]
    if not cand:
        raise FileNotFoundError(f"未找到 {date} 的实际结果 xlsx")
    path = os.path.join(cfg.UPLOAD_DIR, cand[0])
    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[0]
    sh = xl.parse(sheet)
    tc = [c for c in sh.columns if str(c).count(':') == 1]
    # 按列名定位数据项（2026-08-13 修复：不再假设第 0 列；同一数据项多企业取均值）
    if '数据项' not in sh.columns:
        raise ValueError(f"文件缺『数据项』列: {os.path.basename(path)}")
    item_rows = {}
    for item, _fname in cfg.ACTUAL_ITEM_MAP.items():
        sub = sh[sh['数据项'].astype(str) == item][tc]
        if not len(sub):
            continue
        vals = (sub.apply(pd.to_numeric, errors='coerce')
                .mean(axis=0).values.astype(np.float32))
        item_rows[item] = vals

    mx = cfg.ACTUAL_MATRIX
    os.makedirs(mx, exist_ok=True)
    for item, fname in cfg.ACTUAL_ITEM_MAP.items():
        if item not in item_rows:
            print(f"  警告: 数据项缺失: {item}"); continue
        vals = item_rows[item]
        row = pd.DataFrame([vals], index=[date], columns=[str(c) for c in tc])
        fpath = os.path.join(mx, f"{fname}.feather")
        if os.path.exists(fpath):
            f = pd.read_feather(fpath); f.index = f.index.astype(str)
            if date in f.index:
                f.loc[date] = vals
            else:
                f.loc[date] = vals
                f = f.sort_index()
        else:
            f = row
        if not dry_run:
            f.to_feather(fpath)
        print(f"  {fname}: 追加/更新 {date}")

    # 同步宽表 实际运行结果-用电侧.feather（Date×Time 索引, 列为 4 数据项）
    sync_actual_wide(dry_run)

    # 重建 spread_label（DA - RT）
    rebuild_spread_label(dry_run)
    # 重建 sp_wow
    if cfg.REBUILD_SP_WOW:
        rebuild_sp_wow(dry_run)


def sync_actual_wide(dry_run):
    """从逐通道矩阵重建 实际运行结果-用电侧.feather 宽表。

    宽表结构: MultiIndex(Date, Time) × 4 列 (日前统一结算价/实时统一结算价/日前成交电量/实际用电量)。
    逐通道矩阵是事实来源（含 96/24 点），宽表始终与矩阵一致，避免两者日期脱节。"""
    raw_path = cfg.ACTUAL_RAW
    items = list(cfg.ACTUAL_ITEM_MAP.values())
    mx = cfg.ACTUAL_MATRIX
    # 用第一个矩阵确定时间点模板
    dfs = {}
    for fname in items:
        p = os.path.join(mx, f"{fname}.feather")
        if not os.path.exists(p):
            continue
        df = pd.read_feather(p)
        df.index = df.index.astype(str)
        dfs[fname] = df
    if not dfs:
        print("  警告: 实际矩阵为空，跳过宽表同步")
        return
    # 时间列取矩阵列名（形如 00:00..23:00 或 00:00..23:45）
    time_cols = [str(c) for c in next(iter(dfs.values())).columns]
    # 需要转成 (Date, Time) 长格式
    parts = []
    for fname, df in dfs.items():
        s = df.stack().rename(fname)
        parts.append(s)
    wide = pd.concat(parts, axis=1)
    wide.index = wide.index.rename(['Date', 'Time'])
    wide = wide.sort_index()
    if not dry_run:
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        wide.to_feather(raw_path)
    print(f"  实际宽表: {len(wide)} 行 (日期 {wide.index.get_level_values(0).min()} ~ "
          f"{wide.index.get_level_values(0).max()})")


def rebuild_spread_label(dry_run):
    """spread_label.feather = 日前统一结算价 - 实时统一结算价 (date×24)"""
    da_p = os.path.join(cfg.ACTUAL_MATRIX, '日前统一结算价.feather')
    rt_p = os.path.join(cfg.ACTUAL_MATRIX, '实时统一结算价.feather')
    da = pd.read_feather(da_p); rt = pd.read_feather(rt_p)
    da.index = da.index.astype(str); rt.index = rt.index.astype(str)
    spread = (da - rt).astype(np.float32)
    spread.columns = [str(c) for c in spread.columns]
    os.makedirs(cfg.LABEL_DIR, exist_ok=True)
    if not dry_run:
        spread.to_feather(os.path.join(cfg.LABEL_DIR, 'spread_label.feather'))
    print(f"  spread_label: {spread.shape[0]}天×{spread.shape[1]}h")


def rebuild_sp_wow(dry_run):
    """sp_wow_abs = spread.shift(7d)-spread.shift(14d);  rate 为相对变化（21_features 同款）"""
    sp = pd.read_feather(os.path.join(cfg.LABEL_DIR, 'spread_label.feather'))
    EPS = 1e-6
    sp_wow_abs = sp.shift(7) - sp.shift(14)
    sp_wow_rate = (sp.shift(7) - sp.shift(14)) / (sp.shift(14).abs() + EPS)
    os.makedirs(cfg.FACTOR_DIR, exist_ok=True)
    for name, df in [('sp_wow_abs', sp_wow_abs), ('sp_wow_rate', sp_wow_rate)]:
        df.index = df.index.astype(str)
        df.columns = [str(c) for c in df.columns]
        if not dry_run:
            df.to_feather(os.path.join(cfg.FACTOR_DIR, f'{name}.fea'))
    print("  sp_wow_abs / sp_wow_rate 已更新")


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='导入每日新 xlsx')
    ap.add_argument('--dry-run', action='store_true', help='只检查不写入')
    ap.add_argument('--force-date', type=str, default=None,
                    help='强制只处理指定日期 (YYYY-MM-DD)，用于补数据/验证')
    args = ap.parse_args()

    cfg.validate()
    os.makedirs(cfg.UPLOAD_DIR, exist_ok=True)
    man = load_manifest()

    files = sorted(os.listdir(cfg.UPLOAD_DIR))
    xlsx_files = [f for f in files if f.endswith('.xlsx')]
    print(f"upload 目录发现 {len(xlsx_files)} 个 xlsx")

    any_new = False
    for filename in xlsx_files:
        ftype = classify_file(filename)
        date = extract_date(filename)
        if not ftype or not date:
            print(f"  [跳过] 无法识别: {filename}")
            continue
        if args.force_date and date != args.force_date:
            continue

        if ftype == 'disclosure':
            key = f"disclosure:{date}"
            if key in man['disclosure'] and not args.force_date:
                print(f"  [跳过] 已导入过: {filename}")
                continue
            print(f"▶ 处理披露 {date}: {filename}")
            rows = parse_disclosure_xlsx(os.path.join(cfg.UPLOAD_DIR, filename), date)
            if not rows:
                # 空解析 = 该文件没有可用的预测 sheet（可能被误判）→ 跳过, 不写 NaN
                print(f"  警告: 跳过 {filename}: 未解析出任何预测通道（文件可能不是披露预测数据）")
                continue
            cols = ingest_disclosure(date, rows, args.dry_run)
            if not args.dry_run and key not in man['disclosure']:
                man['disclosure'].append(key)
            any_new = True

        elif ftype == 'actual':
            key = f"actual:{date}"
            if key in man['actual'] and not args.force_date:
                print(f"  [跳过] 已导入过: {filename}")
                continue
            print(f"▶ 处理实际结果 {date}: {filename}")
            if not _has_actual_items(os.path.join(cfg.UPLOAD_DIR, filename)):
                # 缺 4 个结算价数据项 → 不是标准实际结果文件 → 跳过
                print(f"  警告: 跳过 {filename}: 缺少日前/实时统一结算价等数据项（可能不是实际结果文件）")
                continue
            ingest_actual(date, args.dry_run)
            if not args.dry_run and key not in man['actual']:
                man['actual'].append(key)
            any_new = True

    if not args.dry_run:
        save_manifest(man)

    print("\n完成。")
    if args.dry_run:
        print("（--dry-run 模式，未写入任何文件）")
    elif not any_new:
        print("没有新文件需要处理（已在 manifest 中）。")


if __name__ == '__main__':
    main()
