# ============================================================
# Yonon/v9_atr.py — v9.1 ATR 波动率过滤器（规则层）+ 硬风险规则
#
# 需求（用户 2026-08-19，最紧急）：触发交易前计算过去 N 小时（默认 24h）的 ATR。
#   规则：若当前 ATR < 过去 20 日 ATR 均值的 90%（判定为震荡市）→ 禁止开仓。
#   目的：避开 OOS 首周那种 7 笔中性 + 1 笔押反的震荡行情。
#
# 口径（因果，无前视）：
#   每小时真实波幅 TR(t) ≈ |spread(t) − spread(t−1)|（小时单值序列的波幅近似）
#   ATR_24(t)   = 过去 n_hours 小时 TR 的均值（rolling）
#   baseline(t) = 过去 baseline_days 天的 ATR_24 均值（rolling, baseline_days×24 小时）
#   日 D 的判定用「截至 D-1 末」的 ATR_24 与 baseline（shift 24 小时），
#   因为交易发生在 D 之前，当时只知道 ≤D-1 的实际价差：
#       allow(D) = (ATR_24 ≥ ratio × baseline)  且  (ATR_24 ≥ abs_atr_floor)
#   - ratio 为相对比例（默认 0.90，2026-08-19 用户拍板，原 0.80 不够拦 07-25 押反）
#   - abs_atr_floor 为「绝对 ATR 下限」硬规则：即使相对比例满足，ATR 低于绝对下限仍禁开仓
#     （治"持续震荡期基线也降、相对比例自我放松"的漏洞）
#   allow 为 per-date 布尔；数据不足（滚动窗未满）时保守返回 False（禁交易）。
#
# 另提供月累计亏损硬停：monthly_loss_hard_stop_mask() — 当月累计亏损达阈值即停开仓至月末。
# ============================================================
import numpy as np
import pandas as pd


def build_atr_allow(spread_df, n_hours=24, baseline_days=20, ratio=0.90,
                    abs_atr_floor=25.0):
    """按实际价差宽表构造逐日 allow 过滤器（相对比例 + 绝对 ATR 下限）。

    参数:
      spread_df      date×24 实际价差宽表（索引为可排序日期，列小时）
      n_hours        当前 ATR 的窗口小时数（默认 24）
      baseline_days  基线窗口天数（默认 20）
      ratio          相对阈值比例（默认 0.90）
      abs_atr_floor  绝对 ATR 下限（元/MWh，默认 25.0）；ATR 低于此值即使比例满足也禁开仓
    返回:
      Series[date] → bool，True=允许交易（该日非震荡市且波动不低于绝对下限）
    """
    sp = spread_df.sort_index()
    cols = [str(c) for c in sp.columns]
    s = sp.stack().astype(float)                      # (date,hour)
    s.index = s.index.set_names(['date', 'hour'])
    s = s.sort_index()

    tr = s.diff().abs()                               # |Δspread|，首小时 NaN
    atr = tr.rolling(n_hours, min_periods=n_hours).mean()
    baseline = atr.rolling(baseline_days * 24, min_periods=baseline_days * 24).mean()

    # 因果化：判定日 D 只用 ≤ D-1 的数据 → 平移 24 小时（1 天）
    atr_prev = atr.shift(24)
    base_prev = baseline.shift(24)

    df = pd.DataFrame({'atr': atr_prev, 'base': base_prev})
    # 取每天 23:00 的值作为当日判定（即截至 D-1 23:00 的窗口）
    last = df.groupby(level=0).last()
    allow = last['atr'] >= ratio * last['base']
    if abs_atr_floor is not None:
        allow &= (last['atr'] >= abs_atr_floor)
    return allow.fillna(False)


def monthly_loss_hard_stop_mask(sel, pnl, dates, max_monthly_loss=-5000.0):
    """月累计亏损硬停：按自然月累计实际成交盈亏，一旦当月累计 < max_monthly_loss
    （即当月已亏达 5000 元/MWh），当月剩余候选小时全部禁开仓（硬停，不回撤）。

    参数:
      sel              候选出手掩码 (bool array，未出手 False)
      pnl              逐小时"若成交"的盈亏 (float array；只对 sel=True 有意义)
      dates            对齐的日期字符串 array (YYYY-MM-DD，须按时间升序)
      max_monthly_loss 硬停阈值（默认 -5000）
    返回:
      exec_mask  (bool array) True=该笔实际成交；已在 sel 基础上叠加硬停
    """
    sel = np.asarray(sel, dtype=bool)
    pnl = np.asarray(pnl, dtype=float)
    exec_mask = np.zeros(len(sel), dtype=bool)
    month_pl = 0.0
    cur_month = None
    for i in range(len(sel)):
        m = str(dates[i])[:7]
        if cur_month is not None and m != cur_month:
            month_pl = 0.0          # 跨月重置
        cur_month = m
        if not sel[i]:
            continue
        if month_pl < max_monthly_loss:
            continue                # 硬停：当月剩余禁开仓
        exec_mask[i] = True
        month_pl += float(pnl[i])
    return exec_mask


def build_atr_gate_from_label(label_path, **kw):
    """从 spread_label.feather 直接构建过滤器（便捷入口）。"""
    sp = pd.read_feather(label_path)
    sp.index = pd.to_datetime(sp.index)
    sp = sp[~sp.index.duplicated(keep='last')]
    return build_atr_allow(sp, **kw)
