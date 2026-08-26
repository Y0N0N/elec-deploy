#!/usr/bin/env python
# ============================================================
# deploy/2B_inference.py — 步骤 2B：推理（用已有模型 + 最新因子 预测）
#
# 作用：
#   用最近一次重建的模型，对"预测窗口"内每一天的 24 小时输出：
#     预警等级 (大负/负/正常/正/大正) + 预测价差 (元/MWh) + 是否预警
#   写入 output/预测_YYYY-MM-DD_YYYY-MM-DD.csv
#
# 预测窗口（默认 D-4 ~ D+1，D = 最新披露日的前一天，见 0_config.py 五）：
#   例：最新披露日 2026-07-27 → D=07-26 → 窗口 [07-22, 07-27]
#
# 用法：
#   python 2B_inference.py
#   python 2B_inference.py --model v7        # 指定模型（默认用 config 的 ACTIVE_MODEL）
#   python 2B_inference.py --dates 2026-07-25 2026-07-26   # 手动指定日期
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
from _cfg import cfg, latest_disclosure_date
from _verify import (CHECK_LABEL, check_flag_for_dates, output_filename,
                     verify_rows)
from v9_atr import build_atr_gate_from_label

import numpy as np
import pandas as pd


LEVEL_MAP = {
    0: '大负偏差', 1: '负偏差', 2: '正常',
    3: '正偏差', 4: '大正偏差',
}


# ════════════════════════════════════════════════════════════
# 模型查找
# ════════════════════════════════════════════════════════════
def find_latest_model(model_key):
    """按同 key 前缀的最新模型解析路径（0_config 统一实现；旧的扫目录逻辑已并入）。"""
    path = cfg.resolve_latest_model(model_key)
    if path:
        return path
    # 兜底：用固定名
    reg = cfg.MODEL_REGISTRY.get(model_key) or {}
    pattern = reg.get('pattern', f'{model_key}.joblib')
    prefix = pattern.split('{ts}')[0]
    fixed = prefix.rstrip('_') + '.joblib'
    if os.path.exists(os.path.join(cfg.MODEL_DIR, fixed)):
        return os.path.join(cfg.MODEL_DIR, fixed)
    raise FileNotFoundError(
        f"未找到模型文件（前缀 {pattern}）。请先运行 2A_rebuild.py 重建模型。")


def _fmt_metric(v, nd=3):
    """指标值格式化：None/NaN → '—'，否则保留 nd 位小数。"""
    if v is None:
        return '—'
    try:
        if np.isnan(float(v)):
            return '—'
    except (TypeError, ValueError):
        return '—'
    return f'{float(v):.{nd}f}'


def _print_model_metrics(model):
    """打印模型 Valid 集指标（控制台）。新模型含 metrics.valid，旧模型仅平铺 2 项。

    v9.1 的指标字段与 v9 不同：顶层存 valid_c_atr_hit/valid_c_atr_trigger，
    metrics.valid = {c_base, c_atr}（无 dir_head 明细）。按版本兼容取数，
    避免 v9.1 下整行打全 nan（BUG-1 修复）。"""
    if model.get('model_type') == 'v9_direction':
        is_v91 = model.get('version') == 'v9.1' or 'valid_c_atr_hit' in model
        if is_v91:
            hit = model.get('valid_c_atr_hit')
            trig = model.get('valid_c_atr_trigger')
            dhit = ((model.get('metrics') or {}).get('valid') or {}) \
                .get('c_atr', {}).get('dir_hit')
        else:
            hit = model.get('valid_c_hit')
            trig = model.get('valid_trigger_rate')
            dhit = model.get('valid_dir_hit')

        def _fmt(v):
            return 'nan' if v is None else f'{float(v):.3f}'
        print(f"  Valid: 方向命中(C规则层)={_fmt(hit)} | 触发率={_fmt(trig)} "
              f"| 方向头命中={_fmt(dhit)}")
    vm = (model.get('metrics') or {}).get('valid')
    if vm:
        keys = ['acc', 'sign_hit', 'big_f1', 'big_recall', 'big_precision',
                'trigger_rate', 'num_recall', 'num_precision', 'tier_acc',
                'small_f1', 'big_f1_num', 'trig_hit50', 'rmse', 'rmse_cond']
        line = '  Valid 指标: ' + ' | '.join(
            f"{k}={_fmt_metric(vm.get(k))}" for k in keys if k in vm)
        print(line)
    elif 'valid_sign_hit' in model:
        print(f"  Valid 指标: sign_hit={model['valid_sign_hit']:.3f} "
              f"| big_f1={model['valid_big_f1']:.3f}（旧模型仅 2 项）")
    # 附带模型时间/阈值信息
    ts = model.get('trained_at')
    if ts:
        print(f"  训练时间: {ts} | τ_minor={model.get('threshold_minor', cfg.SPREAD_THRESHOLD)}"
              f" | τ_big={model.get('threshold_big', cfg.SPREAD_THRESHOLD_BIG)}")


# ════════════════════════════════════════════════════════════
# 特征加载
# ════════════════════════════════════════════════════════════
def load_features(feature_names, dates):
    """从因子库加载 (date,hour) 特征矩阵，只保留指定日期"""
    X_parts = []
    for name in feature_names:
        # 脏因子名兼容：形如 'ev_gas_limit_active 2' 的历史固化特征名
        # （pandas 重复命名残留），因子库里已无此文件。其内容与去掉
        # 空格数字后缀的干净文件完全一致，读取干净文件、保留脏名作列名，
        # 使 X 列数与模型内置特征名一致（否则 XGBoost 报 feature_names
        # mismatch）。正常特征（含 v9/v9.1 全部 404 个）不受影响。
        clean_name = re.sub(r' \d+$', '', str(name))
        p = os.path.join(cfg.FACTOR_DIR, f'{clean_name}.fea')
        if not os.path.exists(p):
            print(f"  警告: 因子缺失: {name}，置 NaN")
            df = pd.DataFrame(index=pd.Index(dates), columns=[f'{h:02d}:00' for h in range(24)])
        else:
            if clean_name != str(name):
                print(f"  兼容脏特征名: {name} → {clean_name}")
            df = pd.read_feather(p)
            df.index = df.index.astype(str)
            df.columns = [str(c) for c in df.columns]
        s = df.loc[[d for d in dates if d in df.index]].stack()
        s.name = name
        X_parts.append(s)
    X = pd.concat(X_parts, axis=1)
    X.index = X.index.rename(['date', 'hour'])
    return X


def resolve_dates(disclosure_matrix_dir, back, fwd, manual=None):
    """计算预测窗口日期列表（升序）。

    参考日 D = 最新披露日 - 1 天（因为披露信息是 D+1 的）。
    窗口 = [D - back, D + fwd]。
    例：最新披露日 2026-07-28 → D=07-27 → back=4/fwd=1 → [07-23, 07-28]。

    最新披露日取 披露预测数据.feather 宽表（每次导入同步维护），
    不再扫矩阵 sorted()[:30]（旧实现会漏掉按字母排序靠后的供给通道，
    导致预测窗口永远停在旧的 22~27）。"""
    if manual:
        return sorted(manual)
    latest = latest_disclosure_date()
    if latest is None:
        raise RuntimeError("无法确定最新披露日期（披露矩阵为空？）")
    ref = pd.Timestamp(latest) - pd.Timedelta(days=1)   # D = 最新披露日前一天
    start = ref - pd.Timedelta(days=back)
    end = ref + pd.Timedelta(days=fwd)
    dates = [(start + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
             for i in range((end - start).days + 1)]
    return dates


# ════════════════════════════════════════════════════════════
# 推理
# ════════════════════════════════════════════════════════════
def predict_spread(model, X, model_key=None):
    """价差模型预测：返回 DataFrame (date,hour,等级,价差,是否预警)。

    按模型自带的 trigger_rule 分派：
      - 'value_driven' (v8.1, 3类 {0,2,4} + 条件回归): 触发/等级由回归值判定
          |reg| <  τ_minor(50)   → 正常, 不预警, 不输出值
          τ_minor ≤ |reg| < τ_big(100) → 小偏差预警, 输出值 (方向取 reg 符号)
          |reg| ≥  τ_big(100)    → 大偏差预警, 输出值 (方向取 reg 符号)
      - 其它 (v7 5类 {0..4}): 原分类驱动, alert = c != neu, 等级/方向由分类决定
    """
    if model.get('model_type') == 'v9_direction':
        return predict_spread_v9(model, X)
    clf = model['clf']; reg = model['reg']
    yc = clf.predict(X)
    yv = reg.predict(X)
    t1 = model.get('threshold_minor', cfg.SPREAD_THRESHOLD)
    t2 = model.get('threshold_big', cfg.SPREAD_THRESHOLD_BIG)
    rule = model.get('trigger_rule') or cfg.model_trigger_rule(model_key)
    rows = []
    for (date, hour), cls, val in zip(X.index, yc, yv):
        v = float(val)
        if rule == 'value_driven':
            # ── v8.1 数值驱动: 触发/等级由 |reg| vs 50/100 判定, 方向取 reg 符号 ──
            av = abs(v)
            if av < t1:
                level, alert = '正常', False
            elif av < t2:
                level = '正偏差' if v >= 0 else '负偏差'
                alert = True
            else:
                level = '大正偏差' if v >= 0 else '大负偏差'
                alert = True
            c = 4 if (v >= 0 and av >= t2) else 3 if (v >= 0 and alert) else \
                0 if (v < 0 and av >= t2) else 1 if (v < 0 and alert) else 2
            rows.append({
                '日期': date, '小时': hour,
                '预警等级': level,
                '预测价差(元/MWh)': round(v, 2) if alert else None,
                '是否预警': '是' if alert else '否',
                'class_code': c,
            })
        else:
            # ── v7 原分类驱动: alert = c != neu, 等级/方向由分类决定 ──
            c = int(cls)
            alert = c != cfg.SPREAD_CLASS_MAP['neu']
            rows.append({
                '日期': date, '小时': hour,
                '预警等级': LEVEL_MAP[c],
                '预测价差(元/MWh)': round(v, 2) if alert else None,
                '是否预警': '是' if alert else '否',
                'class_code': c,
            })
    return pd.DataFrame(rows)


def predict_price(model, X):
    """v5.4/v5.4r 日前价格模型：返回 DataFrame (date,hour,预测价格)"""
    mdl = model['model']
    yv = mdl.predict(X)
    rows = []
    for (date, hour), val in zip(X.index, yv):
        rows.append({'日期': date, '小时': hour,
                     '预测日前价格(元/MWh)': round(float(val), 2)})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════
# v9 方向信号模型推理（交接文档 §4.1 / §5 step 4）
# ════════════════════════════════════════════════════════════
_DIR_LABEL = {0: '负偏差', 1: '中性', 2: '正偏差'}


def _atr_allow_for_dates(model, dates):
    """模型带 atr_filter 时，按 ≤D-1 实际价差重算每日 ATR 门（per-date allow）。

    返回 {date_str: bool}；无 atr_filter 配置（v9 旧模型）返回 None（不过滤）。
    ATR 口径与 v9_atr.build_atr_allow 一致（因果：判定日 D 只用 ≤D-1 数据）。
    """
    atr_cfg = model.get('atr_filter')
    if not atr_cfg:
        return None
    try:
        allow = build_atr_gate_from_label(
            os.path.join(cfg.LABEL_DIR, 'spread_label.feather'),
            n_hours=atr_cfg.get('n_hours', 24),
            baseline_days=atr_cfg.get('baseline_days', 20),
            ratio=atr_cfg.get('ratio', 0.90),
            abs_atr_floor=atr_cfg.get('abs_atr_floor', 25.0))
    except Exception as e:
        print(f"  警告: ATR 门计算失败，本次不过滤: {e}")
        return None
    return {str(d)[:10]: bool(v) for d, v in allow.items()}


def predict_spread_v9(model, X):
    """v9：输出 模型方向(方向头) + 量级(元/MWh) + 置信度 + 小时先验 + 是否出手 + 交易建议。

    规则层（不靠模型方向）：
      出手条件 = 量级头触发(|mag| ≥ τ)  且  小时先验方向明确
               （v9.1 叠加：且 当日 ATR 门放行，见模型 atr_filter 配置）
      交易方向 = 小时先验（负→日前买/实时卖；正→日前卖/实时买）
    """
    clf = model['clf']; reg = model['reg']
    t1 = model.get('threshold_minor', cfg.SPREAD_THRESHOLD)
    mag_raw = np.asarray(reg.predict(X), dtype=float)
    if model.get('mag_transform') == 'log1p':          # 量级头 log1p 目标 → expm1 还原
        mag_raw = np.expm1(mag_raw)
    dir_codes = np.asarray(clf.predict(X), dtype=int)   # 0=负 1=中性 2=正
    conf = np.asarray(clf.predict_proba(X)).max(axis=1)
    hour_prior = model.get('hour_prior', {})
    # v9.1 ATR 门（per-date allow）
    pred_dates = sorted({str(d)[:10] for d in X.index.get_level_values('date')})
    atr_allow = _atr_allow_for_dates(model, pred_dates)
    rows = []
    for (date, hour), mag, dc, c in zip(X.index, mag_raw, dir_codes, conf):
        hh = str(hour)[:2]
        prior = int(hour_prior.get(hh, 0)) if isinstance(hour_prior, dict) else 0
        d = str(date)[:10]
        atr_ok = atr_allow is None or atr_allow.get(d, False)
        triggered = bool(mag >= t1 and prior != 0 and atr_ok)
        if prior < 0:
            prior_s, action = '负', ('日前买/实时卖' if triggered else '不动')
        elif prior > 0:
            prior_s, action = '正', ('日前卖/实时买' if triggered else '不动')
        else:
            prior_s, action = '不明', '不动'
        rows.append({
            '日期': date, '小时': hour,
            '模型方向': _DIR_LABEL.get(int(dc), '?'),
            '量级(元/MWh)': round(float(mag), 1),
            '置信度': round(float(c), 3),
            '小时先验': prior_s,
            '是否出手': '是' if triggered else '否',
            '交易建议': action,
        })
    out = pd.DataFrame(rows)
    return add_actual_verification(out)


def add_actual_verification(out):
    """用真实结算价核对方向 + 套利结果（口径统一在 _verify.verify_rows）。

    只对已有实际结算价的日期填充（实际结果截止日之后为「待实际」），不阻塞推理。
    补齐 实际价差 / 方向核对 / 套利时机 / 套利结果 / 套利盈亏 五列。
    """
    return verify_rows(out, True)


def main():
    ap = argparse.ArgumentParser(description='推理预测')
    ap.add_argument('--model', type=str, default=cfg.ACTIVE_MODEL,
                    help=f'模型 key (默认 {cfg.ACTIVE_MODEL})')
    ap.add_argument('--dates', nargs='*', default=None, help='手动指定预测日期 YYYY-MM-DD ...')
    args = ap.parse_args()

    cfg.validate()
    if args.model not in cfg.MODEL_REGISTRY:
        sys.exit(f"错误: 模型 '{args.model}' 不在 MODEL_REGISTRY，请检查 0_config.py")
    reg = cfg.MODEL_REGISTRY[args.model]

    model_path = find_latest_model(args.model)
    print(f"模型: {args.model} ({reg['label']})")
    print(f"  路径: {model_path}")

    import joblib
    model = joblib.load(model_path)
    features = model['features']
    print(f"  特征数: {len(features)}")
    # 模型 Valid 指标（控制台展示；旧模型仅 2 项降级）
    _print_model_metrics(model)

    dates = resolve_dates(cfg.DISCLOSURE_MATRIX, cfg.PREDICT_BACK_DAYS,
                          cfg.PREDICT_FWD_DAYS, args.dates)
    print(f"预测窗口: {dates[0]} ~ {dates[-1]} (共 {len(dates)} 天)")

    X = load_features(features, dates)
    # 报告缺日
    have_dates = set(X.index.get_level_values('date'))
    for d in dates:
        if d not in have_dates:
            print(f"  警告: {d} 的因子数据缺失，将不输出该天")
    X = X.loc[[d for d in dates if d in X.index]]

    is_v9 = model.get('model_type') == 'v9_direction'
    if reg['predict_mode'] == 'spread':
        out = predict_spread(model, X, args.model)
    else:
        out = predict_price(model, X)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # 新命名带 模型 + 核对标记（a/b/c）：预测_<model>_<start>_<end>_<flag>.csv
    flag = check_flag_for_dates(dates)
    out_path = os.path.join(cfg.OUTPUT_DIR,
                            output_filename(args.model, dates[0], dates[-1], flag))
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n已保存: {out_path}  ({len(out)} 行)  [核对标记 {CHECK_LABEL[flag]}]")

    # 控制台汇总
    print("\n预测汇总（按天）:")
    for d in dates:
        if d not in have_dates:
            continue
        sub = out[out['日期'] == d]
        if is_v9:
            n_act = (sub['是否出手'] == '是').sum()
            print(f"  {d}: {len(sub)}h | 出手 {n_act}h")
            for _, r in sub.iterrows():
                if r['是否出手'] == '是':
                    print(f"     {r['小时']} 方向头{r['模型方向']:3s} 量级{r['量级(元/MWh)']:>6.1f} "
                          f"置信{r['置信度']:.2f} 先验{r['小时先验']:2s} → {r['交易建议']}")
        elif reg['predict_mode'] == 'spread':
            n_alert = (sub['是否预警'] == '是').sum()
            print(f"  {d}: {len(sub)}h | 预警 {n_alert}h")
            for _, r in sub.iterrows():
                v = r['预测价差(元/MWh)']
                val = f"{v:+.1f}" if (v is not None and not pd.isna(v)) else "—"
                print(f"     {r['小时']} {r['预警等级']:10s} {val:>8s} {r['是否预警']}")
        else:
            print(f"  {d}: 预测日前价格前3h {sub['预测日前价格(元/MWh)'].iloc[:3].tolist()}")


if __name__ == '__main__':
    main()
