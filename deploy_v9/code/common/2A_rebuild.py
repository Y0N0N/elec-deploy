#!/usr/bin/env python
# ============================================================
# deploy/2A_rebuild.py — 步骤 2A：重建（重算因子 + 重训模型）
#
# 作用：
#   1) 重算因子（供给侧 s_* / 日内 h_* / 电网约束 gc_* / 事件 ev_* / 价差状态 sp_wow）
#   2) 用最新数据重训模型，保存为带时间戳的模型文件
#       如 models/xgb_v7_20260811_0930.joblib
#   3) 更新"最新模型指针"，供 2B_inference.py 推理使用
#
# 因子模式（0_config.py 六）：
#   FACTOR_REBUILD_MODE = 'full'  → 全量重算所有因子
#   FACTOR_REBUILD_MODE = 'tail'  → 只重算最近 TAIL_DAYS 天，与既有因子库合并
#
# 用法：
#   python 2A_rebuild.py                # 重建因子 + 重训模型
#   python 2A_rebuild.py --factors      # 只重算因子，不重训模型
#   python 2A_rebuild.py --model        # 只重训模型，不重算因子
# ============================================================
import argparse
import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(_HERE)
while not os.path.exists(os.path.join(_ROOT, '_cfg.py')):
    _ROOT = os.path.dirname(_ROOT)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from _cfg import cfg, latest_disclosure_date

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, mean_squared_error)


# ════════════════════════════════════════════════════════════
# 一、因子重建
# ════════════════════════════════════════════════════════════
def get_latest_disclosure_date():
    """披露矩阵最新日期（优先取 披露预测数据.feather 宽表，秒级；
    旧实现扫 sorted()[:30] 会漏掉按字母排序靠后的供给通道，导致因子只算到旧日期）。"""
    latest = latest_disclosure_date()
    if latest is not None:
        return latest
    raise RuntimeError("披露矩阵为空，无法确定日期范围")


def rebuild_factors():
    import _factors_supply

    mode = cfg.FACTOR_REBUILD_MODE
    latest = get_latest_disclosure_date()
    print("=" * 64)
    print(f"  [因子重建] 模式={mode} | 最新披露日={latest}")
    print("=" * 64)

    # 尾部窗口日期（tail 模式下 s_* 只写这些天）
    tail_dates = None
    env_start = '2025-01-01'
    if mode == 'tail':
        start = (pd.Timestamp(latest) - pd.Timedelta(days=cfg.TAIL_DAYS - 1)).strftime('%Y-%m-%d')
        tail_dates = [(pd.Timestamp(start) + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                      for i in range((pd.Timestamp(latest) - pd.Timestamp(start)).days + 1)]
        env_start = start
        print(f"  尾部窗口: {start} ~ {latest} ({len(tail_dates)} 天)")

    env = dict(os.environ)
    env['DEPLOY_FACTOR_START'] = env_start
    env['DEPLOY_FACTOR_END'] = latest

    # 1) 供给侧 s_* + is_peak/is_valley（移植版，已验证与既有 .fea 一致）
    print("\n[1/5] 供给侧因子 (s_*) ...")
    _factors_supply.build_supply_factors(tail_dates=tail_dates)

    # 2) 日内因子 h_* + 时间特征（动态日期，直接跑）
    print("\n[2/5] 日内因子 (h_*) ...")
    _run_py('50_tool_intraday_factors_v3.py', env)

    # 3) 电网约束 gc_*（日期范围由环境变量扩展到最新披露日）
    print("\n[3/5] 电网约束因子 (gc_*) ...")
    _run_py('50_tool_grid_constraint_factors_v1.py', env)

    # 4) 事件因子 ev_*
    print("\n[4/5] 事件因子 (ev_*) ...")
    _run_py('51_tool_event_factors_v1.py', env)

    # 5) 价差状态 sp_wow（依赖最新实际结果 → spread_label）
    print("\n[5/5] 价差状态因子 (sp_wow) ...")
    if cfg.REBUILD_SP_WOW:
        _rebuild_sp_wow()
    else:
        print("  （REBUILD_SP_WOW=False，跳过）")

    # 6) v9 新增特征（hour_00~23 one-hot + sp_regime_mean7/abs7）。
    #    幂等、廉价，总是构建——这些 .fea 是 v9 训练/推理的必需输入，
    #    缺失时 load_X 会静默跳过，导致重训出的 v9 模型丢核心特征。
    print("\n[6/6] v9 新增特征（hour one-hot + 近期价差 regime）...")
    import subprocess
    v9_add = os.path.join(_HERE, '..', 'v9', 'v9_add_features.py')
    r = subprocess.run([sys.executable, v9_add], cwd=_ROOT)
    if r.returncode != 0:
        print(f"  警告: v9_add_features.py 运行返回非零 ({r.returncode})")

    # 记录本次因子重建日期到 manifest（BUG-6 修复：rebuilt_factors_dates 此前是死字段，
    # 只有 1_ingest 初始化空数组，从未被写入。重建成功后追加最新披露日，供审计/排查用）
    try:
        man_path = cfg.MANIFEST_FILE
        if os.path.exists(man_path):
            man = json.load(open(man_path, encoding='utf-8'))
        else:
            man = {'disclosure': [], 'actual': [], 'rebuilt_factors_dates': []}
        dates = man.setdefault('rebuilt_factors_dates', [])
        if latest not in dates:
            dates.append(latest)
            dates.sort()
        with open(man_path, 'w', encoding='utf-8') as f:
            json.dump(man, f, ensure_ascii=False, indent=2)
        print(f"  manifest 已记录因子重建日期: {latest}")
    except Exception as e:
        print(f"  警告: 更新 manifest.rebuilt_factors_dates 失败: {e}")

    print("\n完成 因子重建完成。")


def _run_py(script, env):
    """运行既有生成脚本（子进程，保持原逻辑）。脚本与 2A 同目录（deploy/），
    因为移植版已 _cfg 化、不再依赖项目根。"""
    path = os.path.join(_HERE, script)
    if not os.path.exists(path):
        print(f"  警告: 未找到 {path}，跳过")
        return
    r = subprocess.run([sys.executable, path], env=env,
                       cwd=_HERE)
    if r.returncode != 0:
        print(f"  警告: {script} 运行返回非零 ({r.returncode})")


def _rebuild_sp_wow():
    """sp_wow_abs/rate = spread 的 d-7/d-14 变化（21_features 同款）"""
    sp_p = os.path.join(cfg.LABEL_DIR, 'spread_label.feather')
    if not os.path.exists(sp_p):
        print("  警告: 无 spread_label.feather，请先运行 1_ingest_xlsx.py 导入实际结果")
        return
    sp = pd.read_feather(sp_p)
    EPS = 1e-6
    sp_wow_abs = sp.shift(7) - sp.shift(14)
    sp_wow_rate = (sp.shift(7) - sp.shift(14)) / (sp.shift(14).abs() + EPS)
    os.makedirs(cfg.FACTOR_DIR, exist_ok=True)
    for name, df in [('sp_wow_abs', sp_wow_abs), ('sp_wow_rate', sp_wow_rate)]:
        df.index = df.index.astype(str)
        df.columns = [str(c) for c in df.columns]
        df.to_feather(os.path.join(cfg.FACTOR_DIR, f'{name}.fea'))
    print(f"  sp_wow_abs / sp_wow_rate 已更新 ({sp.shape[0]} 天)")


# ════════════════════════════════════════════════════════════
# 二、模型重训
# ════════════════════════════════════════════════════════════
def get_feature_list(model_key):
    """从既有模型 joblib 取特征清单（唯一来源，避免读 the local data directory 下的文件）。
    首次无模型时，回退到 deploy/{model_key}_features.json（由 v7_features.json 提供引导）。"""
    import joblib
    path = _find_latest_model(model_key)
    if path:
        m = joblib.load(path)
        if 'features' in m:
            return m['features'], path
    # 回退：特征清单 JSON（首次冷启动用；v7_features.json 等引导文件在部署根）
    feat_json = os.path.join(_ROOT, f'{model_key}_features.json')
    if os.path.exists(feat_json):
        try:
            fl = json.load(open(feat_json)).get('features')
            if fl:
                return list(fl), None
        except Exception:
            pass
    raise FileNotFoundError(
        f"无法确定 {model_key} 的特征清单：没有任何既有模型，也缺少 {model_key}_features.json。\n"
        f"请先放置一个 {model_key} 模型（joblib 自含特征清单）或补上特征 JSON 再运行重建。")


def _find_latest_model(model_key):
    """按同 key 前缀的最新模型解析路径（0_config 统一实现，避免读可能悬空的指针文件）。"""
    return cfg.resolve_latest_model(model_key)


def _to_class5(spr, t1, t2):
    spr = np.asarray(spr, dtype=float)
    return np.where(spr < -t2, 0,
           np.where(spr < -t1, 1,
           np.where(spr <= t1, 2,
           np.where(spr <= t2, 3, 4))))


def evaluate_valid(y_va, yc_va, yc_va_pred, yv, t1, t2):
    """valid 集完整指标（对齐 Yonon/31_model_v8.1_penalty_run.py evaluate_nn 口径）。

    输入:
      y_va        真实价差值 (array)
      yc_va       真实 5 类码 (array)
      yc_va_pred  预测 5 类码 (array)
      yv          回归头预测值 (array)
      t1, t2      τ_minor / τ_big 阈值
    返回 dict: acc / sign_hit / big_* / trigger_rate / 数值驱动分级指标 / rmse 等。
    关键口径: sign_hit 用「阈值感知方向」 sign(y)*(|y|>t1)，只统计真值非中性样本，
              与部署侧 value_driven 语义一致（避免旧口径把小幅中性误判为方向命中）。
    """
    ys = np.asarray(y_va, dtype=float)
    yc_true = np.asarray(yc_va, dtype=int)
    yc_pred = np.asarray(yc_va_pred, dtype=int)
    yv = np.asarray(yv, dtype=float)
    neu = cfg.SPREAD_CLASS_MAP['neu']   # 2

    acc = accuracy_score(yc_true, yc_pred)
    # ── 方向命中（阈值感知，只统计真值非中性）──
    nonneu = yc_true != neu
    true_dir = np.sign(ys) * (np.abs(ys) > t1)
    pred_dir = np.sign(yc_pred - neu)
    sign_hit = (pred_dir[nonneu] == true_dir[nonneu]).mean() if nonneu.sum() else np.nan
    # ── 分类驱动：大偏差类 (big_neg=0 / big_pos=4) ──
    yt_big = ((yc_true == 0) | (yc_true == 4)).astype(int)
    yp_big = ((yc_pred == 0) | (yc_pred == 4)).astype(int)
    big_f1 = f1_score(yt_big, yp_big, zero_division=0)
    big_recall = recall_score(yt_big, yp_big, zero_division=0)
    big_precision = precision_score(yt_big, yp_big, zero_division=0)
    trigger_rate = (yc_pred != neu).mean()
    # ── 数值驱动分级（部署侧判定口径: |reg| vs t1/t2）──
    v_abs = np.abs(yv)
    num_tier = np.where(v_abs >= t2, 2, np.where(v_abs >= t1, 1, 0))
    yt_tier = np.where(np.abs(ys) >= t2, 2, np.where(np.abs(ys) >= t1, 1, 0))
    trig_num = num_tier != 0
    num_recall = recall_score((yt_tier != 0).astype(int), trig_num.astype(int), zero_division=0)
    num_precision = precision_score((yt_tier != 0).astype(int), trig_num.astype(int), zero_division=0)
    tier_acc = (num_tier[trig_num] == yt_tier[trig_num]).mean() if trig_num.sum() else np.nan
    small_f1 = f1_score((yt_tier == 1).astype(int), (num_tier == 1).astype(int), zero_division=0)
    big_f1_num = f1_score((yt_tier == 2).astype(int), (num_tier == 2).astype(int), zero_division=0)
    trig_mask = yc_pred != neu
    trig_hit50 = (v_abs[trig_mask] >= t1).mean() if trig_mask.sum() else np.nan
    trigger_rate_num = trig_num.mean()
    # ── 误差 ──
    rmse = float(np.sqrt(mean_squared_error(ys, yv)))
    m_cond = np.abs(ys) > t1
    rmse_cond = float(np.sqrt(mean_squared_error(ys[m_cond], yv[m_cond]))) if m_cond.sum() else np.nan
    return dict(
        acc=float(acc), sign_hit=float(sign_hit),
        big_f1=float(big_f1), big_recall=float(big_recall), big_precision=float(big_precision),
        trigger_rate=float(trigger_rate),
        num_recall=float(num_recall), num_precision=float(num_precision),
        tier_acc=float(tier_acc), small_f1=float(small_f1), big_f1_num=float(big_f1_num),
        trig_hit50=float(trig_hit50), trigger_rate_num=float(trigger_rate_num),
        rmse=rmse, rmse_cond=rmse_cond,
        n=len(ys),
    )


def _rebuild_v9():
    """v9 方向信号模型：走专用训练脚本（方向头+量级头+规则层），
    避免被通用级联训练覆盖成错误形状（v9 需 model_type='v9_direction'）。

    v9 训练脚本在 code/v9/（31_model_v9_direction_run.py），随部署落地。"""
    import subprocess
    v9_script = os.path.join(_HERE, '..', 'v9', '31_model_v9_direction_run.py')
    print(f"  [模型重训] v9 → 专用脚本 {v9_script}")
    if not os.path.exists(v9_script):
        raise FileNotFoundError(f"缺 v9 训练脚本 {v9_script}")
    r = subprocess.run([sys.executable, v9_script, '--valid-days', '60'],
                       cwd=os.path.dirname(v9_script))
    if r.returncode != 0:
        raise RuntimeError(f"v9 训练脚本返回非零 ({r.returncode})")
    return


def _rebuild_v9_1():
    """v9.1 = v9 + ATR 波动率过滤器（震荡市禁开仓）。

    v9.1 训练脚本在 code/v9/（31_model_v9.1_atr_run.py），
    内部复用 31_model_v9_direction_run 的方向头/量级头/小时先验训练，
    再叠加 ATR 过滤器（见 v9_atr.py）。"""
    import subprocess
    script = os.path.join(_HERE, '..', 'v9', '31_model_v9.1_atr_run.py')
    print(f"  [模型重训] v9.1 → 专用脚本 {script}")
    if not os.path.exists(script):
        raise FileNotFoundError(f"缺 v9.1 训练脚本 {script}")
    r = subprocess.run([sys.executable, script, '--valid-days', '60'],
                       cwd=os.path.dirname(script))
    if r.returncode != 0:
        raise RuntimeError(f"v9.1 训练脚本返回非零 ({r.returncode})")
    return


def rebuild_model(model_key):
    """用全部可得标签数据重训级联模型，保存为带时间戳的文件。"""
    if model_key == 'v9':
        return _rebuild_v9()
    if model_key == 'v9.1':
        return _rebuild_v9_1()
    import gc
    import joblib
    import xgboost as xgb

    reg = cfg.MODEL_REGISTRY[model_key]
    print("=" * 64)
    print(f"  [模型重训] {model_key} ({reg['label']})")
    print("=" * 64)

    features, old_path = get_feature_list(model_key)
    # 加上 gas_limit 事件因子（v7 用；若特征清单来自旧模型已含则不再重复）
    # 只收集「干净因子名」（无空格数字后缀，如 'ev_dispatch_active 2.fea' 这类
    # pandas 重复命名残留会被排除），避免把 object 列混进 X 导致 XGBoost 报 dtype 错
    gas_names = sorted([n[:-4] for n in os.listdir(cfg.FACTOR_DIR)
                        if (n.startswith('ev_gas_limit') or n.startswith('ev_burst_gas'))
                        and n.endswith('.fea')
                        and not re.search(r' \d+\.fea$', n)])
    features = list(dict.fromkeys(features + gas_names))
    print(f"  特征: {len(features)}")

    # ---- 加载特征 ----
    fl = []
    for name in features:
        p = os.path.join(cfg.FACTOR_DIR, f'{name}.fea')
        if not os.path.exists(p):
            print(f"  警告: 因子缺失 {name}，置 NaN")
            continue
        df = pd.read_feather(p)
        df.index = df.index.astype(str)
        df.columns = [str(c) for c in df.columns]
        s = df.stack(); s.name = name
        fl.append(s)
    X = pd.concat(fl, axis=1)
    X.index = X.index.rename(['date', 'hour'])
    X = X.sort_index()
    print(f"  X: {X.shape} ({X.index.get_level_values('date').min()} ~ "
          f"{X.index.get_level_values('date').max()})")

    # ---- 加载标签（spread_label，来自矩阵实际结果，不碰 the local data directory） ----
    sp_p = os.path.join(cfg.LABEL_DIR, 'spread_label.feather')
    spread = pd.read_feather(sp_p)
    spread.index = pd.to_datetime(spread.index).strftime('%Y-%m-%d')
    spread.columns = [str(c) for c in spread.columns]
    y = spread.stack(); y.index = y.index.rename(['date', 'hour'])
    y.index = pd.MultiIndex.from_arrays(
        [y.index.get_level_values('date').astype(str), y.index.get_level_values('hour')],
        names=['date', 'hour'])

    xd = X.index.get_level_values('date')
    yd = y.index.get_level_values('date')
    def subset(which):
        common = which.index.intersection(y.index)
        return which.loc[common], y.loc[common]
    X_full, y_full = subset(X)

    # ---- 数据划分（红线：valid 绝不能混入 train） ----
    y_dates = sorted(set(y_full.index.get_level_values('date')))
    valid_dates = set(y_dates[-cfg.TRAIN_VALID_DAYS:])
    tr_m = X_full.index.get_level_values('date').isin([d for d in y_dates if d not in valid_dates])
    va_m = X_full.index.get_level_values('date').isin(valid_dates)
    X_tr, y_tr = X_full.loc[tr_m], y_full.loc[tr_m]
    X_va, y_va = X_full.loc[va_m], y_full.loc[va_m]
    assert not set(X_tr.index.get_level_values('date')) & valid_dates, "红线: valid 混入 train!"
    print(f"  train {len(X_tr)} | valid {len(X_va)} (valid={cfg.TRAIN_VALID_DAYS}天, 已排除)")

    t1, t2 = cfg.SPREAD_THRESHOLD, cfg.SPREAD_THRESHOLD_BIG
    yc_tr = _to_class5(y_tr, t1, t2)
    yc_va = _to_class5(y_va, t1, t2)
    ncls = len(cfg.SPREAD_CLASSES)

    # ---- 训练 ----
    FIXED = cfg.MODEL_FIXED_PARAMS
    cls_counts = np.bincount(yc_tr, minlength=ncls)
    w = len(yc_tr) / (ncls * cls_counts)

    clf = xgb.XGBClassifier(n_estimators=cfg.MODEL_N_ESTIMATORS,
        early_stopping_rounds=cfg.MODEL_EARLY_STOPPING,
        eval_metric='mlogloss', random_state=cfg.MODEL_RANDOM_STATE, verbosity=0, **FIXED)
    clf.fit(X_tr, yc_tr, sample_weight=w[yc_tr], eval_set=[(X_va, yc_va)], verbose=False)
    print(f"  clf best_iteration: {clf.best_iteration}")

    # ---- 回归头：Plan C' 加权回归（防数值塌缩，v8.1 数值驱动触发的前提） ----
    # v8.1 的触发/等级完全由回归值判定：|reg|≥τ_minor 才预警。
    # 若回归头只在 |y|>τ 的样本上训练（条件回归），会数值塌缩到 ±τ 内 → 永远不触发。
    # 正确做法（与 Yonon/31_model_v8.1_penalty_run.py 一致）：全量样本训练，
    #   |y| > τ_minor → 权重 1.0（保大偏差量级）；|y| ≤ τ_minor → 权重 0.2（轻量保留中性结构）。
    # 早停：不用 rmse 早停——rmse 在头几轮就停滞导致 best_iteration≈2，数值塌缩。
    #   直接跑满 n_estimators 轮（实测 300 轮能学到正确量级，|y|≥50 约 51%）。
    W_NEU, W_BIG = 0.2, 1.0
    y_vals_tr = np.abs(y_tr.values)
    y_vals_va = np.abs(y_va.values)
    sw_tr = np.where(y_vals_tr > t1, W_BIG, W_NEU).astype(np.float32)
    sw_va = np.where(y_vals_va > t1, W_BIG, W_NEU).astype(np.float32)
    reg_head = xgb.XGBRegressor(n_estimators=cfg.MODEL_N_ESTIMATORS,
        random_state=cfg.MODEL_RANDOM_STATE, verbosity=0, **FIXED)
    reg_head.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
    print(f"  reg: 全量 {cfg.MODEL_N_ESTIMATORS} 轮（无 rmse 早停，Plan C' 加权: "
          f"|y|>{t1}→{W_BIG}, |y|≤{t1}→{W_NEU}，防数值塌缩）")

    # ---- 诚实评估（valid 已排除出 train，完整指标） ----
    yc_va_pred = clf.predict(X_va)
    yv = reg_head.predict(X_va)
    m_valid = evaluate_valid(y_va.values, yc_va, yc_va_pred, yv, t1, t2)
    sign_hit = m_valid['sign_hit']
    big_f1 = m_valid['big_f1']
    print(f"  Valid 指标: 5类acc={m_valid['acc']:.3f} | 方向命中={sign_hit:.3f} | "
          f"大偏差F1={big_f1:.3f}(R{m_valid['big_recall']:.2f}/P{m_valid['big_precision']:.2f}) "
          f"| 触发率={m_valid['trigger_rate']:.3f}")
    print(f"         数值驱动: 触发R={m_valid['num_recall']:.2f}/P={m_valid['num_precision']:.2f} "
          f"| 分级acc={m_valid['tier_acc']:.3f} | 小F1={m_valid['small_f1']:.3f} "
          f"大F1={m_valid['big_f1_num']:.3f} | 触发中|值|≥τ={m_valid['trig_hit50']:.3f}")
    print(f"         条件RMSE(|y|>{t1})={m_valid['rmse_cond']:.1f} | 全量RMSE={m_valid['rmse']:.1f} "
          f"| n={m_valid['n']}")

    # ---- 保存（时间戳命名 + 更新指针） ----
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    fname = reg['pattern'].format(ts=ts)
    save_path = os.path.join(cfg.MODEL_DIR, fname)
    joblib.dump({
        'clf': clf,
        'reg': reg_head,
        'features': features,
        'threshold_minor': t1,
        'threshold_big': t2,
        'classes': cfg.SPREAD_CLASSES,
        'trigger_rule': cfg.model_trigger_rule(model_key),
        'decision_rule': (
            f"|spread| ≤ {t1} → neu (正常, 不预警); "
            f"{t1} < |spread| ≤ {t2} → neg/pos (有偏差, 预警+输出值); "
            f"|spread| > {t2} → big_neg/big_pos (大偏差, 预警+输出值)"),
        'fixed_params': FIXED,
        'train_end_date': y_dates[-1],
        'data_end_date': X.index.get_level_values('date').max(),
        'n_features': len(features),
        'metrics': {'valid': m_valid},
        'valid_sign_hit': float(sign_hit),
        'valid_big_f1': float(big_f1),
        'trained_at': ts,
    }, save_path)
    print(f"\n  模型已保存: {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

    # 更新指针
    ptr = {}
    if os.path.exists(cfg.LATEST_MODEL_FILE):
        try:
            ptr = json.load(open(cfg.LATEST_MODEL_FILE))
        except Exception:
            pass
    ptr[model_key] = save_path
    json.dump(ptr, open(cfg.LATEST_MODEL_FILE, 'w'), ensure_ascii=False, indent=2)
    print(f"  最新模型指针已更新: {cfg.LATEST_MODEL_FILE}")


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='重建因子 + 重训模型')
    ap.add_argument('--factors', action='store_true', help='只重算因子')
    ap.add_argument('--model', action='store_true', help='只重训模型')
    ap.add_argument('--model-key', type=str, default=cfg.ACTIVE_MODEL,
                    help=f'模型 key (默认 {cfg.ACTIVE_MODEL})')
    args = ap.parse_args()

    cfg.validate()
    do_factors = args.factors or not args.model
    do_model = args.model or not args.factors

    if do_factors:
        rebuild_factors()
    if do_model:
        rebuild_model(args.model_key)

    print("\n完成 2A_rebuild 完成。接下来可运行 2B_inference.py 推理。")


if __name__ == '__main__':
    main()
