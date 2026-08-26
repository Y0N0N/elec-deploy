# ============================================================
# deploy/v9_wrappers.py — v9 模型预测包装类 (独立模块, 供 joblib 序列化/反序列化)
#
# 为什么独立成模块:
#   v9 模型保存为 joblib, 包装类若定义在训练脚本 (__main__) 里,
#   deploy/其它进程反序列化时会报 "Can't get attribute 'BoosterDir'".
#   独立模块保证 pickle 能用稳定路径 `v9_wrappers.BoosterDir` 解析。
#
# v9 结构 (交接文档 §4):
#   clf = 方向头 (3类: 0=负偏差 / 1=中性 / 2=正偏差, 在 ±τ 处切, 成本矩阵 loss)
#   reg = 量级头 (回归, 预测 |实际价差| 元/MWh, 供触发/置信度/仓位)
#   规则层 (在 2B 侧): 出手 = 量级头触发(|reg|≥τ) 且 小时先验方向明确; 方向 = 小时先验
#
# 调用契约:
#   model['clf'].predict(X_df)       → 方向码 {0,1,2}
#   model['clf'].predict_proba(X_df) → (n,3) 概率 (置信度用)
#   model['reg'].predict(X_df)       → 预测量级 (元/MWh, ≥0)
#   X_df 为 (date,hour) 特征 DataFrame, NaN 由 XGBoost 原生处理
# ============================================================
import numpy as np
import xgboost as xgb


def dir_hit_metric(y_true, y_score, sample_weight=None):
    """自定义 eval：带方向的命中率（阈值感知，只统计真值非中性），higher=better。

    定义在本独立模块（而非训练脚本 __main__），保证 joblib 反序列化时
    能用稳定路径 v9_wrappers.dir_hit_metric 解析。sklearn 回调可能带
    sample_weight 关键字，签名兼容。"""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score)
    pred_c = y_score.argmax(1) if y_score.ndim > 1 else np.asarray(y_score, dtype=int)
    non = y_true != 1
    if non.sum() == 0:
        return 0.0
    return float(np.mean(pred_c[non] == y_true[non]))


class BoosterDir:
    """包装分类 Booster (multi:softprob + 成本矩阵 fobj) → 方向码 {0,1,2} + 概率"""

    def __init__(self, booster):
        self._bst = booster

    def _proba(self, X_df):
        pr = self._bst.predict(xgb.DMatrix(X_df))
        if pr.ndim == 1:
            pr = pr.reshape(1, -1)
        p = np.exp(pr - pr.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X_df):
        return self.predict_proba(X_df).argmax(axis=1)   # 0=负 / 1=中性 / 2=正

    def predict_proba(self, X_df):
        return self._proba(X_df)


class BoosterMag:
    """包装回归 Booster → 预测量级 (|价差| 元/MWh)"""

    def __init__(self, booster, scale=1.0):
        self._bst = booster
        self._scale = scale

    def predict(self, X_df):
        return self._bst.predict(xgb.DMatrix(X_df))
