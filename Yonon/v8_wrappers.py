# ============================================================
# v8_wrappers.py — 模型预测包装类 (独立模块, 供 joblib 序列化/反序列化)
#
# 为什么独立成模块:
#   v8 模型保存为 joblib, 包装类若定义在训练脚本 (__main__) 里,
#   deploy/其它进程反序列化时会报 "Can't get attribute '_BoosterClf'".
#   独立模块保证 pickle 能用稳定路径 `v8_wrappers.BoosterClf` 解析。
#
# 调用契约 (与 deploy/2B_inference.py 一致):
#   model['clf'].predict(X_df)  → 项目编码数组 {0,2,4}  (neu=2)
#   model['reg'].predict(X_df)  → 连续预测值数组
#   X_df 为 (date,hour) 特征 DataFrame, NaN 由 XGBoost 原生处理
# ============================================================
import numpy as np
import xgboost as xgb


class BoosterClf:
    """包装分类 Booster → predict(X_df) 返回项目编码 {0,2,4}"""

    def __init__(self, code_of_cls, booster):
        self._codes = code_of_cls
        self._bst = booster

    def predict(self, X_df):
        pr = self._bst.predict(xgb.DMatrix(X_df))
        if pr.ndim == 1:
            pr = pr.reshape(1, -1)
        return np.array([self._codes[i] for i in pr.argmax(axis=1)])


class BoosterReg:
    """包装回归 Booster → predict(X_df) 返回连续预测值 (原量纲, 元/MWh)"""

    def __init__(self, booster, scale=50.0):
        self._bst = booster
        self._scale = scale  # 兼容字段 (原生 RMSE 回归不用变换)

    def predict(self, X_df):
        return self._bst.predict(xgb.DMatrix(X_df))
