import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from torch.optim import Adam
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as f

from .loss import *
from .models import Model, lr_scheduler


# class LinearRegressionModel:
#     """
#     线性回归模型封装类，提供与XGBoost类相似的接口
#
#     参数:
#     task: str, 任务类型，'reg'表示回归，'cls'表示分类（逻辑回归）
#     method: str, 回归方法，可选'ols', 'lasso', 'ridge'
#     alpha: float, 正则化系数
#     max_iter: int, 最大迭代次数
#     """
#     def __init__(self, lin_model=None, task: str = "reg", method: str = "ols",
#                  alpha: float = 1e-3, max_iter: int = 1000):
#         self.task = task
#         self.method = method
#         self.alpha = alpha
#         self.max_iter = max_iter
#         self.lin_model = lin_model

#     def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame = None,
#             y_valid: pd.Series = None):
#         """
#         训练线性模型（保持接口统一性，验证集参数可选）
#         """
#         from sklearn.linear_model import LinearRegression, Lasso, Ridge
#
#         if self.method == 'ols':
#             self.lin_model = LinearRegression()
#         elif self.method == 'lasso':
#             self.lin_model = Lasso(alpha=self.alpha, max_iter=self.max_iter)
#         elif self.method == 'ridge':
#             self.lin_model = Ridge(alpha=self.alpha, max_iter=self.max_iter)
#
#         self.lin_model.fit(x_train, y_train)
#         return self

#     def predict(self, x_test: pd.DataFrame) -> list:
#         """生成预测结果"""
#         if self.lin_model is None:
#             raise ValueError("模型尚未训练，请先调用fit方法")
#         return self.lin_model.predict(x_test).tolist()

#     def predict_pandas(self, x: pd.DataFrame) -> pd.Series:
#         """生成带索引的预测序列"""
#         index = x.index
#         # 将列表中的单个元素解包
#         pred = pd.Series([item[0] if isinstance(item, list) else item for item in self.predict(x)],
#                         index=index)
#         return pred

#     def save(self, file_path: str):
#         """保存模型到指定目录"""
#         import pickle
#         pickle.dump(self.lin_model, open(file_path, 'wb'))

#     def load(self, file_path: str):
#         """从目录加载模型"""
#         import pickle
#         self.lin_model = pickle.load(open(file_path, 'rb'))

#     def explain_model(self, index=None):
#         """解释模型系数"""
#         if self.lin_model is None:
#             raise ValueError("模型尚未训练，请先调用fit方法")
#
#         print('Linear Model Coefficients:')
#         coef = pd.Series(self.lin_model.coef_, index=index)
#         print(coef.sort_values(ascending=False))


class LinearRegressionModel:
    """
    线性回归模型封装类，支持PyTorch张量输入和批量时间步训练

    参数:
    task: str, 任务类型，'reg'表示回归，'cls'表示分类（逻辑回归）
    method: str, 回归方法，可选'ols', 'lasso', 'ridge'
    alpha: float, 正则化系数
    max_iter: int, 最大迭代次数
    batch_size: int, 训练时的时间步批量大小，默认1
    device: str, 设备选择，'cuda'或'cpu'
    """
    def __init__(self, lin_model=None, task: str = "reg", method: str = "ols",
                 alpha: float = 1e-3, max_iter: int = 1000, batch_size: int = 1,
                 device: str = None):
        self.task = task
        self.method = method
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lin_model = lin_model
        # 自动选择设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # 存储因子名称（如果有）
        self.feature_names = None

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        """
        训练线性模型，支持DataFrame和PyTorch张量输入

        参数:
        x_train: 训练数据，可以是pd.DataFrame或torch.Tensor [时间, 股票, 因子]
        y_train: 训练标签，可以是pd.Series或torch.Tensor [时间, 股票]
        x_valid: 验证数据（可选）
        y_valid: 验证标签（可选）
        """
        # 处理PyTorch张量输入
        if isinstance(x_train, torch.Tensor) and isinstance(y_train, torch.Tensor):
            return self._fit_tensor(x_train, y_train, x_valid, y_valid)
        # 保持原有DataFrame接口
        elif isinstance(x_train, pd.DataFrame) and isinstance(y_train, (pd.Series, pd.DataFrame)):
            self.feature_names = x_train.columns.tolist()

            # 过滤掉NaN值
            # 对于DataFrame，我们需要确保X和y中的NaN值被同时过滤
            if isinstance(y_train, pd.Series):
                # 合并X和y以便同时过滤NaN
                combined = pd.concat([x_train, y_train], axis=1)
                combined = combined.dropna()
                if not combined.empty:
                    x_train_clean = combined.drop(columns=y_train.name)
                    y_train_clean = combined[y_train.name]
                else:
                    raise ValueError("过滤后的数据为空，请检查输入数据")
            else:  # DataFrame情况
                combined = x_train.join(y_train)
                combined = combined.dropna()
                if not combined.empty:
                    x_train_clean = combined[x_train.columns]
                    y_train_clean = combined[y_train.columns]
                else:
                    raise ValueError("过滤后的数据为空，请检查输入数据")

            if self.method == 'ols':
                self.lin_model = LinearRegression()
            elif self.method == 'lasso':
                self.lin_model = Lasso(alpha=self.alpha, max_iter=self.max_iter)
            elif self.method == 'ridge':
                self.lin_model = Ridge(alpha=self.alpha, max_iter=self.max_iter)

            self.lin_model.fit(x_train_clean, y_train_clean)
            return self
        else:
            raise TypeError("输入类型不支持，请使用pd.DataFrame/pd.Series或torch.Tensor")

    def _fit_tensor(self, x_train_tensor, y_train_tensor, x_valid=None, y_valid=None):
        """
        使用PyTorch张量训练模型，按时间步批量处理

        x_train_tensor形状: [时间步数, 股票数量, 因子数量]
        y_train_tensor形状: [时间步数, 股票数量]
        """
        # 获取维度信息
        total_time_steps, num_stocks, num_factors = x_train_tensor.shape

        # 初始化模型权重
        if self.lin_model is None:
            if self.method == 'ols':
                self.lin_model = LinearRegression()
            elif self.method == 'lasso':
                self.lin_model = Lasso(alpha=self.alpha, max_iter=self.max_iter)
            elif self.method == 'ridge':
                self.lin_model = Ridge(alpha=self.alpha, max_iter=self.max_iter)

        # 按时间步批量训练
        for t in range(0, total_time_steps, self.batch_size):
            # 获取当前批次
            end_t = min(t + self.batch_size, total_time_steps)

            # 处理批次内的所有时间步
            for time_idx in range(t, end_t):
                # 获取当前时间步的输入和标签 [1, N, K] -> [N, K]
                x_batch = x_train_tensor[time_idx].cpu().numpy().reshape(-1, num_factors)
                # 获取当前时间步的标签 [1, N] -> [N]
                y_batch = y_train_tensor[time_idx].cpu().numpy().reshape(-1)

                # 过滤掉NaN值
                valid_mask = ~(np.isnan(x_batch).any(axis=1) | np.isnan(y_batch))
                if valid_mask.any():
                    x_valid_batch = x_batch[valid_mask]
                    y_valid_batch = y_batch[valid_mask]

                    # 在线学习模式下，更新模型
                    if hasattr(self.lin_model, 'partial_fit'):
                        self.lin_model.partial_fit(x_valid_batch, y_valid_batch)
                    else:
                        # 对于不支持partial_fit的模型，使用所有已处理数据重新训练
                        # 注意：这不是真正的在线学习，而是累积数据的批处理
                        if t > 0 or time_idx > 0:
                            # 这里可以根据需要实现更好的增量学习策略
                            pass
                        self.lin_model.fit(x_valid_batch, y_valid_batch)

        return self

    def predict(self, x_test):
        """
        生成预测结果，支持DataFrame和PyTorch张量输入

        参数:
        x_test: 测试数据，可以是pd.DataFrame或torch.Tensor [时间, 股票, 因子]

        返回:
        对于DataFrame输入返回list，对于张量输入返回torch.Tensor [时间, 股票]
        """
        if self.lin_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        # 处理PyTorch张量输入
        if isinstance(x_test, torch.Tensor):
            return self._predict_tensor(x_test)
        # 保持原有DataFrame接口
        elif isinstance(x_test, pd.DataFrame):
            # 检查是否有NaN值
            if x_test.isna().any().any():
                # 记录原始索引，用于重建结果
                original_index = x_test.index
                # 创建完整的预测结果数组
                predictions = np.full(len(x_test), np.nan)
                # 找出有效数据的索引
                valid_mask = ~x_test.isna().any(axis=1)
                if valid_mask.any():
                    # 对有效数据进行预测
                    x_valid = x_test[valid_mask]
                    valid_preds = self.lin_model.predict(x_valid)
                    # 将预测结果放回到正确位置
                    predictions[valid_mask] = valid_preds
                return predictions.tolist()
            else:
                # 没有NaN值时，直接预测
                return self.lin_model.predict(x_test).tolist()
        else:
            raise TypeError("输入类型不支持，请使用pd.DataFrame或torch.Tensor")

    def _predict_tensor(self, x_test_tensor):
        """
        使用PyTorch张量进行预测，过滤NaN值

        x_test_tensor形状: [时间, 股票, 因子]
        返回形状: [时间, 股票]
        """
        # 确保输入形状正确 [时间, 股票, 因子]
        if len(x_test_tensor.shape) != 3:
            raise ValueError(f"输入张量形状应为[时间, 股票, 因子]，实际为{x_test_tensor.shape}")

        # 获取维度信息
        num_time_steps = x_test_tensor.shape[0]
        num_stocks = x_test_tensor.shape[1]
        num_factors = x_test_tensor.shape[2]

        # 创建输出张量
        y_pred = torch.full((num_time_steps, num_stocks), np.nan, dtype=torch.float32, device=self.device)

        # 逐时间步处理
        for t in range(num_time_steps):
            # 获取当前时间步的输入 [1, N, K] -> [N, K]
            x_batch = x_test_tensor[t].cpu().numpy().reshape(-1, num_factors)

            # 创建当前时间步的预测结果数组，初始化为NaN
            y_pred_batch = np.full(num_stocks, np.nan)

            # 找出有效数据的索引
            valid_mask = ~np.isnan(x_batch).any(axis=1)
            if valid_mask.any():
                # 对有效数据进行预测
                x_valid_batch = x_batch[valid_mask]
                valid_preds = self.lin_model.predict(x_valid_batch)
                # 将预测结果放回到正确位置
                y_pred_batch[valid_mask] = valid_preds

            # 将当前时间步的预测结果放入输出张量
            y_pred[t] = torch.tensor(y_pred_batch, dtype=torch.float32, device=self.device)

        return y_pred

    def predict_pandas(self, x: pd.DataFrame) -> pd.Series:
        """
        生成带索引的预测序列（仅支持DataFrame输入）
        """
        if not isinstance(x, pd.DataFrame):
            raise TypeError("predict_pandas仅支持pd.DataFrame输入")

        index = x.index
        pred = pd.Series([item[0] if isinstance(item, list) else item for item in self.predict(x)],
                        index=index)
        return pred

    def save(self, file_path: str):
        """
        保存模型到指定目录
        """
        import pickle
        # 保存模型和配置信息
        model_dict = {
            'lin_model': self.lin_model,
            'task': self.task,
            'method': self.method,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
            'feature_names': self.feature_names
        }
        pickle.dump(model_dict, open(file_path, 'wb'))

    def load(self, file_path: str):
        """
        从目录加载模型
        """
        import pickle
        model_dict = pickle.load(open(file_path, 'rb'))
        self.lin_model = model_dict['lin_model']
        self.task = model_dict.get('task', 'reg')
        self.method = model_dict.get('method', 'ols')
        self.alpha = model_dict.get('alpha', 1e-3)
        self.max_iter = model_dict.get('max_iter', 1000)
        self.batch_size = model_dict.get('batch_size', 1)
        self.feature_names = model_dict.get('feature_names', None)

    def explain_model(self, index=None):
        """
        解释模型系数
        """
        if self.lin_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        print('Linear Model Coefficients:')
        # 使用保存的特征名或传入的索引
        coef_index = self.feature_names if self.feature_names else index
        coef = pd.Series(self.lin_model.coef_, index=coef_index)
        print(coef.sort_values(ascending=False))


# 接收张量形状为[D, T, N, K]，其中D是天数，T是每天的时间步数，N是股票数，K是因子数
class HF_LinearRegressionModel:
    """
    高频线性回归模型封装类，支持PyTorch张量输入和批量时间步训练
    接收张量形状为[D, T, N, K]，其中D是天数，T是每天的时间步数，N是股票数，K是因子数

    参数:
    task: str, 任务类型，'reg'表示回归，'cls'表示分类（逻辑回归）
    method: str, 回归方法，可选'ols', 'lasso', 'ridge'
    alpha: float, 正则化系数
    max_iter: int, 最大迭代次数
    batch_size: int, 训练时的时间步批量大小，默认1
    device: str, 设备选择，'cuda'或'cpu'
    """
    def __init__(self, lin_model=None, task: str = "reg", method: str = "ols",
                 alpha: float = 1e-3, max_iter: int = 1000, batch_size: int = 1,
                 device: str = None):
        self.task = task
        self.method = method
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lin_model = lin_model
        # 自动选择设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # 存储因子名称（如果有）
        self.feature_names = None

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        """
        训练线性模型，支持DataFrame和PyTorch张量输入

        参数:
        x_train: 训练数据，可以是pd.DataFrame或torch.Tensor [D, T, N, K]
        y_train: 训练标签，可以是pd.Series或torch.Tensor [D, T, N]
        x_valid: 验证数据（可选）
        y_valid: 验证标签（可选）
        """
        # 处理PyTorch张量输入
        if isinstance(x_train, torch.Tensor) and isinstance(y_train, torch.Tensor):
            return self._fit_tensor(x_train, y_train, x_valid, y_valid)
        # 保持原有DataFrame接口
        elif isinstance(x_train, pd.DataFrame) and isinstance(y_train, (pd.Series, pd.DataFrame)):
            self.feature_names = x_train.columns.tolist()

            # 过滤掉NaN值
            # 对于DataFrame，我们需要确保X和y中的NaN值被同时过滤
            if isinstance(y_train, pd.Series):
                # 合并X和y以便同时过滤NaN
                combined = pd.concat([x_train, y_train], axis=1)
                combined = combined.dropna()
                if not combined.empty:
                    x_train_clean = combined.drop(columns=y_train.name)
                    y_train_clean = combined[y_train.name]
                else:
                    raise ValueError("过滤后的数据为空，请检查输入数据")
            else:  # DataFrame情况
                combined = x_train.join(y_train)
                combined = combined.dropna()
                if not combined.empty:
                    x_train_clean = combined[x_train.columns]
                    y_train_clean = combined[y_train.columns]
                else:
                    raise ValueError("过滤后的数据为空，请检查输入数据")

            if self.method == 'ols':
                self.lin_model = LinearRegression()
            elif self.method == 'lasso':
                self.lin_model = Lasso(alpha=self.alpha, max_iter=self.max_iter)
            elif self.method == 'ridge':
                self.lin_model = Ridge(alpha=self.alpha, max_iter=self.max_iter)

            self.lin_model.fit(x_train_clean, y_train_clean)
            return self
        else:
            raise TypeError("输入类型不支持，请使用pd.DataFrame/pd.Series或torch.Tensor")

    def _fit_tensor(self, x_train_tensor, y_train_tensor, x_valid=None, y_valid=None):
        """
        使用PyTorch张量训练模型，按时间步批量处理

        x_train_tensor形状: [D, T, N, K] - D天数, T每日时间步数, N股票数量, K因子数量
        y_train_tensor形状: [D, T, N] - D天数, T每日时间步数, N股票数量
        """
        # 获取维度信息
        num_days, num_time_steps_per_day, num_stocks, num_factors = x_train_tensor.shape

        # 初始化模型权重
        if self.lin_model is None:
            if self.method == 'ols':
                self.lin_model = LinearRegression()
            elif self.method == 'lasso':
                self.lin_model = Lasso(alpha=self.alpha, max_iter=self.max_iter)
            elif self.method == 'ridge':
                self.lin_model = Ridge(alpha=self.alpha, max_iter=self.max_iter)

        # 按天数和时间步批量训练
        for d in range(num_days):
            for t in range(0, num_time_steps_per_day, self.batch_size):
                # 获取当前批次
                end_t = min(t + self.batch_size, num_time_steps_per_day)

                # 处理当前天的批次内所有时间步
                for time_idx in range(t, end_t):
                    # 获取当前时间步的输入和标签 [1, 1, N, K] -> [N, K]
                    x_batch = x_train_tensor[d, time_idx].cpu().numpy().reshape(-1, num_factors)
                    # 获取当前时间步的标签 [1, 1, N] -> [N]
                    y_batch = y_train_tensor[d, time_idx].cpu().numpy().reshape(-1)

                    # 过滤掉NaN值
                    valid_mask = ~(np.isnan(x_batch).any(axis=1) | np.isnan(y_batch))
                    if valid_mask.any():
                        x_valid_batch = x_batch[valid_mask]
                        y_valid_batch = y_batch[valid_mask]

                        # 在线学习模式下，更新模型
                        if hasattr(self.lin_model, 'partial_fit'):
                            self.lin_model.partial_fit(x_valid_batch, y_valid_batch)
                        else:
                            # 对于不支持partial_fit的模型，使用所有已处理数据重新训练
                            self.lin_model.fit(x_valid_batch, y_valid_batch)

        return self

    def predict(self, x_test):
        """
        生成预测结果，支持DataFrame和PyTorch张量输入

        参数:
        x_test: 测试数据，可以是pd.DataFrame或torch.Tensor [D, T, N, K]

        返回:
        对于DataFrame输入返回list，对于张量输入返回torch.Tensor [D, T, N]
        """
        if self.lin_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        # 处理PyTorch张量输入
        if isinstance(x_test, torch.Tensor):
            return self._predict_tensor(x_test)
        # 保持原有DataFrame接口
        elif isinstance(x_test, pd.DataFrame):
            # 检查是否有NaN值
            if x_test.isna().any().any():
                # 记录原始索引，用于重建结果
                original_index = x_test.index
                # 创建完整的预测结果数组
                predictions = np.full(len(x_test), np.nan)
                # 找出有效数据的索引
                valid_mask = ~x_test.isna().any(axis=1)
                if valid_mask.any():
                    # 对有效数据进行预测
                    x_valid = x_test[valid_mask]
                    valid_preds = self.lin_model.predict(x_valid)
                    # 将预测结果放回到正确位置
                    predictions[valid_mask] = valid_preds
                return predictions.tolist()
            else:
                # 没有NaN值时，直接预测
                return self.lin_model.predict(x_test).tolist()
        else:
            raise TypeError("输入类型不支持，请使用pd.DataFrame或torch.Tensor")

    def _predict_tensor(self, x_test_tensor):
        """
        使用PyTorch张量进行预测，过滤NaN值

        x_test_tensor形状: [D, T, N, K] - D天数, T每日时间步数, N股票数量, K因子数量
        返回形状: [D, T, N] - D天数, T每日时间步数, N股票数量
        """
        # 确保输入形状正确 [D, T, N, K]
        if len(x_test_tensor.shape) != 4:
            raise ValueError(f"输入张量形状应为[D, T, N, K]，实际为{x_test_tensor.shape}")

        # 获取维度信息
        num_days, num_time_steps_per_day, num_stocks, num_factors = x_test_tensor.shape

        # 创建输出张量
        y_pred = torch.full((num_days, num_time_steps_per_day, num_stocks), np.nan,
                           dtype=torch.float32, device=self.device)

        # 逐天逐时间步处理
        for d in range(num_days):
            for t in range(num_time_steps_per_day):
                # 获取当前时间步的输入 [1, 1, N, K] -> [N, K]
                x_batch = x_test_tensor[d, t].cpu().numpy().reshape(-1, num_factors)

                # 创建当前时间步的预测结果数组，初始化为NaN
                y_pred_batch = np.full(num_stocks, np.nan)

                # 找出有效数据的索引
                valid_mask = ~np.isnan(x_batch).any(axis=1)
                if valid_mask.any():
                    # 对有效数据进行预测
                    x_valid_batch = x_batch[valid_mask]
                    valid_preds = self.lin_model.predict(x_valid_batch)
                    # 将预测结果放回到正确位置
                    y_pred_batch[valid_mask] = valid_preds

                # 将当前时间步的预测结果放入输出张量
                y_pred[d, t] = torch.tensor(y_pred_batch, dtype=torch.float32, device=self.device)

        return y_pred

    def predict_pandas(self, x: pd.DataFrame) -> pd.Series:
        """
        生成带索引的预测序列（仅支持DataFrame输入）
        """
        if not isinstance(x, pd.DataFrame):
            raise TypeError("predict_pandas仅支持pd.DataFrame输入")

        index = x.index
        pred = pd.Series([item[0] if isinstance(item, list) else item for item in self.predict(x)],
                        index=index)
        return pred

    def save(self, file_path: str):
        """
        保存模型到指定目录
        """
        import pickle
        # 保存模型和配置信息
        model_dict = {
            'lin_model': self.lin_model,
            'task': self.task,
            'method': self.method,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
            'feature_names': self.feature_names
        }
        pickle.dump(model_dict, open(file_path, 'wb'))

    def load(self, file_path: str):
        """
        从目录加载模型
        """
        import pickle
        model_dict = pickle.load(open(file_path, 'rb'))
        self.lin_model = model_dict['lin_model']
        self.task = model_dict.get('task', 'reg')
        self.method = model_dict.get('method', 'ols')
        self.alpha = model_dict.get('alpha', 1e-3)
        self.max_iter = model_dict.get('max_iter', 1000)
        self.batch_size = model_dict.get('batch_size', 1)
        self.feature_names = model_dict.get('feature_names', None)

    def explain_model(self, index=None):
        """
        解释模型系数
        """
        if self.lin_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        print('Linear Model Coefficients:')
        # 使用保存的特征名或传入的索引
        coef_index = self.feature_names if self.feature_names else index
        coef = pd.Series(self.lin_model.coef_, index=coef_index)
        print(coef.sort_values(ascending=False))



class GARCH11(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.8))

    def forward(self, returns):
        # 确保参数非负（GARCH模型要求）
        omega = nn.functional.relu(self.omega)
        alpha = nn.functional.relu(self.alpha)
        beta = nn.functional.relu(self.beta)

        T = len(returns)
        # 初始化方差序列
        sigma2 = torch.zeros_like(returns)
        sigma2[0] = omega + alpha * returns[0]**2 + beta * (omega / (1 - alpha - beta))

        # 不使用原地操作，而是创建一个新的计算图
        for t in range(1, T):
            # 计算当前方差
            current_sigma2 = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            # 使用索引赋值，但在每次循环中创建一个新的sigma2
            sigma2 = sigma2 * 1  # 创建一个新张量
            sigma2[t] = current_sigma2
        return sigma2

def negative_log_likelihood(model, returns):
    sigma2 = model(returns)
    loglik = 0.5 * (torch.log(sigma2) + (returns**2) / sigma2)
    return loglik.sum()


# model = GARCH11()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(1000):
#     optimizer.zero_grad()
#     loss = negative_log_likelihood(model, returns)
#     loss.backward()
#     optimizer.step()
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}: NLL = {loss.item():.4f}")


class HF_GARCH11(nn.Module):
    def __init__(self, omega_init=0.1, alpha_init=0.1, beta_init=0.8,
                  lr=0.01, max_iter=1000, batch_size=1, device=None, loss='mse_loss'):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(omega_init))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.loss = loss  # 添加损失函数参数
        self.grad_clip = 1.0  # 添加梯度裁剪参数

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # 移动模型到指定设备

    def forward(self, returns):
        # 确保参数非负（GARCH模型要求）
        omega = nn.functional.relu(self.omega)
        alpha = nn.functional.relu(self.alpha)
        beta = nn.functional.relu(self.beta)

        T = len(returns)

        # 使用更安全的GARCH计算方式，避免inplace操作
        # 初始化方差序列
        sigma2 = torch.zeros_like(returns)

        # 计算长期方差作为初始值
        long_run_variance = omega / (1 - alpha - beta + 1e-8)  # 加小常数避免除零
        sigma2[0] = omega + alpha * returns[0] + beta * long_run_variance

        # 使用循环但避免inplace操作问题
        # 我们创建一个新的tensor来存储结果
        result = [sigma2[0]]  # 存储每个时间点的方差

        for t in range(1, T):
            next_sigma2 = omega + alpha * returns[t-1] + beta * result[-1]
            result.append(next_sigma2)

        # 将结果连接成一个tensor
        sigma2 = torch.stack(result)

        return sigma2.reshape(T, -1)

    def init_model(self):
        """初始化模型和优化器"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, **kwargs):
        """
        训练GARCH模型，使用x_train和y_train进行训练
        在GARCH模型中，y_train通常代表收益率序列

        参数:
        x_train: 训练数据，torch.Tensor [时间, 股票, 因子]
        y_train: 训练标签，torch.Tensor [时间, 股票] - 收益率序列
        x_valid: 验证数据（可选）
        y_valid: 验证标签（可选）
        """
        if isinstance(x_train, torch.Tensor) and isinstance(y_train, torch.Tensor):
            return self._fit_tensor(x_train, y_train, x_valid, y_valid)
        else:
            raise TypeError("GARCH11需要PyTorch张量输入，形状为[时间, 股票, 因子]和[时间, 股票]")

    def _fit_tensor(self, x_train_tensor, y_train_tensor, x_valid=None, y_valid=None, **kwargs):
        """
        使用PyTorch张量训练GARCH模型
        对每个时间步的收益率序列进行建模

        x_train_tensor形状: [天数, 时间步数, 股票数量, 因子数量]
        y_train_tensor形状: [天数, 时间步数, 股票数量]
        """
        # 更新参数
        if kwargs.get('epochs') is not None:
            self.epochs = kwargs.get('epochs')
        if kwargs.get('early_stopping') is not None:
            self.early_stopping = kwargs.get('early_stopping')
        if kwargs.get('lr') is not None:
            self.lr = kwargs.get('lr')
        if kwargs.get('weight_decay') is not None:
            self.weight_decay = kwargs.get('weight_decay')

        self.init_model()

        x_train_tensor = x_train_tensor.squeeze(-1)
        x_valid = x_valid.squeeze(-1) if x_valid is not None else None
        has_valid = x_valid is not None and y_valid is not None

        # 获取维度信息
        D, T, N = x_train_tensor.shape

        # 按时间步批量训练
        for epoch in tqdm(range(self.max_iter)):
            self.train()
            epoch_loss = 0.0
            batch_count = 0

            for d in range(0, D):
                epoch_start_time = time.time()

                # 获取当前时间步的收益率 [T, N]
                returns = x_train_tensor[d].to(self.device)
                label = y_train_tensor[d].to(self.device)

                predictions = self(returns)

                # 计算损失
                if isinstance(self.loss, str):
                    loss = eval("f." + self.loss + "(predictions, label)")
                else:
                    loss = self.loss(predictions, label)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪（如果需要）
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            # 学习率调度
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # 计算平均训练损失
            avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')

            # 验证集损失
            if has_valid:
                self.eval()
                val_loss = 0.0
                val_batch_count = 0
                with torch.no_grad():
                    for d in range(0, D):
                        # 获取当前时间步的收益率 [T, N]
                        returns = x_train_tensor[d].to(self.device)
                        label = y_train_tensor[d].to(self.device)

                        predictions = self(returns)

                        # 计算损失
                        if isinstance(self.loss, str):
                            loss = eval("f." + self.loss + "(predictions, label)")
                        else:
                            loss = self.loss(predictions, label)

                        val_loss += loss.item()
                        val_batch_count += 1

                avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

                # 早停逻辑
                if avg_val_loss < self.best_loss_val:
                    self.best_loss_val = avg_val_loss
                    best_epoch = epoch + 1
                    early_stop_count = 0
                    # 保存最佳模型参数
                    best_model_state = self.model.state_dict()
                    if epoch % 10 == 0:
                        print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_val_loss:.4f} (Best), Time: {epoch_duration:.2f}s")
                else:
                    early_stop_count += 1
                    if epoch % 10 == 0:
                        print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_val_loss:.4f}, Early Stop Count: {early_stop_count}, Time: {epoch_duration:.2f}s")

                # 检查是否需要早停
                if self.early_stopping and early_stop_count >= self.early_stopping:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                    break
            else:
                # 没有验证集时使用训练集损失
                if avg_train_loss < self.best_loss_val:
                    self.best_loss_val = avg_train_loss
                    best_epoch = epoch + 1
                    early_stop_count = 0
                    # 保存最佳模型参数
                    best_model_state = self.model.state_dict()
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_train_loss:.4f} (Best), Time: {epoch_duration:.2f}s")
                else:
                    early_stop_count += 1
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_train_loss:.4f}, Early Stop Count: {early_stop_count}, Time: {epoch_duration:.2f}s")

                # 检查是否需要早停
                if self.early_stopping and early_stop_count >= self.early_stopping:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                    break

            # 记录每个epoch的结束时间并计算耗时
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

        # 恢复最佳模型参数
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        # 打印最优结果
        print(f"\nTraining completed!")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Loss: {self.best_loss_val:.4f}")

        return self

    def predict(self, x_test):
        """
        生成预测结果，返回条件方差预测

        参数:
        x_test: 测试数据，torch.Tensor [时间, 股票, 因子]

        返回:
        torch.Tensor [时间, 股票] - 条件方差预测
        """
        if isinstance(x_test, torch.Tensor):
            return self._predict_tensor(x_test)
        else:
            raise TypeError("输入类型不支持，请使用torch.Tensor")

    def _predict_tensor(self, x_test_tensor):
        """
        使用PyTorch张量进行预测，返回条件方差

        x_test_tensor形状: [日期, 时间, 股票, 因子]
        返回形状: [日期, 时间, 股票] - 条件方差预测
        """
        # 确保输入形状正确 [时间, 股票, 因子]
        if len(x_test_tensor.shape) != 4:
            raise ValueError(f"输入张量形状应为[日期, 时间, 股票, 因子]，实际为{x_test_tensor.shape}")

        # 获取维度信息
        x_test_tensor = x_test_tensor.squeeze(-1)
        D, T, N = x_test_tensor.shape

        # 创建输出张量
        y_pred = torch.full((D, T, N), np.nan, dtype=torch.float32, device=self.device)

        # 逐时间步处理
        for d in range(D):
            returns = x_test_tensor[d].to(self.device)

            # 计算条件方差
            with torch.no_grad():
                sigma2 = self(returns)
                # 将预测结果放回到正确位置
                y_pred[d] = sigma2

        return y_pred

    def predict_conditional_volatility(self, returns):
        """
        直接对给定收益率序列预测条件方差

        参数:
        returns: 收益率序列，torch.Tensor

        返回:
        torch.Tensor - 条件方差预测
        """
        with torch.no_grad():
            return self(returns)

    def save(self, file_path: str):
        """
        保存模型到指定路径
        """
        import pickle
        # 保存模型参数和配置信息
        model_dict = {
            'omega': self.omega.item(),
            'alpha': self.alpha.item(),
            'beta': self.beta.item(),
            'lr': self.lr,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
            'device': self.device
        }
        pickle.dump(model_dict, open(file_path, 'wb'))

    def load(self, file_path: str):
        """
        从指定路径加载模型
        """
        import pickle
        model_dict = pickle.load(open(file_path, 'rb'))

        # 更新参数值
        with torch.no_grad():
            self.omega.copy_(torch.tensor(model_dict['omega']))
            self.alpha.copy_(torch.tensor(model_dict['alpha']))
            self.beta.copy_(torch.tensor(model_dict['beta']))

        self.lr = model_dict.get('lr', 0.01)
        self.max_iter = model_dict.get('max_iter', 1000)
        self.batch_size = model_dict.get('batch_size', 1)
        self.device = model_dict.get('device', 'cpu')
        self.to(self.device)

    def get_params(self):
        """
        获取模型参数
        """
        with torch.no_grad():
            return {
                'omega': nn.functional.relu(self.omega).item(),
                'alpha': nn.functional.relu(self.alpha).item(),
                'beta': nn.functional.relu(self.beta).item()
            }

def negative_log_likelihood(model, returns):
    sigma2 = model(returns)
    # 避免log(0)的情况
    sigma2 = torch.clamp(sigma2, min=1e-8)
    loglik = 0.5 * (torch.log(sigma2) + (returns**2) / sigma2)
    return loglik.sum()


class HF_SARIMA:
    """
    高频季节性自回归积分滑动平均模型(SARIMA)，支持PyTorch张量输入和批量时间步训练
    专门用于处理已实现波动率等时间序列预测
    接收张量形状为[D, T, N, K]，其中D是天数，T是每天的时间步数，N是股票数，K是因子数
    对所有股票使用相同的SARIMA(1,1,1)x(P,D,Q,S)模型参数

    参数:
    order: tuple, ARIMA阶数 (p,d,q)，默认为(1,1,1)
    seasonal_order: tuple, 季节性阶数 (P,D,Q,S)，其中S为季节周期
    max_iter: int, 最大迭代次数（对于参数估计）
    batch_size: int, 训练时的时间步批量大小，默认1
    device: str, 设备选择，'cuda'或'cpu'
    """
    def __init__(self, order=(1,1,1), seasonal_order=(0,0,0,7), max_iter=100, batch_size=1, device=None):
        self.order = order  # (p, d, q)
        self.seasonal_order = seasonal_order  # (P, D, Q, S) - P:季节AR阶数, D:季节差分阶数, Q:季节MA阶数, S:季节周期
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.model = None  # 存储共享的SARIMA模型
        self.fitted_values = {}  # 存储拟合值
        # 自动选择设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        """
        训练SARIMA模型，专门用于时间序列预测
        对于SARIMA，我们主要关注y_train作为时间序列

        参数:
        x_train: 训练数据，可以是torch.Tensor [D, T, N, K]
        y_train: 训练标签，torch.Tensor [D, T, N] - 时间序列数据
        x_valid: 验证数据（可选）
        y_valid: 验证标签（可选）
        """
        if isinstance(x_train, torch.Tensor) and isinstance(y_train, torch.Tensor):
            return self._fit_tensor(x_train, y_train, x_valid, y_valid)
        else:
            raise TypeError("SARIMA需要PyTorch张量输入，形状为[D, T, N, K]和[D, T, N]")

    def _fit_tensor(self, x_train_tensor, y_train_tensor, x_valid=None, y_valid=None):
        """
        使用PyTorch张量训练SARIMA模型
        对所有股票使用相同的SARIMA模型参数

        y_train_tensor形状: [D, T, N] - D天数, T每日时间步数, N股票数量
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import numpy as np

        # 获取维度信息
        num_days, num_time_steps_per_day, num_stocks = y_train_tensor.shape

        # 合并所有股票的时间序列数据形成一个大的时间序列
        # shape: [D*T*N] - 所有天的所有时间步的所有股票
        all_ts = y_train_tensor.flatten().cpu().numpy()

        # 过滤掉NaN值
        valid_data = all_ts[~np.isnan(all_ts)]

        if len(valid_data) > max(self.order[0], self.order[2]) + self.seasonal_order[3] + 10:
            try:
                # 创建SARIMA模型 (p,d,q)x(P,D,Q,S)
                model = SARIMAX(
                    valid_data,
                    order=self.order,  # (p, d, q)
                    seasonal_order=self.seasonal_order,  # (P, D, Q, S)
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                # 拟合模型
                fitted_model = model.fit(disp=False)

                # 保存模型 - 所有股票共享同一个模型
                self.model = fitted_model

                # 保存拟合值
                self.fitted_values['all'] = fitted_model.fittedvalues

            except Exception as e:
                print(f"SARIMA模型拟合失败: {str(e)}")

        return self

    def predict(self, x_test):
        """
        生成预测结果

        参数:
        x_test: 测试数据，torch.Tensor [D, T, N, K] 或 [D, T, N]

        返回:
        torch.Tensor [D, T, N] - 预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        if isinstance(x_test, torch.Tensor):
            # 如果x_test是完整的4维张量，我们只关心时间序列部分
            if len(x_test.shape) == 4:
                # 提取时间序列部分
                y_test_tensor = x_test[:, :, :, 0] if x_test.shape[3] > 0 else x_test[:, :, :, :]
            elif len(x_test.shape) == 3:
                y_test_tensor = x_test
            else:
                raise ValueError(f"x_test应为[D, T, N, K]或[D, T, N]形状，实际为{x_test.shape}")

            return self._predict_tensor(y_test_tensor)
        else:
            raise TypeError("输入类型不支持，请使用torch.Tensor")

    def _predict_tensor(self, y_test_tensor):
        """
        使用PyTorch张量进行预测

        y_test_tensor形状: [D, T, N] - D天数, T每日时间步数, N股票数量
        返回形状: [D, T, N] - D天数, T每日时间步数, N股票数量
        """
        import numpy as np

        # 获取维度信息
        num_days, num_time_steps_per_day, num_stocks = y_test_tensor.shape

        # 创建输出张量
        y_pred = torch.full((num_days, num_time_steps_per_day, num_stocks), np.nan,
                           dtype=torch.float32, device=self.device)

        try:
            # 使用共享的模型进行预测
            fitted_model = self.model

            # 预测未来D*T个时间步
            steps = num_days * num_time_steps_per_day
            forecast = fitted_model.forecast(steps=steps)

            # 将预测结果重塑为[D, T]形状
            forecast_reshaped = forecast.reshape(num_days, num_time_steps_per_day)

            # 将相同的预测结果复制到所有股票
            for n in range(num_stocks):
                y_pred[:, :, n] = torch.tensor(forecast_reshaped,
                                              dtype=torch.float32,
                                              device=self.device)
        except Exception as e:
            print(f"预测失败: {str(e)}")

        return y_pred

    def forecast(self, steps=1):
        """
        对所有股票进行未来若干步预测（使用共享参数）

        参数:
        steps: int, 预测步数

        返回:
        numpy array, 预测值
        """
        if self.model is None:
            raise ValueError("没有训练好的模型")

        forecast = self.model.forecast(steps=steps)
        return forecast

    def predict_pandas(self, x: pd.DataFrame) -> pd.Series:
        """
        生成带索引的预测序列（仅支持DataFrame输入）
        注意：对于SARIMA，这通常不是主要用途
        """
        if not isinstance(x, pd.DataFrame):
            raise TypeError("predict_pandas仅支持pd.DataFrame输入")

        # SARIMA通常处理时间序列数据，而不是横截面数据
        # 这里简单返回空序列
        index = x.index
        return pd.Series([np.nan] * len(index), index=index)

    def save(self, file_path: str):
        """
        保存模型到指定目录
        """
        import pickle
        # 保存模型和配置信息
        model_dict = {
            'model': self.model,
            'fitted_values': self.fitted_values,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
        }
        pickle.dump(model_dict, open(file_path, 'wb'))

    def load(self, file_path: str):
        """
        从目录加载模型
        """
        import pickle
        model_dict = pickle.load(open(file_path, 'rb'))
        self.model = model_dict['model']
        self.fitted_values = model_dict['fitted_values']
        self.order = model_dict.get('order', (1, 1, 1))
        self.seasonal_order = model_dict.get('seasonal_order', (1, 1, 1, 24))
        self.max_iter = model_dict.get('max_iter', 100)
        self.batch_size = model_dict.get('batch_size', 1)

    def get_model_summary(self):
        """
        获取模型摘要信息
        """
        if self.model is None:
            return "模型尚未训练"

        summary = f"共享SARIMA模型摘要:\n"
        summary += f"非季节性参数: ARIMA{self.order}\n"
        summary += f"季节性参数: ({self.seasonal_order[0]},{self.seasonal_order[1]},{self.seasonal_order[2]},{self.seasonal_order[3]})\n"
        summary += f"此模型参数适用于所有股票\n"
        summary += str(self.model.summary())

        return summary






if __name__ == '__main__':
    x_train_tensor = torch.randn(848, 3182, 154)
    y_train_tensor = torch.randn(848, 3182)
    x_test_tensor = torch.randn(1, 3182, 154)

    # 示例用法
    # 创建模型实例，设置batch_size
    model = LinearRegressionModel(method='ols', batch_size=5, device='cuda')

    # 训练模型（输入PyTorch张量）
    # x_train_tensor形状: [848, 3182, 154]，y_train_tensor形状: [848, 3182]
    model.fit(x_train_tensor, y_train_tensor)

    # 预测（输入单个时间步的张量）
    # x_test_tensor形状: [1, N, K]
    y_pred = model.predict(x_test_tensor)  # 返回形状: [1, N]