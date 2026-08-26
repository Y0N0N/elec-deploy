import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from torch.optim import Adam
import warnings
import xgboost
import pickle

import matplotlib.pyplot as plt

import torch
import torch.nn as nn



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



class XGBoost:
    """
    XGBoost模型封装类，提供与hybrid类似的接口，支持PyTorch张量输入和批量时间步训练

    参数:
    task: str, 任务类型，'reg'表示回归，'cls'表示分类
    xgb_params: dict, XGBoost模型参数
    max_iter: int, 最大迭代次数
    batch_size: int, 训练时的时间步批量大小，默认1
    device: str, 设备选择，'cuda'或'cpu'
    """
    def __init__(self, xgb_model=None, task: str = "reg", xgb_params: dict = None,
                 max_iter: int = 1000, batch_size: int = 1, device: str = None):
        self.task = task
        self.xgb_params = xgb_params
        self.xgb_model = xgb_model
        self.max_iter = max_iter
        self.batch_size = batch_size
        # 自动选择设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # 存储因子名称（如果有）
        self.feature_names = None
        # 只保留全局模型，移除时间步模型字典
        self.global_model = None

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        """
        训练XGBoost模型，支持DataFrame和PyTorch张量输入

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

            # 处理验证集
            x_valid_clean, y_valid_clean = None, None
            if x_valid is not None and y_valid is not None:
                if isinstance(x_valid, pd.DataFrame) and isinstance(y_valid, (pd.Series, pd.DataFrame)):
                    if isinstance(y_valid, pd.Series):
                        combined_valid = pd.concat([x_valid, y_valid], axis=1)
                        combined_valid = combined_valid.dropna()
                        if not combined_valid.empty:
                            x_valid_clean = combined_valid.drop(columns=y_valid.name)
                            y_valid_clean = combined_valid[y_valid.name]
                    else:
                        combined_valid = x_valid.join(y_valid)
                        combined_valid = combined_valid.dropna()
                        if not combined_valid.empty:
                            x_valid_clean = combined_valid[x_valid.columns]
                            y_valid_clean = combined_valid[y_valid.columns]

            if self.xgb_params is None:
                # 默认参数
                est = 800
                eta = 0.0421
                colsamp = 0.9325
                subsamp = 0.8785
                max_depth = 6
                l1 = 0.25
                l2 = 0.5
                early_stopping_rounds = 20
            else:
                # 使用用户提供的参数
                est = self.xgb_params.get('est', 800)
                eta = self.xgb_params.get('eta', 0.0421)
                colsamp = self.xgb_params.get('colsamp', 0.9325)
                subsamp = self.xgb_params.get('subsamp', 0.8785)
                max_depth = self.xgb_params.get('max_depth', 6)
                l1 = self.xgb_params.get('l1', 0.25)
                l2 = self.xgb_params.get('l2', 0.5)
                early_stopping_rounds = self.xgb_params.get('early_stopping_rounds', 20)

            if self.task == 'reg':
                xgb = xgboost.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=est,
                    learning_rate=eta,  # 更新参数名
                    colsample_bytree=colsamp,
                    subsample=subsamp,
                    reg_alpha=l1,
                    reg_lambda=l2,
                    max_depth=max_depth,
                    early_stopping_rounds=early_stopping_rounds
                )
                eval_set = [(x_valid_clean, y_valid_clean)] if x_valid_clean is not None and y_valid_clean is not None else None
                self.global_model = xgb.fit(x_train_clean, y_train_clean,
                                         eval_set=eval_set,
                                         verbose=False)
            else:
                xgb = xgboost.XGBClassifier(
                    n_estimators=est,
                    learning_rate=eta,  # 更新参数名
                    colsample_bytree=colsamp,
                    subsample=subsamp,
                    reg_alpha=l1,
                    reg_lambda=l2,
                    max_depth=max_depth,
                    early_stopping_rounds=early_stopping_rounds
                )
                eval_set = [(x_valid_clean, y_valid_clean)] if x_valid_clean is not None and y_valid_clean is not None else None
                self.global_model = xgb.fit(x_train_clean, y_train_clean,
                                         eval_set=eval_set,
                                         verbose=False)

            return self
        else:
            raise TypeError("输入类型不支持，请使用pd.DataFrame/pd.Series或torch.Tensor")

    def _fit_tensor(self, x_train_tensor, y_train_tensor, x_valid=None, y_valid=None):
        """
        使用PyTorch张量训练XGBoost模型，将所有时间步的数据合并为一个全局模型

        x_train_tensor形状: [时间步数, 股票数量, 因子数量]
        y_train_tensor形状: [时间步数, 股票数量]
        """
        # 获取维度信息
        total_time_steps, num_stocks, num_factors = x_train_tensor.shape

        # 将所有时间步的数据合并为一个大的训练集
        # 重塑数据为 [总样本数, 特征数]
        x_all = x_train_tensor.cpu().numpy().reshape(-1, num_factors)
        y_all = y_train_tensor.cpu().numpy().reshape(-1)

        # 过滤掉NaN值
        valid_mask = ~(np.isnan(x_all).any(axis=1) | np.isnan(y_all))
        if not valid_mask.any():
            raise ValueError("过滤后没有有效的训练数据")

        x_valid_combined = x_all[valid_mask]
        y_valid_combined = y_all[valid_mask]

        # 如果提供了验证集，也进行同样的处理
        eval_set = None
        if x_valid is not None and y_valid is not None:
            if isinstance(x_valid, torch.Tensor) and isinstance(y_valid, torch.Tensor):
                # 合并验证集的所有时间步数据
                x_valid_flat = x_valid.cpu().numpy().reshape(-1, num_factors)
                y_valid_flat = y_valid.cpu().numpy().reshape(-1)

                # 过滤验证集的NaN值
                valid_mask_v = ~(np.isnan(x_valid_flat).any(axis=1) | np.isnan(y_valid_flat))
                if valid_mask_v.any():
                    x_valid_eval = x_valid_flat[valid_mask_v]
                    y_valid_eval = y_valid_flat[valid_mask_v]
                    eval_set = [(x_valid_eval, y_valid_eval)]

        # 设置XGBoost参数
        if self.xgb_params is None:
            # 默认参数
            est = 800
            eta = 0.0421
            colsamp = 0.9325
            subsamp = 0.8785
            max_depth = 6
            l1 = 0.25
            l2 = 0.5
            early_stopping_rounds = 20
        else:
            # 使用用户提供的参数
            est = self.xgb_params.get('est', 800)
            eta = self.xgb_params.get('eta', 0.0421)
            colsamp = self.xgb_params.get('colsamp', 0.9325)
            subsamp = self.xgb_params.get('subsamp', 0.8785)
            max_depth = self.xgb_params.get('max_depth', 6)
            l1 = self.xgb_params.get('l1', 0.25)
            l2 = self.xgb_params.get('l2', 0.5)
            early_stopping_rounds = self.xgb_params.get('early_stopping_rounds', 20)

        # 如果没有验证集，禁用早停
        if eval_set is None:
            early_stopping_rounds = None

        if self.task == 'reg':
            xgb = xgboost.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=est,
                learning_rate=eta,  # 更新参数名
                colsample_bytree=colsamp,
                subsample=subsamp,
                reg_alpha=l1,
                reg_lambda=l2,
                max_depth=max_depth,
                early_stopping_rounds=early_stopping_rounds
            )
        else:
            xgb = xgboost.XGBClassifier(
                n_estimators=est,
                learning_rate=eta,  # 更新参数名
                colsample_bytree=colsamp,
                subsample=subsamp,
                reg_alpha=l1,
                reg_lambda=l2,
                max_depth=max_depth,
                early_stopping_rounds=early_stopping_rounds
            )

        # 训练全局模型
        if eval_set is not None:
            self.global_model = xgb.fit(x_valid_combined, y_valid_combined,
                                       eval_set=eval_set, verbose=False)
        else:
            self.global_model = xgb.fit(x_valid_combined, y_valid_combined, verbose=False)

        return self

    def predict(self, x_test):
        """
        生成预测结果，支持DataFrame和PyTorch张量输入

        参数:
        x_test: 测试数据，可以是pd.DataFrame或torch.Tensor [时间, 股票, 因子]

        返回:
        对于DataFrame输入返回list，对于张量输入返回torch.Tensor [时间, 股票]
        """
        if self.global_model is None:
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
                    valid_preds = self.global_model.predict(x_valid)
                    # 将预测结果放回到正确位置
                    predictions[valid_mask] = valid_preds
                return predictions.tolist()
            else:
                # 没有NaN值时，直接预测
                return self.global_model.predict(x_test).tolist()
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

                # 使用全局模型进行预测
                valid_preds = self.global_model.predict(x_valid_batch)
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
        # 保存模型和配置信息
        model_dict = {
            'global_model': self.global_model,
            'task': self.task,
            'xgb_params': self.xgb_params,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
            'feature_names': self.feature_names
        }
        pickle.dump(model_dict, open(file_path, 'wb'))

    def load(self, file_path: str):
        """
        从目录加载模型
        """
        model_dict = pickle.load(open(file_path, 'rb'))
        self.global_model = model_dict['global_model']
        self.task = model_dict.get('task', 'reg')
        self.xgb_params = model_dict.get('xgb_params', None)
        self.max_iter = model_dict.get('max_iter', 1000)
        self.batch_size = model_dict.get('batch_size', 1)
        self.feature_names = model_dict.get('feature_names', None)

    def explain_model(self, index=None):
        """
        解释模型，展示特征重要性
        """
        if self.global_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        print('XGBoost Feature Importance:')
        xgboost.plot_importance(self.global_model)
        plt.show()

        importance = self.global_model.feature_importances_
        if index is not None:
            importance = pd.Series(importance, index=index).sort_values(ascending=False)
            print(importance)



class HF_XGBoost:
    """
    高频XGBoost模型封装类，支持PyTorch张量输入和批量时间步训练
    接收张量形状为[D, T, N, K]，其中D是天数，T是每天的时间步数，N是股票数，K是因子数

    参数:
    task: str, 任务类型，'reg'表示回归，'cls'表示分类
    xgb_params: dict, XGBoost模型参数
    max_iter: int, 最大迭代次数（映射到n_estimators）
    batch_size: int, 训练时的时间步批量大小，默认1
    device: str, 设备选择，'cuda'或'cpu'
    """
    def __init__(self, xgb_model=None, task: str = "reg", xgb_params: dict = None,
                 max_iter: int = 800, batch_size: int = 1, device: str = None):
        self.task = task
        self.xgb_params = xgb_params
        self.xgb_model = xgb_model
        self.max_iter = max_iter  # 映射到n_estimators
        self.batch_size = batch_size
        # 自动选择设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # 存储因子名称（如果有）
        self.feature_names = None
        # 全局模型
        self.global_model = None

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        """
        训练XGBoost模型，支持DataFrame和PyTorch张量输入

        参数:
        x_train: 训练数据，可以是pd.DataFrame或torch.Tensor [D, T, N, K]
        y_train: 训练标签，可以是pd.Series或torch.Tensor [D, T, N]
        x_valid: 验证数据（可选）
        y_valid: 验证标签（可选）
        """
        # 处理PyTorch张量输入
        if isinstance(x_train, torch.Tensor) and isinstance(y_train, torch.Tensor):
            if len(x_train.shape) == 4:  # [D, T, N, K]
                return self._fit_tensor(x_train, y_train, x_valid, y_valid)
            else:
                raise ValueError(f"期望输入张量形状为[D, T, N, K]，实际为{x_train.shape}")
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

            # 处理验证集
            x_valid_clean, y_valid_clean = None, None
            if x_valid is not None and y_valid is not None:
                if isinstance(x_valid, pd.DataFrame) and isinstance(y_valid, (pd.Series, pd.DataFrame)):
                    if isinstance(y_valid, pd.Series):
                        combined_valid = pd.concat([x_valid, y_valid], axis=1)
                        combined_valid = combined_valid.dropna()
                        if not combined_valid.empty:
                            x_valid_clean = combined_valid.drop(columns=y_valid.name)
                            y_valid_clean = combined_valid[y_valid.name]
                    else:
                        combined_valid = x_valid.join(y_valid)
                        combined_valid = combined_valid.dropna()
                        if not combined_valid.empty:
                            x_valid_clean = combined_valid[x_valid.columns]
                            y_valid_clean = combined_valid[y_valid.columns]

            # 设置XGBoost参数
            if self.xgb_params is None:
                # 默认参数
                est = self.max_iter
                eta = 0.0421
                colsamp = 0.9325
                subsamp = 0.8785
                max_depth = 6
                l1 = 0.25
                l2 = 0.5
                early_stopping_rounds = 20
            else:
                # 使用用户提供的参数
                est = self.xgb_params.get('est', self.max_iter)
                eta = self.xgb_params.get('eta', 0.0421)
                colsamp = self.xgb_params.get('colsamp', 0.9325)
                subsamp = self.xgb_params.get('subsamp', 0.8785)
                max_depth = self.xgb_params.get('max_depth', 6)
                l1 = self.xgb_params.get('l1', 0.25)
                l2 = self.xgb_params.get('l2', 0.5)
                early_stopping_rounds = self.xgb_params.get('early_stopping_rounds', 20)

            if self.task == 'reg':
                xgb = xgboost.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=est,
                    learning_rate=eta,  # 更新参数名
                    colsample_bytree=colsamp,
                    subsample=subsamp,
                    reg_alpha=l1,
                    reg_lambda=l2,
                    max_depth=max_depth,
                    early_stopping_rounds=early_stopping_rounds
                )
                eval_set = [(x_valid_clean, y_valid_clean)] if x_valid_clean is not None and y_valid_clean is not None else None
                self.global_model = xgb.fit(x_train_clean, y_train_clean,
                                         eval_set=eval_set,
                                         verbose=False)
            else:
                xgb = xgboost.XGBClassifier(
                    n_estimators=est,
                    learning_rate=eta,  # 更新参数名
                    colsample_bytree=colsamp,
                    subsample=subsamp,
                    reg_alpha=l1,
                    reg_lambda=l2,
                    max_depth=max_depth,
                    early_stopping_rounds=early_stopping_rounds
                )
                eval_set = [(x_valid_clean, y_valid_clean)] if x_valid_clean is not None and y_valid_clean is not None else None
                self.global_model = xgb.fit(x_train_clean, y_train_clean,
                                         eval_set=eval_set,
                                         verbose=False)

            return self
        else:
            raise TypeError("输入类型不支持，请使用pd.DataFrame/pd.Series或torch.Tensor")

    def _fit_tensor(self, x_train_tensor, y_train_tensor, x_valid=None, y_valid=None):
        """
        使用PyTorch张量训练XGBoost模型，按天数和时间步批量处理

        x_train_tensor形状: [D, T, N, K] - D天数, T每日时间步数, N股票数量, K因子数量
        y_train_tensor形状: [D, T, N] - D天数, T每日时间步数, N股票数量
        """
        # 获取维度信息
        num_days, num_time_steps_per_day, num_stocks, num_factors = x_train_tensor.shape

        # 将所有时间步的数据合并为一个大的训练集
        # 重塑数据为 [总样本数, 特征数]
        x_all = x_train_tensor.cpu().numpy().reshape(-1, num_factors)
        y_all = y_train_tensor.cpu().numpy().reshape(-1)

        # 过滤掉NaN值
        valid_mask = ~(np.isnan(x_all).any(axis=1) | np.isnan(y_all))
        if not valid_mask.any():
            raise ValueError("过滤后没有有效的训练数据")

        x_valid_combined = x_all[valid_mask]
        y_valid_combined = y_all[valid_mask]

        # 如果提供了验证集，也进行同样的处理
        eval_set = None
        if x_valid is not None and y_valid is not None:
            if isinstance(x_valid, torch.Tensor) and isinstance(y_valid, torch.Tensor):
                # 合并验证集的所有时间步数据
                x_valid_flat = x_valid.cpu().numpy().reshape(-1, num_factors)
                y_valid_flat = y_valid.cpu().numpy().reshape(-1)

                # 过滤验证集的NaN值
                valid_mask_v = ~(np.isnan(x_valid_flat).any(axis=1) | np.isnan(y_valid_flat))
                if valid_mask_v.any():
                    x_valid_eval = x_valid_flat[valid_mask_v]
                    y_valid_eval = y_valid_flat[valid_mask_v]
                    eval_set = [(x_valid_eval, y_valid_eval)]

        # 设置XGBoost参数
        if self.xgb_params is None:
            # 默认参数
            est = self.max_iter
            eta = 0.0421
            colsamp = 0.9325
            subsamp = 0.8785
            max_depth = 6
            l1 = 0.25
            l2 = 0.5
            early_stopping_rounds = 20
        else:
            # 使用用户提供的参数
            est = self.xgb_params.get('est', self.max_iter)
            eta = self.xgb_params.get('eta', 0.0421)
            colsamp = self.xgb_params.get('colsamp', 0.9325)
            subsamp = self.xgb_params.get('subsamp', 0.8785)
            max_depth = self.xgb_params.get('max_depth', 6)
            l1 = self.xgb_params.get('l1', 0.25)
            l2 = self.xgb_params.get('l2', 0.5)
            early_stopping_rounds = self.xgb_params.get('early_stopping_rounds', 20)

        # 如果没有验证集，禁用早停
        if eval_set is None:
            early_stopping_rounds = None

        if self.task == 'reg':
            xgb = xgboost.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=est,
                learning_rate=eta,  # 更新参数名
                colsample_bytree=colsamp,
                subsample=subsamp,
                reg_alpha=l1,
                reg_lambda=l2,
                max_depth=max_depth,
                early_stopping_rounds=early_stopping_rounds
            )
        else:
            xgb = xgboost.XGBClassifier(
                n_estimators=est,
                learning_rate=eta,  # 更新参数名
                colsample_bytree=colsamp,
                subsample=subsamp,
                reg_alpha=l1,
                reg_lambda=l2,
                max_depth=max_depth,
                early_stopping_rounds=early_stopping_rounds
            )

        # 训练全局模型
        if eval_set is not None:
            self.global_model = xgb.fit(x_valid_combined, y_valid_combined,
                                       eval_set=eval_set, verbose=False)
        else:
            self.global_model = xgb.fit(x_valid_combined, y_valid_combined, verbose=False)

        return self

    def predict(self, x_test):
        """
        生成预测结果，支持DataFrame和PyTorch张量输入

        参数:
        x_test: 测试数据，可以是pd.DataFrame或torch.Tensor [D, T, N, K]

        返回:
        对于DataFrame输入返回list，对于张量输入返回torch.Tensor [D, T, N]
        """
        if self.global_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        # 处理PyTorch张量输入
        if isinstance(x_test, torch.Tensor):
            if len(x_test.shape) == 4:  # [D, T, N, K]
                return self._predict_tensor(x_test)
            else:
                raise ValueError(f"期望输入张量形状为[D, T, N, K]，实际为{x_test.shape}")
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
                    valid_preds = self.global_model.predict(x_valid)
                    # 将预测结果放回到正确位置
                    predictions[valid_mask] = valid_preds
                return predictions.tolist()
            else:
                # 没有NaN值时，直接预测
                return self.global_model.predict(x_test).tolist()
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

                    # 使用全局模型进行预测
                    valid_preds = self.global_model.predict(x_valid_batch)
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
        # 保存模型和配置信息
        model_dict = {
            'global_model': self.global_model,
            'task': self.task,
            'xgb_params': self.xgb_params,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
            'feature_names': self.feature_names
        }
        pickle.dump(model_dict, open(file_path, 'wb'))

    def load(self, file_path: str):
        """
        从目录加载模型
        """
        model_dict = pickle.load(open(file_path, 'rb'))
        self.global_model = model_dict['global_model']
        self.task = model_dict.get('task', 'reg')
        self.xgb_params = model_dict.get('xgb_params', None)
        self.max_iter = model_dict.get('max_iter', 800)
        self.batch_size = model_dict.get('batch_size', 1)
        self.feature_names = model_dict.get('feature_names', None)

    def explain_model(self, index=None):
        """
        解释模型，展示特征重要性
        """
        if self.global_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        print('XGBoost Feature Importance:')
        xgboost.plot_importance(self.global_model)
        plt.show()

        importance = self.global_model.feature_importances_
        if index is not None:
            importance = pd.Series(importance, index=index).sort_values(ascending=False)
            print(importance)



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