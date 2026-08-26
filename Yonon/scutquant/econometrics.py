import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.groupby import Grouper
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from .models.models import Model
from linearmodels.panel import PanelOLS


class GARCH(Model):
    def __init__(self, p: int = 1, q: int = 1, mean: str = 'Zero', forecast_horizon: int = 0, *args, **kwargs):
        """
        GARCH模型用于波动率预测

        参数:
        p: GARCH模型的p参数，表示条件方差的自回归阶数
        q: GARCH模型的q参数，表示残差平方的移动平均阶数
        mean: 均值模型，可选'Zero'、'Constant'、'AR'等
        forecast_horizon: 预测未来几天的波动率
        """
        super().__init__(*args, **kwargs)
        self.p = p
        self.q = q
        self.mean = mean
        self.forecast_horizon = forecast_horizon
        self.models = {}  # 存储每个金融工具的GARCH模型
        self.model_fits = {}  # 存储每个金融工具的拟合结果
        self.panel_results = None  # 存储面板数据分析结果

    def init_model(self):
        # GARCH模型不需要初始化PyTorch模型
        pass

    def _prepare_panel_data(self, x_train, returns_col='pct_chg'):
        """
        准备面板数据用于初步分析 - 优化版本，使用向量化操作代替循环

        参数:
        x_train: 训练数据，包含收益率列
        returns_col: 收益率列名

        返回:
        面板数据DataFrame
        """
        x_train = x_train.sort_index()
        returns = x_train[returns_col].copy()
        returns = returns.dropna()
        min_data_points = max(self.p, self.q) + 10
        data_counts = returns.groupby(level=1).count()
        valid_instruments = data_counts[data_counts > min_data_points].index

        if len(valid_instruments) == 0:
            raise ValueError("没有足够的数据来构建面板数据集")

        valid_returns = returns[returns.index.get_level_values(1).isin(valid_instruments)]
        panel_data = pd.DataFrame({'returns': valid_returns})

        return panel_data

    def fit(self, x_train, returns_col='pct_chg', use_panel=True, batch_size=50):
        """
        训练GARCH模型并返回训练集上的条件方差预测

        参数:
        x_train: 训练数据，包含收益率列
        returns_col: 收益率列名
        use_panel: 是否使用面板数据进行初步分析
        batch_size: 批处理大小，用于并行处理

        返回:
        训练集上的条件方差预测DataFrame
        """
        from arch import arch_model

        # 确保x_train是DataFrame
        if not isinstance(x_train, DataFrame):
            raise ValueError("x_train必须是DataFrame")

        # 获取所有唯一的金融工具
        instruments = x_train.index.get_level_values(1).unique()

        # 如果使用面板数据进行初步分析
        if use_panel:
            try:
                print("准备面板数据进行初步分析...")
                panel_data = self._prepare_panel_data(x_train, returns_col)

                # 使用PanelOLS进行初步分析，可以帮助识别共同趋势
                from linearmodels.panel import PanelOLS

                # 创建滞后项作为特征
                for lag in range(1, max(self.p, self.q) + 1):
                    panel_data[f'returns_lag_{lag}'] = panel_data.groupby(level=1)['returns'].shift(lag)
                    panel_data[f'returns_squared_lag_{lag}'] = panel_data[f'returns_lag_{lag}']**2

                # 删除包含NaN的行
                panel_data = panel_data.dropna()

                # 准备因变量和自变量
                y = panel_data['returns']**2  # 使用平方收益率作为波动率代理
                X = panel_data[[col for col in panel_data.columns if 'lag' in col]]

                # 添加常数项
                from linearmodels.panel.model import PanelModelData
                X = PanelModelData(X).add_constant()

                # 训练PanelOLS模型
                print("使用PanelOLS进行初步分析...")
                panel_model = PanelOLS(y, X)
                self.panel_results = panel_model.fit(cov_type='clustered', cluster_entity=True)

                print(f"面板数据分析完成，R-squared: {self.panel_results.rsquared}")

                # 可以使用面板分析结果来初始化GARCH模型的参数
                # 这里只是示例，实际上GARCH模型的参数初始化比较复杂

            except Exception as e:
                print(f"面板数据分析出错: {e}")
                print("继续使用传统GARCH方法...")

        total_loss = 0
        valid_models = 0

        # 存储训练集上的条件方差预测
        volatility_predictions = []

        # 将金融工具分成批次进行处理，提高并行效率
        instrument_batches = [instruments[i:i + batch_size] for i in range(0, len(instruments), batch_size)]

        for batch in tqdm(instrument_batches, desc="训练GARCH模型（批处理）"):
            batch_results = []

            # 使用并行处理批次内的金融工具
            try:
                from joblib import Parallel, delayed

                def process_instrument(instrument):
                    try:
                        instrument_data = x_train.xs(instrument, level=1)
                        instrument_data = instrument_data.sort_index()
                        returns = instrument_data[returns_col]
                        returns = returns.dropna()

                        if len(returns) <= max(self.p, self.q) + 10:
                            return None, None, None, []

                        # 如果有面板分析结果，可以用来初始化GARCH模型参数
                        if self.panel_results is not None:
                            # 这里只是示例，实际上需要更复杂的逻辑
                            model = arch_model(returns, vol='Garch', p=self.p, q=self.q, mean=self.mean)
                        else:
                            model = arch_model(returns, vol='Garch', p=self.p, q=self.q, mean=self.mean)

                        model_fit = model.fit(disp='off')

                        # 获取训练集上的条件方差预测
                        conditional_variance = model_fit.conditional_volatility**2

                        # 创建预测结果
                        predictions = []
                        for date, variance in zip(conditional_variance.index, conditional_variance.values):
                            predictions.append({
                                'datetime': date,
                                'instrument': instrument,
                                'conditional_variance': variance
                            })

                        return instrument, model, model_fit, predictions

                    except Exception as e:
                        print(f"处理 {instrument} 时出错: {e}")
                        return None, None, None, []

                # 并行处理批次内的金融工具
                batch_results = Parallel(n_jobs=-1)(
                    delayed(process_instrument)(instrument) for instrument in batch
                )

            except ImportError:
                print("警告: 未安装joblib，使用串行处理...")

                # 串行处理批次内的金融工具
                for instrument in batch:
                    try:
                        instrument_data = x_train.xs(instrument, level=1)
                        instrument_data = instrument_data.sort_index()
                        returns = instrument_data[returns_col]
                        returns = returns.dropna()

                        if len(returns) <= max(self.p, self.q) + 10:
                            continue

                        model = arch_model(returns, vol='Garch', p=self.p, q=self.q, mean=self.mean)
                        model_fit = model.fit(disp='off')

                        # 获取训练集上的条件方差预测
                        conditional_variance = model_fit.conditional_volatility**2

                        # 创建预测结果
                        predictions = []
                        for date, variance in zip(conditional_variance.index, conditional_variance.values):
                            predictions.append({
                                'datetime': date,
                                'instrument': instrument,
                                'conditional_variance': variance
                            })

                        batch_results.append((instrument, model, model_fit, predictions))

                    except Exception as e:
                        print(f"处理 {instrument} 时出错: {e}")

            # 处理批次结果
            for instrument, model, model_fit, predictions in batch_results:
                if instrument is not None:
                    self.models[instrument] = model
                    self.model_fits[instrument] = model_fit

                    # 计算损失（负对数似然）
                    total_loss += model_fit.loglikelihood
                    valid_models += 1

                    # 添加预测结果
                    volatility_predictions.extend(predictions)

        if valid_models > 0:
            print(f"训练完成，平均对数似然: {total_loss / valid_models}")
        else:
            print("警告: 没有成功训练任何模型")

        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)

        # 设置多重索引
        if not vol_df.empty:
            vol_df = vol_df.set_index(['datetime', 'instrument'])

        return vol_df

    def predict_pandas(self, x: DataFrame, returns_col='pct_chg') -> DataFrame:
        """
        预测每个金融工具的条件方差

        参数:
        x: 包含收益率数据的DataFrame
        returns_col: 收益率列名

        返回:
        预测的条件方差DataFrame，索引与原始x相同
        """
        if not self.models:
            raise ValueError("模型尚未训练，请先调用fit方法")

        instruments = x.index.get_level_values(1).unique()
        volatility_predictions = []

        # 将金融工具分成批次进行处理
        instrument_batches = [instruments[i:i + 50] for i in range(0, len(instruments), 50)]

        for batch in tqdm(instrument_batches, desc="预测条件方差（批处理）"):
            batch_predictions = []

            # 使用并行处理批次内的金融工具
            try:
                from joblib import Parallel, delayed

                def process_instrument(instrument):
                    if instrument not in self.models:
                        return []

                    try:
                        # 提取该金融工具的数据
                        instrument_data = x.xs(instrument, level=1)
                        instrument_data = instrument_data.sort_index()

                        # 获取模型拟合结果
                        model_fit = self.model_fits[instrument]

                        # 获取条件方差预测
                        conditional_variance = model_fit.conditional_volatility**2

                        # 创建预测结果
                        predictions = []
                        for date, variance in zip(conditional_variance.index, conditional_variance.values):
                            predictions.append({
                                'datetime': date,
                                'instrument': instrument,
                                'conditional_variance': variance
                            })

                        # 如果需要预测未来的条件方差
                        if self.forecast_horizon > 0:
                            forecast = model_fit.forecast(horizon=self.forecast_horizon)

                            # 获取最后一个预测点
                            last_date = instrument_data.index[-1]

                            # 添加未来预测的条件方差
                            for h in range(1, self.forecast_horizon + 1):
                                future_variance = forecast.variance.iloc[-1, h-1]
                                # 这里假设日期是按天递增的，可以根据实际情况调整
                                future_date = pd.Timestamp(last_date) + pd.Timedelta(days=h)

                                predictions.append({
                                    'datetime': future_date,
                                    'instrument': instrument,
                                    'conditional_variance': future_variance,
                                    'is_forecast': True  # 标记为预测值
                                })

                        return predictions

                    except Exception as e:
                        print(f"预测 {instrument} 时出错: {e}")
                        return []

                # 并行处理批次内的金融工具
                batch_results = Parallel(n_jobs=-1)(
                    delayed(process_instrument)(instrument) for instrument in batch
                )

                # 合并批次结果
                for result in batch_results:
                    batch_predictions.extend(result)

            except ImportError:
                # 串行处理批次内的金融工具
                for instrument in batch:
                    if instrument not in self.models:
                        print(f"警告: {instrument} 没有对应的模型，跳过")
                        continue

                    try:
                        # 提取该金融工具的数据
                        instrument_data = x.xs(instrument, level=1)
                        instrument_data = instrument_data.sort_index()

                        # 获取模型拟合结果
                        model_fit = self.model_fits[instrument]

                        # 获取条件方差预测
                        conditional_variance = model_fit.conditional_volatility**2

                        # 创建预测结果
                        for date, variance in zip(conditional_variance.index, conditional_variance.values):
                            batch_predictions.append({
                                'datetime': date,
                                'instrument': instrument,
                                'conditional_variance': variance
                            })

                        # 如果需要预测未来的条件方差
                        if self.forecast_horizon > 0:
                            forecast = model_fit.forecast(horizon=self.forecast_horizon)

                            # 获取最后一个预测点
                            last_date = instrument_data.index[-1]

                            # 添加未来预测的条件方差
                            for h in range(1, self.forecast_horizon + 1):
                                future_variance = forecast.variance.iloc[-1, h-1]
                                # 这里假设日期是按天递增的，可以根据实际情况调整
                                future_date = pd.Timestamp(last_date) + pd.Timedelta(days=h)

                                batch_predictions.append({
                                    'datetime': future_date,
                                    'instrument': instrument,
                                    'conditional_variance': future_variance,
                                    'is_forecast': True  # 标记为预测值
                                })

                    except Exception as e:
                        print(f"预测 {instrument} 时出错: {e}")

            # 添加批次预测结果
            volatility_predictions.extend(batch_predictions)

        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)

        # 设置多重索引
        if not vol_df.empty:
            if 'is_forecast' in vol_df.columns:
                vol_df = vol_df.set_index(['datetime', 'instrument', 'is_forecast'])
            else:
                vol_df = vol_df.set_index(['datetime', 'instrument'])

        return vol_df

    def rolling_predict(self, x: DataFrame, returns_col='pct_chg', w=252, step=1, parallel=False, n_jobs=-1) -> DataFrame:
        """
        使用滚动窗口方法预测条件方差

        参数:
        x: 包含收益率数据的DataFrame
        returns_col: 收益率列名
        w: 滚动窗口大小，默认为252（约一年的交易日）
        step: 滚动步长，默认为1，可以设置更大的值来减少计算量
        parallel: 是否使用并行计算，默认为False
        n_jobs: 并行计算的作业数，默认为-1（使用所有可用CPU）

        返回:
        滚动预测的条件方差DataFrame
        """
        from arch import arch_model

        # 确保x是DataFrame
        if not isinstance(x, DataFrame):
            raise ValueError("x必须是DataFrame")

        # 获取所有唯一的金融工具
        instruments = x.index.get_level_values(1).unique()

        # 存储滚动预测结果
        volatility_predictions = []

        # 定义单个金融工具的处理函数
        def process_instrument(instrument):
            instrument_predictions = []
            try:
                # 提取该金融工具的数据
                instrument_data = x.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()

                # 获取收益率数据
                returns = instrument_data[returns_col]
                returns = returns.dropna()

                if len(returns) <= max(self.p, self.q) + 10 + w:
                    print(f"警告: {instrument} 的数据点不足，跳过")
                    return []

                # 获取所有日期
                dates = returns.index

                # 对每个滚动窗口进行预测，使用步长来减少计算量
                for i in range(w, len(dates), step):
                    # 获取当前窗口的数据
                    window_data = returns.iloc[i-w:i]
                    current_date = dates[i]

                    # 创建并拟合GARCH模型
                    model = arch_model(window_data, vol='Garch', p=self.p, q=self.q, mean=self.mean)
                    model_fit = model.fit(disp='off', show_warning=False, options={'maxiter': 100})

                    # 获取最后一天的条件方差
                    conditional_variance = model_fit.conditional_volatility[-1]**2

                    # 存储预测结果
                    instrument_predictions.append({
                        'datetime': current_date,
                        'instrument': instrument,
                        'conditional_variance': conditional_variance
                    })

                    # 如果需要预测未来的条件方差
                    if self.forecast_horizon > 0:
                        forecast = model_fit.forecast(horizon=self.forecast_horizon)

                        # 添加未来预测的条件方差
                        for h in range(1, self.forecast_horizon + 1):
                            future_variance = forecast.variance.iloc[-1, h-1]
                            future_date = pd.Timestamp(current_date) + pd.Timedelta(days=h)

                            instrument_predictions.append({
                                'datetime': future_date,
                                'instrument': instrument,
                                'conditional_variance': future_variance,
                                'is_forecast': True
                            })

            except Exception as e:
                print(f"处理 {instrument} 时出错: {e}")

            return instrument_predictions

        # 使用并行计算处理多个金融工具
        if parallel:
            try:
                from joblib import Parallel, delayed

                # 并行处理所有金融工具
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_instrument)(instrument) for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})")
                )

                # 合并结果
                for result in results:
                    volatility_predictions.extend(result)

            except ImportError:
                print("警告: 未安装joblib，无法使用并行计算。请使用 'pip install joblib' 安装。")
                # 回退到串行处理
                for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})"):
                    volatility_predictions.extend(process_instrument(instrument))
        else:
            # 串行处理所有金融工具
            for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})"):
                volatility_predictions.extend(process_instrument(instrument))

        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)

        # 设置多重索引
        if not vol_df.empty:
            if 'is_forecast' in vol_df.columns:
                vol_df = vol_df.set_index(['datetime', 'instrument', 'is_forecast'])
            else:
                vol_df = vol_df.set_index(['datetime', 'instrument'])

        return vol_df


class EGARCH(Model):
    def __init__(self, p: int = 1, q: int = 1, o: int = 1, mean: str = 'Zero', forecast_horizon: int = 0, *args, **kwargs):
        """
        EGARCH模型用于波动率预测，能够捕捉杠杆效应

        参数:
        p: EGARCH模型的p参数，表示条件方差的自回归阶数
        q: EGARCH模型的q参数，表示残差平方的移动平均阶数
        o: EGARCH模型的o参数，表示杠杆效应项的阶数
        mean: 均值模型，可选'Zero'、'Constant'、'AR'等
        forecast_horizon: 预测未来几天的波动率
        """
        super().__init__(*args, **kwargs)
        self.p = p
        self.q = q
        self.o = o
        self.mean = mean
        self.forecast_horizon = forecast_horizon
        self.models = {}  # 存储每个金融工具的EGARCH模型
        self.model_fits = {}  # 存储每个金融工具的拟合结果
        self.panel_results = None  # 存储面板数据分析结果

    def init_model(self):
        # EGARCH模型不需要初始化PyTorch模型
        pass

    def _prepare_panel_data(self, x_train, returns_col='pct_chg'):
        """
        准备面板数据用于初步分析 - 优化版本，使用向量化操作代替循环

        参数:
        x_train: 训练数据，包含收益率列
        returns_col: 收益率列名

        返回:
        面板数据DataFrame
        """
        x_train = x_train.sort_index()
        returns = x_train[returns_col].copy()
        returns = returns.dropna()
        min_data_points = max(self.p, self.q, self.o) + 10
        data_counts = returns.groupby(level=1).count()
        valid_instruments = data_counts[data_counts > min_data_points].index

        if len(valid_instruments) == 0:
            raise ValueError("没有足够的数据来构建面板数据集")

        valid_returns = returns[returns.index.get_level_values(1).isin(valid_instruments)]
        panel_data = pd.DataFrame({'returns': valid_returns})

        return panel_data

    def fit(self, x_train, returns_col='pct_chg', use_panel=True, batch_size=50):
        """
        训练EGARCH模型并返回训练集上的条件方差预测

        参数:
        x_train: 训练数据，包含收益率列
        returns_col: 收益率列名
        use_panel: 是否使用面板数据进行初步分析
        batch_size: 批处理大小，用于并行处理

        返回:
        训练集上的条件方差预测DataFrame
        """
        from arch import arch_model

        # 确保x_train是DataFrame
        if not isinstance(x_train, DataFrame):
            raise ValueError("x_train必须是DataFrame")

        # 获取所有唯一的金融工具
        instruments = x_train.index.get_level_values(1).unique()

        # 如果使用面板数据进行初步分析
        if use_panel:
            try:
                print("准备面板数据进行初步分析...")
                panel_data = self._prepare_panel_data(x_train, returns_col)

                # 使用PanelOLS进行初步分析，可以帮助识别共同趋势
                from linearmodels.panel import PanelOLS

                # 创建滞后项作为特征
                for lag in range(1, max(self.p, self.q, self.o) + 1):
                    panel_data[f'returns_lag_{lag}'] = panel_data.groupby(level=1)['returns'].shift(lag)
                    panel_data[f'returns_squared_lag_{lag}'] = panel_data[f'returns_lag_{lag}']**2
                    # 添加杠杆效应特征
                    panel_data[f'returns_sign_lag_{lag}'] = np.sign(panel_data[f'returns_lag_{lag}']) * panel_data[f'returns_squared_lag_{lag}']

                # 删除包含NaN的行
                panel_data = panel_data.dropna()

                # 准备因变量和自变量
                y = panel_data['returns']**2  # 使用平方收益率作为波动率代理
                X = panel_data[[col for col in panel_data.columns if 'lag' in col]]

                # 添加常数项
                from linearmodels.panel.model import PanelModelData
                X = PanelModelData(X).add_constant()

                # 训练PanelOLS模型
                print("使用PanelOLS进行初步分析...")
                panel_model = PanelOLS(y, X)
                self.panel_results = panel_model.fit(cov_type='clustered', cluster_entity=True)

                print(f"面板数据分析完成，R-squared: {self.panel_results.rsquared}")

            except Exception as e:
                print(f"面板数据分析出错: {e}")
                print("继续使用传统EGARCH方法...")

        total_loss = 0
        valid_models = 0

        # 存储训练集上的条件方差预测
        volatility_predictions = []

        # 将金融工具分成批次进行处理，提高并行效率
        instrument_batches = [instruments[i:i + batch_size] for i in range(0, len(instruments), batch_size)]

        for batch in tqdm(instrument_batches, desc="训练EGARCH模型（批处理）"):
            batch_results = []

            # 使用并行处理批次内的金融工具
            try:
                from joblib import Parallel, delayed

                def process_instrument(instrument):
                    try:
                        instrument_data = x_train.xs(instrument, level=1)
                        instrument_data = instrument_data.sort_index()
                        returns = instrument_data[returns_col]
                        returns = returns.dropna()

                        if len(returns) <= max(self.p, self.q, self.o) + 10:
                            return None, None, None, []

                        # 如果有面板分析结果，可以用来初始化EGARCH模型参数
                        if self.panel_results is not None:
                            # 这里只是示例，实际上需要更复杂的逻辑
                            model = arch_model(returns, vol='EGARCH', p=self.p, q=self.q, o=self.o, mean=self.mean)
                        else:
                            model = arch_model(returns, vol='EGARCH', p=self.p, q=self.q, o=self.o, mean=self.mean)

                        model_fit = model.fit(disp='off')

                        # 获取训练集上的条件方差预测
                        conditional_variance = model_fit.conditional_volatility**2

                        # 创建预测结果
                        predictions = []
                        for date, variance in zip(conditional_variance.index, conditional_variance.values):
                            predictions.append({
                                'datetime': date,
                                'instrument': instrument,
                                'conditional_variance': variance
                            })

                        return instrument, model, model_fit, predictions

                    except Exception as e:
                        print(f"处理 {instrument} 时出错: {e}")
                        return None, None, None, []

                # 并行处理批次内的金融工具
                batch_results = Parallel(n_jobs=-1)(
                    delayed(process_instrument)(instrument) for instrument in batch
                )

            except ImportError:
                print("警告: 未安装joblib，使用串行处理...")

                # 串行处理批次内的金融工具
                for instrument in batch:
                    try:
                        instrument_data = x_train.xs(instrument, level=1)
                        instrument_data = instrument_data.sort_index()
                        returns = instrument_data[returns_col]
                        returns = returns.dropna()

                        if len(returns) <= max(self.p, self.q, self.o) + 10:
                            continue

                        model = arch_model(returns, vol='EGARCH', p=self.p, q=self.q, o=self.o, mean=self.mean)
                        model_fit = model.fit(disp='off')

                        # 获取训练集上的条件方差预测
                        conditional_variance = model_fit.conditional_volatility**2

                        # 创建预测结果
                        predictions = []
                        for date, variance in zip(conditional_variance.index, conditional_variance.values):
                            predictions.append({
                                'datetime': date,
                                'instrument': instrument,
                                'conditional_variance': variance
                            })

                        batch_results.append((instrument, model, model_fit, predictions))

                    except Exception as e:
                        print(f"处理 {instrument} 时出错: {e}")

            # 处理批次结果
            for instrument, model, model_fit, predictions in batch_results:
                if instrument is not None:
                    self.models[instrument] = model
                    self.model_fits[instrument] = model_fit

                    # 计算损失（负对数似然）
                    total_loss += model_fit.loglikelihood
                    valid_models += 1

                    # 添加预测结果
                    volatility_predictions.extend(predictions)

        if valid_models > 0:
            print(f"训练完成，平均对数似然: {total_loss / valid_models}")
        else:
            print("警告: 没有成功训练任何模型")

        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)

        # 设置多重索引
        if not vol_df.empty:
            vol_df = vol_df.set_index(['datetime', 'instrument'])

        return vol_df

    def predict_pandas(self, x: DataFrame, returns_col='pct_chg') -> DataFrame:
        """
        预测每个金融工具的条件方差

        参数:
        x: 包含收益率数据的DataFrame
        returns_col: 收益率列名

        返回:
        预测的条件方差DataFrame，索引与原始x相同
        """
        if not self.models:
            raise ValueError("模型尚未训练，请先调用fit方法")

        instruments = x.index.get_level_values(1).unique()
        volatility_predictions = []

        # 将金融工具分成批次进行处理
        instrument_batches = [instruments[i:i + 50] for i in range(0, len(instruments), 50)]

        for batch in tqdm(instrument_batches, desc="预测条件方差（批处理）"):
            batch_predictions = []

            # 使用并行处理批次内的金融工具
            try:
                from joblib import Parallel, delayed

                def process_instrument(instrument):
                    if instrument not in self.models:
                        return []

                    try:
                        # 提取该金融工具的数据
                        instrument_data = x.xs(instrument, level=1)
                        instrument_data = instrument_data.sort_index()

                        # 获取模型拟合结果
                        model_fit = self.model_fits[instrument]

                        # 获取条件方差预测
                        conditional_variance = model_fit.conditional_volatility**2

                        # 创建预测结果
                        predictions = []
                        for date, variance in zip(conditional_variance.index, conditional_variance.values):
                            predictions.append({
                                'datetime': date,
                                'instrument': instrument,
                                'conditional_variance': variance
                            })

                        # 如果需要预测未来的条件方差
                        if self.forecast_horizon > 0:
                            forecast = model_fit.forecast(horizon=self.forecast_horizon)

                            # 获取最后一个预测点
                            last_date = instrument_data.index[-1]

                            # 添加未来预测的条件方差
                            for h in range(1, self.forecast_horizon + 1):
                                future_variance = forecast.variance.iloc[-1, h-1]
                                # 这里假设日期是按天递增的，可以根据实际情况调整
                                future_date = pd.Timestamp(last_date) + pd.Timedelta(days=h)

                                predictions.append({
                                    'datetime': future_date,
                                    'instrument': instrument,
                                    'conditional_variance': future_variance,
                                    'is_forecast': True  # 标记为预测值
                                })

                        return predictions

                    except Exception as e:
                        print(f"预测 {instrument} 时出错: {e}")
                        return []

                # 并行处理批次内的金融工具
                batch_results = Parallel(n_jobs=-1)(
                    delayed(process_instrument)(instrument) for instrument in batch
                )

                # 合并批次结果
                for result in batch_results:
                    batch_predictions.extend(result)

            except ImportError:
                # 串行处理批次内的金融工具
                for instrument in batch:
                    if instrument not in self.models:
                        print(f"警告: {instrument} 没有对应的模型，跳过")
                        continue

                    try:
                        # 提取该金融工具的数据
                        instrument_data = x.xs(instrument, level=1)
                        instrument_data = instrument_data.sort_index()

                        # 获取模型拟合结果
                        model_fit = self.model_fits[instrument]

                        # 获取条件方差预测
                        conditional_variance = model_fit.conditional_volatility**2

                        # 创建预测结果
                        for date, variance in zip(conditional_variance.index, conditional_variance.values):
                            batch_predictions.append({
                                'datetime': date,
                                'instrument': instrument,
                                'conditional_variance': variance
                            })

                        # 如果需要预测未来的条件方差
                        if self.forecast_horizon > 0:
                            forecast = model_fit.forecast(horizon=self.forecast_horizon)

                            # 获取最后一个预测点
                            last_date = instrument_data.index[-1]

                            # 添加未来预测的条件方差
                            for h in range(1, self.forecast_horizon + 1):
                                future_variance = forecast.variance.iloc[-1, h-1]
                                # 这里假设日期是按天递增的，可以根据实际情况调整
                                future_date = pd.Timestamp(last_date) + pd.Timedelta(days=h)

                                batch_predictions.append({
                                    'datetime': future_date,
                                    'instrument': instrument,
                                    'conditional_variance': future_variance,
                                    'is_forecast': True  # 标记为预测值
                                })

                    except Exception as e:
                        print(f"预测 {instrument} 时出错: {e}")

            # 添加批次预测结果
            volatility_predictions.extend(batch_predictions)

        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)

        # 设置多重索引
        if not vol_df.empty:
            if 'is_forecast' in vol_df.columns:
                vol_df = vol_df.set_index(['datetime', 'instrument', 'is_forecast'])
            else:
                vol_df = vol_df.set_index(['datetime', 'instrument'])

        return vol_df

    def rolling_predict(self, x: DataFrame, returns_col='pct_chg', w=252, step=1, parallel=False, n_jobs=-1) -> DataFrame:
        """
        使用滚动窗口方法预测条件方差

        参数:
        x: 包含收益率数据的DataFrame
        returns_col: 收益率列名
        w: 滚动窗口大小，默认为252（约一年的交易日）
        step: 滚动步长，默认为1，可以设置更大的值来减少计算量
        parallel: 是否使用并行计算，默认为False
        n_jobs: 并行计算的作业数，默认为-1（使用所有可用CPU）

        返回:
        滚动预测的条件方差DataFrame
        """
        from arch import arch_model
        import warnings

        # 忽略ARCH模型的警告，提高效率
        warnings.filterwarnings("ignore", category=UserWarning)

        # 确保x是DataFrame
        if not isinstance(x, DataFrame):
            raise ValueError("x必须是DataFrame")

        # 获取所有唯一的金融工具
        instruments = x.index.get_level_values(1).unique()

        # 存储滚动预测结果
        volatility_predictions = []

        # 定义单个金融工具的处理函数
        def process_instrument(instrument):
            instrument_predictions = []
            try:
                # 提取该金融工具的数据
                instrument_data = x.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()

                # 获取收益率数据
                returns = instrument_data[returns_col]
                returns = returns.dropna()

                if len(returns) <= max(self.p, self.q, self.o) + 10 + w:
                    print(f"警告: {instrument} 的数据点不足，跳过")
                    return []

                # 获取所有日期
                dates = returns.index

                # 对每个滚动窗口进行预测，使用步长来减少计算量
                for i in range(w, len(dates), step):
                    # 获取当前窗口的数据
                    window_data = returns.iloc[i-w:i]
                    current_date = dates[i]

                    # 创建并拟合EGARCH模型
                    model = arch_model(window_data, vol='EGARCH', p=self.p, q=self.q, o=self.o, mean=self.mean)
                    model_fit = model.fit(disp='off', show_warning=False, options={'maxiter': 100})

                    # 获取最后一天的条件方差
                    conditional_variance = model_fit.conditional_volatility[-1]**2

                    # 存储预测结果
                    instrument_predictions.append({
                        'datetime': current_date,
                        'instrument': instrument,
                        'conditional_variance': conditional_variance
                    })

                    # 如果需要预测未来的条件方差
                    if self.forecast_horizon > 0:
                        forecast = model_fit.forecast(horizon=self.forecast_horizon)

                        # 添加未来预测的条件方差
                        for h in range(1, self.forecast_horizon + 1):
                            future_variance = forecast.variance.iloc[-1, h-1]
                            future_date = pd.Timestamp(current_date) + pd.Timedelta(days=h)

                            instrument_predictions.append({
                                'datetime': future_date,
                                'instrument': instrument,
                                'conditional_variance': future_variance,
                                'is_forecast': True
                            })

            except Exception as e:
                print(f"处理 {instrument} 时出错: {e}")

            return instrument_predictions

        # 使用并行计算处理多个金融工具
        if parallel:
            try:
                from joblib import Parallel, delayed

                # 并行处理所有金融工具
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_instrument)(instrument) for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})")
                )

                # 合并结果
                for result in results:
                    volatility_predictions.extend(result)

            except ImportError:
                print("警告: 未安装joblib，无法使用并行计算。请使用 'pip install joblib' 安装。")
                # 回退到串行处理
                for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})"):
                    volatility_predictions.extend(process_instrument(instrument))
        else:
            # 串行处理所有金融工具
            for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})"):
                volatility_predictions.extend(process_instrument(instrument))

        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)

        # 设置多重索引
        if not vol_df.empty:
            if 'is_forecast' in vol_df.columns:
                vol_df = vol_df.set_index(['datetime', 'instrument', 'is_forecast'])
            else:
                vol_df = vol_df.set_index(['datetime', 'instrument'])

        return vol_df


class HAR(Model):
    def __init__(self, lags=[1, 5, 20], forecast_horizon: int = 0, *args, **kwargs):
        """
        HAR (Heterogeneous Autoregressive) 模型用于波动率预测
        # 使用预先计算好的RV数据
        har_model = HAR(lags=[1, 5, 22])
        har_model.fit(data, rv_col='RV', use_precomputed_rv=True)
        predictions = har_model.predict_pandas(test_data, rv_col='RV', use_precomputed_rv=True)

        参数:
        lags: 滞后期列表，默认为[1, 5, 22]，分别代表日度、周度和月度
        forecast_horizon: 预测未来几天的波动率
        """
        super().__init__(*args, **kwargs)
        self.lags = lags
        self.forecast_horizon = forecast_horizon
        self.models = {}  # 存储每个金融工具的HAR模型参数
        self.panel_model = None  # 存储面板数据模型

    def init_model(self):
        # HAR模型不需要初始化PyTorch模型
        pass

    def _prepare_har_features(self, vol_series, use_precomputed_rv=True):
        """
        准备HAR模型的特征

        参数:
        vol_series: 波动率序列或收益率序列
        use_precomputed_rv: 是否使用预先计算好的RV，如果为True，则vol_series应该是RV序列

        返回:
        特征矩阵X和目标变量y
        """
        # 如果使用预先计算好的RV，直接使用输入序列
        if use_precomputed_rv:
            vol_proxy = vol_series
        else:
            # 计算波动率代理变量（原有逻辑）
            vol_proxy = vol_series**2

        # 计算不同滞后期的平均波动率
        X = pd.DataFrame(index=vol_proxy.index)

        for lag in self.lags:
            # 计算过去lag天的平均波动率
            X[f'vol_lag_{lag}'] = vol_proxy.rolling(window=lag).mean().shift(1)

        # 删除包含NaN的行
        X = X.dropna()

        # 准备目标变量
        y = vol_proxy.loc[X.index]

        return X, y

    def _prepare_panel_data(self, x_train, rv_col='RV', returns_col='pct_chg', use_precomputed_rv=True):
        """
        准备面板数据用于PanelOLS估计 - 优化版本，使用向量化操作代替循环

        参数:
        x_train: 训练数据，包含RV列或收益率列
        rv_col: RV列名，当use_precomputed_rv=True时使用
        returns_col: 收益率列名，当use_precomputed_rv=False时使用
        use_precomputed_rv: 是否使用预先计算好的RV

        返回:
        面板数据DataFrame，包含目标变量和特征
        """
        # 确保数据已排序
        x_train = x_train.sort_index()

        # 根据参数选择使用RV列或收益率列
        if use_precomputed_rv:
            vol_series = x_train[rv_col].copy()
        else:
            vol_series = x_train[returns_col].copy()

        # 过滤掉NaN值
        vol_series = vol_series.dropna()
        max_lag = max(self.lags)
        data_counts = vol_series.groupby(level=1).count()
        valid_instruments = data_counts[data_counts > max_lag + 10].index

        if len(valid_instruments) == 0:
            raise ValueError("没有足够的数据来构建面板数据集")

        valid_vol = vol_series[vol_series.index.get_level_values(1).isin(valid_instruments)]

        panel_data = pd.DataFrame()
        if use_precomputed_rv:
            vol_proxy = valid_vol
        else:
            vol_proxy = valid_vol**2

        panel_data['y'] = vol_proxy

        for lag in self.lags:
            # 按金融工具分组计算滚动平均
            panel_data[f'vol_lag_{lag}'] = vol_proxy.groupby(level=1).transform(
                lambda x: x.rolling(window=lag).mean().shift(1)
            )

        panel_data = panel_data.dropna()

        return panel_data

    def fit(self, x_train, rv_col='RV', returns_col='pct_chg', use_precomputed_rv=True):
        """
        使用PanelOLS训练HAR模型并返回训练集上的条件方差预测

        参数:
        x_train: 训练数据，包含RV列或收益率列
        rv_col: RV列名，当use_precomputed_rv=True时使用
        returns_col: 收益率列名，当use_precomputed_rv=False时使用
        use_precomputed_rv: 是否使用预先计算好的RV，默认为True

        返回:
        训练集上的条件方差预测DataFrame
        """

        # 确保x_train是DataFrame
        if not isinstance(x_train, DataFrame):
            raise ValueError("x_train必须是DataFrame")

        try:
            # 准备面板数据
            print("准备面板数据...")
            panel_data = self._prepare_panel_data(x_train, rv_col, returns_col, use_precomputed_rv)

            # 提取特征和目标变量
            y = panel_data['y']
            X = panel_data.drop('y', axis=1)

            # 添加常数项
            from linearmodels.panel.model import PanelModelData
            X = PanelModelData(X).add_constant()

            # 训练PanelOLS模型
            print("训练PanelOLS模型...")
            self.panel_model = PanelOLS(y, X)
            panel_results = self.panel_model.fit(cov_type='clustered', cluster_entity=True)

            print(f"模型训练完成，R-squared: {panel_results.rsquared}")

            # 存储模型参数
            self.model_params = {
                'coefficients': panel_results.params,
                'feature_names': list(panel_results.params.index)
            }

            # 预测条件方差
            y_pred = panel_results.predict()

            # 创建预测结果DataFrame
            vol_df = pd.DataFrame(y_pred, columns=['conditional_variance'])

            return vol_df

        except Exception as e:
            print(f"训练PanelOLS模型时出错: {e}")
            # 如果PanelOLS失败，回退到原始方法
            print("回退到原始HAR实现...")
            return self._fit_original(x_train, rv_col, returns_col, use_precomputed_rv)

    def _fit_original(self, x_train, rv_col='RV', returns_col='pct_chg', use_precomputed_rv=True):
        """原始的HAR模型训练方法，作为备选"""
        from sklearn.linear_model import LinearRegression

        # 获取所有唯一的金融工具
        instruments = x_train.index.get_level_values(1).unique()

        # 存储训练集上的条件方差预测
        volatility_predictions = []

        # 对每个金融工具单独建模
        for instrument in tqdm(instruments, desc="训练HAR模型"):
            try:
                # 提取该金融工具的数据
                instrument_data = x_train.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()

                # 获取波动率或收益率数据
                if use_precomputed_rv:
                    vol_series = instrument_data[rv_col]
                else:
                    vol_series = instrument_data[returns_col]

                vol_series = vol_series.dropna()

                # 最大滞后期
                max_lag = max(self.lags)

                if len(vol_series) <= max_lag + 10:
                    print(f"警告: {instrument} 的数据点不足，跳过")
                    continue

                # 准备特征和目标变量
                X, y = self._prepare_har_features(vol_series, use_precomputed_rv)

                # 训练线性回归模型
                model = LinearRegression()
                model.fit(X, y)

                # 存储模型参数
                self.models[instrument] = {
                    'intercept': model.intercept_,
                    'coefficients': model.coef_,
                    'feature_names': X.columns.tolist()
                }

                # 预测条件方差
                y_pred = model.predict(X)

                # 创建预测结果
                for date, variance in zip(X.index, y_pred):
                    volatility_predictions.append({
                        'datetime': date,
                        'instrument': instrument,
                        'conditional_variance': variance
                    })

            except Exception as e:
                print(f"处理 {instrument} 时出错: {e}")

        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)

        # 设置多重索引
        if not vol_df.empty:
            vol_df = vol_df.set_index(['datetime', 'instrument'])

        return vol_df

    def predict_pandas(self, x: DataFrame, rv_col='RV', returns_col='pct_chg', use_precomputed_rv=True) -> DataFrame:
        """
        使用训练好的模型预测每个金融工具的条件方差

        参数:
        x: 包含RV数据或收益率数据的DataFrame
        rv_col: RV列名，当use_precomputed_rv=True时使用
        returns_col: 收益率列名，当use_precomputed_rv=False时使用
        use_precomputed_rv: 是否使用预先计算好的RV，默认为True

        返回:
        预测的条件方差DataFrame
        """
        from linearmodels.panel import PanelOLS

        # 如果使用了PanelOLS模型
        if hasattr(self, 'panel_model') and self.panel_model is not None:
            try:
                # 准备面板数据
                panel_data = self._prepare_panel_data(x, rv_col, returns_col, use_precomputed_rv)

                # 提取特征
                X = panel_data.drop('y', axis=1)

                # 添加常数项
                from linearmodels.panel.model import PanelModelData
                X = PanelModelData(X).add_constant()

                # 使用模型参数进行预测
                coeffs = self.model_params['coefficients']

                # 创建预测结果
                predictions = pd.DataFrame(index=X.index)

                # 计算预测值
                predictions['conditional_variance'] = 0
                for feature in coeffs.index:
                    if feature in X.columns:
                        predictions['conditional_variance'] += X[feature] * coeffs[feature]

                return predictions

            except Exception as e:
                print(f"使用PanelOLS预测时出错: {e}")
                print("回退到原始预测方法...")

        # 回退到原始方法
        return self._predict_original(x, rv_col, returns_col, use_precomputed_rv)

    def _predict_original(self, x: DataFrame, rv_col='RV', returns_col='pct_chg', use_precomputed_rv=True) -> DataFrame:
        """原始的HAR模型预测方法，作为备选"""
        if not self.models:
            raise ValueError("模型尚未训练，请先调用fit方法")

        instruments = x.index.get_level_values(1).unique()
        volatility_predictions = []

        for instrument in tqdm(instruments, desc="预测条件方差"):
            if instrument not in self.models:
                print(f"警告: {instrument} 没有对应的模型，跳过")
                continue

            try:
                # 提取该金融工具的数据
                instrument_data = x.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()

                # 获取波动率或收益率数据
                if use_precomputed_rv:
                    vol_series = instrument_data[rv_col]
                else:
                    vol_series = instrument_data[returns_col]

                vol_series = vol_series.dropna()

                # 准备特征
                X, _ = self._prepare_har_features(vol_series, use_precomputed_rv)

                # 获取模型参数
                model_params = self.models[instrument]
                intercept = model_params['intercept']
                coefficients = model_params['coefficients']

                # 预测条件方差
                y_pred = intercept + np.dot(X, coefficients)

                # 创建预测结果
                for date, variance in zip(X.index, y_pred):
                    volatility_predictions.append({
                        'datetime': date,
                        'instrument': instrument,
                        'conditional_variance': variance
                    })

                # 如果需要预测未来的条件方差
                if self.forecast_horizon > 0:
                    # 获取最后一个预测点的特征
                    last_features = X.iloc[-1].values
                    last_date = X.index[-1]

                    # 预测未来的条件方差
                    for h in range(1, self.forecast_horizon + 1):
                        # 简单地使用最后一个预测作为未来预测
                        future_variance = intercept + np.dot(last_features, coefficients)
                        future_date = pd.Timestamp(last_date) + pd.Timedelta(days=h)

                        volatility_predictions.append({
                            'datetime': future_date,
                            'instrument': instrument,
                            'conditional_variance': future_variance,
                            'is_forecast': True  # 标记为预测值
                        })

            except Exception as e:
                print(f"预测 {instrument} 时出错: {e}")

        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)

        # 设置多重索引
        if not vol_df.empty:
            if 'is_forecast' in vol_df.columns:
                vol_df = vol_df.set_index(['datetime', 'instrument', 'is_forecast'])
            else:
                vol_df = vol_df.set_index(['datetime', 'instrument'])

        return vol_df

    def rolling_predict(self, x: DataFrame, returns_col='pct_chg', volatility_proxy='RV',
                        w=252, step=1, parallel=False, n_jobs=-1) -> DataFrame:
        """
        使用滚动窗口方法预测条件方差

        参数:
        x: 包含收益率数据的DataFrame
        returns_col: 收益率列名
        volatility_proxy: 波动率代理变量
        w: 滚动窗口大小，默认为252（约一年的交易日）
        step: 滚动步长，默认为1，可以设置更大的值来减少计算量
        parallel: 是否使用并行计算，默认为False
        n_jobs: 并行计算的作业数，默认为-1（使用所有可用CPU）

        返回:
        滚动预测的条件方差DataFrame
        """
        from linearmodels.panel import PanelOLS

        # 确保x是DataFrame
        if not isinstance(x, DataFrame):
            raise ValueError("x必须是DataFrame")

        # 获取所有唯一的金融工具和日期
        instruments = x.index.get_level_values(1).unique()
        dates = x.index.get_level_values(0).unique()

        # 如果日期数量不足以进行滚动预测，则回退到原始方法
        if len(dates) <= w + max(self.lags):
            print("日期数量不足以使用PanelOLS进行滚动预测，回退到原始方法...")
            return self._rolling_predict_original(x, returns_col, volatility_proxy, w, step, parallel, n_jobs)

        # 存储滚动预测结果
        volatility_predictions = []

        try:
            # 对每个滚动窗口进行预测
            for i in range(w, len(dates), step):
                # 获取当前窗口的数据
                window_start = dates[i-w]
                window_end = dates[i]
                window_data = x.loc[(slice(window_start, window_end), slice(None)), :]

                # 训练当前窗口的模型
                self.fit(window_data, returns_col, volatility_proxy)

                # 获取当前日期的数据
                current_date = dates[i]
                current_data = x.loc[(current_date, slice(None)), :]

                # 预测当前日期的条件方差
                predictions = self.predict_pandas(current_data, returns_col, volatility_proxy)

                # 添加到结果中
                volatility_predictions.append(predictions)

            # 合并所有预测结果
            vol_df = pd.concat(volatility_predictions)

            return vol_df

        except Exception as e:
            print(f"使用PanelOLS进行滚动预测时出错: {e}")
            print("回退到原始滚动预测方法...")
            return self._rolling_predict_original(x, returns_col, volatility_proxy, w, step, parallel, n_jobs)

    def _rolling_predict_original(self, x: DataFrame, returns_col='pct_chg', volatility_proxy='RV',
                                w=252, step=1, parallel=False, n_jobs=-1) -> DataFrame:
        """原始的HAR模型滚动预测方法，作为备选"""
        from sklearn.linear_model import LinearRegression

        # 获取所有唯一的金融工具
        instruments = x.index.get_level_values(1).unique()

        # 存储滚动预测结果
        volatility_predictions = []

        # 定义单个金融工具的处理函数
        def process_instrument(instrument):
            instrument_predictions = []
            try:
                # 提取该金融工具的数据
                instrument_data = x.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()

                # 获取收益率数据
                returns = instrument_data[returns_col]
                returns = returns.dropna()

                # 最大滞后期
                max_lag = max(self.lags)

                if len(returns) <= max_lag + 10 + w:
                    print(f"警告: {instrument} 的数据点不足，跳过")
                    return []

                # 获取所有日期
                all_dates = returns.index

                # 对每个滚动窗口进行预测，使用步长来减少计算量
                for i in range(w + max_lag, len(all_dates), step):
                    # 获取当前窗口的数据
                    window_returns = returns.iloc[i-w:i]
                    current_date = all_dates[i]

                    # 准备特征和目标变量
                    X, y = self._prepare_har_features(window_returns, volatility_proxy)

                    # 只使用窗口内的数据
                    X = X.iloc[-w:]
                    y = y.iloc[-w:]

                    # 训练线性回归模型
                    model = LinearRegression()
                    model.fit(X, y)

                    # 预测最后一天的条件方差
                    last_features = X.iloc[-1].values.reshape(1, -1)
                    conditional_variance = model.predict(last_features)[0]

                    # 存储预测结果
                    instrument_predictions.append({
                        'datetime': current_date,
                        'instrument': instrument,
                        'conditional_variance': conditional_variance
                    })

                    # 如果需要预测未来的条件方差
                    if self.forecast_horizon > 0:
                        # 简单地使用最后一个预测作为未来预测
                        for h in range(1, self.forecast_horizon + 1):
                            future_variance = conditional_variance  # 简化处理
                            future_date = pd.Timestamp(current_date) + pd.Timedelta(days=h)

                            instrument_predictions.append({
                                'datetime': future_date,
                                'instrument': instrument,
                                'conditional_variance': future_variance,
                                'is_forecast': True
                            })

            except Exception as e:
                print(f"处理 {instrument} 时出错: {e}")

            return instrument_predictions

        # 使用并行计算处理多个金融工具
        if parallel:
            try:
                from joblib import Parallel, delayed

                # 并行处理所有金融工具
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_instrument)(instrument) for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})")
                )

                # 合并结果
                for result in results:
                    volatility_predictions.extend(result)

            except ImportError:
                print("警告: 未安装joblib，无法使用并行计算。请使用 'pip install joblib' 安装。")
                # 回退到串行处理
                for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})"):
                    volatility_predictions.extend(process_instrument(instrument))
        else:
            # 串行处理所有金融工具
            for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})"):
                volatility_predictions.extend(process_instrument(instrument))

        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)

        # 设置多重索引
        if not vol_df.empty:
            if 'is_forecast' in vol_df.columns:
                vol_df = vol_df.set_index(['datetime', 'instrument', 'is_forecast'])
            else:
                vol_df = vol_df.set_index(['datetime', 'instrument'])

        return vol_df


class ARFIMA(Model):
    def __init__(self, p: int = 1, d: float = 0.5, q: int = 1, mean: str = 'Zero', forecast_horizon: int = 0, *args, **kwargs):
        """
        ARFIMA模型用于长记忆时间序列预测
        参数:
        p: 自回归阶数
        d: 差分阶数（支持分数阶差分）
        q: 移动平均阶数
        mean: 均值模型，可选'Zero'、'Constant'等
        forecast_horizon: 预测步长
        """
        super().__init__(*args, **kwargs)
        self.p = p
        self.d = d
        self.q = q
        self.mean = mean
        self.forecast_horizon = forecast_horizon
        self.models = {}  # 存储每个序列的ARFIMA模型
        self.model_fits = {}  # 存储拟合结果
        self.panel_results = None  # 保持与GARCH类结构一致

    def init_model(self):
        pass

    def _prepare_panel_data(self, x_train, series_col='pct_chg'):
        """准备面板数据（保持与GARCH类结构一致）"""
        x_train = x_train.sort_index()
        series = x_train[series_col].copy().dropna()
        min_data_points = max(self.p, self.q) + 10
        data_counts = series.groupby(level=1).count()
        valid_instruments = data_counts[data_counts > min_data_points].index

        if len(valid_instruments) == 0:
            raise ValueError("没有足够的数据来构建面板数据集")

        return series[series.index.get_level_values(1).isin(valid_instruments)]

    def fit(self, x_train, series_col='pct_chg', use_panel=False, batch_size=50):
        """训练ARFIMA模型"""
        from statsmodels.tsa.arima.model import ARIMA
        from pmdarima import auto_arima  # 需要安装pmdarima

        instruments = x_train.index.get_level_values(1).unique()
        predictions = []

        instrument_batches = [instruments[i:i + batch_size] for i in range(0, len(instruments), batch_size)]

        for batch in tqdm(instrument_batches, desc="训练ARFIMA模型（批处理）"):
            batch_results = []

            try:
                from joblib import Parallel, delayed

                def process_instrument(instrument):
                    try:
                        instrument_data = x_train.xs(instrument, level=1)
                        series = instrument_data[series_col].dropna()

                        if len(series) < max(self.p, self.q)*10 + 10:
                            return None, None, []

                        # 自动选择最优参数（如果允许）
                        model = auto_arima(series, d=self.d, max_p=self.p, max_q=self.q,
                                          seasonal=False, trace=False, error_action='ignore')

                        # 存储模型和拟合结果
                        fitted_values = model.predict_in_sample()

                        preds = [{
                            'datetime': date,
                            'instrument': instrument,
                            'fitted_value': value
                        } for date, value in zip(series.index, fitted_values)]

                        return instrument, model, preds

                    except Exception as e:
                        print(f"处理 {instrument} 出错: {e}")
                        return None, None, []

                batch_results = Parallel(n_jobs=-1)(
                    delayed(process_instrument)(inst) for inst in batch
                )

            except ImportError:
                for instrument in batch:
                    try:
                        instrument_data = x_train.xs(instrument, level=1)
                        series = instrument_data[series_col].dropna()

                        if len(series) < max(self.p, self.q)*10 + 10:
                            continue

                        model = auto_arima(series, d=self.d, max_p=self.p, max_q=self.q,
                                          seasonal=False, trace=False)

                        fitted_values = model.predict_in_sample()

                        preds = [{
                            'datetime': date,
                            'instrument': instrument,
                            'fitted_value': value
                        } for date, value in zip(series.index, fitted_values)]

                        batch_results.append((instrument, model, preds))

                    except Exception as e:
                        print(f"处理 {instrument} 出错: {e}")

            # 存储结果
            for instrument, model, preds in batch_results:
                if instrument is not None:
                    self.models[instrument] = model
                    predictions.extend(preds)

        # 返回训练集预测结果
        pred_df = pd.DataFrame(predictions)
        if not pred_df.empty:
            pred_df = pred_df.set_index(['datetime', 'instrument'])
        return pred_df

    def predict_pandas(self, x: pd.DataFrame, series_col='pct_chg') -> pd.DataFrame:
        """生成预测"""
        if not self.models:
            raise ValueError("模型尚未训练，请先调用fit方法")

        predictions = []
        instruments = x.index.get_level_values(1).unique()

        for instrument in instruments:
            if instrument not in self.models:
                continue

            try:
                model = self.models[instrument]
                instrument_data = x.xs(instrument, level=1)
                series = instrument_data[series_col]

                # 生成预测
                forecast = model.predict(n_periods=self.forecast_horizon)

                # 获取最后一个日期
                last_date = instrument_data.index[-1]

                for i, value in enumerate(forecast):
                    pred_date = last_date + pd.Timedelta(days=i+1)
                    predictions.append({
                        'datetime': pred_date,
                        'instrument': instrument,
                        'forecast_value': value,
                        'is_forecast': True
                    })

            except Exception as e:
                print(f"预测 {instrument} 时出错: {e}")

        pred_df = pd.DataFrame(predictions)
        if not pred_df.empty:
            return pred_df.set_index(['datetime', 'instrument', 'is_forecast'])
        return pred_df

    def rolling_predict(self, x: pd.DataFrame, series_col='pct_chg',
                       w=252, step=1, parallel=False, n_jobs=-1) -> pd.DataFrame:
        """滚动窗口预测"""
        predictions = []
        instruments = x.index.get_level_values(1).unique()

        def process_instrument(instrument):
            instrument_predictions = []
            try:
                data = x.xs(instrument, level=1)[series_col].dropna()
                dates = data.index

                for i in range(w, len(dates), step):
                    train_data = data.iloc[i-w:i]

                    # 训练模型
                    model = auto_arima(train_data, d=self.d, max_p=self.p, max_q=self.q,
                                      seasonal=False, trace=False)

                    # 预测
                    forecast = model.predict(n_periods=self.forecast_horizon)

                    for h, value in enumerate(forecast):
                        pred_date = dates[i] + pd.Timedelta(days=h+1)
                        instrument_predictions.append({
                            'datetime': pred_date,
                            'instrument': instrument,
                            'forecast_value': value,
                            'is_forecast': True
                        })

            except Exception as e:
                print(f"处理 {instrument} 出错: {e}")
            return instrument_predictions

        # 并行处理
        if parallel:
            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_instrument)(inst) for inst in instruments
                )
                for res in results:
                    predictions.extend(res)
            except ImportError:
                print("未安装joblib，使用串行处理")
                for inst in instruments:
                    predictions.extend(process_instrument(inst))
        else:
            for inst in instruments:
                predictions.extend(process_instrument(inst))

        return pd.DataFrame(predictions).set_index(['datetime', 'instrument', 'is_forecast'])


def adf_test(series, title='', figsize=(12, 8), dpi=100, save_path=None, group_by_instrument=True):
    """
    对时间序列进行ADF检验并绘制相关图表

    参数:
    series: pandas Series，需要检验的时间序列
    title: 图表标题，默认为空
    figsize: 图表大小，默认为(12, 8)
    dpi: 图表分辨率，默认为100
    save_path: 图表保存路径，默认为None（不保存）
    group_by_instrument: 是否按金融工具分组进行检验，默认为True

    返回:
    dict: 包含ADF检验结果的字典
    """
    from statsmodels.tsa.stattools import adfuller
    import matplotlib.pyplot as plt
    import numpy as np
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # 确保输入是pandas Series
    if not isinstance(series, pd.Series):
        if isinstance(series, pd.DataFrame) and series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            raise ValueError("输入必须是pandas Series或单列DataFrame")

    # 检查是否有多重索引
    has_multiindex = isinstance(series.index, pd.MultiIndex)

    # 如果有多重索引且需要按金融工具分组
    if has_multiindex and group_by_instrument and 'instrument' in series.index.names:
        # 获取所有唯一的金融工具
        instruments = series.index.get_level_values('instrument').unique()

        # 存储所有金融工具的ADF检验结果
        all_results = {}

        # 对每个金融工具进行ADF检验
        for instrument in instruments:
            print(f"\n分析金融工具: {instrument}")
            # 提取该金融工具的数据
            instrument_series = series.xs(instrument, level='instrument')

            # 确保数据按时间排序
            instrument_series = instrument_series.sort_index()

            # 去除NaN值
            instrument_series = instrument_series.dropna()

            if len(instrument_series) < 20:  # 至少需要20个数据点才能进行有意义的检验
                print(f"警告: {instrument} 的数据点不足，跳过")
                continue

            # 进行ADF检验
            result = adfuller(instrument_series.values)

            # 提取结果
            adf_stat = result[0]
            p_value = result[1]
            critical_values = result[4]

            # 创建结果字典
            adf_result = {
                'ADF统计量': adf_stat,
                'p值': p_value,
                '临界值': critical_values,
                '样本数': len(instrument_series),
                '是否平稳': p_value < 0.05
            }

            # 打印结果
            print('ADF检验结果:')
            print(f'ADF统计量: {adf_stat:.4f}')
            print(f'p值: {p_value:.4f}')
            print('临界值:')
            for key, value in critical_values.items():
                print(f'  {key}: {value:.4f}')

            if p_value < 0.05:
                print('结论: 序列是平稳的 (拒绝单位根假设)')
            else:
                print('结论: 序列不是平稳的 (无法拒绝单位根假设)')

            # 绘制图表
            instrument_title = f"{title} - {instrument}" if title else instrument
            fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi)

            # 绘制原始时间序列
            axes[0].plot(instrument_series, color='blue')
            axes[0].set_title(f'Time Series - {instrument_title}')
            axes[0].grid(True)

            # 绘制自相关函数(ACF)
            plot_acf(instrument_series, ax=axes[1], lags=min(40, len(instrument_series)//2))
            axes[1].set_title('ACF')
            axes[1].grid(True)

            # 绘制偏自相关函数(PACF)
            plot_pacf(instrument_series, ax=axes[2], lags=min(40, len(instrument_series)//2))
            axes[2].set_title('PACF')
            axes[2].grid(True)

            plt.tight_layout()

            # 保存图表
            if save_path:
                # 为每个金融工具创建单独的文件名
                instrument_save_path = save_path.replace('.', f'_{instrument}.')
                plt.savefig(instrument_save_path, dpi=dpi, bbox_inches='tight')

            plt.show()

            # 存储结果
            all_results[instrument] = adf_result

        return all_results

    else:
        # 原始的ADF检验逻辑，用于单一时间序列
        # 去除NaN值
        series = series.dropna()

        # 进行ADF检验
        result = adfuller(series.values)

        # 提取结果
        adf_stat = result[0]
        p_value = result[1]
        critical_values = result[4]

        # 创建结果字典
        adf_result = {
            'ADF统计量': adf_stat,
            'p值': p_value,
            '临界值': critical_values,
            '样本数': len(series),
            '是否平稳': p_value < 0.05
        }

        # 打印结果
        print('ADF检验结果:')
        print(f'ADF统计量: {adf_stat:.4f}')
        print(f'p值: {p_value:.4f}')
        print('临界值:')
        for key, value in critical_values.items():
            print(f'  {key}: {value:.4f}')

        if p_value < 0.05:
            print('结论: 序列是平稳的 (拒绝单位根假设)')
        else:
            print('结论: 序列不是平稳的 (无法拒绝单位根假设)')

        # 绘制图表
        fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi)

        # 绘制原始时间序列
        axes[0].plot(series, color='blue')
        axes[0].set_title(f'时间序列 {"- " + title if title else ""}')
        axes[0].grid(True)

        # 绘制自相关函数(ACF)
        plot_acf(series, ax=axes[1], lags=min(40, len(series)//2))
        axes[1].set_title('自相关函数(ACF)')
        axes[1].grid(True)

        # 绘制偏自相关函数(PACF)
        plot_pacf(series, ax=axes[2], lags=min(40, len(series)//2))
        axes[2].set_title('偏自相关函数(PACF)')
        axes[2].grid(True)

        plt.tight_layout()

        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

        plt.show()

        return adf_result

def adf_test_diff(series, max_diff=3, title='', figsize=(12, 8), dpi=100, save_path=None, group_by_instrument=True):
    """
    对时间序列进行差分并进行ADF检验，直到序列平稳或达到最大差分次数

    参数:
    series: pandas Series，需要检验的时间序列
    max_diff: 最大差分次数，默认为3
    title: 图表标题，默认为空
    figsize: 图表大小，默认为(12, 8)
    dpi: 图表分辨率，默认为100
    save_path: 图表保存路径，默认为None（不保存）
    group_by_instrument: 是否按金融工具分组进行检验，默认为True

    返回:
    dict: 包含ADF检验结果的字典
    """
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import adfuller

    # 确保输入是pandas Series
    if not isinstance(series, pd.Series):
        if isinstance(series, pd.DataFrame) and series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            raise ValueError("输入必须是pandas Series或单列DataFrame")

    # 检查是否有多重索引
    has_multiindex = isinstance(series.index, pd.MultiIndex)

    # 如果有多重索引且需要按金融工具分组
    if has_multiindex and group_by_instrument and 'instrument' in series.index.names:
        # 获取所有唯一的金融工具
        instruments = series.index.get_level_values('instrument').unique()

        # 存储所有金融工具的ADF检验结果
        all_results = {}

        # 对每个金融工具进行ADF检验
        for instrument in instruments:
            print(f"\n分析金融工具: {instrument}")
            # 提取该金融工具的数据
            instrument_series = series.xs(instrument, level='instrument')

            # 确保数据按时间排序
            instrument_series = instrument_series.sort_index()

            # 去除NaN值
            instrument_series = instrument_series.dropna()

            if len(instrument_series) < 20:  # 至少需要20个数据点才能进行有意义的检验
                print(f"警告: {instrument} 的数据点不足，跳过")
                continue

            # 创建图表
            instrument_title = f"{title} - {instrument}" if title else instrument
            fig, axes = plt.subplots(max_diff + 1, 2, figsize=figsize, dpi=dpi)

            # 对原始序列进行ADF检验
            result = adfuller(instrument_series.values)
            p_value = result[1]

            # 绘制原始序列
            axes[0, 0].plot(instrument_series, color='blue')
            axes[0, 0].set_title(f'Original Series - {instrument_title}')
            axes[0, 0].grid(True)

            # 绘制原始序列的直方图
            axes[0, 1].hist(instrument_series, bins=30, color='blue', alpha=0.7)
            axes[0, 1].set_title(f'Original Series Histogram (p-value: {p_value:.4f})')
            axes[0, 1].grid(True)

            # 如果原始序列已经平稳，则不需要差分
            if p_value < 0.05:
                print(f'原始序列已经是平稳的 (p值: {p_value:.4f})')
                plt.tight_layout()

                # 保存图表
                if save_path:
                    # 为每个金融工具创建单独的文件名
                    instrument_save_path = save_path.replace('.', f'_{instrument}.')
                    plt.savefig(instrument_save_path, dpi=dpi, bbox_inches='tight')

                plt.show()

                all_results[instrument] = {'差分次数': 0, 'p值': p_value, '是否平稳': True}
                continue

            # 进行差分并检验
            diff_series = instrument_series
            diff_count = 0

            for i in range(max_diff):
                diff_count += 1
                diff_series = diff_series.diff().dropna()

                # 进行ADF检验
                result = adfuller(diff_series.values)
                p_value = result[1]

                # 绘制差分序列
                axes[diff_count, 0].plot(diff_series, color='green')
                axes[diff_count, 0].set_title(f'{diff_count}st Order Difference Series')
                axes[diff_count, 0].grid(True)

                # 绘制差分序列的直方图
                axes[diff_count, 1].hist(diff_series, bins=30, color='green', alpha=0.7)
                axes[diff_count, 1].set_title(f'{diff_count}st Order Difference Histogram (p-value: {p_value:.4f})')
                axes[diff_count, 1].grid(True)

                # 如果序列平稳，则停止差分
                if p_value < 0.05:
                    print(f'序列在{diff_count}阶差分后平稳 (p值: {p_value:.4f})')
                    break

            # 如果达到最大差分次数仍不平稳
            if p_value >= 0.05:
                print(f'警告: 序列在{max_diff}阶差分后仍不平稳 (p值: {p_value:.4f})')

            plt.tight_layout()

            # 保存图表
            if save_path:
                # 为每个金融工具创建单独的文件名
                instrument_save_path = save_path.replace('.', f'_{instrument}.')
                plt.savefig(instrument_save_path, dpi=dpi, bbox_inches='tight')

            plt.show()

            all_results[instrument] = {'差分次数': diff_count, 'p值': p_value, '是否平稳': p_value < 0.05}

        return all_results

    else:
        # 原始的ADF差分检验逻辑，用于单一时间序列
        # 去除NaN值
        series = series.dropna()

        # 创建图表
        fig, axes = plt.subplots(max_diff + 1, 2, figsize=figsize, dpi=dpi)

        # 对原始序列进行ADF检验
        result = adfuller(series.values)
        p_value = result[1]

        # 绘制原始序列
        axes[0, 0].plot(series, color='blue')
        axes[0, 0].set_title(f'Original Series {"- " + title if title else ""}')
        axes[0, 0].grid(True)

        # 绘制原始序列的直方图
        axes[0, 1].hist(series, bins=30, color='blue', alpha=0.7)
        axes[0, 1].set_title(f'Original Series Histogram (p-value: {p_value:.4f})')
        axes[0, 1].grid(True)

        # 如果原始序列已经平稳，则不需要差分
        if p_value < 0.05:
            print(f'原始序列已经是平稳的 (p值: {p_value:.4f})')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.show()
            return {'差分次数': 0, 'p值': p_value, '是否平稳': True}

        # 进行差分并检验
        diff_series = series
        diff_count = 0

        for i in range(max_diff):
            diff_count += 1
            diff_series = diff_series.diff().dropna()

            # 进行ADF检验
            result = adfuller(diff_series.values)
            p_value = result[1]

            # 绘制差分序列
            axes[diff_count, 0].plot(diff_series, color='green')
            # 根据差分次数选择正确的序数词后缀
            if diff_count == 1:
                suffix = "st"
            elif diff_count == 2:
                suffix = "nd"
            elif diff_count == 3:
                suffix = "rd"
            else:
                suffix = "th"

            axes[diff_count, 0].set_title(f'{diff_count}{suffix} Order Difference Series')
            axes[diff_count, 0].grid(True)

            # 绘制差分序列的直方图
            axes[diff_count, 1].hist(diff_series, bins=30, color='green', alpha=0.7)
            axes[diff_count, 1].set_title(f'{diff_count}{suffix} Order Difference Histogram (p-value: {p_value:.4f})')
            axes[diff_count, 1].grid(True)

            # 如果序列平稳，则停止差分
            if p_value < 0.05:
                print(f'序列在{diff_count}阶差分后平稳 (p值: {p_value:.4f})')
                break

        # 如果达到最大差分次数仍不平稳
        if p_value >= 0.05:
            print(f'警告: 序列在{max_diff}阶差分后仍不平稳 (p值: {p_value:.4f})')

        plt.tight_layout()

        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

        plt.show()

        return {'差分次数': diff_count, 'p值': p_value, '是否平稳': p_value < 0.05}
