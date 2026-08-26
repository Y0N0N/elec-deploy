import numpy as np
import pandas as pd
from pandas import Series, DataFrame, concat, Grouper

import torch
from torch import Tensor
from typing import Union


# 构建图结构
def calc_tensor_corr(x: Tensor, y: Tensor):
    if x.shape != y.shape:
        raise ValueError("The shapes of x and y must be the same.")
    mask = ~torch.isnan(y)
    mean_x = torch.mean(x[mask])
    mean_y = torch.mean(y[mask])
    std_x = torch.std(x[mask])
    std_y = torch.std(y[mask])
    return torch.mean((x[mask] - mean_x) * (y[mask] - mean_y)) / (std_x * std_y)


def make_corr_matrix(x: Series, threshold: float = 0.5, shift: int = 0):
    if shift > 0:
        x = x.groupby(level=1).shift(shift).fillna(0)
    corr_matrix = x.unstack().corr().fillna(0)
    corr_matrix[abs(corr_matrix) < threshold] = 0
    corr_matrix[corr_matrix != 0] = 1
    return corr_matrix


def make_binary_corr_matrix_tensor(y_tensor: Tensor, threshold: float = 0.5) -> Tensor:
    """
    为PyTorch张量生成相关系数矩阵

    参数:
    y_tensor: 输入张量，形状为[T, N]
                 T: 时间序列长度
                 N: 股票数量
    threshold: 相关系数阈值，绝对值小于该阈值的相关系数将被置为0

    返回:
    corr_matrix: 相关系数矩阵，形状为[N, N]，值为0或1
    """
    # 确保输入是二维张量
    assert len(y_tensor.shape) == 2, f"输入张量应为2维，当前形状为{y_tensor.shape}"

    # 标准化：减去均值
    y_mean = torch.mean(y_tensor, dim=0, keepdim=True)  # [1, N]
    y_centered = y_tensor - y_mean  # [T, N]

    # 计算标准差
    y_std = torch.std(y_tensor, dim=0, keepdim=True)  # [1, N]
    # 防止除零错误
    y_std = torch.where(y_std == 0, torch.ones_like(y_std), y_std)

    # 标准化
    y_normalized = y_centered / y_std  # [T, N]

    # 计算相关系数矩阵: (1/T) * X^T * X
    T = y_tensor.shape[0]
    corr_matrix = torch.matmul(y_normalized.transpose(0, 1), y_normalized) / T  # [N, N]

    # 填充NaN值为0
    corr_matrix = torch.where(torch.isnan(corr_matrix), torch.zeros_like(corr_matrix), corr_matrix)

    # 应用阈值
    corr_matrix_abs = torch.abs(corr_matrix)
    corr_matrix_thresholded = torch.where(corr_matrix_abs < threshold, torch.zeros_like(corr_matrix), corr_matrix)

    # 将非零值设为1
    corr_matrix_binary = torch.where(corr_matrix_thresholded != 0, torch.ones_like(corr_matrix_thresholded),
                                     torch.zeros_like(corr_matrix_thresholded))

    return corr_matrix_binary


def make_corr_matrix_tensor(y_tensor: Tensor) -> Tensor:
    """
    为PyTorch张量生成相关系数矩阵

    参数:
    y_tensor: 输入张量，形状为[T, N]
                 T: 时间序列长度
                 N: 股票数量
    threshold: 相关系数阈值，绝对值小于该阈值的相关系数将被置为0

    返回:
    corr_matrix: 相关系数矩阵，形状为[N, N]
    """
    # 确保输入是二维张量
    assert len(y_tensor.shape) == 2, f"输入张量应为2维，当前形状为{y_tensor.shape}"

    # 标准化：减去均值
    y_mean = torch.mean(y_tensor, dim=0, keepdim=True)  # [1, N]
    y_centered = y_tensor - y_mean  # [T, N]

    # 计算标准差
    y_std = torch.std(y_tensor, dim=0, keepdim=True)  # [1, N]
    # 防止除零错误
    y_std = torch.where(y_std == 0, torch.ones_like(y_std), y_std)

    # 标准化
    y_normalized = y_centered / y_std  # [T, N]

    # 计算相关系数矩阵: (1/T) * X^T * X
    T = y_tensor.shape[0]
    corr_matrix = torch.matmul(y_normalized.transpose(0, 1), y_normalized) / T  # [N, N]

    # 填充NaN值为0
    corr_matrix = torch.where(torch.isnan(corr_matrix), torch.zeros_like(corr_matrix), corr_matrix)

    return corr_matrix


def from_matrix_tensor_to_edge(corr_matrix: Tensor, layout: str = None, trading_days: list = None) -> list:
    mat_list = []

    if layout == "csr":
        edge_index = corr_matrix.to_sparse_csr()
    elif layout == "coo":
        edge_index = corr_matrix.to_sparse_coo()
    else:
        edge_index = corr_matrix

    for d in range(len(trading_days)):
        if layout == "csr":
            mat_list.append(edge_index)
        elif layout == "coo":
            mat_list.append(edge_index)
        else:
            mat_list.append(edge_index)
    return mat_list


def from_corrmatrix_to_edge(x: Series, corr_matrix, threshold: float = 0.5, layout: str = "csr", shift: int = 0,
                        select_instrument: bool = True) -> list:

    inst = x.groupby(level=0).apply(lambda a: a.index.get_level_values(1).unique().values.tolist()).values
    mat_list = []

    for d in range(len(inst)):
        if select_instrument:
            in_col = corr_matrix.columns.isin(inst[d])
            relation_ = corr_matrix[corr_matrix.columns[in_col]]
            relation_ = relation_[relation_.index.isin(inst[d])]
        else:
            relation_ = corr_matrix
        tensor = torch.from_numpy(relation_.values).to(torch.float32)
        if layout == "csr":
            mat_list.append(tensor.to_sparse_csr())
        else:
            mat_list.append(tensor.to_sparse_coo())
    return mat_list


def from_series_to_edge(x: Series, threshold: float = 0.5, layout: str = "csr", shift: int = 0,
                        select_instrument: bool = True) -> list:
    """
    计算x的相关系数矩阵. 如果相关性 < threshold则两个资产没有相关性, 值为0; 若>=threshold则有相关性, 值为1

    默认返回inst_day * inst_day的矩阵, 压缩成csr格式; 如果关掉select_instrument则返回inst * inst的矩阵, inst包含所有instrument
    """
    if shift > 0:
        x = x.groupby(level=1).shift(shift).fillna(0)
    corr_matrix = x.unstack().corr().fillna(0)
    corr_matrix[abs(corr_matrix) < threshold] = 0
    corr_matrix[corr_matrix != 0] = 1
    inst = x.groupby(level=0).apply(lambda a: a.index.get_level_values(1).unique().values.tolist()).values
    mat_list = []

    # plt.figure(figsize=(10, 10))
    # plt.imshow(corr_matrix, cmap='Blues')

    for d in range(len(inst)):
        if select_instrument:
            in_col = corr_matrix.columns.isin(inst[d])
            relation_ = corr_matrix[corr_matrix.columns[in_col]]
            relation_ = relation_[relation_.index.isin(inst[d])]
        else:
            # 获取完整的相关矩阵
            relation_ = corr_matrix.copy()

            # 将当日不交易的标的对应的行列设置为0
            trading_instruments = inst[d]
            # relation_.columns = relation_.columns.get_level_values(1)
            # relation_.index = relation_.index.get_level_values(1)
            non_trading_mask = ~relation_.index.isin(trading_instruments)

            # 将不交易标的的行设为0
            relation_.loc[non_trading_mask, :] = 0

            # 将不交易标的的列设为0
            relation_.loc[:, relation_.columns[~relation_.columns.isin(trading_instruments)]] = 0

        tensor = torch.from_numpy(relation_.values).to(torch.float32)
        if layout == "csr":
            mat_list.append(tensor.to_sparse_csr())
        else:
            mat_list.append(tensor.to_sparse_coo())
    return mat_list


def from_volatility_to_edge(x: Series, var_lags: int = 2, forecast_horizon: int = 10,
                           threshold: float = 0.1, layout: str = "csr",
                           select_instrument: bool = True) -> dict:
    """
    基于已实现波动率序列计算Diebold-Yilmaz (DY)波动率溢出矩阵。

    参数:
    x: Series，输入数据，每日每只股票的已实现波动率序列
    var_lags: int，VAR模型的滞后阶数
    forecast_horizon: int，预测步长，用于方差分解
    threshold: float，溢出效应阈值，低于此值的溢出效应将被设为0
    layout: str，返回矩阵的格式，"csr"或"coo"
    select_instrument: bool，是否为每天选择当天存在的资产

    返回:
    dict，日期为键，波动率溢出矩阵为值的字典
    """
    try:
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.vector_ar.var_model import VARResults
    except ImportError:
        raise ImportError("请安装statsmodels库: pip install statsmodels")

    # 获取所有资产列表和日期列表
    all_instruments = x.index.get_level_values(1).unique()
    days = x.index.get_level_values(0).unique()
    inst = x.groupby(level=0).apply(lambda a: a.index.get_level_values(1).unique().values.tolist()).values

    # 创建结果字典
    result_dict = {}

    # 对每个交易日计算溢出矩阵
    for d in range(len(days)):
        current_date = days[d]
        current_instruments = inst[d]

        # 创建资产映射字典，用于后续填充完整矩阵
        instrument_map = {instr: i for i, instr in enumerate(all_instruments)}

        # 如果当天资产数量太少，无法进行VAR建模，则使用相关系数矩阵代替
        if len(current_instruments) < 3:
            # 使用简单的相关系数矩阵
            vol_data = x.loc[days[d]]
            if len(vol_data) < 2:
                # 如果数据不足，创建全零矩阵
                spillover_matrix = np.zeros((len(all_instruments), len(all_instruments)))
            else:
                # 计算相关系数
                vol_corr = vol_data.unstack().corr().fillna(0).abs()

                # 创建完整的溢出矩阵（包含所有资产）
                spillover_matrix = np.zeros((len(all_instruments), len(all_instruments)))

                # 填充相关系数
                for i, instr_i in enumerate(vol_corr.index):
                    for j, instr_j in enumerate(vol_corr.columns):
                        if instr_i in instrument_map and instr_j in instrument_map:
                            spillover_matrix[instrument_map[instr_i], instrument_map[instr_j]] = vol_corr.loc[instr_i, instr_j]
        else:
            try:
                # 准备VAR模型的数据
                # 获取过去20天的数据用于建模
                start_date = days[max(0, d-20)]
                end_date = days[d]

                # 选择当前日期之前的数据
                mask = (x.index.get_level_values(0) >= start_date) & (x.index.get_level_values(0) <= end_date)
                vol_data = x[mask]

                # 只选择当天存在的资产
                if select_instrument:
                    vol_data = vol_data[vol_data.index.get_level_values(1).isin(current_instruments)]

                # 将数据重塑为适合VAR模型的格式
                vol_matrix = vol_data.unstack().fillna(method='ffill').fillna(0)

                # 确保数据足够进行VAR建模
                if vol_matrix.shape[0] <= var_lags + 1:
                    # 数据不足，使用相关系数矩阵
                    vol_corr = vol_matrix.corr().fillna(0).abs()

                    # 创建完整的溢出矩阵（包含所有资产）
                    spillover_matrix = np.zeros((len(all_instruments), len(all_instruments)))

                    # 填充相关系数
                    for i, instr_i in enumerate(vol_corr.index):
                        for j, instr_j in enumerate(vol_corr.columns):
                            if instr_i in instrument_map and instr_j in instrument_map:
                                spillover_matrix[instrument_map[instr_i], instrument_map[instr_j]] = vol_corr.loc[instr_i, instr_j]
                else:
                    # 拟合VAR模型
                    model = VAR(vol_matrix)
                    results = model.fit(maxlags=var_lags)

                    # 计算方差分解
                    fevd = results.fevd(forecast_horizon)

                    # 创建完整的溢出矩阵（包含所有资产）
                    spillover_matrix = np.zeros((len(all_instruments), len(all_instruments)))

                    # 填充溢出矩阵
                    for i, instr_i in enumerate(vol_matrix.columns):
                        decomp = fevd.decomp[i]
                        # 取最后一个预测步长的方差分解
                        contrib = decomp[-1, :]
                        # 归一化贡献
                        contrib = contrib / np.sum(contrib)

                        # 将贡献填充到完整矩阵中
                        if instr_i in instrument_map:
                            row_idx = instrument_map[instr_i]
                            for j, instr_j in enumerate(vol_matrix.columns):
                                if instr_j in instrument_map:
                                    col_idx = instrument_map[instr_j]
                                    spillover_matrix[row_idx, col_idx] = contrib[j]

            except Exception as e:
                print(f"计算日期 {days[d]} 的DY溢出矩阵时出错: {e}")
                # 出错时使用相关系数矩阵
                vol_data = x.loc[days[d]]
                vol_corr = vol_data.unstack().corr().fillna(0).abs()

                # 创建完整的溢出矩阵（包含所有资产）
                spillover_matrix = np.zeros((len(all_instruments), len(all_instruments)))

                # 填充相关系数
                for i, instr_i in enumerate(vol_corr.index):
                    for j, instr_j in enumerate(vol_corr.columns):
                        if instr_i in instrument_map and instr_j in instrument_map:
                            spillover_matrix[instrument_map[instr_i], instrument_map[instr_j]] = vol_corr.loc[instr_i, instr_j]

        # 应用阈值
        spillover_matrix[spillover_matrix < threshold] = 0

        # 转换为张量
        tensor = torch.from_numpy(spillover_matrix).to(torch.float32)

        # 根据指定格式返回稀疏矩阵
        if layout == "csr":
            result_dict[current_date] = tensor.to_sparse_csr()
        else:
            result_dict[current_date] = tensor.to_sparse_coo()

    return result_dict


def split_volatility_dict(volatility_dict, train_start_date, train_end_date, valid_end_date):
    """
    将波动率溢出字典按照给定的时间范围划分为训练集、验证集和测试集

    参数:
    volatility_dict: dict，日期为键，波动率溢出矩阵为值的字典
    train_end_date: datetime，训练集结束日期
    valid_end_date: datetime，验证集结束日期

    返回:
    tuple，包含三个列表：训练集矩阵列表、验证集矩阵列表、测试集矩阵列表
    """
    train_list = []
    valid_list = []
    test_list = []

    # 按日期排序
    sorted_dates = sorted(volatility_dict.keys())

    for date in sorted_dates:
        if (date >= train_start_date) & (date <= train_end_date):
            train_list.append(volatility_dict[date])
        elif date <= valid_end_date:
            valid_list.append(volatility_dict[date])
        else:
            test_list.append(volatility_dict[date])

    return train_list, valid_list, test_list


def make_rolling_corr_matrix(data: Series, freq: str = "M", threshold: float = 0.5) -> list[torch.Tensor]:
    monthly_corr_matrices = {}
    for month, group in data.groupby(Grouper(freq=freq, level=0)):
        name = str(month)[:7]
        returns_unstack = group.unstack()
        corr_matrix = returns_unstack.corr().fillna(0)
        corr_matrix[abs(corr_matrix) < threshold] = 0
        # corr_matrix[corr_matrix != 0] = 1
        corr_matrix = abs(corr_matrix)
        monthly_corr_matrices[name] = corr_matrix
    trade_days = data.index.get_level_values(0).unique()
    inst = data.groupby(level=0).apply(lambda x: x.index.get_level_values(1).unique().values.tolist()).values
    edges = []
    for d in range(len(trade_days)):
        mat_this_month = monthly_corr_matrices[str(trade_days[d])[0:7]]
        today_inst = inst[d]
        in_col = mat_this_month.columns.isin(today_inst)
        mat = mat_this_month[mat_this_month.columns[in_col]]
        mat = mat[mat.index.isin(today_inst)]
        edges.append(torch.from_numpy(mat.values).to(torch.float32).to_sparse_csr())
    return edges
