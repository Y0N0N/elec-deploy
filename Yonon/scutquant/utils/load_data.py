import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Function
import random
import threading
import sys

from scutquant.operators import *


def get_stock_list(factor_names, factor_files):
    stock_list = []
    for factor_name, factor_file in zip(factor_names, factor_files):
        factor_data = pd.read_feather(factor_file)
        stock_list = set(stock_list) | set(factor_data.columns)
    return sorted(list(stock_list))

def get_factor_dict(factor_names, factor_files):
    factor_dict = {}

    stock_list = get_stock_list(factor_names, factor_files)

    for factor_name, factor_file in zip(factor_names, factor_files):
        factor_dict[factor_name] = pd.read_feather(factor_file)
        factor_dict[factor_name].columns = stock_list

    return factor_dict

def split_factor_dict(factor_dict, train_start_date, valid_start_date, test_start_date, trading_days):
    train_end_date = trading_days[trading_days.index(valid_start_date) - 1]
    valid_end_date = trading_days[trading_days.index(test_start_date) - 1]
    test_end_date = trading_days[-1]

    factor_dict_train = {}
    factor_dict_valid = {}
    factor_dict_test = {}
    for factor_name, factor_df in factor_dict.items():
        factor_dict_train[factor_name] = factor_df[(factor_df.index >= train_start_date) & (factor_df.index <= train_end_date)]
        factor_dict_valid[factor_name] = factor_df[(factor_df.index >= valid_start_date) & (factor_df.index <= valid_end_date)]
        factor_dict_test[factor_name] = factor_df[(factor_df.index >= test_start_date) & (factor_df.index <= test_end_date)]
    return factor_dict_train, factor_dict_valid, factor_dict_test

def from_factor_dict_to_tensor(factor_dict):
    """
    将因子字典转换为三维张量

    参数:
        factor_dict: dict, key为因子名, value为DataFrame
                    DataFrame的index为日期(T), columns为股票代码(N)

    返回:
        torch.Tensor: 形状为 (T, N, K) 的三维张量
    """
    # 获取所有因子值并转换为张量
    factor_tensors = []

    for factor_name, factor_df in factor_dict.items():
        factor_df = inf_mask(factor_df.astype(float))
        # 将DataFrame转换为torch.Tensor (形状: T x N)
        factor_tensor = torch.tensor(factor_df.values, dtype=torch.float32)
        factor_tensors.append(factor_tensor)

    # 在最后一个维度上堆叠所有因子 (形状: T x N x K)
    result_tensor = torch.stack(factor_tensors, dim=-1)

    return result_tensor

def tensor_fill_mean(tensor):
    '''跟实盘对齐调整后的处理方式'''
    tensor[torch.isinf(tensor)]=torch.nan  #将inf转为空
    mask = torch.isnan(tensor)
    # 计算每列的中值
    column_mean = torch.nanmean(tensor.clamp(-3,3), dim=0, keepdim=True)
    # 填充中值到缺失值的位置
    filled_tensor = torch.where(mask, column_mean, tensor)
    return filled_tensor

def nanstd(tensor, dim=None, keepdim=False):
    # Compute the mean while ignoring NaNs
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)

    # Compute the squared differences from the mean
    squared_diff = torch.pow(tensor - mean, 2)

    # Compute the mean of squared differences while ignoring NaNs
    variance = torch.nanmean(squared_diff, dim=dim, keepdim=keepdim)

    # Return the square root of the variance (standard deviation)
    return torch.sqrt(variance)

def tensor_normalize(tensor, norm_method="zscore", fill_mean=True):
    '''
    对张量进行归一化处理

    参数:
        tensor: torch.Tensor, 输入张量，形状为 (D, T, N, K) 或 (T, N, K) 或 (N, K) 或 (N,)
                D为天数，T为时间维度，N为股票维度，K为因子维度
        norm_method: str, 归一化方法
                    "zscore": 使用标准z-score归一化
                    "robust_zscore": 使用稳健z-score归一化
                    "scale": 使用0-1缩放
                    "rank": 使用排名归一化

    返回:
        torch.Tensor: 归一化后的张量，形状与输入相同
    '''
    # 处理形状，确保至少为2D张量
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)  # 转换为 (N, 1)

    # 复制张量以避免修改原始数据
    result = tensor.clone()

    # 1. 替换inf为nan (inf_mask的功能)
    result[torch.isinf(result)] = torch.nan

    # 2. 进行mad_winsor处理
    # 根据张量维度计算中位数和MAD
    if tensor.dim() == 4:  # (D, T, N, K)
        # 对每个天、时间步和因子计算股票维度的中位数和MAD
        med = torch.nanmedian(result, dim=2, keepdim=True)[0]  # 在N维度上计算
        mad = torch.nanmedian(torch.abs(result - med), dim=2, keepdim=True)[0]  # 在N维度上计算
        up = med + 3 * mad * 1.4826
        down = med - 3 * mad * 1.4826
    elif tensor.dim() == 3:  # (T, N, K)
        med = torch.nanmedian(result, dim=1, keepdim=True)[0]  # 在N维度上计算
        mad = torch.nanmedian(torch.abs(result - med), dim=1, keepdim=True)[0]  # 在N维度上计算
        up = med + 3 * mad * 1.4826
        down = med - 3 * mad * 1.4826
    else:  # (N, K) 或 (N,)
        med = torch.nanmedian(result, dim=0, keepdim=True)[0]  # 在N维度上计算
        mad = torch.nanmedian(torch.abs(result - med), dim=0, keepdim=True)[0]  # 在N维度上计算
        up = med + 3 * mad * 1.4826
        down = med - 3 * mad * 1.4826

    # 进行winsorize
    result = torch.clamp(result, down, up)

    # 3. 进行归一化处理
    if norm_method == "zscore":
        if tensor.dim() == 4:
            mean = torch.nanmean(result, dim=2, keepdim=True)  # 在N维度上计算
            std = nanstd(result, dim=2, keepdim=True)  # 在N维度上计算
        elif tensor.dim() == 3:
            mean = torch.nanmean(result, dim=1, keepdim=True)  # 在N维度上计算
            std = nanstd(result, dim=1, keepdim=True)  # 在N维度上计算
        else:
            mean = torch.nanmean(result, dim=0, keepdim=True)  # 在N维度上计算
            std = nanstd(result, dim=0, keepdim=True)  # 在N维度上计算
        std = torch.where(std == 0, torch.tensor(1.0, device=std.device), std)
        result = (result - mean) / std

    elif norm_method == "robust_zscore":
        if tensor.dim() == 4:
            med = torch.nanmedian(result, dim=2, keepdim=True)[0]  # 在N维度上计算
            mad = torch.nanmedian(torch.abs(result - med), dim=2, keepdim=True)[0]  # 在N维度上计算
        elif tensor.dim() == 3:
            med = torch.nanmedian(result, dim=1, keepdim=True)[0]  # 在N维度上计算
            mad = torch.nanmedian(torch.abs(result - med), dim=1, keepdim=True)[0]  # 在N维度上计算
        else:
            med = torch.nanmedian(result, dim=0, keepdim=True)[0]  # 在N维度上计算
            mad = torch.nanmedian(torch.abs(result - med), dim=0, keepdim=True)[0]  # 在N维度上计算
        mad = torch.where(mad == 0, torch.tensor(1.0, device=mad.device), mad)
        result = (result - med) / (mad * 1.4826)

    elif norm_method == "scale":
        if tensor.dim() == 4:
            min_val = torch.nanmin(result, dim=2, keepdim=True)[0]  # 在N维度上计算
            max_val = torch.nanmax(result, dim=2, keepdim=True)[0]  # 在N维度上计算
        elif tensor.dim() == 3:
            min_val = torch.nanmin(result, dim=1, keepdim=True)[0]  # 在N维度上计算
            max_val = torch.nanmax(result, dim=1, keepdim=True)[0]  # 在N维度上计算
        else:
            min_val = torch.nanmin(result, dim=0, keepdim=True)[0]  # 在N维度上计算
            max_val = torch.nanmax(result, dim=0, keepdim=True)[0]  # 在N维度上计算
        # 避免除以零
        denom = max_val - min_val
        denom = torch.where(denom == 0, torch.tensor(1.0, device=denom.device), denom)
        result = (result - min_val) / denom

    else:  # rank
        # 计算排名
        if tensor.dim() == 4:
            # 对每个天、时间步和因子，计算股票的排名
            rank_shape = result.shape
            # 将形状转换为 (D*T*K, N)
            result_2d = result.permute(0, 1, 3, 2).reshape(-1, rank_shape[2])
            # 计算排名
            _, indices = torch.sort(result_2d, dim=1)
            _, ranks = torch.sort(indices, dim=1)
            # 将形状转换回 (D, T, N, K)
            ranks = ranks.reshape(rank_shape[0], rank_shape[1], rank_shape[3], rank_shape[2]).permute(0, 1, 3, 2)
            # 转换为0-1范围
            result = ranks.float() / (rank_shape[2] - 1)
        elif tensor.dim() == 3:
            # 对每个时间步和因子，计算股票的排名
            rank_shape = result.shape
            # 将形状转换为 (T*K, N)
            result_2d = result.permute(0, 2, 1).reshape(-1, rank_shape[1])
            # 计算排名
            _, indices = torch.sort(result_2d, dim=1)
            _, ranks = torch.sort(indices, dim=1)
            # 将形状转换回 (T, N, K)
            ranks = ranks.reshape(rank_shape[0], rank_shape[2], rank_shape[1]).permute(0, 2, 1)
            # 转换为0-1范围
            result = ranks.float() / (rank_shape[1] - 1)
        else:
            # 对每个因子，计算股票的排名
            _, indices = torch.sort(result, dim=0)
            _, ranks = torch.sort(indices, dim=0)
            # 转换为0-1范围
            result = ranks.float() / (result.shape[0] - 1)

    if fill_mean:
        result = tensor_fill_mean(result)
    result[torch.isnan(result)] = 0

    return result


# ---- 原scutquant ----
def get_daily_inter(data: Series | DataFrame, shuffle: bool = False):
    '''
    计算数据按天分组的索引和计数

    参数:
        data: Series或DataFrame，输入数据，索引为(datetime, instrument)
        shuffle: 是否随机打乱分组顺序

    返回:
        daily_index: 按天分组的起始索引
        daily_count: 按天分组的样本数
    '''
    daily_count = data.groupby(level=0).size().values
    daily_index = np.roll(np.cumsum(daily_count), 1)
    daily_index[0] = 0
    if shuffle:
        daily_shuffle = list(zip(daily_index, daily_count))
        np.random.shuffle(daily_shuffle)
        daily_index, daily_count = zip(*daily_shuffle)
    return daily_index, daily_count

def from_pandas_to_list(x, for_cnn: bool = False, fillna: bool = False):
    """
    将DataFrame或Series拆成list, 并根据模型类型对数据的shape进行调整
    for cnn: [inst * 1 * feat]
    else:    [inst * feat]

    参数:
    x: DataFrame或Series，输入数据
    for_cnn: 是否为CNN模型准备数据
    fillna: 是否将缺失值填充为0
    """
    if isinstance(x, DataFrame) or isinstance(x, Series):
        # 使用unstack确保每天的instrument维度一致
        if isinstance(x, Series):
            n_feat = 1
        else:
            n_feat = len(x.columns)

        # 先unstack，确保每天的instrument维度一致
        if fillna:
            x_unstack = x.unstack(level=0)  # instrument为索引，日期为列
            x_unstack = x_unstack.fillna(0)
            x_3d = x_unstack.values.reshape(x_unstack.shape[0], n_feat, -1)  # inst * feat * date
            x_list = np.split(x_3d, x_3d.shape[-1], axis=-1)
            x_tensor_list = [torch.tensor(x.squeeze(axis=-1), dtype=torch.float32) for x in x_list]
            return x_tensor_list
        else:
            dataset = []
            daily_index, daily_count = get_daily_inter(x)
            for index, count in zip(daily_index, daily_count):
                batch = slice(index, index + count)
                data_slice = x.iloc[batch]
                if for_cnn:
                    value = data_slice.values.reshape(-1, 1, data_slice.shape[1])  # instrument * 1 * feat
                    # print(value.shape)
                else:
                    value = data_slice.values  # instrument * feat
                if value.ndim == 1:
                    dataset.append(torch.from_numpy(np.squeeze(value)).to(torch.float32).view(-1, 1))
                else:
                    dataset.append(torch.from_numpy(value).to(torch.float32))
            # print(dataset[-1].shape)
            return dataset
    else:
        return x

def from_pandas_to_rnn(x: DataFrame | Series, fillna: bool = False):
    """
    [inst, date, feat]
    """
    if isinstance(x, Series):
        n_feat = 1
    else:
        n_feat = len(x.columns)

    x_unstack = x.unstack(level=0)
    if fillna:
        x_unstack = x_unstack.fillna(0)

    x_3d = x_unstack.values.reshape(x_unstack.shape[0], n_feat, -1)  # inst * feat * date
    tensor = torch.from_numpy(x_3d).to(torch.float32).permute(0, 2, 1)  # Tensor with shape inst * date * feat
    return tensor

def calc_kernel_size(f_in: int, f_out: int, stride: int = 2) -> int:
    # f_out = (f_in - kernel_size) / stride + 1
    # f_in - kernel_size = (f_out - 1) * stride
    # kernel_size = f_in - (f_out - 1) * stride
    assert f_out > 1
    return int(f_in - (f_out - 1) * stride)

def transform_data(x_train, y_train, x_valid, y_valid, z_train=None, z_valid=None,
                   for_cnn: bool = False, for_rnn: bool = False,
                   fillna: bool = False):
    """
    将DataFrame或Series拆成list, 并根据模型类型对数据的shape进行调整
    """
    if isinstance(x_train, DataFrame) or isinstance(x_train, Series):
        if not for_rnn:
            x_train, x_valid = from_pandas_to_list(x_train, for_cnn, fillna=fillna), from_pandas_to_list(x_valid, for_cnn, fillna=fillna)
            y_train, y_valid = from_pandas_to_list(y_train, fillna=fillna), from_pandas_to_list(y_valid, fillna=fillna)
        else:
            x_train, x_valid = from_pandas_to_rnn(x_train, fillna=True), from_pandas_to_rnn(x_valid, fillna=True)
            y_train, y_valid = from_pandas_to_rnn(y_train), from_pandas_to_rnn(y_valid)
        if z_train is not None:
            z_train = from_pandas_to_list(z_train)
        if z_valid is not None:
            z_valid = from_pandas_to_list(z_valid)
    return x_train, y_train, x_valid, y_valid, z_train, z_valid
