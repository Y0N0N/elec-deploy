import os
import math
import time

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as f
import numpy as np
from pandas import DataFrame, Series, concat, Grouper
from typing import List, Optional

from .loss import *
from .models import Model, lr_scheduler
from ..utils import get_daily_inter, from_pandas_to_list, from_pandas_to_rnn, calc_kernel_size

"""
L_out = (L_in + 2 * padding - (kernel_size - 1) * stride) / stride + 1
"""


class CNN(Model):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, output_shape: int = 1,
                 batch_size: int = 1, device: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_shape = output_shape
        self.batch_size = batch_size
        # 自动选择设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # [B, 1, F_in] -> [B, filters, F_out], kernel_size = f_in - (f_out - 1) * stride
        self.input_conv = torch.nn.Conv1d(1, 16,
                                          kernel_size=calc_kernel_size(self.input_channels, self.hidden_channels, 3),
                                          stride=3)
        self.hidden_conv = torch.nn.Conv1d(16, 32,
                                           kernel_size=calc_kernel_size(self.hidden_channels, self.output_channels),
                                           stride=2)

        self.bn = torch.nn.BatchNorm1d(self.hidden_channels)
        self.bn_1 = torch.nn.BatchNorm1d(self.output_channels)
        self.flatten = torch.nn.Flatten()

        self.out_layer = torch.nn.Linear(self.output_channels * 32, self.output_shape)

        self.optimizer = None
        self.init_weights()  # 初始化权重

    def init_weights(self):
        # 使用Xavier/Glorot初始化
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, **kwargs):
        x = f.hardswish(self.input_conv(x))
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = f.hardswish(self.hidden_conv(x))
        x = self.bn_1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.flatten(x)

        x = self.out_layer(x)
        return x

    def init_model(self):
        self.model = CNN(input_channels=self.input_channels, hidden_channels=self.hidden_channels,
                         output_channels=self.output_channels, output_shape=self.output_shape,
                         batch_size=self.batch_size, device=self.device,
                         epochs=self.epochs, loss=self.loss, lr=self.lr,
                         weight_decay=self.weight_decay, dropout=self.dropout).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, z_train=None, z_valid=None, **kwargs):
        """
        训练模型，支持PyTorch张量输入和按时间步批量训练

        参数:
        x_train: 训练数据，可以是pd.DataFrame或torch.Tensor [时间, 股票, 因子]
        y_train: 训练标签，可以是pd.Series或torch.Tensor [时间, 股票]
        x_valid: 验证数据（可选）
        y_valid: 验证标签（可选）
        z_train: 额外的训练数据（可选）
        z_valid: 额外的验证数据（可选）
        """
        # 处理PyTorch张量输入
        if isinstance(x_train, torch.Tensor) and isinstance(y_train, torch.Tensor):
            return self._fit_tensor(x_train, y_train, x_valid, y_valid, z_train, z_valid, **kwargs)
        # 保持原有接口
        else:
            return super().fit(x_train, y_train, x_valid, y_valid, z_train, z_valid, **kwargs)

    def _fit_tensor(self, x_train_tensor, y_train_tensor, x_valid=None, y_valid=None, z_train=None, z_valid=None, **kwargs):
        """
        使用PyTorch张量训练模型，按时间步批量处理

        x_train_tensor形状: [时间步数, 股票数量, 因子数量]
        y_train_tensor形状: [时间步数, 股票数量]
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

        if self.model is None:
            self.init_model()

        # 早停机制初始化
        self.best_loss_val = float('inf')  # 初始化为一个很大的值
        best_epoch = 0
        early_stop_count = 0
        has_valid = x_valid is not None and y_valid is not None

        # 保存最佳模型参数
        best_model_state = None

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.to(self.device)

        # 获取维度信息
        total_time_steps, num_stocks, num_factors = x_train_tensor.shape

        # 按时间步批量训练
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            for t in range(0, total_time_steps, self.batch_size):
                # 获取当前批次
                end_t = min(t + self.batch_size, total_time_steps)

                # 获取当前批次的输入和标签 [T, N, K] -> [T*N, K]
                x_batch = x_train_tensor[t:end_t].reshape(-1, num_factors).to(self.device)
                # 获取当前批次的标签 [T, N] -> [T*N]
                y_batch = y_train_tensor[t:end_t].reshape(-1, 1).to(self.device)

                # 过滤掉NaN值
                valid_mask = ~(torch.isnan(x_batch).any(dim=1) | torch.isnan(y_batch).any(dim=1))
                if valid_mask.any():
                    x_valid_batch = x_batch[valid_mask]
                    y_valid_batch = y_batch[valid_mask]

                    # 为CNN调整输入形状: [B, K] -> [B, 1, K]
                    x_valid_batch = x_valid_batch.unsqueeze(1)

                    # 前向传播
                    predictions = self.model(x_valid_batch)

                    # 计算损失
                    if isinstance(self.loss, str):
                        loss = eval("f." + self.loss + "(predictions, y_valid_batch)")
                    else:
                        loss = self.loss(predictions, y_valid_batch)

                    # 反向传播和优化
                    self.optimizer.zero_grad()
                    loss.backward()

                    # 梯度裁剪（如果需要）
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    self.optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

            # 学习率调度
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # 计算平均训练损失
            avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')

            # 验证集评估
            if has_valid:
                self.model.eval()
                val_loss = 0.0
                val_batch_count = 0

                with torch.no_grad():
                    # 处理验证集
                    val_time_steps = x_valid.shape[0]
                    for t in range(0, val_time_steps, self.batch_size):
                        end_t = min(t + self.batch_size, val_time_steps)

                        x_val_batch = x_valid[t:end_t].reshape(-1, num_factors).to(self.device)
                        y_val_batch = y_valid[t:end_t].reshape(-1, 1).to(self.device)

                        # 过滤NaN值
                        val_valid_mask = ~(torch.isnan(x_val_batch).any(dim=1) | torch.isnan(y_val_batch).any(dim=1))
                        if val_valid_mask.any():
                            x_val_valid = x_val_batch[val_valid_mask]
                            y_val_valid = y_val_batch[val_valid_mask]

                            # 为CNN调整输入形状: [B, K] -> [B, 1, K]
                            x_val_valid = x_val_valid.unsqueeze(1)

                            val_preds = self.model(x_val_valid)

                            if isinstance(self.loss, str):
                                val_batch_loss = eval("f." + self.loss + "(val_preds, y_val_valid)")
                            else:
                                val_batch_loss = self.loss(val_preds, y_val_valid)

                            val_loss += val_batch_loss.item()
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
                        print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_val_loss:.4f} (Best)")
                else:
                    early_stop_count += 1
                    if epoch % 10 == 0:
                        print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_val_loss:.4f}, Early Stop Count: {early_stop_count}")

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
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_train_loss:.4f} (Best)")
                else:
                    early_stop_count += 1
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_train_loss:.4f}, Early Stop Count: {early_stop_count}")

                # 检查是否需要早停
                if self.early_stopping and early_stop_count >= self.early_stopping:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                    break

        # 恢复最佳模型参数
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # 打印最优结果
        print(f"\nTraining completed!")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Loss: {self.best_loss_val:.4f}")

        return self

    def predict(self, x_test):
        """
        生成预测结果，支持DataFrame和PyTorch张量输入

        参数:
        x_test: 测试数据，可以是pd.DataFrame或torch.Tensor [时间, 股票, 因子]

        返回:
        对于DataFrame输入返回list，对于张量输入返回torch.Tensor [时间, 股票]
        """
        # 处理PyTorch张量输入
        if isinstance(x_test, torch.Tensor):
            return self._predict_tensor(x_test)
        # 保持原有DataFrame接口
        else:
            return super().predict(x_test)

    def _predict_tensor(self, x_test_tensor):
        """
        使用PyTorch张量进行预测，支持批量处理

        x_test_tensor形状: [时间, 股票, 因子]
        返回形状: [时间, 股票]
        """
        # 确保输入形状正确 [时间, 股票, 因子]
        if len(x_test_tensor.shape) != 3:
            raise ValueError(f"输入张量形状应为[时间, 股票, 因子]，实际为{x_test_tensor.shape}")

        # 将模型移动到指定设备
        self.model.to(self.device)
        self.model.eval()

        # 获取维度信息
        num_time_steps = x_test_tensor.shape[0]
        num_stocks = x_test_tensor.shape[1]
        num_factors = x_test_tensor.shape[2]

        # 创建输出张量
        y_pred = torch.full((num_time_steps, num_stocks), np.nan, dtype=torch.float32, device=self.device)

        # 按时间步批量预测
        with torch.no_grad():
            for t in range(0, num_time_steps, self.batch_size):
                # 获取当前批次
                end_t = min(t + self.batch_size, num_time_steps)

                # 获取当前批次的输入 [T, N, K] -> [T*N, K]
                x_batch = x_test_tensor[t:end_t].reshape(-1, num_factors).to(self.device)

                # 创建当前批次的预测结果数组，初始化为NaN
                batch_pred = torch.full((x_batch.shape[0], 1), np.nan, dtype=torch.float32, device=self.device)

                # 找出有效数据的索引
                valid_mask = ~torch.isnan(x_batch).any(dim=1)
                if valid_mask.any():
                    # 对有效数据进行预测
                    x_valid_batch = x_batch[valid_mask]
                    # 为CNN调整输入形状: [B, K] -> [B, 1, K]
                    x_valid_batch = x_valid_batch.unsqueeze(1)
                    valid_preds = self.model(x_valid_batch)
                    # 将预测结果放回到正确位置
                    batch_pred[valid_mask] = valid_preds

                # 将当前批次的预测结果重塑并放入输出张量 [T*N, 1] -> [T, N]
                y_pred[t:end_t] = batch_pred.reshape(end_t - t, num_stocks)

        return y_pred

    def save(self, path: str):
        """
        保存模型参数到指定路径

        参数:
        path: 模型参数保存路径
        """
        torch.save(self.model.state_dict(), path)



class Chomp1d(nn.Module):
    """
    Chomp1d层用于移除因果卷积后多余的填充部分
    在因果卷积中，为了保证不使用未来信息，会在序列前面进行填充，
    这会导致输出序列比输入序列长，Chomp1d用于移除这些多余的填充
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        移除末尾的填充部分

        参数:
        x: 输入张量，形状为 [B, C, T] 其中B是批次大小，C是通道数，T是时间步数

        返回:
        输出张量，形状为 [B, C, T-chomp_size]
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN中的时序块，包含两个因果卷积层和残差连接

    输入形状: [B, C_in, T] 其中B是批次大小，C_in是输入通道数（因子数量），T是时间步数
    输出形状: [B, C_out, T] 其中C_out是输出通道数
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        初始化TemporalBlock

        参数:
        n_inputs: 输入通道数 (C_in，即因子数量)
        n_outputs: 输出通道数 (C_out)
        kernel_size: 卷积核大小
        stride: 步长
        dilation: 膨胀系数，用于控制感受野大小
        padding: 填充大小，通常为 (kernel_size-1) * dilation
        dropout: Dropout比率
        """
        super(TemporalBlock, self).__init__()
        # 第一个因果卷积层
        # 输入: [B, C_in, T]
        # 输出: [B, C_out, T+(kernel_size-1)*dilation-padding] (经过填充和卷积后)
        #       [B, C_out, T] (经过Chomp1d后)
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)  # 移除多余填充，保持时间序列长度不变
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个因果卷积层
        # 输入: [B, C_out, T]
        # 输出: [B, C_out, T+(kernel_size-1)*dilation-padding] (经过填充和卷积后)
        #       [B, C_out, T] (经过Chomp1d后)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)  # 移除多余填充，保持时间序列长度不变
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 组合层
        # 整体输入: [B, C_in, T]
        # 整体输出: [B, C_out, T]
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 残差连接
        # 当输入通道数与输出通道数不一致时，使用1x1卷积进行通道变换
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """初始化网络权重"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        前向传播

        参数:
        x: 输入张量，形状为 [B, C_in, T]
           B: 批次大小（股票数量）
           C_in: 输入通道数（因子数量）
           T: 时间序列长度

        返回:
        输出张量，形状为 [B, C_out, T]
           B: 批次大小（股票数量）
           C_out: 输出通道数
           T: 时间序列长度
        """
        # 主路径: [B, C_in, T] -> [B, C_out, T]
        out = self.net(x)

        # 残差路径: [B, C_in, T] -> [B, C_out, T] (如果需要通道变换)
        res = x if self.downsample is None else self.downsample(x)

        # 残差连接: [B, C_out, T] + [B, C_out, T] -> [B, C_out, T]
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """时序卷积网络"""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)  # [B, C_in, T] -> [B, C_out, T]


class TCN(Model):
    """时序卷积网络模型，适用于金融时间序列预测"""
    def __init__(self, input_channels: int, num_channels: List[int], output_shape: int=1,
                 kernel_size: int = 3, dropout: float = 0.2,
                 time_steps: int = 32, device: str = None, *args, **kwargs):
        """
        初始化TCN模型

        参数:
        input_channels: 输入特征维度（因子数量）
        num_channels: 每个层级的通道数列表，例如 [25, 25, 25] 表示3个层级，每层25个通道
        output_shape: 输出维度（通常是1，表示单个预测值）
        kernel_size: 卷积核大小
        dropout: Dropout比率
        batch_size: 批次大小（时间步数）
        device: 设备 ('cuda' 或 'cpu')
        """
        super(TCN, self).__init__(*args, **kwargs)

        self.input_channels = input_channels  # 因子数量C
        self.output_shape = output_shape
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.time_steps = time_steps
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 构建TCN网络
        self.tcn = TemporalConvNet(input_channels, num_channels, kernel_size, dropout)

        # 输出层：将TCN输出映射到最终输出维度
        self.linear = nn.Linear(num_channels[-1], output_shape)

        # 初始化权重
        self.init_weights()

        # 优化器将在init_model中初始化
        self.optimizer = None

    def init_weights(self):
        """初始化网络权重"""
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        """
        前向传播

        参数:
        x: 输入张量，形状为 [B, T, C]
           B: 批次大小（股票数量）
           T: 时间序列长度
           C: 特征维度（因子数量）

        返回:
        输出张量，形状为 [B, output_size]
        """
        # 转换输入形状为 [B, C, T] 以适应Conv1d
        x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]

        # 通过TCN网络
        y = self.tcn(x)  # [B, C, T] -> [B, C_out, T]

        # 取最后一个时间步的输出
        o = self.linear(y[:, :, -1])  # [B, C_out, T] -> [B, C_out] -> [B, output_size]
        return o

    def init_model(self):
        """初始化模型和优化器"""
        self.model = TCN(
            input_channels=self.input_channels,
            output_shape=self.output_shape,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            time_steps=self.time_steps,
            device=self.device,
            epochs=self.epochs,
            loss=self.loss,
            lr=self.lr,
            weight_decay=self.weight_decay
        ).to(torch.float32)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, z_train=None, z_valid=None, **kwargs):
        """
        训练模型，支持PyTorch张量输入和按时间步批量训练

        参数:
        x_train: 训练数据，可以是pd.DataFrame或torch.Tensor [T, N, C]
                 T: 时间序列长度
                 N: 股票数量
                 C: 因子数量
        y_train: 训练标签，可以是pd.Series或torch.Tensor [T, N]
        x_valid: 验证数据（可选）
        y_valid: 验证标签（可选）
        z_train: 额外的训练数据（可选）
        z_valid: 额外的验证数据（可选）
        """
        # 处理PyTorch张量输入
        if isinstance(x_train, torch.Tensor) and isinstance(y_train, torch.Tensor):
            return self._fit_tensor(x_train, y_train, x_valid, y_valid, z_train, z_valid, **kwargs)
        # 保持原有接口
        else:
            return super().fit(x_train, y_train, x_valid, y_valid, z_train, z_valid, **kwargs)

    def _fit_tensor(self, x_train_tensor, y_train_tensor, x_valid=None, y_valid=None, z_train=None, z_valid=None, **kwargs):
        """
        使用PyTorch张量训练模型，按时间步批量处理

        x_train_tensor形状: [T, N, C]
                           T: 时间序列长度
                           N: 股票数量
                           C: 因子数量
        y_train_tensor形状: [T, N]
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

        if self.model is None:
            self.init_model()

        # 早停机制初始化
        self.best_loss_val = float('inf')
        best_epoch = 0
        early_stop_count = 0
        has_valid = x_valid is not None and y_valid is not None

        # 保存最佳模型参数
        best_model_state = None

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.to(self.device)

        # 获取维度信息
        num_time_steps, num_stocks, num_factors = x_train_tensor.shape  # [T, N, C]

        # 按时间步批量训练
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            # 记录每个epoch的开始时间
            epoch_start_time = time.time()

            for t in range(0, num_time_steps - self.time_steps + 1, 1):
                # 获取当前批次
                end_t = min(t + self.time_steps, num_time_steps)

                # 获取当前批次的输入和标签
                x_batch = x_train_tensor[t:end_t].to(self.device)  # [T_batch, N, C]
                y_batch = y_train_tensor[t:end_t].to(self.device)  # [T_batch, N]

                # 过滤掉NaN值
                # 修改：正确地检查每个时间步和股票的NaN值
                valid_mask = ~(torch.isnan(x_batch).any(dim=2).any(dim=0) | torch.isnan(y_batch).any(dim=0))
                if valid_mask.any():
                    x_valid_batch = x_batch[:, valid_mask, :]  # [T_batch, N_valid, C]
                    y_valid_batch = y_batch[:, valid_mask]     # [T_batch, N_valid]

                    # 重塑为适合TCN的格式: [N_valid, T_batch, C]
                    # 交换维度以适应TCN: [T_batch, N_valid, C] -> [N_valid, T_batch, C]
                    x_for_tcn = x_valid_batch.transpose(0, 1)  # [N_valid, T_batch, C]
                    y_for_tcn = y_valid_batch.transpose(0, 1)  # [N_valid, T_batch]

                    # 前向传播
                    predictions = self.model(x_for_tcn)  # [N_valid, output_size]

                    # 计算损失 - 修正：确保标签形状与预测形状匹配
                    y_target = y_for_tcn[:, -1]  # [N_valid] - 取最后一个时间步的真实值
                    if predictions.shape[1] == 1:  # 如果输出维度为1
                        y_target = y_target.unsqueeze(1)  # [N_valid, 1]

                    # 计算损失
                    if isinstance(self.loss, str):
                        loss = eval("f." + self.loss + "(predictions, y_target)")
                    else:
                        loss = self.loss(predictions, y_target)

                    # 反向传播和优化
                    self.optimizer.zero_grad()
                    loss.backward()

                    # 梯度裁剪（如果需要）
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    self.optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

            # 记录每个epoch的结束时间并计算耗时
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            # 学习率调度
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # 计算平均训练损失
            avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')

            # 验证集评估
            if has_valid:
                self.model.eval()
                val_loss = 0.0
                val_batch_count = 0

                with torch.no_grad():
                    # 处理验证集
                    val_time_steps = x_valid.shape[0]
                    for t in range(0, val_time_steps - self.time_steps + 1, 1):
                        end_t = min(t + self.time_steps, val_time_steps)

                        x_val_batch = x_valid[t:end_t].to(self.device)  # [T_batch, N, C]
                        y_val_batch = y_valid[t:end_t].to(self.device)  # [T_batch, N]

                        # 过滤NaN值 - 修正：正确的NaN检查方式
                        val_valid_mask = ~(torch.isnan(x_val_batch).any(dim=2).any(dim=0) | torch.isnan(y_val_batch).any(dim=0))
                        if val_valid_mask.any():
                            x_val_valid = x_val_batch[:, val_valid_mask, :]  # [T_batch, N_valid, C]
                            y_val_valid = y_val_batch[:, val_valid_mask]     # [T_batch, N_valid]

                            # 准备TCN输入 - 修正：正确的维度转换
                            x_val_for_tcn = x_val_valid.transpose(0, 1)  # [N_valid, T_batch, C]
                            y_val_for_tcn = y_val_valid.transpose(0, 1)  # [N_valid, T_batch]

                            val_preds = self.model(x_val_for_tcn)  # [N_valid, output_size]

                            # 修正：确保标签形状与预测形状匹配
                            y_val_target = y_val_for_tcn[:, -1]  # [N_valid] - 取最后一个时间步的真实值
                            if val_preds.shape[1] == 1:  # 如果输出维度为1
                                y_val_target = y_val_target.unsqueeze(1)  # [N_valid, 1]

                            if isinstance(self.loss, str):
                                val_batch_loss = eval("f." + self.loss + "(val_preds, y_val_target)")
                            else:
                                val_batch_loss = self.loss(val_preds, y_val_target)

                            val_loss += val_batch_loss.item()
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

        # 恢复最佳模型参数
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # 打印最优结果
        print(f"\nTraining completed!")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Loss: {self.best_loss_val:.4f}")

        return self

    def predict(self, x_test):
        """
        生成预测结果，支持DataFrame和PyTorch张量输入

        参数:
        x_test: 测试数据，可以是pd.DataFrame或torch.Tensor [T, N, C]
                T: 时间序列长度
                N: 股票数量
                C: 因子数量

        返回:
        对于DataFrame输入返回list，对于张量输入返回torch.Tensor [T, N]
        """
        # 处理PyTorch张量输入
        if isinstance(x_test, torch.Tensor):
            return self._predict_tensor(x_test)
        # 保持原有DataFrame接口
        else:
            return super().predict(x_test)

    def _predict_tensor(self, x_test_tensor):
        """
        使用PyTorch张量进行预测，支持批量处理

        x_test_tensor形状: [T, N, C]
                          T: 时间序列长度
                          N: 股票数量
                          C: 因子数量
        返回形状: [T-N_time_steps+1, N] 或者更准确地说是 [T', N] 其中T'是有效预测时间步数
        """
        # 确保输入形状正确 [T, N, C]
        if len(x_test_tensor.shape) != 3:
            raise ValueError(f"输入张量形状应为[T, N, C]，实际为{x_test_tensor.shape}")

        # 将模型移动到指定设备
        self.model.to(self.device)
        self.model.eval()

        # 获取维度信息
        num_time_steps = x_test_tensor.shape[0]  # T
        num_stocks = x_test_tensor.shape[1]      # N
        num_factors = x_test_tensor.shape[2]     # C

        # 检查时间序列长度是否足够
        if num_time_steps < self.time_steps:
            raise ValueError(f"输入时间序列长度({num_time_steps})小于所需时间步长({self.time_steps})")

        # 创建输出张量，形状为[T-N_time_steps+1, N]
        y_pred = torch.full((num_time_steps - self.time_steps + 1, num_stocks), np.nan, dtype=torch.float32, device=self.device)

        # 按时间步批量预测
        with torch.no_grad():
            for t in range(0, num_time_steps - self.time_steps + 1, 1):
                # 获取当前批次
                end_t = t + self.time_steps

                # 获取当前批次的输入
                x_batch = x_test_tensor[t:end_t].to(self.device)  # [time_steps, N, C]

                # 找出有效数据的索引（没有NaN值的股票）
                valid_mask = ~torch.isnan(x_batch).any(dim=2).any(dim=0)  # 检查每个股票在所有时间步和因子上是否有NaN

                if valid_mask.any():
                    # 对有效数据进行预测
                    x_valid_batch = x_batch[:, valid_mask, :]  # [time_steps, N_valid, C]

                    # 交换维度以适应TCN: [time_steps, N_valid, C] -> [N_valid, time_steps, C]
                    x_for_prediction = x_valid_batch.transpose(0, 1)  # [N_valid, time_steps, C]

                    # 模型预测
                    valid_preds = self.model(x_for_prediction)  # [N_valid, output_size]

                    # 将预测结果放回到正确位置
                    batch_pred = torch.full((num_stocks,), np.nan, dtype=torch.float32, device=self.device)
                    batch_pred[valid_mask] = valid_preds.squeeze(-1)  # [N_valid] -> 放回对应位置

                    # 将当前批次的预测结果放入输出张量
                    y_pred[t] = batch_pred

        return y_pred

    def save(self, path: str):
        """
        保存模型参数到指定路径

        参数:
        path: 模型参数保存路径
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """
        从指定路径加载模型参数

        参数:
        path: 模型参数保存路径
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
