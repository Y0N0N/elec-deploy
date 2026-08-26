import torch
import torch.nn as nn
import os
import math
import time
from torch import Tensor
import torch.nn.functional as f
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, concat, Grouper
from typing import Union, Tuple, Optional

from sklearn.preprocessing import StandardScaler

from .loss import *
from .models import Model, lr_scheduler
from ..utils import get_daily_inter, from_pandas_to_list, from_pandas_to_rnn, calc_kernel_size, transform_data


class GRU(Model):
    def __init__(self, input_channels: int, hidden_channels: int, output_shape: int = 1, n_layers: int = 1,
                 time_steps: int = 10, device: str = None, *args, **kwargs):
        super(GRU, self).__init__(*args, **kwargs)
        """
        GRU模型，适用于金融时间序列预测

        输入形状: [T, N, C]
        T: 时间步数
        N: 批次大小（股票数量）
        C: 输入特征维度（因子数量）
        """
        self.input_channels = input_channels      # 输入特征维度（因子数量）
        self.hidden_channels = hidden_channels    # 隐藏层维度
        self.output_shape = output_shape    # 输出维度
        self.time_steps = time_steps          # 时间步数
        self.n_layers = n_layers            # GRU层数
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 构建GRU网络
        self.gru = torch.nn.GRU(
            input_size=self.input_channels,
            hidden_size=self.hidden_channels,
            num_layers=self.n_layers,
            batch_first=True,
            bias=False,
            dropout=self.dropout if self.n_layers > 1 else 0  # 只有在多层时才使用dropout
        )

        # 批归一化层
        self.bn = torch.nn.BatchNorm1d(self.hidden_channels)

        # 输出层：将GRU输出映射到最终输出维度
        self.linear = torch.nn.Linear(self.hidden_channels, self.output_shape)

        # 初始化权重
        self.init_weights()

        # 优化器将在init_model中初始化
        self.optimizer = None

    def init_weights(self):
        """初始化网络权重"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        """
        前向传播

        参数:
        x: 输入张量，形状为 [N, T, F_in]
           N: 批次大小（股票数量）
           T: 时间序列长度（时间步数）
           F_in: 输入特征维度（因子数量）

        返回:
        输出张量，形状为 [N, output_shape]
        """
        # 通过GRU网络
        x, _ = self.gru(x)  # [N, T, F_in] -> [N, T, H_hidden]

        # 取最后一个时间步的输出并应用激活函数
        x = f.relu(x[:, -1, :])  # [N, T, H_hidden] -> [N, H_hidden]

        # 批归一化
        x = self.bn(x)  # [N, H_hidden]

        # 线性变换到输出维度
        x = self.linear(x)  # [N, H_hidden] -> [N, output_shape]

        return x

    def init_model(self):
        """初始化模型和优化器"""
        self.model = GRU(
            input_channels=self.input_channels,
            hidden_channels=self.hidden_channels,
            output_shape=self.output_shape,
            n_layers=self.n_layers,
            time_steps=self.time_steps,
            device=self.device,
            epochs=self.epochs,
            loss=self.loss,
            lr=self.lr,
            weight_decay=self.weight_decay,
            dropout=self.dropout
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
                valid_mask = ~(torch.isnan(x_batch).any(dim=2).any(dim=0) | torch.isnan(y_batch).any(dim=0))
                if valid_mask.any():
                    x_valid_batch = x_batch[:, valid_mask, :]  # [T_batch, N_valid, C]
                    y_valid_batch = y_batch[:, valid_mask]     # [T_batch, N_valid]

                    # 重塑为适合GRU的格式: [N_valid, T_batch, C]
                    x_for_gru = x_valid_batch.transpose(0, 1)  # [N_valid, T_batch, C]
                    y_for_gru = y_valid_batch.transpose(0, 1)  # [N_valid, T_batch]

                    # 前向传播
                    predictions = self.model(x_for_gru)  # [N_valid, output_shape]

                    # 计算损失 - 确保标签形状与预测形状匹配
                    y_target = y_for_gru[:, -1]  # [N_valid] - 取最后一个时间步的真实值
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

                        # 过滤NaN值
                        val_valid_mask = ~(torch.isnan(x_val_batch).any(dim=2).any(dim=0) | torch.isnan(y_val_batch).any(dim=0))
                        if val_valid_mask.any():
                            x_val_valid = x_val_batch[:, val_valid_mask, :]  # [T_batch, N_valid, C]
                            y_val_valid = y_val_batch[:, val_valid_mask]     # [T_batch, N_valid]

                            # 准备GRU输入
                            x_val_for_gru = x_val_valid.transpose(0, 1)  # [N_valid, T_batch, C]
                            y_val_for_gru = y_val_valid.transpose(0, 1)  # [N_valid, T_batch]

                            val_preds = self.model(x_val_for_gru)  # [N_valid, output_shape]

                            # 确保标签形状与预测形状匹配
                            y_val_target = y_val_for_gru[:, -1]  # [N_valid] - 取最后一个时间步的真实值
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

                    # 交换维度以适应GRU: [time_steps, N_valid, C] -> [N_valid, time_steps, C]
                    x_for_prediction = x_valid_batch.transpose(0, 1)  # [N_valid, time_steps, C]

                    # 模型预测
                    valid_preds = self.model(x_for_prediction)  # [N_valid, output_shape]

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



class LSTM(Model):
    def __init__(self, input_channels: int, hidden_channels: int, output_shape: int = 1, n_layers: int = 1,
                 time_steps: int = 10, device: str = None, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
        """
        LSTM模型，适用于金融时间序列预测

        输入形状: [T, N, C]
        T: 时间步数
        N: 批次大小（股票数量）
        C: 输入特征维度（因子数量）
        """
        self.input_channels = input_channels      # 输入特征维度（因子数量）
        self.hidden_channels = hidden_channels    # 隐藏层维度
        self.output_shape = output_shape    # 输出维度
        self.time_steps = time_steps          # 时间步数
        self.n_layers = n_layers            # LSTM层数
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 构建LSTM网络
        self.lstm = torch.nn.LSTM(
            input_size=self.input_channels,
            hidden_size=self.hidden_channels,
            num_layers=self.n_layers,
            batch_first=True,
            bias=False,
            dropout=self.dropout if self.n_layers > 1 else 0  # 只有在多层时才使用dropout
        )

        # 批归一化层
        self.bn = torch.nn.BatchNorm1d(self.hidden_channels)

        # 输出层：将LSTM输出映射到最终输出维度
        self.linear = torch.nn.Linear(self.hidden_channels, self.output_shape)

        # 初始化权重
        self.init_weights()

        # 优化器将在init_model中初始化
        self.optimizer = None

    def init_weights(self):
        """初始化网络权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        """
        前向传播

        参数:
        x: 输入张量，形状为 [N, T, F_in]
           N: 批次大小（股票数量）
           T: 时间序列长度（时间步数）
           F_in: 输入特征维度（因子数量）

        返回:
        输出张量，形状为 [N, output_shape]
        """
        # 通过LSTM网络
        x, _ = self.lstm(x)  # [N, T, F_in] -> [N, T, H_hidden]

        # 取最后一个时间步的输出并应用激活函数
        x = f.relu(x[:, -1, :])  # [N, T, H_hidden] -> [N, H_hidden]

        # 批归一化
        x = self.bn(x)  # [N, H_hidden]

        # 线性变换到输出维度
        x = self.linear(x)  # [N, H_hidden] -> [N, output_shape]

        return x

    def init_model(self):
        """初始化模型和优化器"""
        self.model = LSTM(
            input_channels=self.input_channels,
            hidden_channels=self.hidden_channels,
            output_shape=self.output_shape,
            n_layers=self.n_layers,
            time_steps=self.time_steps,
            device=self.device,
            epochs=self.epochs,
            loss=self.loss,
            lr=self.lr,
            weight_decay=self.weight_decay,
            dropout=self.dropout
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
                valid_mask = ~(torch.isnan(x_batch).any(dim=2).any(dim=0) | torch.isnan(y_batch).any(dim=0))
                if valid_mask.any():
                    x_valid_batch = x_batch[:, valid_mask, :]  # [T_batch, N_valid, C]
                    y_valid_batch = y_batch[:, valid_mask]     # [T_batch, N_valid]

                    # 重塑为适合LSTM的格式: [N_valid, T_batch, C]
                    x_for_lstm = x_valid_batch.transpose(0, 1)  # [N_valid, T_batch, C]
                    y_for_lstm = y_valid_batch.transpose(0, 1)  # [N_valid, T_batch]

                    # 前向传播
                    predictions = self.model(x_for_lstm)  # [N_valid, output_shape]

                    # 计算损失 - 确保标签形状与预测形状匹配
                    y_target = y_for_lstm[:, -1]  # [N_valid] - 取最后一个时间步的真实值
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

                        # 过滤NaN值
                        val_valid_mask = ~(torch.isnan(x_val_batch).any(dim=2).any(dim=0) | torch.isnan(y_val_batch).any(dim=0))
                        if val_valid_mask.any():
                            x_val_valid = x_val_batch[:, val_valid_mask, :]  # [T_batch, N_valid, C]
                            y_val_valid = y_val_batch[:, val_valid_mask]     # [T_batch, N_valid]

                            # 准备LSTM输入
                            x_val_for_lstm = x_val_valid.transpose(0, 1)  # [N_valid, T_batch, C]
                            y_val_for_lstm = y_val_valid.transpose(0, 1)  # [N_valid, T_batch]

                            val_preds = self.model(x_val_for_lstm)  # [N_valid, output_shape]

                            # 确保标签形状与预测形状匹配
                            y_val_target = y_val_for_lstm[:, -1]  # [N_valid] - 取最后一个时间步的真实值
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

                    # 交换维度以适应LSTM: [time_steps, N_valid, C] -> [N_valid, time_steps, C]
                    x_for_prediction = x_valid_batch.transpose(0, 1)  # [N_valid, time_steps, C]

                    # 模型预测
                    valid_preds = self.model(x_for_prediction)  # [N_valid, output_shape]

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
