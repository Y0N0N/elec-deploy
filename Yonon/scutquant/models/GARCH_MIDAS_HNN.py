import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as f

import os
import math
import time

import numpy as np
from pandas import DataFrame, Series, concat, Grouper
from typing import List, Optional
from tqdm import tqdm

from .loss import *
from .models import Model, lr_scheduler
from ..utils import get_daily_inter, from_pandas_to_list, from_pandas_to_rnn, calc_kernel_size, transform_data

from sklearn.covariance import GraphicalLasso

def garch_nll(eps, sigma2):
    """
    Negative log-likelihood under Gaussian assumption
    """
    return 0.5 * (torch.log(sigma2) + eps.pow(2) / sigma2)


class HF_GARCH11(Model):
    def __init__(self, omega_init=0.1, alpha_init=0.1, beta_init=0.8,
                  lr=0.01, epochs=1000, batch_size=1, device=None, loss='mse_loss'):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(omega_init))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.lr = lr
        self.epochs = epochs
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
            self.parameters(),
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
            self.weight_decay = kwargs.get('weight_decay', 5e-4)

        # 早停机制初始化
        self.best_loss_val = float('inf')
        best_epoch = 0
        early_stop_count = 0

        # 保存最佳模型参数
        best_model_state = None

        self.init_model()
        self.to(self.device)

        # 早停机制初始化
        self.best_loss_val = float('inf')
        best_epoch = 0
        early_stop_count = 0
        has_valid = x_valid is not None and y_valid is not None

        # 保存最佳模型参数
        best_model_state = None

        # 获取维度信息
        D, T, N = x_train_tensor.shape

        # 按时间步批量训练
        for epoch in tqdm(range(self.epochs)):
            self.train()
            epoch_loss = 0.0
            batch_count = 0

            epoch_start_time = time.time()

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

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪（如果需要）
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

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
                    best_model_state = self.state_dict()
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
                    best_model_state = self.state_dict()
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
            'epochs': self.epochs,
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
        self.epochs = model_dict.get('epochs', 1000)
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


class HF_NNParamGARCH(Model):
    def __init__(
        self,
        input_channels=3,          # RV_{t-1}, r_{t-1}, I{r_{t-1}<0}
        hidden_channels=16,
        lr=0.01,
        epochs=200,
        loss='mse_loss',
        dropout_rate=0.02,
        device=None
    ):
        super().__init__()

        # -------- 参数估计网络 f_phi --------
        layers = [
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_channels, 3)  # 输出 tilde_omega, tilde_alpha, tilde_beta
        ]
        self.param_net = nn.Sequential(*layers)

        self._init_weights()

        self.lr = lr
        self.epochs = epochs
        self.loss = loss
        self.grad_clip = 1.0

        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.to(self.device)

    def _init_weights(self):
        """初始化神经网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _harmonize_params(self, raw_params):
        """
        raw_params: [T, N, 3]
        """
        tilde_omega = raw_params[..., 0]
        tilde_alpha = raw_params[..., 1]
        tilde_beta  = raw_params[..., 2]

        omega = torch.exp(tilde_omega)
        alpha = torch.sigmoid(tilde_alpha)
        beta  = (1.0 - alpha) * torch.sigmoid(tilde_beta)

        return omega, alpha, beta

    def forward(self, x, returns, pred_log_rv=True):
        """
        x       : [T, N, K]
        returns  : [T, N]
        """
        T, N = returns.shape

        # 神经网络输出原始参数
        raw_params = self.param_net(x)

        if pred_log_rv:
            omega = raw_params[..., 0]
            alpha = raw_params[..., 1]
            beta = raw_params[..., 2]
        else:
            omega, alpha, beta = self._harmonize_params(raw_params)

        # 初始化条件方差
        sigma2_list = []

        # 长期方差（数值稳定）
        initial_sigma2 = omega[0] / (1.0 - alpha[0] - beta[0] + 1e-8)
        first_sigma2 = omega[0] + alpha[0] * returns[0] ** 2 + beta[0] * initial_sigma2
        sigma2_list.append(first_sigma2)

        # GARCH 递推
        for t in range(1, T):
            next_sigma2 = (
                omega[t]
                + alpha[t] * returns[t] ** 2
                + beta[t] * sigma2_list[t-1]
            )
            sigma2_list.append(next_sigma2)

        # 将列表转换为张量
        sigma2 = torch.stack(sigma2_list, dim=0)

        return sigma2.unsqueeze(-1)

    def init_model(self):
        """初始化模型和优化器"""
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def fit(self, x_train_tensor, returns_train_tensor, y_train_tensor,
             x_valid=None, returns_valid=None, y_valid=None,
             pred_log_rv = True, **kwargs):
        """
        x       : [D, T, N, K]
        returns  : [D, T, N]
        label    : [D, T, N]
        """
        if kwargs.get('epochs', 100) is not None:
            self.epochs = kwargs.get('epochs', 100)
        if kwargs.get('early_stopping', 40) is not None:
            self.early_stopping = kwargs.get('early_stopping', 40)
        if kwargs.get('lr', 1e-3) is not None:
            self.lr = kwargs.get('lr', 1e-3)
        if kwargs.get('weight_decay', 5e-4) is not None:
            self.weight_decay = kwargs.get('weight_decay', 5e-4)
        if kwargs.get('grad_clip', 1) is not None:
            self.grad_clip = kwargs.get('grad_clip', 1)

        # 早停机制初始化
        self.best_loss_val = float('inf')
        best_epoch = 0
        early_stop_count = 0

        # 保存最佳模型参数
        best_model_state = None

        self.init_model()
        self.to(self.device)

        has_valid = x_valid is not None and y_valid is not None

        # 获取维度信息
        D, T, N, K = x_train_tensor.shape

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            batch_count = 0

            epoch_start_time = time.time()

            for d in range(0, D):
                # 获取当前时间步的收益率 [T, N]
                x_train_day = x_train_tensor[d].to(self.device)                 # [T, N, K]
                returns_train_day = returns_train_tensor[d].to(self.device)     # [T, N]
                label_day = y_train_tensor[d].to(self.device)

                predictions = self(x_train_day, returns_train_day, pred_log_rv)

                # 计算损失
                if isinstance(self.loss, str):
                    loss = eval("f." + self.loss + "(predictions.squeeze(-1), label_day)")
                else:
                    loss = self.loss(predictions.squeeze(-1), label_day)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

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

            # 验证集损失
            if has_valid:
                self.eval()
                val_loss = 0.0
                val_batch_count = 0

                D_valid = x_valid.shape[0]

                with torch.no_grad():
                    for d in range(0, D_valid):
                        # 获取当前时间步的收益率 [T, N]
                        x_valid_day = x_valid[d].to(self.device)
                        returns_valid_day = returns_valid[d].to(self.device)
                        label_day = y_valid[d].to(self.device)

                        predictions = self(x_valid_day, returns_valid_day)

                        # 计算损失
                        if isinstance(self.loss, str):
                            loss = eval("f." + self.loss + "(predictions.squeeze(-1), label_day)")
                        else:
                            loss = self.loss(predictions.squeeze(-1), label_day)

                        val_loss += loss.item()
                        val_batch_count += 1

                    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

                    # 早停逻辑
                    if avg_val_loss < self.best_loss_val:
                        self.best_loss_val = avg_val_loss
                        best_epoch = epoch + 1
                        early_stop_count = 0
                        # 保存最佳模型参数
                        best_model_state = self.state_dict()
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
                    best_model_state = self.state_dict()
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
            self.load_state_dict(best_model_state)

        # 打印最优结果
        print(f"\nTraining completed!")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Loss: {self.best_loss_val:.4f}")

        return self

    def predict(self, x_test_tensor, returns_test_tensor, pred_log_rv=True):
        """
        生成预测结果，返回条件方差预测

        参数:
        x_test: 测试数据，torch.Tensor [时间, 股票, 因子]

        返回:
        torch.Tensor [时间, 股票] - 条件方差预测
        """
        if isinstance(x_test_tensor, torch.Tensor) and isinstance(returns_test_tensor, torch.Tensor):
            return self._predict_tensor(x_test_tensor, returns_test_tensor, pred_log_rv)
        else:
            raise TypeError("输入类型不支持，请使用torch.Tensor")

    def _predict_tensor(self, x_test_tensor, returns_test_tensor, pred_log_rv=True):
        """
        使用PyTorch张量进行预测，返回条件方差

        x_test_tensor形状: [日期, 时间, 股票, 因子]
        返回形状: [日期, 时间, 股票] - 条件方差预测
        """
        # 确保输入形状正确 [时间, 股票, 因子]
        if len(x_test_tensor.shape) != 4:
            raise ValueError(f"输入张量形状应为[日期, 时间, 股票, 因子]，实际为{x_test_tensor.shape}")

        # 获取维度信息
        D, T, N, K = x_test_tensor.shape

        # 创建输出张量
        y_pred = torch.full((D, T, N), np.nan, dtype=torch.float32, device=self.device)

        # 逐时间步处理
        for d in range(D):
            x_test_d = x_test_tensor[d].to(self.device)
            returns_test_d = returns_test_tensor[d].to(self.device)

            # 计算条件方差
            with torch.no_grad():
                sigma2 = self(x_test_d, returns_test_d, pred_log_rv)
                # 将预测结果放回到正确位置
                y_pred[d] = sigma2.squeeze(-1)

        return y_pred


class HF_RNNParamGARCH(Model):
    """
    RNN-based time-varying parameter GARCH(1,1)
    """

    def __init__(
        self,
        input_channels=3,      # x_{t-1}: RV, r, I{r<0}
        hidden_channels=16,    # RNN hidden state dimension
        rnn_type='GRU',        # 'RNN', 'LSTM', 'GRU'
        dropout_rate=0.02,
        device=None
    ):
        super().__init__()

        # -------- RNN module f_RNN --------
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(
                input_channels,
                hidden_channels,
                batch_first=True
            )
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_channels,
                hidden_channels,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_channels,
                hidden_channels,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type must be RNN, LSTM, or GRU")

        # -------- 参数生成网络 f_phi --------
        layers = [
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_channels, 3)  # 输出 tilde_omega, tilde_alpha, tilde_beta
        ]
        self.param_net = nn.Sequential(*layers)

        self._init_weights()

        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.to(self.device)

    def _init_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # -------- 参数调和映射（与个体范式完全一致）--------
    def _harmonize_params(self, raw_params):
        """
        raw_params: [T, N, 3]
        """
        tilde_omega = raw_params[..., 0]
        tilde_alpha = raw_params[..., 1]
        tilde_beta  = raw_params[..., 2]

        omega = torch.exp(tilde_omega)
        alpha = torch.sigmoid(tilde_alpha)
        beta  = (1.0 - alpha) * torch.sigmoid(tilde_beta)

        return omega, alpha, beta

    def forward(self, x, returns, pred_log_rv=False):
        """
        x       : [T, N, K]   -> x_{t-1}
        returns : [T, N]
        """

        T, N, _ = x.shape

        # -------- RNN hidden state evolution (Eq. 11) --------
        # reshape to batch_first: [N, T, K]
        x_rnn = x.permute(1, 0, 2)

        h_seq, _ = self.rnn(x_rnn)
        # h_seq: [N, T, hidden_channels]

        # back to [T, N, hidden_channels]
        h_seq = h_seq.permute(1, 0, 2)

        # -------- Parameter generation (Eq. 12) --------
        raw_params = self.param_net(h_seq)

        if pred_log_rv:
            omega = raw_params[..., 0]
            alpha = raw_params[..., 1]
            beta  = raw_params[..., 2]
        else:
            omega, alpha, beta = self._harmonize_params(raw_params)

        # -------- GARCH recursion --------
        sigma2_list = []

        # unconditional variance for initialization
        initial_sigma2 = omega[0] / (1.0 - alpha[0] - beta[0] + 1e-8)

        first_sigma2 = (
            omega[0]
            + alpha[0] * returns[0] ** 2
            + beta[0] * initial_sigma2
        )
        sigma2_list.append(first_sigma2)

        for t in range(1, T):
            sigma2_t = (
                omega[t]
                + alpha[t] * returns[t] ** 2
                + beta[t] * sigma2_list[t-1]
            )
            sigma2_list.append(sigma2_t)

        sigma2 = torch.stack(sigma2_list, dim=0)

        return sigma2.unsqueeze(-1)

    def init_model(self):
        """初始化模型和优化器"""
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def fit(self, x_train_tensor, returns_train_tensor, y_train_tensor,
             x_valid=None, returns_valid=None, y_valid=None,
             pred_log_rv = True, **kwargs):
        """
        x       : [D, T, N, K]
        returns  : [D, T, N]
        label    : [D, T, N]
        """
        if kwargs.get('epochs', 100) is not None:
            self.epochs = kwargs.get('epochs', 100)
        if kwargs.get('early_stopping', 40) is not None:
            self.early_stopping = kwargs.get('early_stopping', 40)
        if kwargs.get('lr', 1e-3) is not None:
            self.lr = kwargs.get('lr', 1e-3)
        if kwargs.get('weight_decay', 5e-4) is not None:
            self.weight_decay = kwargs.get('weight_decay', 5e-4)
        if kwargs.get('grad_clip', 2) is not None:
            self.grad_clip = kwargs.get('grad_clip', 2)

        # 早停机制初始化
        self.best_loss_val = float('inf')
        best_epoch = 0
        early_stop_count = 0

        # 保存最佳模型参数
        best_model_state = None

        self.init_model()
        self.to(self.device)

        has_valid = x_valid is not None and y_valid is not None

        # 获取维度信息
        D, T, N, K = x_train_tensor.shape

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            batch_count = 0

            epoch_start_time = time.time()

            for d in range(0, D):
                # 获取当前时间步的收益率 [T, N]
                x_train_day = x_train_tensor[d].to(self.device)                 # [T, N, K]
                returns_train_day = returns_train_tensor[d].to(self.device)     # [T, N]
                label_day = y_train_tensor[d].to(self.device)

                predictions = self(x_train_day, returns_train_day, pred_log_rv)

                # 计算损失
                if isinstance(self.loss, str):
                    loss = eval("f." + self.loss + "(predictions.squeeze(-1), label_day)")
                else:
                    loss = self.loss(predictions.squeeze(-1), label_day)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

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

            # 验证集损失
            if has_valid:
                self.eval()
                val_loss = 0.0
                val_batch_count = 0

                D_valid = x_valid.shape[0]

                with torch.no_grad():
                    for d in range(0, D_valid):
                        # 获取当前时间步的收益率 [T, N]
                        x_valid_day = x_valid[d].to(self.device)
                        returns_valid_day = returns_valid[d].to(self.device)
                        label_day = y_valid[d].to(self.device)

                        predictions = self(x_valid_day, returns_valid_day)

                        # 计算损失
                        if isinstance(self.loss, str):
                            loss = eval("f." + self.loss + "(predictions.squeeze(-1), label_day)")
                        else:
                            loss = self.loss(predictions.squeeze(-1), label_day)

                        val_loss += loss.item()
                        val_batch_count += 1

                    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

                    # 早停逻辑
                    if avg_val_loss < self.best_loss_val:
                        self.best_loss_val = avg_val_loss
                        best_epoch = epoch + 1
                        early_stop_count = 0
                        # 保存最佳模型参数
                        best_model_state = self.state_dict()
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
                    best_model_state = self.state_dict()
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
            self.load_state_dict(best_model_state)

        # 打印最优结果
        print(f"\nTraining completed!")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Loss: {self.best_loss_val:.4f}")

        return self

    def predict(self, x_test_tensor, returns_test_tensor, pred_log_rv=True):
        """
        生成预测结果，返回条件方差预测

        参数:
        x_test: 测试数据，torch.Tensor [时间, 股票, 因子]

        返回:
        torch.Tensor [时间, 股票] - 条件方差预测
        """
        if isinstance(x_test_tensor, torch.Tensor) and isinstance(returns_test_tensor, torch.Tensor):
            return self._predict_tensor(x_test_tensor, returns_test_tensor, pred_log_rv)
        else:
            raise TypeError("输入类型不支持，请使用torch.Tensor")

    def _predict_tensor(self, x_test_tensor, returns_test_tensor, pred_log_rv=True):
        """
        使用PyTorch张量进行预测，返回条件方差

        x_test_tensor形状: [日期, 时间, 股票, 因子]
        返回形状: [日期, 时间, 股票] - 条件方差预测
        """
        # 确保输入形状正确 [时间, 股票, 因子]
        if len(x_test_tensor.shape) != 4:
            raise ValueError(f"输入张量形状应为[日期, 时间, 股票, 因子]，实际为{x_test_tensor.shape}")

        # 获取维度信息
        D, T, N, K = x_test_tensor.shape

        # 创建输出张量
        y_pred = torch.full((D, T, N), np.nan, dtype=torch.float32, device=self.device)

        # 逐时间步处理
        for d in range(D):
            x_test_d = x_test_tensor[d].to(self.device)
            returns_test_d = returns_test_tensor[d].to(self.device)

            # 计算条件方差
            with torch.no_grad():
                sigma2 = self(x_test_d, returns_test_d, pred_log_rv)
                # 将预测结果放回到正确位置
                y_pred[d] = sigma2.squeeze(-1)

        return y_pred


class HF_RNNParamGARCH_v2(Model):
    """
    RNN-based time-varying parameter GARCH(1,1)
    """

    def __init__(
        self,
        input_channels=3,      # x_{t-1}: RV, r, I{r<0}
        hidden_channels=16,    # RNN hidden state dimension
        rnn_type='GRU',        # 'RNN', 'LSTM', 'GRU'
        dropout_rate=0.02,
        device=None
    ):
        super().__init__()

        preprocess_layers = [
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        ]
        self.preprocess_net = nn.Sequential(*preprocess_layers)

        # -------- RNN module f_RNN --------
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(
                hidden_channels,
                hidden_channels,
                batch_first=True
            )
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                hidden_channels,
                hidden_channels,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                hidden_channels,
                hidden_channels,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type must be RNN, LSTM, or GRU")

        # -------- 参数生成网络 f_phi --------
        layers = [
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_channels, 3)  # 输出 tilde_omega, tilde_alpha, tilde_beta
        ]
        self.param_net = nn.Sequential(*layers)

        self._init_weights()

        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.to(self.device)

    def _init_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # -------- 参数调和映射（与个体范式完全一致）--------
    def _harmonize_params(self, raw_params):
        """
        raw_params: [T, N, 3]
        """
        tilde_omega = raw_params[..., 0]
        tilde_alpha = raw_params[..., 1]
        tilde_beta  = raw_params[..., 2]

        omega = torch.exp(tilde_omega)
        alpha = torch.sigmoid(tilde_alpha)
        beta  = (1.0 - alpha) * torch.sigmoid(tilde_beta)

        return omega, alpha, beta

    def forward(self, x, returns, pred_log_rv=False):
        """
        x       : [T, N, K]   -> x_{t-1}
        returns : [T, N]
        """

        T, N, _ = x.shape

        # -------- RNN hidden state evolution (Eq. 11) --------
        # reshape to batch_first: [N, T, K]
        x = self.preprocess_net(x)
        x_rnn = x.permute(1, 0, 2)

        h_seq, _ = self.rnn(x_rnn)
        # h_seq: [N, T, hidden_channels]

        # back to [T, N, hidden_channels]
        h_seq = h_seq.permute(1, 0, 2)

        # -------- Parameter generation (Eq. 12) --------
        raw_params = self.param_net(torch.cat([x, h_seq], dim=-1))

        if pred_log_rv:
            omega = raw_params[..., 0]
            alpha = raw_params[..., 1]
            beta  = raw_params[..., 2]
        else:
            omega, alpha, beta = self._harmonize_params(raw_params)

        # -------- GARCH recursion --------
        sigma2_list = []

        # unconditional variance for initialization
        initial_sigma2 = omega[0] / (1.0 - alpha[0] - beta[0] + 1e-8)

        first_sigma2 = (
            omega[0]
            + alpha[0] * returns[0] ** 2
            + beta[0] * initial_sigma2
        )
        sigma2_list.append(first_sigma2)

        for t in range(1, T):
            sigma2_t = (
                omega[t]
                + alpha[t] * returns[t] ** 2
                + beta[t] * sigma2_list[t-1]
            )
            sigma2_list.append(sigma2_t)

        sigma2 = torch.stack(sigma2_list, dim=0)

        return sigma2.unsqueeze(-1)

    def init_model(self):
        """初始化模型和优化器"""
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def fit(self, x_train_tensor, returns_train_tensor, y_train_tensor,
             x_valid=None, returns_valid=None, y_valid=None,
             pred_log_rv = True, **kwargs):
        """
        x       : [D, T, N, K]
        returns  : [D, T, N]
        label    : [D, T, N]
        """
        if kwargs.get('epochs', 100) is not None:
            self.epochs = kwargs.get('epochs', 100)
        if kwargs.get('early_stopping', 40) is not None:
            self.early_stopping = kwargs.get('early_stopping', 40)
        if kwargs.get('lr', 1e-3) is not None:
            self.lr = kwargs.get('lr', 1e-3)
        if kwargs.get('weight_decay', 5e-4) is not None:
            self.weight_decay = kwargs.get('weight_decay', 5e-4)
        if kwargs.get('grad_clip', 1) is not None:
            self.grad_clip = kwargs.get('grad_clip', 1)

        # 早停机制初始化
        self.best_loss_val = float('inf')
        best_epoch = 0
        early_stop_count = 0

        # 保存最佳模型参数
        best_model_state = None

        self.init_model()
        self.to(self.device)

        has_valid = x_valid is not None and y_valid is not None

        # 获取维度信息
        D, T, N, K = x_train_tensor.shape

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            batch_count = 0

            epoch_start_time = time.time()

            for d in range(0, D):
                # 获取当前时间步的收益率 [T, N]
                x_train_day = x_train_tensor[d].to(self.device)                 # [T, N, K]
                returns_train_day = returns_train_tensor[d].to(self.device)     # [T, N]
                label_day = y_train_tensor[d].to(self.device)

                predictions = self(x_train_day, returns_train_day, pred_log_rv)

                # 计算损失
                if isinstance(self.loss, str):
                    loss = eval("f." + self.loss + "(predictions.squeeze(-1), label_day)")
                else:
                    loss = self.loss(predictions.squeeze(-1), label_day)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

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

            # 验证集损失
            if has_valid:
                self.eval()
                val_loss = 0.0
                val_batch_count = 0

                D_valid = x_valid.shape[0]

                with torch.no_grad():
                    for d in range(0, D_valid):
                        # 获取当前时间步的收益率 [T, N]
                        x_valid_day = x_valid[d].to(self.device)
                        returns_valid_day = returns_valid[d].to(self.device)
                        label_day = y_valid[d].to(self.device)

                        predictions = self(x_valid_day, returns_valid_day)

                        # 计算损失
                        if isinstance(self.loss, str):
                            loss = eval("f." + self.loss + "(predictions.squeeze(-1), label_day)")
                        else:
                            loss = self.loss(predictions.squeeze(-1), label_day)

                        val_loss += loss.item()
                        val_batch_count += 1

                    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

                    # 早停逻辑
                    if avg_val_loss < self.best_loss_val:
                        self.best_loss_val = avg_val_loss
                        best_epoch = epoch + 1
                        early_stop_count = 0
                        # 保存最佳模型参数
                        best_model_state = self.state_dict()
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
                    best_model_state = self.state_dict()
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
            self.load_state_dict(best_model_state)

        # 打印最优结果
        print(f"\nTraining completed!")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Loss: {self.best_loss_val:.4f}")

        return self

    def predict(self, x_test_tensor, returns_test_tensor, pred_log_rv=True):
        """
        生成预测结果，返回条件方差预测

        参数:
        x_test: 测试数据，torch.Tensor [时间, 股票, 因子]

        返回:
        torch.Tensor [时间, 股票] - 条件方差预测
        """
        if isinstance(x_test_tensor, torch.Tensor) and isinstance(returns_test_tensor, torch.Tensor):
            return self._predict_tensor(x_test_tensor, returns_test_tensor, pred_log_rv)
        else:
            raise TypeError("输入类型不支持，请使用torch.Tensor")

    def _predict_tensor(self, x_test_tensor, returns_test_tensor, pred_log_rv=True):
        """
        使用PyTorch张量进行预测，返回条件方差

        x_test_tensor形状: [日期, 时间, 股票, 因子]
        返回形状: [日期, 时间, 股票] - 条件方差预测
        """
        # 确保输入形状正确 [时间, 股票, 因子]
        if len(x_test_tensor.shape) != 4:
            raise ValueError(f"输入张量形状应为[日期, 时间, 股票, 因子]，实际为{x_test_tensor.shape}")

        # 获取维度信息
        D, T, N, K = x_test_tensor.shape

        # 创建输出张量
        y_pred = torch.full((D, T, N), np.nan, dtype=torch.float32, device=self.device)

        # 逐时间步处理
        for d in range(D):
            x_test_d = x_test_tensor[d].to(self.device)
            returns_test_d = returns_test_tensor[d].to(self.device)

            # 计算条件方差
            with torch.no_grad():
                sigma2 = self(x_test_d, returns_test_d, pred_log_rv)
                # 将预测结果放回到正确位置
                y_pred[d] = sigma2.squeeze(-1)

        return y_pred


class NNParamGARCH(Model):
    def __init__(
        self,
        input_channels=3,          # RV_{t-1}, r_{t-1}, I{r_{t-1}<0}
        hidden_channels=16,
        lr=0.01,
        epochs=200,
        loss='mse_loss',
        dropout_rate=0.02,
        device=None
    ):
        super().__init__()

        # -------- 参数估计网络 f_phi --------
        layers = [
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_channels, 3)  # 输出 tilde_omega, tilde_alpha, tilde_beta
        ]
        self.param_net = nn.Sequential(*layers)

        self._init_weights()

        self.lr = lr
        self.epochs = epochs
        self.loss = loss
        self.grad_clip = 1.0

        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.to(self.device)

    def _init_weights(self):
        """初始化神经网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _harmonize_params(self, raw_params):
        """
        raw_params: [T, N, 3]
        """
        tilde_omega = raw_params[..., 0]
        tilde_alpha = raw_params[..., 1]
        tilde_beta  = raw_params[..., 2]

        omega = torch.exp(tilde_omega)
        alpha = torch.sigmoid(tilde_alpha)
        beta  = (1.0 - alpha) * torch.sigmoid(tilde_beta)

        return omega, alpha, beta

    def forward(self, x, returns, prev_sigma2=None, pred_log_rv=True):
        """
        x       : [T, N, K]
        returns  : [T, N]
        """
        T, N = returns.shape

        # 神经网络输出原始参数
        raw_params = self.param_net(x)

        if pred_log_rv:
            omega = raw_params[..., 0]
            alpha = raw_params[..., 1]
            beta = raw_params[..., 2]
        else:
            omega, alpha, beta = self._harmonize_params(raw_params)

        # 初始化条件方差
        sigma2_list = []

        if prev_sigma2 is None:
            sigma2_0 = omega[0] / (1.0 - alpha[0] - beta[0] + 1e-8)
        else:
            sigma2_0 = prev_sigma2

        first_sigma2 = (
            omega[0]
            + alpha[0] * returns[0] ** 2
            + beta[0] * sigma2_0
        )

        sigma2_list = [first_sigma2]

        # GARCH 递推
        for t in range(1, T):
            next_sigma2 = (
                omega[t]
                + alpha[t] * returns[t] ** 2
                + beta[t] * sigma2_list[t-1]
            )
            sigma2_list.append(next_sigma2)

        # 将列表转换为张量
        sigma2 = torch.stack(sigma2_list, dim=0)

        return sigma2.unsqueeze(-1)

    def init_model(self):
        """初始化模型和优化器"""
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def fit(self, x_train_tensor, returns_train_tensor, y_train_tensor,
             x_valid=None, returns_valid=None, y_valid=None,
             pred_log_rv = True, **kwargs):
        """
        x       : [D, T, N, K]
        returns  : [D, T, N]
        label    : [D, T, N]
        """
        if kwargs.get('epochs', 100) is not None:
            self.epochs = kwargs.get('epochs', 100)
        if kwargs.get('early_stopping', 40) is not None:
            self.early_stopping = kwargs.get('early_stopping', 40)
        if kwargs.get('lr', 1e-3) is not None:
            self.lr = kwargs.get('lr', 1e-3)
        if kwargs.get('weight_decay', 5e-4) is not None:
            self.weight_decay = kwargs.get('weight_decay', 5e-4)
        if kwargs.get('grad_clip', 1) is not None:
            self.grad_clip = kwargs.get('grad_clip', 1)

        # 早停机制初始化
        self.best_loss_val = float('inf')
        best_epoch = 0
        early_stop_count = 0

        # 保存最佳模型参数
        best_model_state = None

        self.init_model()
        self.to(self.device)

        has_valid = x_valid is not None and y_valid is not None

        # 获取维度信息
        D, T, N, K = x_train_tensor.shape

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            batch_count = 0

            epoch_start_time = time.time()
            prev_sigma2 = None

            for d in range(0, D):
                # 获取当前时间步的收益率 [T, N]
                x_train_day = x_train_tensor[d].to(self.device)                 # [T, N, K]
                returns_train_day = returns_train_tensor[d].to(self.device)     # [T, N]
                label_day = y_train_tensor[d].to(self.device)

                predictions = self(x_train_day, returns_train_day, prev_sigma2, pred_log_rv)
                prev_sigma2 = predictions[-1].detach().squeeze(-1)

                # 计算损失
                if isinstance(self.loss, str):
                    loss = eval("f." + self.loss + "(predictions.squeeze(-1), label_day)")
                else:
                    loss = self.loss(predictions.squeeze(-1), label_day)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

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

            # 验证集损失
            if has_valid:
                self.eval()
                val_loss = 0.0
                val_batch_count = 0

                D_valid = x_valid.shape[0]

                with torch.no_grad():
                    for d in range(0, D_valid):
                        # 获取当前时间步的收益率 [T, N]
                        x_valid_day = x_valid[d].to(self.device)
                        returns_valid_day = returns_valid[d].to(self.device)
                        label_day = y_valid[d].to(self.device)

                        predictions = self(x_valid_day, returns_valid_day, prev_sigma2, pred_log_rv)
                        prev_sigma2 = predictions[-1].detach().squeeze(-1)

                        # 计算损失
                        if isinstance(self.loss, str):
                            loss = eval("f." + self.loss + "(predictions.squeeze(-1), label_day)")
                        else:
                            loss = self.loss(predictions.squeeze(-1), label_day)

                        val_loss += loss.item()
                        val_batch_count += 1

                    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

                    # 早停逻辑
                    if avg_val_loss < self.best_loss_val:
                        self.best_loss_val = avg_val_loss
                        best_epoch = epoch + 1
                        early_stop_count = 0
                        # 保存最佳模型参数
                        best_model_state = self.state_dict()
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
                    best_model_state = self.state_dict()
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
            self.load_state_dict(best_model_state)

        # 打印最优结果
        print(f"\nTraining completed!")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Loss: {self.best_loss_val:.4f}")

        return self

    def predict(self, x_test_tensor, returns_test_tensor, pred_log_rv=True):
        """
        生成预测结果，返回条件方差预测

        参数:
        x_test: 测试数据，torch.Tensor [时间, 股票, 因子]

        返回:
        torch.Tensor [时间, 股票] - 条件方差预测
        """
        if isinstance(x_test_tensor, torch.Tensor) and isinstance(returns_test_tensor, torch.Tensor):
            return self._predict_tensor(x_test_tensor, returns_test_tensor, pred_log_rv)
        else:
            raise TypeError("输入类型不支持，请使用torch.Tensor")

    def _predict_tensor(self, x_test_tensor, returns_test_tensor, pred_log_rv=True):
        """
        使用PyTorch张量进行预测，返回条件方差

        x_test_tensor形状: [日期, 时间, 股票, 因子]
        返回形状: [日期, 时间, 股票] - 条件方差预测
        """
        # 确保输入形状正确 [时间, 股票, 因子]
        if len(x_test_tensor.shape) != 4:
            raise ValueError(f"输入张量形状应为[日期, 时间, 股票, 因子]，实际为{x_test_tensor.shape}")

        # 获取维度信息
        D, T, N, K = x_test_tensor.shape

        # 创建输出张量
        y_pred = torch.full((D, T, N), np.nan, dtype=torch.float32, device=self.device)
        prev_sigma2 = None

        # 逐时间步处理
        for d in range(D):
            x_test_d = x_test_tensor[d].to(self.device)
            returns_test_d = returns_test_tensor[d].to(self.device)

            # 计算条件方差
            with torch.no_grad():
                sigma2 = self(x_test_d, returns_test_d, prev_sigma2, pred_log_rv)
                prev_sigma2 = sigma2[-1].detach().squeeze(-1)
                # 将预测结果放回到正确位置
                y_pred[d] = sigma2.squeeze(-1)

        return y_pred


class RNNParamGARCH(Model):
    """
    RNN-based time-varying parameter GARCH(1,1)
    """

    def __init__(
        self,
        input_channels=3,      # x_{t-1}: RV, r, I{r<0}
        hidden_channels=16,    # RNN hidden state dimension
        rnn_type='GRU',        # 'RNN', 'LSTM', 'GRU'
        dropout_rate=0.02,
        device=None
    ):
        super().__init__()

        # -------- RNN module f_RNN --------
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(
                input_channels,
                hidden_channels,
                batch_first=True
            )
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_channels,
                hidden_channels,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_channels,
                hidden_channels,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type must be RNN, LSTM, or GRU")

        # -------- 参数生成网络 f_phi --------
        layers = [
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_channels, 3)  # 输出 tilde_omega, tilde_alpha, tilde_beta
        ]
        self.param_net = nn.Sequential(*layers)

        self._init_weights()

        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.to(self.device)

    def _init_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # -------- 参数调和映射（与个体范式完全一致）--------
    def _harmonize_params(self, raw_params):
        """
        raw_params: [T, N, 3]
        """
        tilde_omega = raw_params[..., 0]
        tilde_alpha = raw_params[..., 1]
        tilde_beta  = raw_params[..., 2]

        omega = torch.exp(tilde_omega)
        alpha = torch.sigmoid(tilde_alpha)
        beta  = (1.0 - alpha) * torch.sigmoid(tilde_beta)

        return omega, alpha, beta

    def forward(self, x, returns, prev_sigma2=None, pred_log_rv=False):
        """
        x       : [T, N, K]   -> x_{t-1}
        returns : [T, N]
        """

        T, N, _ = x.shape

        # -------- RNN hidden state evolution (Eq. 11) --------
        # reshape to batch_first: [N, T, K]
        x_rnn = x.permute(1, 0, 2)

        h_seq, _ = self.rnn(x_rnn)
        # h_seq: [N, T, hidden_channels]

        # back to [T, N, hidden_channels]
        h_seq = h_seq.permute(1, 0, 2)

        # -------- Parameter generation (Eq. 12) --------
        raw_params = self.param_net(h_seq)

        if pred_log_rv:
            omega = raw_params[..., 0]
            alpha = raw_params[..., 1]
            beta  = raw_params[..., 2]
        else:
            omega, alpha, beta = self._harmonize_params(raw_params)

        # -------- GARCH recursion --------
        # 初始化条件方差
        sigma2_list = []

        if prev_sigma2 is None:
            sigma2_0 = omega[0] / (1.0 - alpha[0] - beta[0] + 1e-8)
        else:
            sigma2_0 = prev_sigma2

        first_sigma2 = (
            omega[0]
            + alpha[0] * returns[0] ** 2
            + beta[0] * sigma2_0
        )

        sigma2_list = [first_sigma2]

        for t in range(1, T):
            sigma2_t = (
                omega[t]
                + alpha[t] * returns[t] ** 2
                + beta[t] * sigma2_list[t-1]
            )
            sigma2_list.append(sigma2_t)

        sigma2 = torch.stack(sigma2_list, dim=0)

        return sigma2.unsqueeze(-1)

    def init_model(self):
        """初始化模型和优化器"""
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def fit(self, x_train_tensor, returns_train_tensor, y_train_tensor,
             x_valid=None, returns_valid=None, y_valid=None,
             pred_log_rv = True, **kwargs):
        """
        x       : [D, T, N, K]
        returns  : [D, T, N]
        label    : [D, T, N]
        """
        if kwargs.get('epochs', 100) is not None:
            self.epochs = kwargs.get('epochs', 100)
        if kwargs.get('early_stopping', 40) is not None:
            self.early_stopping = kwargs.get('early_stopping', 40)
        if kwargs.get('lr', 1e-3) is not None:
            self.lr = kwargs.get('lr', 1e-3)
        if kwargs.get('weight_decay', 5e-4) is not None:
            self.weight_decay = kwargs.get('weight_decay', 5e-4)
        if kwargs.get('grad_clip', 2) is not None:
            self.grad_clip = kwargs.get('grad_clip', 2)

        # 早停机制初始化
        self.best_loss_val = float('inf')
        best_epoch = 0
        early_stop_count = 0

        # 保存最佳模型参数
        best_model_state = None

        self.init_model()
        self.to(self.device)

        has_valid = x_valid is not None and y_valid is not None

        # 获取维度信息
        D, T, N, K = x_train_tensor.shape

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            batch_count = 0

            epoch_start_time = time.time()
            prev_sigma2 = None

            for d in range(0, D):
                # 获取当前时间步的收益率 [T, N]
                x_train_day = x_train_tensor[d].to(self.device)                 # [T, N, K]
                returns_train_day = returns_train_tensor[d].to(self.device)     # [T, N]
                label_day = y_train_tensor[d].to(self.device)

                predictions = self(x_train_day, returns_train_day, prev_sigma2, pred_log_rv)
                # 更新prev_sigma2为当前时间步的预测值
                prev_sigma2 = predictions[-1].detach().squeeze(-1)

                # 计算损失
                if isinstance(self.loss, str):
                    loss = eval("f." + self.loss + "(predictions.squeeze(-1), label_day)")
                else:
                    loss = self.loss(predictions.squeeze(-1), label_day)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

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

            # 验证集损失
            if has_valid:
                self.eval()
                val_loss = 0.0
                val_batch_count = 0

                D_valid = x_valid.shape[0]

                with torch.no_grad():
                    for d in range(0, D_valid):
                        # 获取当前时间步的收益率 [T, N]
                        x_valid_day = x_valid[d].to(self.device)
                        returns_valid_day = returns_valid[d].to(self.device)
                        label_day = y_valid[d].to(self.device)

                        predictions = self(x_valid_day, returns_valid_day, prev_sigma2, pred_log_rv)
                        # 更新prev_sigma2为当前时间步的预测值
                        prev_sigma2 = predictions[-1].detach().squeeze(-1)

                        # 计算损失
                        if isinstance(self.loss, str):
                            loss = eval("f." + self.loss + "(predictions.squeeze(-1), label_day)")
                        else:
                            loss = self.loss(predictions.squeeze(-1), label_day)

                        val_loss += loss.item()
                        val_batch_count += 1

                    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

                    # 早停逻辑
                    if avg_val_loss < self.best_loss_val:
                        self.best_loss_val = avg_val_loss
                        best_epoch = epoch + 1
                        early_stop_count = 0
                        # 保存最佳模型参数
                        best_model_state = self.state_dict()
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
                    best_model_state = self.state_dict()
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
            self.load_state_dict(best_model_state)

        # 打印最优结果
        print(f"\nTraining completed!")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Loss: {self.best_loss_val:.4f}")

        return self

    def predict(self, x_test_tensor, returns_test_tensor, pred_log_rv=True):
        """
        生成预测结果，返回条件方差预测

        参数:
        x_test: 测试数据，torch.Tensor [时间, 股票, 因子]

        返回:
        torch.Tensor [时间, 股票] - 条件方差预测
        """
        if isinstance(x_test_tensor, torch.Tensor) and isinstance(returns_test_tensor, torch.Tensor):
            return self._predict_tensor(x_test_tensor, returns_test_tensor, pred_log_rv)
        else:
            raise TypeError("输入类型不支持，请使用torch.Tensor")

    def _predict_tensor(self, x_test_tensor, returns_test_tensor, pred_log_rv=True):
        """
        使用PyTorch张量进行预测，返回条件方差

        x_test_tensor形状: [日期, 时间, 股票, 因子]
        返回形状: [日期, 时间, 股票] - 条件方差预测
        """
        # 确保输入形状正确 [时间, 股票, 因子]
        if len(x_test_tensor.shape) != 4:
            raise ValueError(f"输入张量形状应为[日期, 时间, 股票, 因子]，实际为{x_test_tensor.shape}")

        # 获取维度信息
        D, T, N, K = x_test_tensor.shape

        # 创建输出张量
        y_pred = torch.full((D, T, N), np.nan, dtype=torch.float32, device=self.device)
        prev_sigma2 = None

        # 逐时间步处理
        for d in range(D):
            x_test_d = x_test_tensor[d].to(self.device)
            returns_test_d = returns_test_tensor[d].to(self.device)

            # 计算条件方差
            with torch.no_grad():
                sigma2 = self(x_test_d, returns_test_d, prev_sigma2, pred_log_rv)
                # 更新prev_sigma2为当前时间步的预测值
                prev_sigma2 = sigma2[-1].detach().squeeze(-1)
                # 将预测结果放回到正确位置
                y_pred[d] = sigma2.squeeze(-1)

        return y_pred


def GLasso(
    X: torch.Tensor,
    alpha: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-4,
    standardize: bool = True,
    ):
    """
    Graphical Lasso for [T, N] tensor

    Parameters
    ----------
    X : torch.Tensor
        [T, N]，T=时间，N=股票
    alpha : float
        L1 正则强度
    standardize : bool
        是否标准化（强烈建议 True）

    Returns
    -------
    precision : torch.Tensor
        [N, N] 稀疏精度矩阵（图结构）
    covariance : torch.Tensor
        [N, N] 协方差矩阵
    """

    assert X.dim() == 2, "Input must be [T,N]"

    # 转 numpy
    X_np = X.detach().cpu().numpy()

    # ===== 标准化（非常重要）=====
    if standardize:
        X_np = (X_np - X_np.mean(axis=0)) / (X_np.std(axis=0) + 1e-8)

    # ===== Graphical Lasso =====
    model = GraphicalLasso(
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
    )

    model.fit(X_np)

    precision = torch.from_numpy(model.precision_).float()
    covariance = torch.from_numpy(model.covariance_).float()

    return precision, covariance


class NN_Beta_MIDAS(Model):
    """
    输入:  RV_lags [K,N]
    输出:  tau [N]
    """

    def __init__(
        self,
        K=7,
        hidden_size=16,
        dropout_rate=0.0,
        device=None,
    ):
        super().__init__()

        self.K = K

        # ===== 权重生成网络 =====
        self.theta_net = nn.Sequential(
            nn.Linear(K, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate>0 else nn.Identity(),
            nn.Linear(hidden_size, 1)
        )

        self._init_weights()

        # Beta 网格
        k = torch.arange(1, K+1).float()
        self.register_buffer("beta_grid", k / K)

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _beta_weights(self, theta):
        """
        theta: [N]
        return: [N,K]
        """
        x = self.beta_grid  # [K]

        w = (x**(theta.unsqueeze(-1)-1))

        w = w / (w.sum(-1,keepdim=True)+1e-8)

        return w

    def forward(self, RV_lags):
        """
        RV_lags: [K,N]
                 第0行=最远期
                 第K-1行=最近一期

        return:
            tau: [N]
        """

        RV_lags = RV_lags.to(self.device)

        # 转为 [N,K] 给网络
        z = RV_lags.T  # [N,K]

        # NN 输出
        tilde_theta = self.theta_net(z)  # [N,1]
        theta = torch.exp(tilde_theta[:,0]) + 1

        # Beta 权重
        w = self._beta_weights(theta)  # [N,K]

        # MIDAS 聚合
        tau = (w * z).sum(-1)

        return tau

    def init_model(self):
        """初始化模型和优化器"""
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)


class Softmax_MIDAS(Model):
    """
    输入:  RV_lags [K,N]
    输出:  tau [N]
    """

    def __init__(
        self,
        K=7,
        hidden_size=16,
        dropout_rate=0.0,
        device=None,
    ):
        super().__init__()

        self.K = K

        # ===== 权重得分生成网络 =====
        self.score_net = nn.Sequential(
            nn.Linear(K, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_size, K)  # 输出 K 个score
        )

        self.softmax = nn.Softmax(dim=-1)

        self._init_weights()

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, RV_lags):
        """
        RV_lags: [K,N]
                 第0行=最远期
                 第K-1行=最近一期

        return:
            tau: [N]
        """

        RV_lags = RV_lags.to(self.device)

        # 转为 [N,K]
        z = RV_lags.T  # [N,K]

        # ===== NN生成权重得分 =====
        s = self.score_net(z)  # [N,K]

        # ===== Softmax归一化 =====
        w = self.softmax(s)  # [N,K]

        # ===== MIDAS聚合 =====
        tau = (w * z).sum(-1)

        return tau


class Attention_MIDAS(Model):
    """
    输入:  RV_lags [K,N]
    输出:  tau [N]
    """

    def __init__(
        self,
        K=7,
        d_attn=16,
        dropout_rate=0.0,
        device=None,
    ):
        super().__init__()

        self.K = K
        self.d_attn = d_attn

        # ===== Query: 由完整滞后向量生成 =====
        self.W_q = nn.Linear(K, d_attn)

        # ===== Key / Value: 由单个滞后值生成 =====
        self.W_k = nn.Linear(1, d_attn)
        self.W_v = nn.Linear(1, d_attn)

        # 输出层（把attention聚合后的向量变成标量）
        self.out_proj = nn.Linear(d_attn, 1)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.softmax = nn.Softmax(dim=-1)

        self._init_weights()

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, RV_lags):
        """
        RV_lags: [K,N]
                 第0行=最远期
                 第K-1行=最近一期

        return:
            tau: [N]
        """

        RV_lags = RV_lags.to(self.device)

        # ===== 形状整理 =====
        z = RV_lags.T              # [N,K]
        N = z.shape[0]

        # ===== Query =====
        q = self.W_q(z)            # [N,d]

        # ===== Keys / Values =====
        x = z.unsqueeze(-1)        # [N,K,1]

        k = self.W_k(x)            # [N,K,d]
        v = self.W_v(x)            # [N,K,d]

        # ===== Attention score =====
        # q·k / sqrt(d)
        scores = torch.sum(
            k * q.unsqueeze(1),    # broadcast
            dim=-1
        ) / math.sqrt(self.d_attn) # [N,K]

        w = self.softmax(scores)   # [N,K]

        # ===== 加权聚合 =====
        context = torch.sum(
            w.unsqueeze(-1) * v,
            dim=1
        )                          # [N,d]

        context = self.dropout(context)

        # ===== 标量化 + 指数映射 =====
        tau = torch.exp(
            self.out_proj(context).squeeze(-1)
        )                          # [N]

        return tau




class GARCH_MIDAS_Model(Model):
    def __init__(
        self,
        garch_model_name,
        midas_model_name,
        garch_params=None,
        midas_params=None,
        device=None,
    ):
        super().__init__()

        garch_params = garch_params or {}
        midas_params = midas_params or {}

        # ===== eval 初始化子模型 =====
        self.garch_model = eval(garch_model_name)(**garch_params)
        self.midas_model = eval(midas_model_name)(**midas_params)

        self.garch_model._init_weights()
        self.midas_model._init_weights()

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.to(self.device)

    def forward(self, x, returns,
                 RV_lags, prev_sigma2=None, pred_log_rv=True):
        """
        x        : [T,N,K]
        returns  : [T,N]

        return:
            pred_log_rv : [T,N]
        """
        x = x.to(self.device)
        returns = returns.to(self.device)

        # ===== GARCH 部分 =====
        garch_pred = self.garch_model(
            x,        # [T,N,K]
            returns,   # [T,N]
            prev_sigma2,
            pred_log_rv
        ).squeeze(-1)   # -> [T,N]

        # ===== MIDAS 部分 =====
        midas_pred = self.midas_model(RV_lags)  # -> [N]

        # ===== 相加 =====
        total_pred = garch_pred + midas_pred             # [T,N]

        return total_pred  # [T,N]

    def init_model(self):
        """初始化模型和优化器"""
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def fit(self, x_train_tensor, returns_train_tensor, y_train_tensor,
             x_valid=None, returns_valid=None, y_valid=None,
             pred_log_rv = True, **kwargs):
        """
        x       : [D, T, N, K]
        returns  : [D, T, N]
        label    : [D, T, N]
        """
        if kwargs.get('epochs', 100) is not None:
            self.epochs = kwargs.get('epochs', 100)
        if kwargs.get('early_stopping', 40) is not None:
            self.early_stopping = kwargs.get('early_stopping', 40)
        if kwargs.get('lr', 1e-3) is not None:
            self.lr = kwargs.get('lr', 1e-3)
        if kwargs.get('weight_decay', 5e-4) is not None:
            self.weight_decay = kwargs.get('weight_decay', 5e-4)
        if kwargs.get('grad_clip', 1) is not None:
            self.grad_clip = kwargs.get('grad_clip', 1)

        # 早停机制初始化
        self.best_loss_val = float('inf')
        best_epoch = 0
        early_stop_count = 0

        # 保存最佳模型参数
        best_model_state = None

        self.init_model()
        self.to(self.device)

        has_valid = x_valid is not None and y_valid is not None

        # 获取维度信息
        D, T, N, K = x_train_tensor.shape

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            batch_count = 0

            epoch_start_time = time.time()
            prev_sigma2 = None
            RV_lags = y_train_tensor[0].to(self.device)

            for d in range(1, D):
                # 获取当前时间步的收益率 [T, N]
                x_train_day = x_train_tensor[d].to(self.device)                 # [T, N, K]
                returns_train_day = returns_train_tensor[d].to(self.device)     # [T, N]
                label_day = y_train_tensor[d].to(self.device)

                predictions = self(x_train_day, returns_train_day, RV_lags, prev_sigma2, pred_log_rv)
                prev_sigma2 = predictions[-1].detach().squeeze(-1)
                RV_lags = label_day

                # 计算损失
                if isinstance(self.loss, str):
                    loss = eval("f." + self.loss + "(predictions.squeeze(-1), label_day)")
                else:
                    loss = self.loss(predictions.squeeze(-1), label_day)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

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

            # 验证集损失
            if has_valid:
                self.eval()
                val_loss = 0.0
                val_batch_count = 0

                D_valid = x_valid.shape[0]

                with torch.no_grad():
                    for d in range(0, D_valid):
                        # 获取当前时间步的收益率 [T, N]
                        x_valid_day = x_valid[d].to(self.device)
                        returns_valid_day = returns_valid[d].to(self.device)
                        label_day = y_valid[d].to(self.device)

                        predictions = self(x_valid_day, returns_valid_day, RV_lags, prev_sigma2, pred_log_rv)
                        prev_sigma2 = predictions[-1].detach().squeeze(-1)
                        RV_lags = label_day

                        # 计算损失
                        if isinstance(self.loss, str):
                            loss = eval("f." + self.loss + "(predictions.squeeze(-1), label_day)")
                        else:
                            loss = self.loss(predictions.squeeze(-1), label_day)

                        val_loss += loss.item()
                        val_batch_count += 1

                    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

                    # 早停逻辑
                    if avg_val_loss < self.best_loss_val:
                        self.best_loss_val = avg_val_loss
                        best_epoch = epoch + 1
                        early_stop_count = 0
                        # 保存最佳模型参数
                        best_model_state = self.state_dict()
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
                    best_model_state = self.state_dict()
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
            self.load_state_dict(best_model_state)

        # 打印最优结果
        print(f"\nTraining completed!")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Loss: {self.best_loss_val:.4f}")

        return self

    def predict(self, x_test_tensor, returns_test_tensor, y_test_tensor, pred_log_rv=True):
        """
        生成预测结果，返回条件方差预测

        参数:
        x_test: 测试数据，torch.Tensor [时间, 股票, 因子]

        返回:
        torch.Tensor [时间, 股票] - 条件方差预测
        """
        if isinstance(x_test_tensor, torch.Tensor) and isinstance(returns_test_tensor, torch.Tensor):
            return self._predict_tensor(x_test_tensor, returns_test_tensor, y_test_tensor, pred_log_rv)
        else:
            raise TypeError("输入类型不支持，请使用torch.Tensor")

    def _predict_tensor(self, x_test_tensor, returns_test_tensor, y_test_tensor, pred_log_rv=True):
        """
        使用PyTorch张量进行预测，返回条件方差

        x_test_tensor形状: [日期, 时间, 股票, 因子]
        返回形状: [日期, 时间, 股票] - 条件方差预测
        """
        # 确保输入形状正确 [时间, 股票, 因子]
        if len(x_test_tensor.shape) != 4:
            raise ValueError(f"输入张量形状应为[日期, 时间, 股票, 因子]，实际为{x_test_tensor.shape}")

        # 获取维度信息
        D, T, N, K = x_test_tensor.shape

        # 创建输出张量
        y_pred = torch.full((D, T, N), np.nan, dtype=torch.float32, device=self.device)
        prev_sigma2 = None
        RV_lags = y_test_tensor[0].to(self.device)

        # 逐时间步处理
        for d in range(1, D):
            x_test_d = x_test_tensor[d].to(self.device)
            returns_test_d = returns_test_tensor[d].to(self.device)
            y_test_d = y_test_tensor[d].to(self.device)

            # 计算条件方差
            with torch.no_grad():
                sigma2 = self(x_test_d, returns_test_d, RV_lags, prev_sigma2, pred_log_rv)
                prev_sigma2 = sigma2[-1].detach().squeeze(-1)
                RV_lags = y_test_d
                # 将预测结果放回到正确位置
                y_pred[d] = sigma2.squeeze(-1)

        return y_pred
