import torch
import os
import math
from torch import Tensor
import torch.nn.functional as f
import numpy as np
from pandas import DataFrame, Series, concat, Grouper

from .loss import *
from .models import Model, lr_scheduler
from ..utils import get_daily_inter, from_pandas_to_list, from_pandas_to_rnn


class MLP(Model):
    def __init__(self, input_shape: int, hidden_shape: int, output_shape: int = 1, batch_size: int = 1, device: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        # 自动选择设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_layer = torch.nn.Linear(self.input_shape, self.hidden_shape)
        self.hid_layer_1 = torch.nn.Linear(self.hidden_shape, self.hidden_shape)
        self.hid_layer_2 = torch.nn.Linear(self.hidden_shape, self.hidden_shape)
        self.output_layer = torch.nn.Linear(self.hidden_shape, self.output_shape)

        self.optimizer = None
        self.init_weights()  # 初始化权重

    def init_weights(self):
        # 使用Xavier/Glorot初始化
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def init_model(self):
        self.model = MLP(input_shape=self.input_shape, hidden_shape=self.hidden_shape,
                         output_shape=self.output_shape, batch_size=self.batch_size, device=self.device,
                         epochs=self.epochs, loss=self.loss, lr=self.lr,
                         weight_decay=self.weight_decay, dropout=self.dropout).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def forward(self, x, **kwargs):
        x = f.relu(self.input_layer(x))
        x = f.dropout(x, p=self.dropout, training=self.training)

        x = f.relu(self.hid_layer_1(x))
        x = f.dropout(x, p=self.dropout, training=self.training)

        x = f.relu(self.hid_layer_2(x))

        return self.output_layer(x)

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


# 未修正
class MLP_v1(Model):
    def __init__(self, input_shape: int, hidden_shape: int, output_shape: int = 1, batch_size: int = 1, device: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        # 自动选择设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.feature_filter = torch.nn.Linear(in_features=self.input_shape, out_features=self.hidden_shape, bias=False)
        self.hidden_layer = torch.nn.Linear(in_features=self.hidden_shape, out_features=self.hidden_shape)
        self.out_layer = torch.nn.Linear(in_features=self.hidden_shape, out_features=self.output_shape)
        self.jump_layer = torch.nn.Linear(in_features=self.input_shape, out_features=self.hidden_shape)
        self.bn = torch.nn.BatchNorm1d(self.hidden_shape)

        self.optimizer = None

    def init_model(self):
        self.model = MLP(input_shape=self.input_shape, hidden_shape=self.hidden_shape,
                         output_shape=self.output_shape, batch_size=self.batch_size, device=self.device,
                         epochs=self.epochs, loss=self.loss, lr=self.lr,
                         weight_decay=self.weight_decay, dropout=self.dropout).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x, **kwargs):
        x1 = f.relu(self.jump_layer(x))
        x = f.dropout(x, p=self.dropout, training=self.training)

        x = f.relu(self.feature_filter(x))
        x = f.dropout(x, p=self.dropout, training=self.training)

        x = f.relu(self.hidden_layer(x))

        return self.out_layer(self.bn(x + x1))

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
                    print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_val_loss:.4f} (Best)")
                else:
                    early_stop_count += 1
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
                    valid_preds = self.model(x_valid_batch)
                    # 将预测结果放回到正确位置
                    batch_pred[valid_mask] = valid_preds

                # 将当前批次的预测结果重塑并放入输出张量 [T*N, 1] -> [T, N]
                y_pred[t:end_t] = batch_pred.reshape(end_t - t, num_stocks)

        return y_pred


if __name__ == '__main__':
    x_train_tensor = torch.randn(848, 3182, 154)
    y_train_tensor = torch.randn(848, 3182)
    x_test_tensor = torch.randn(1, 3182, 154)

    # 示例用法
    # 创建模型实例，设置batch_size和early_stopping
    model = MLP(input_shape=154, hidden_shape=64, output_shape=1, batch_size=5, device='cuda', epochs=10, early_stopping=3)

    # 训练模型（输入PyTorch张量）
    # x_train_tensor形状: [848, 3182, 154]，y_train_tensor形状: [848, 3182]
    model.fit(x_train_tensor, y_train_tensor)

    # 预测（输入单个时间步的张量）
    # x_test_tensor形状: [1, N, K]
    y_pred = model.predict(x_test_tensor)  # 返回形状: [1, N]