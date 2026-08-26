import torch
import os
import math
from torch import Tensor
import torch.nn.functional as f
import numpy as np
from pandas import DataFrame, Series, concat, Grouper
from torch_geometric.nn.conv import SAGEConv, GCNConv, GATConv, GINConv, ClusterGCNConv, GATv2Conv, ChebConv, ARMAConv, SGConv, SSGConv, GCN2Conv, APPNP
from torch_geometric.utils import add_self_loops
from collections import OrderedDict, defaultdict  # 添加在文件顶部

from .loss import *
from ..utils import get_daily_inter, from_pandas_to_list, from_pandas_to_rnn, calc_kernel_size, transform_data

"""
目前models模块使用pytorch实现, 这样可以更好地接入图神经网络模块

输入仍然可以是有多重索引 [(datetime, instrument)] 的DataFrame和Series, 但模型训练之前会自动将数据按天拆成一个list(GRU需要特殊的处理),
并以一天的数据量作为batch(所以batch size是会变的)

增加了style_mse函数, 可以通过调参精确控制预测值与某个变量的相关系数大小
"""


def split_dataset_by_index(dataset: list, train_index, test_index):
    """
    用于滚动训练
    """
    f_array = np.zeros(shape=(len(dataset),)).astype(bool)
    train_mask, test_mask = f_array.copy(), f_array.copy()
    train_mask[train_index] = True
    test_mask[test_index] = True
    d_train = [d for d, mask in zip(dataset, train_mask.tolist()) if mask]
    d_test = [d for d, mask in zip(dataset, test_mask.tolist()) if mask]
    return d_train, d_test


def lr_scheduler(optimizer: torch.optim.Optimizer, kwargs: dict) -> torch.optim.lr_scheduler._LRScheduler:
    """
    根据传入的字典参数选择和返回对应的学习率调度器
    :param optimizer: torch.optim.Optimizer 对象，用于指定调度器需要优化的参数
    :param kwargs: dict, 包含学习率调度器的方法及相关参数
    :return: torch.optim.lr_scheduler._LRScheduler, 选择的学习率调度器
    """
    if kwargs is not None:
        lr_scheduler_method = kwargs.get("lr_scheduler_method", None)  # 学习率调度器方法
        lr_scheduler_kwargs = kwargs.get("lr_scheduler_kwargs", {})  # 学习率调度器的参数
    else:
        return None

    if lr_scheduler_method is not None:
        if lr_scheduler_method == 'StepLR':
            step_size = lr_scheduler_kwargs.get("step_size", 20)
            gamma = lr_scheduler_kwargs.get("gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        elif lr_scheduler_method == 'CosineAnnealingLR':
            T_max = lr_scheduler_kwargs.get("T_max", 50)
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

        elif lr_scheduler_method == 'ReduceLROnPlateau':
            mode = lr_scheduler_kwargs.get("mode", 'min')
            factor = lr_scheduler_kwargs.get("factor", 0.1)
            patience = lr_scheduler_kwargs.get("patience", 10)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)

        else:
            raise ValueError(f"Unknown lr_scheduler_method: {lr_scheduler_method}")
    else:
        return None


def calc_tensor_corr(x: Tensor, y: Tensor):
    if x.shape != y.shape:
        raise ValueError("The shapes of x and y must be the same.")
    mask = ~torch.isnan(y)
    mean_x = torch.mean(x[mask])
    mean_y = torch.mean(y[mask])
    std_x = torch.std(x[mask])
    std_y = torch.std(y[mask])
    return torch.mean((x[mask] - mean_x) * (y[mask] - mean_y)) / (std_x * std_y)


class Model(torch.nn.Module):
    def __init__(self, epochs: int = 10, loss: str = "mse_loss", lr: float = 1e-3, weight_decay: float = 5e-4,
                 lr_scheduler_kwargs: dict = None,
                 dropout: float = 0.2, model=None, adv: bool = False,
                 auto_save: bool = False, save_folder: str = "Model",
                 early_stopping: int = 0, min_delta: int = 0.0001, grad_clip: float = None,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = epochs
        # 修改这里以支持qlike_loss
        if loss == "qlike":  # 添加对"qlike"字符串的处理
            epsilon = kwargs.get("epsilon", 1e-2)
            self.loss = qlike_loss(epsilon=epsilon)
        elif loss not in ["style_mse", "huber_loss", "ic_mse", "qlike_loss"]:
            self.loss = loss
        elif loss == "style_mse":
            self.loss = style_mse()
        elif loss == "huber_loss":
            self.loss = huber_loss()
        elif loss == "ic_mse":
            ic_weight = kwargs.get("ic_weight", 0.1)  # 获取ic_weight参数，默认为0.05
            self.loss = ic_mse(ic_weight=ic_weight)
        elif loss == "qlike_loss":
            epsilon = kwargs.get("epsilon", 1e-2)  # 获取epsilon参数，默认为1e-2
            self.loss = qlike_loss(epsilon=epsilon)

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.model = model
        self.optimizer = None
        self.for_cnn = False
        self.for_rnn = False
        self.adv = adv
        self.output = None
        self.auto_save = auto_save
        self.save_folder = save_folder

        # Early stopping parameters
        self.early_stopping = early_stopping  # Now an int (0 means no early stopping)
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        self.grad_clip = grad_clip

        self.lr_scheduler_kwargs = lr_scheduler_kwargs

    def forward(self, x, **kwargs):
        pass

    def init_model(self):
        pass

    def get_loss(self, x, y, z=None, **kwargs):
        self.model.train()
        if self.adv:
            x.requires_grad = True
        self.optimizer.zero_grad()
        mask = ~torch.isnan(y[:, 0])
        out = self.model(x, **kwargs)

        if isinstance(self.loss, str):
            # 确保字符串损失函数存在于torch.nn.functional中
            if hasattr(f, self.loss):
                loss = eval("f." + self.loss + "(out[mask], y[mask])")
            else:
                raise ValueError(f"损失函数 '{self.loss}' 在torch.nn.functional中不存在")
        elif isinstance(self.loss, style_mse):
            loss = self.loss((out[mask], y[mask], z[mask]))
        elif isinstance(self.loss, huber_loss):
            if z is not None:
                loss = self.loss((out[mask], y[mask], z[mask]))
            else:
                loss = self.loss(out[mask], y[mask])
        elif isinstance(self.loss, ic_mse):
            # ic_mse只需要预测值和真实值
            loss = self.loss(out[mask], y[mask])
        elif isinstance(self.loss, qlike_loss):
            # 处理qlike_loss
            if z is not None:
                loss = self.loss((out[mask], y[mask], z[mask]))
            else:
                loss = self.loss(out[mask], y[mask])
        else:
            loss = self.loss(out[mask], y[mask])

        loss.backward()

        # 添加梯度裁剪
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self, x, y, z=None, **kwargs):
        mask = ~torch.isnan(y[:, 0])
        pred_ = self.predict_(x, **kwargs)
        if isinstance(self.loss, str):
            loss = eval("f." + self.loss + "(pred_[mask], y[mask])")
        elif isinstance(self.loss, style_mse):
            loss = self.loss((pred_[mask], y[mask], z[mask]))
        elif isinstance(self.loss, huber_loss):
            if z is not None:
                loss = self.loss((pred_[mask], y[mask], z[mask]))
            else:
                loss = self.loss(pred_[mask], y[mask])
        elif isinstance(self.loss, ic_mse):
            # ic_mse只需要预测值和真实值
            loss = self.loss(pred_[mask], y[mask])
        else:
            loss = self.loss(pred_[mask], y[mask])
        return float(loss)

    @torch.no_grad()
    def predict_(self, x, **kwargs):
        self.model.eval()
        pred = self.model(x, **kwargs)
        return pred

    @torch.no_grad()
    def test(self, x, y, z=None, **kwargs):
        mask = ~torch.isnan(y[:, 0])
        pred_ = self.predict_(x, **kwargs)
        if isinstance(self.loss, str):
            loss = eval("f." + self.loss + "(pred_[mask], y[mask])")
        elif isinstance(self.loss, style_mse):
            loss = self.loss((pred_[mask], y[mask], z[mask]))
        elif isinstance(self.loss, huber_loss):  # 添加对huber_loss的处理
            if z is not None:
                loss = self.loss((pred_[mask], y[mask], z[mask]))
            else:
                loss = self.loss(pred_[mask], y[mask])
        else:
            loss = self.loss(pred_[mask], y[mask])
        return float(loss)

    def fit(self, x_train, y_train, x_valid, y_valid, z_train=None, z_valid=None, **kwargs):
        if self.model is None:
            self.init_model()

        x_train, y_train, x_valid, y_valid, z_train, z_valid = transform_data(x_train, y_train, x_valid, y_valid,
                                                                              z_train, z_valid, for_cnn=self.for_cnn,
                                                                              for_rnn=self.for_rnn)

        best_val_ic = -float('inf')  # 初始化为一个很小的值
        self.best_loss_val = float('inf')  # 初始化为一个很大的值
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):
            total_loss_train = 0
            total_loss_val = 0
            val_ic = 0

            # 训练
            for i in range(len(x_train)):
                loss_train = self.get_loss(x=x_train[i], y=y_train[i], z=z_train[i] if z_train is not None else None,
                                           **kwargs)
                total_loss_train += loss_train

            # 验证
            for i in range(len(x_valid)):
                loss_val = self.test(x=x_valid[i], y=y_valid[i], z=z_valid[i] if z_valid is not None else None,
                                     **kwargs)
                total_loss_val += loss_val
                val_ic += float(calc_tensor_corr(self.predict_(x_valid[i]), y_valid[i]))

            avg_loss_train = total_loss_train / len(x_train)
            avg_loss_val = total_loss_val / len(x_valid)
            avg_val_ic = val_ic / len(x_valid)

            print("Epoch:", epoch, "loss:", avg_loss_train, "val_loss:",
                  avg_loss_val, "val_ic:", avg_val_ic)

            # 早停检查
            if self.early_stopping > 0:  # 只有当early_stopping大于0时才检查
                if avg_loss_val < self.best_loss_val - self.min_delta:
                    self.early_stop_counter = 0  # 如果有改进，重置计数器
                else:
                    self.early_stop_counter += 1
                    print(f"Early stopping counter: {self.early_stop_counter}/{self.early_stopping}")
                    if self.early_stop_counter >= self.early_stopping:
                        print("Early stopping triggered")
                        break  # 如果超过耐心值，停止训练

            # 学习率调度器
            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # 使用 ReduceLROnPlateau 时，需传入验证集的损失
                    self.lr_scheduler.step(avg_loss_val)
                else:
                    self.lr_scheduler.step()  # 其他调度器使用 step() 进行更新

            # 保存模型
            if self.auto_save:
                save_path = f"{self.save_folder}/epoch_{epoch}.pth"
                folder_path = os.path.dirname(save_path)

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                torch.save(self.state_dict(), save_path)
                print(f"Model saved to {save_path}")

                # 更新并保存最佳模型
                if avg_loss_val < self.best_loss_val:
                    self.best_loss_val = avg_loss_val
                    best_val_ic = avg_val_ic
                    best_epoch = epoch
                    best_save_path = f"{self.save_folder}/best_epoch.pth"
                    torch.save(self.state_dict(), best_save_path)
                    print(f"Best model updated: epoch {best_epoch}, val_loss {self.best_loss_val:.6f}, val_ic {best_val_ic:.6f}")

        # 训练结束后打印最佳结果
        if self.auto_save:
            print(f"Training completed. Best epoch: {best_epoch}, Best val_loss: {self.best_loss_val:.6f}, Best val_ic: {best_val_ic:.6f}")

    def fit_kfold(self, x, y, z=None, k: int = 5, train_size=None, test_size=None, collect: bool = False, **kwargs):
        x_list = from_pandas_to_list(x)
        y_list = from_pandas_to_list(y)
        z_list = from_pandas_to_list(z) if z is not None else None
        from sklearn.model_selection import TimeSeriesSplit
        if self.model is None:
            self.init_model()
        tscv = TimeSeriesSplit(n_splits=k, max_train_size=train_size, test_size=test_size)
        if collect:
            self.output = []
        for fold, (train_index, valid_index) in enumerate(tscv.split(x_list)):
            x_train, x_valid = split_dataset_by_index(x_list, train_index, valid_index)
            y_train, y_valid = split_dataset_by_index(y_list, train_index, valid_index)
            if z is not None:
                z_train, z_valid = split_dataset_by_index(z_list, train_index, valid_index)
            else:
                z_train, z_valid = None, None
            print("fold: ", fold)
            self.fit(x_train, y_train, x_valid, y_valid, z_train=z_train, z_valid=z_valid, **kwargs)
            if collect:
                for i in range(len(x_valid)):
                    self.output.append(Series(self.predict_(x_valid[i]).view(-1, ), **kwargs))
        if collect:
            self.output = concat(self.output, axis=0)
            if isinstance(x, DataFrame):
                self.output.index = x.index[-len(self.output):]

    def predict_pandas(self, x: DataFrame, **kwargs) -> Series:
        index = x.index
        x = from_pandas_to_list(x, self.for_cnn)
        result = []
        for batch in x:
            result.append(Series(self.predict_(batch, **kwargs).view(-1, )))
        series = concat(result, axis=0)
        series.index = index
        return series

    def save(self, path: str = "model.pth"):
        torch.save(self.state_dict(), path)


class MyGAT(Model):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, output_shape: int = 1,
                 heads: int = 1, add_self_loop: bool = False, fillna: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_shape = output_shape
        self.heads = heads
        self.add_self_loops = add_self_loop
        self.fillna = fillna

        # 注意力机制参数
        self.att_weight = torch.nn.Parameter(torch.Tensor(2 * hidden_channels, 1))
        self.weight = torch.nn.Parameter(torch.Tensor(input_channels, hidden_channels * heads))
        self.bias = torch.nn.Parameter(torch.Tensor(hidden_channels * heads))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.att_weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = torch.sparse_csr_tensor(
                crow_indices=torch.tensor([0, x.shape[0]]),
                col_indices=torch.tensor([]),
                values=torch.tensor([]),
                size=(x.shape[0], x.shape[0])
            )

        # 添加自环边
        if self.add_self_loops:
            edge_index = add_self_loops(edge_index)

        # 特征变换
        h = torch.mm(x, self.weight) + self.bias
        h = h.view(-1, self.heads, self.hidden_channels)

        # 计算注意力分数
        row, col = edge_index.to_sparse_coo().indices()
        att_src = h[row]  # [E, heads, hidden]
        att_dst = h[col]  # [E, heads, hidden]
        alpha = (torch.cat([att_src, att_dst], dim=-1) @ self.att_weight).squeeze()  # [E, heads]
        alpha = f.leaky_relu(alpha, 0.2)

        # 创建带注意力权重的稀疏矩阵
        alpha = torch.sparse_coo_tensor(
            indices=edge_index.to_sparse_coo().indices(),
            values=alpha,
            size=edge_index.size()
        ).to_sparse_csr()

        # 注意力加权聚合
        out = torch.spmm(alpha, h.view(-1, self.hidden_channels * self.heads))
        return torch.relu(out)

    def init_model(self):
        self.model = MyGAT(
            input_channels=self.input_channels,
            hidden_channels=self.hidden_channels,
            output_channels=self.output_channels,
            output_shape=self.output_shape,
            heads=self.heads,
            add_self_loop=self.add_self_loops,
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

    def fit(self, x_train, y_train, x_valid, y_valid, z_train=None, z_valid=None, edge_train: list = None,
            edge_valid: list = None):
        if self.model is None:
            self.init_model()

        x_train, y_train, x_valid, y_valid, z_train, z_valid = transform_data(
            x_train, y_train, x_valid, y_valid,
            z_train, z_valid,
            for_cnn=self.for_cnn,
            for_rnn=self.for_rnn,
            fillna=self.fillna
        )

        best_val_ic = -float('inf')
        best_epoch = 0
        best_save_path = None

        for epoch in range(1, self.epochs + 1):
            total_loss_train = 0
            total_loss_val = 0
            val_ic = 0

            # 训练循环
            for i in range(len(x_train)):
                loss_train = self.get_loss(
                    x=x_train[i],
                    y=y_train[i],
                    z=z_train[i] if z_train else None,
                    edge_index=edge_train[i] if edge_train else None
                )
                total_loss_train += loss_train

            # 验证循环
            for i in range(len(x_valid)):
                loss_val = self.test(
                    x=x_valid[i],
                    y=y_valid[i],
                    z=z_valid[i] if z_valid else None,
                    edge_index=edge_valid[i] if edge_valid else None
                )
                total_loss_val += loss_val
                val_ic += float(calc_tensor_corr(
                    self.predict_(x_valid[i], edge_index=edge_valid[i] if edge_valid else None),
                    y_valid[i]
                ))

            # 输出训练信息
            avg_loss_train = total_loss_train / len(x_train)
            avg_loss_val = total_loss_val / len(x_valid)
            avg_val_ic = val_ic / len(x_valid)
            print(f"Epoch: {epoch} loss: {avg_loss_train:.4f} val_loss: {avg_loss_val:.4f} val_ic: {avg_val_ic:.4f}")

            # 早停机制
            if self.early_stopping > 0:
                if avg_loss_val < self.best_val_loss - self.min_delta:
                    self.best_val_loss = avg_loss_val
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.early_stopping:
                        print("Early stopping triggered")
                        break

            # 模型保存
            if self.auto_save:
                save_path = f"{self.save_folder}/epoch_{epoch}.pth"
                torch.save(self.state_dict(), save_path)
                if avg_val_ic > best_val_ic:
                    best_val_ic = avg_val_ic
                    best_epoch = epoch
                    best_save_path = f"{self.save_folder}/best_epoch.pth"
                    torch.save(self.state_dict(), best_save_path)

    def predict_pandas(self, x: DataFrame, edge_index=None) -> Series:
        x_tensor = from_pandas_to_list(x, fillna=self.fillna)
        result = []

        for i in range(len(x_tensor)):
            current_edge = edge_index[i] if edge_index else None
            pred = self.predict_(x_tensor[i], edge_index=current_edge).view(-1)
            result.append(Series(pred.detach().numpy()))

        # 保持与原有GNN相同的输出格式
        days = x.index.get_level_values(0).unique()[-len(result):]
        instruments = x.index.get_level_values(1).unique().tolist()
        return format_predictions(result, days, instruments, x.index.names)
