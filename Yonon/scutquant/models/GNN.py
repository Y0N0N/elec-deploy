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


import torch
import os
import math
from torch import Tensor
import torch.nn.functional as f
import numpy as np
from pandas import DataFrame, Series, concat, Grouper

from .loss import *
from .models import Model, lr_scheduler
from ..utils import get_daily_inter, from_pandas_to_list, from_pandas_to_rnn, calc_tensor_corr
from torch_geometric.nn.conv import SAGEConv, GCNConv, GATConv, GINConv, ClusterGCNConv
from torch_geometric.utils import add_self_loops, dense_to_sparse


# 图归一化，适用于GCN
def normalize_matrix_tensor(adj_matrix):
    """
    对 [N, N] 的邻接矩阵 adj_matrix 做对称归一化：
        Â = D̂^(-1/2) (A + I) D̂^(-1/2)

    参数
    ----
    adj_matrix : torch.Tensor 或 torch.sparse.Tensor
        形状 [N, N]，可以是稠密或稀疏。

    返回
    ----
    torch.Tensor 或 torch.sparse.Tensor
        归一化后的矩阵，格式与输入保持一致。
    """
    # 1. 加自环 A + I
    if adj_matrix.is_sparse:
        # 稀疏情况下，构造稀疏单位矩阵再相加
        N = adj_matrix.size(0)
        idx = torch.arange(N, device=adj_matrix.device)
        I = torch.sparse_coo_tensor(
            torch.stack([idx, idx]),
            torch.ones(N, dtype=adj_matrix.dtype, device=adj_matrix.device),
            size=(N, N)
        )
        A_hat = adj_matrix + I
    else:
        A_hat = adj_matrix + torch.eye(adj_matrix.size(0),
                                       dtype=adj_matrix.dtype,
                                       device=adj_matrix.device)

    # 2. 计算度向量 d = A_hat 按行求和
    if A_hat.is_sparse:
        d = torch.sparse.sum(A_hat, dim=1).to_dense()   # [N]
    else:
        d = A_hat.sum(1)   # [N]

    # 3. 计算 D^(-1/2)，避免除 0
    d_inv_sqrt = torch.pow(d + 1e-10, -0.5)   # [N]

    # 4. 构造 D^(-1/2) 的稀疏/稠密对角矩阵
    if A_hat.is_sparse:
        idx = torch.arange(A_hat.size(0), device=A_hat.device)
        D_inv_sqrt = torch.sparse_coo_tensor(
            torch.stack([idx, idx]),
            d_inv_sqrt,
            size=A_hat.shape
        )
    else:
        D_inv_sqrt = torch.diag(d_inv_sqrt)

    # 5. 做对称归一化
    #    Â = D^(-1/2) @ A_hat @ D^(-1/2)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


class GNN(Model):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, output_shape: int = 1,
                 conv_type: str = "GCN", jump_connection: bool = False, num_layers: int = 1,
                 aggr="mean", add_self_loop: bool = False, fillna: bool = False, batch_size: int = 1, device: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_shape = output_shape
        self.aggr = aggr
        self.add_self_loops = add_self_loop
        self.fillna = fillna
        self.conv_type = conv_type
        self.jump_connection = jump_connection
        self.num_layers = num_layers  # 新增参数：卷积层数量
        self.batch_size = batch_size
        # 自动选择设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 输入线性层
        self.input_linear = torch.nn.Linear(input_channels, hidden_channels)
        self.input_bn = torch.nn.BatchNorm1d(hidden_channels)

        # 创建多个卷积层、BatchNorm和激活函数
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # 第一个卷积层的输入维度是hidden_channels
        in_channels = hidden_channels

        # 中间层的输出维度都是hidden_channels，最后一层的输出维度是output_channels
        for i in range(num_layers):
            out_channels = output_channels if i == num_layers - 1 else hidden_channels

            if conv_type == "GCN":
                self.convs.append(GCNConv(in_channels=in_channels, out_channels=out_channels))
            elif conv_type == "SAGE":
                self.convs.append(SAGEConv(in_channels=in_channels, out_channels=out_channels, aggr=aggr))
            elif conv_type == "GAT":
                self.convs.append(GATConv(in_channels=in_channels, out_channels=out_channels))
            elif conv_type == "GIN":
                nn = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, hidden_channels),
                    torch.nn.BatchNorm1d(hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, out_channels)
                )
                self.convs.append(GINConv(nn))
            elif conv_type == "ClusterGCN":  # 新增ClusterGCN支持
                self.convs.append(ClusterGCNConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    aggr=aggr,
                    num_clusters=kwargs.get("num_clusters", 4)  # 从参数获取集群数，默认4
                ))

            self.bns.append(torch.nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        # 输出层
        self.output_linear = torch.nn.Linear(output_channels, 1)
        self.relu = torch.nn.ReLU()

        # 跳跃连接
        if jump_connection is True:
            self.jump_connection = torch.nn.Linear(input_channels, 1)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化网络权重"""
        # 使用Xavier/Glorot初始化
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (GCNConv, SAGEConv, GATConv, GINConv, ClusterGCNConv)):
                # 对于图卷积层，使用Xavier初始化
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        # 特别初始化图卷积层的参数
        for conv_layer in self.convs:
            if hasattr(conv_layer, 'reset_parameters'):
                conv_layer.reset_parameters()

    def forward(self, x, edge_index=None):
        if edge_index is None:
            # 创建全零邻接矩阵，表示没有边连接
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            if self.add_self_loops:
                edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        else:
            if self.add_self_loops:
                edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        if self.jump_connection:
            x_skip = self.jump_connection(x)

        # 输入层处理
        x = self.input_linear(x)
        x = self.input_bn(x)
        x = self.relu(x)

        # 多层卷积处理
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.relu(x)  # 先激活
            x = self.bns[i](x)  # 再归一化

        # 输出层
        x = self.output_linear(x)

        # 跳跃连接
        if self.jump_connection:
            x = x + x_skip

        return x

    def init_model(self):
        self.model = GNN(input_channels=self.input_channels,
                         hidden_channels=self.hidden_channels,
                         output_channels=self.output_channels,
                         output_shape=self.output_shape,
                         aggr=self.aggr,
                           epochs=self.epochs, conv_type=self.conv_type, loss=self.loss, lr=self.lr,
                           weight_decay=self.weight_decay, num_layers=self.num_layers,  # 传递层数参数
                           dropout=self.dropout, batch_size=self.batch_size, device=self.device).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, z_train=None, z_valid=None, edge_train: list = None,
            edge_valid: list = None, **kwargs):
        """
        训练模型，支持PyTorch张量输入和按时间步批量训练

        参数:
        x_train: 训练数据，可以是pd.DataFrame或torch.Tensor [时间, 股票, 因子]
        y_train: 训练标签，可以是pd.Series或torch.Tensor [时间, 股票]
        x_valid: 验证数据（可选）
        y_valid: 验证标签（可选）
        z_train: 额外的训练数据（可选）
        z_valid: 额外的验证数据（可选）
        edge_train: 训练边索引列表（可选）
        edge_valid: 验证边索引列表（可选）
        """
        # 处理PyTorch张量输入
        if isinstance(x_train, torch.Tensor) and isinstance(y_train, torch.Tensor):
            return self._fit_tensor(x_train, y_train, x_valid, y_valid, z_train, z_valid, edge_train, edge_valid, **kwargs)
        # 保持原有接口
        else:
            # 调用原有的fit方法（需要转换数据格式）
            from ..utils import transform_data, split_dataset_by_index
            x_train, y_train, x_valid, y_valid, z_train, z_valid = transform_data(
                x_train, y_train, x_valid, y_valid,
                z_train, z_valid,
                for_cnn=self.for_cnn,
                for_rnn=self.for_rnn,
                fillna=self.fillna
            )

            # 添加最佳模型跟踪变量
            best_loss_val = float('inf')
            best_val_ic = -float('inf')
            best_epoch = 0
            best_save_path = None

            for epoch in range(1, self.epochs + 1):
                total_loss_train = 0
                total_loss_val = 0
                val_ic = 0

                # 训练阶段
                self.model.train()
                for i in range(len(x_train)):
                    loss_train = self.get_loss(x=x_train[i], y=y_train[i], z=z_train[i] if z_train is not None else None,
                                               edge_index=edge_train[i] if edge_train is not None else None)
                    total_loss_train += loss_train

                # 验证阶段
                self.model.eval()
                with torch.no_grad():
                    for i in range(len(x_valid)):
                        loss_val = self.test(x=x_valid[i], y=y_valid[i], z=z_valid[i] if z_valid is not None else None,
                                             edge_index=edge_valid[i] if edge_valid is not None else None)
                        total_loss_val += loss_val
                        val_ic += float(calc_tensor_corr(
                            self.predict_(x_valid[i], edge_index=edge_valid[i] if edge_valid is not None else None),
                            y_valid[i]))

                avg_loss_train = total_loss_train / len(x_train)
                avg_loss_val = total_loss_val / len(x_valid)
                avg_val_ic = val_ic / len(x_valid)

                print("Epoch:", epoch, "loss:", avg_loss_train, "val_loss:", avg_loss_val, "val_ic:", avg_val_ic)

                # 早停
                if self.early_stopping > 0:
                    if avg_loss_val < best_loss_val - self.min_delta:
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        print(f"Early stopping counter: {self.early_stop_counter}/{self.early_stopping}")
                        if self.early_stop_counter >= self.early_stopping:
                            print("Early stopping triggered")
                            break

                # 学习率调度
                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(avg_loss_val)
                    else:
                        self.lr_scheduler.step()

                # 保存模型
                if self.auto_save:
                    save_path = f"{self.save_folder}/epoch_{epoch}.pth"
                    folder_path = os.path.dirname(save_path)

                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    torch.save(self.state_dict(), save_path)
                    print(f"Model saved to {save_path}")

                    if avg_loss_val < best_loss_val:
                        best_loss_val = avg_loss_val
                        best_epoch = epoch
                        best_save_path = f"{self.save_folder}/best_epoch.pth"
                        torch.save(self.state_dict(), best_save_path)

            if self.auto_save and best_save_path:
                torch.save(self.state_dict(), best_save_path)
                print(f"Best epoch:{best_epoch}, MSE:{best_loss_val}, Best Model saved to {best_save_path}")

            return self

    def _fit_tensor(self, x_train_tensor, y_train_tensor, x_valid=None, y_valid=None, z_train=None, z_valid=None,
                    edge_train: list = None, edge_valid: list = None, **kwargs):
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

                # 按时间步处理
                for step in range(t, end_t):
                    x_batch = x_train_tensor[step].to(self.device)  # [N, K]
                    y_batch = y_train_tensor[step].to(self.device)  # [N]

                    # 获取当前时间步的边索引
                    if edge_train is not None and step < len(edge_train):
                        current_edge = edge_train[step].to(self.device)
                        # 如果是稀疏张量，转换为密集张量
                        if current_edge.is_sparse:
                            current_edge = current_edge.to_dense().long()
                        elif current_edge.is_sparse_csr:
                            current_edge = current_edge.to_dense().long()
                    else:
                        # 如果没有提供边索引，创建一个没有边的图（只有自环）
                        current_edge = torch.zeros((2, 0), dtype=torch.long, device=self.device)

                    # 过滤掉NaN值
                    valid_mask = ~(torch.isnan(x_batch).any(dim=1) | torch.isnan(y_batch))
                    if valid_mask.any():
                        x_valid_batch = x_batch[valid_mask]
                        y_valid_batch = y_batch[valid_mask].unsqueeze(1)  # [N, 1]

                        # 使用current_edge并过滤掉NaN值对应的节点
                        if current_edge.size(1) > 0:  # 如果有边存在
                            # 映射到有效节点的索引
                            valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
                            # 创建映射，将原始节点索引映射到过滤后的新索引
                            node_mapping = torch.full((x_batch.size(0),), -1, dtype=torch.long, device=self.device)
                            node_mapping[valid_indices] = torch.arange(len(valid_indices), device=self.device)

                            # 过滤边，只保留有效节点之间的边
                            edge_mask = (node_mapping[current_edge[0]] != -1) & (node_mapping[current_edge[1]] != -1)
                            filtered_edge = current_edge[:, edge_mask]

                            # 重新映射边的索引
                            edge_idx = torch.stack([
                                node_mapping[filtered_edge[0]],
                                node_mapping[filtered_edge[1]]
                            ], dim=0)
                        else:
                            # 如果没有提供边索引，创建自环边
                            num_valid_nodes = x_valid_batch.size(0)
                            if num_valid_nodes > 1:
                                self_loop_idx = torch.arange(num_valid_nodes, device=self.device)
                                edge_idx = torch.stack([self_loop_idx, self_loop_idx], dim=0)
                            else:
                                edge_idx = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)

                        # 前向传播
                        predictions = self.model(x_valid_batch, edge_idx)

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

                        for step in range(t, end_t):
                            x_val_batch = x_valid[step].to(self.device)
                            y_val_batch = y_valid[step].to(self.device)

                            # 获取当前时间步的边索引
                            if edge_valid is not None and step < len(edge_valid):
                                current_edge = edge_valid[step].to(self.device)
                                # 如果是稀疏张量，转换为密集张量
                                if current_edge.is_sparse:
                                    current_edge = current_edge.to_dense().long()
                                elif current_edge.is_sparse_csr:
                                    current_edge = current_edge.to_dense().long()
                            else:
                                current_edge = torch.zeros((2, 0), dtype=torch.long, device=self.device)

                            # 过滤NaN值
                            val_valid_mask = ~(torch.isnan(x_val_batch).any(dim=1) | torch.isnan(y_val_batch))
                            if val_valid_mask.any():
                                x_val_valid = x_val_batch[val_valid_mask]
                                y_val_valid = y_val_batch[val_valid_mask].unsqueeze(1)

                                # 使用current_edge并过滤掉NaN值对应的节点
                                if current_edge.size(1) > 0:  # 如果有边存在
                                    # 映射到有效节点的索引
                                    valid_indices = torch.nonzero(val_valid_mask, as_tuple=True)[0]
                                    # 创建映射，将原始节点索引映射到过滤后的新索引
                                    node_mapping = torch.full((x_val_batch.size(0),), -1, dtype=torch.long, device=self.device)
                                    node_mapping[valid_indices] = torch.arange(len(valid_indices), device=self.device)

                                    # 过滤边，只保留有效节点之间的边
                                    edge_mask = (node_mapping[current_edge[0]] != -1) & (node_mapping[current_edge[1]] != -1)
                                    filtered_edge = current_edge[:, edge_mask]

                                    # 重新映射边的索引
                                    edge_idx = torch.stack([
                                        node_mapping[filtered_edge[0]],
                                        node_mapping[filtered_edge[1]]
                                    ], dim=0)
                                else:
                                    # 如果没有提供边索引，创建自环边
                                    num_valid_nodes = x_val_valid.size(0)
                                    if num_valid_nodes > 1:
                                        self_loop_idx = torch.arange(num_valid_nodes, device=self.device)
                                        edge_idx = torch.stack([self_loop_idx, self_loop_idx], dim=0)
                                    else:
                                        edge_idx = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)

                                val_preds = self.model(x_val_valid, edge_idx)

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

    def predict(self, x_test, edge_index=None):
        """
        生成预测结果，支持DataFrame和PyTorch张量输入

        参数:
        x_test: 测试数据，可以是pd.DataFrame或torch.Tensor [时间, 股票, 因子]
        edge_index: 边索引列表（可选）

        返回:
        对于DataFrame输入返回list，对于张量输入返回torch.Tensor [时间, 股票]
        """
        # 处理PyTorch张量输入
        if isinstance(x_test, torch.Tensor):
            return self._predict_tensor(x_test, edge_index)
        # 保持原有DataFrame接口
        else:
            return super().predict(x_test)

    def _predict_tensor(self, x_test_tensor, edge_index=None):
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

                for step in range(t, end_t):
                    # 获取当前时间步的输入 [N, K]
                    x_batch = x_test_tensor[step].to(self.device)

                    # 获取当前时间步的边索引
                    if edge_index is not None and step < len(edge_index):
                        current_edge = edge_index[step].to(self.device)
                        # 如果是稀疏张量，转换为密集张量
                        if current_edge.is_sparse:
                            current_edge = current_edge.to_dense().long()
                        elif current_edge.is_sparse_csr:
                            current_edge = current_edge.to_dense().long()
                    else:
                        current_edge = torch.zeros((2, 0), dtype=torch.long, device=self.device)

                    # 创建当前时间步的预测结果数组，初始化为NaN
                    batch_pred = torch.full((x_batch.shape[0],), np.nan, dtype=torch.float32, device=self.device)

                    # 找出有效数据的索引
                    valid_mask = ~torch.isnan(x_batch).any(dim=1)
                    if valid_mask.any():
                        # 对有效数据进行预测
                        x_valid_batch = x_batch[valid_mask]

                        # 使用current_edge并过滤掉NaN值对应的节点
                        if current_edge.size(1) > 0:  # 如果有边存在
                            # 映射到有效节点的索引
                            valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
                            # 创建映射，将原始节点索引映射到过滤后的新索引
                            node_mapping = torch.full((x_batch.size(0),), -1, dtype=torch.long, device=self.device)
                            node_mapping[valid_indices] = torch.arange(len(valid_indices), device=self.device)

                            # 过滤边，只保留有效节点之间的边
                            edge_mask = (node_mapping[current_edge[0]] != -1) & (node_mapping[current_edge[1]] != -1)
                            filtered_edge = current_edge[:, edge_mask]

                            # 重新映射边的索引
                            edge_idx = torch.stack([
                                node_mapping[filtered_edge[0]],
                                node_mapping[filtered_edge[1]]
                            ], dim=0)
                        else:
                            # 如果没有提供边索引，创建自环边
                            num_valid_nodes = x_valid_batch.size(0)
                            if num_valid_nodes > 1:
                                self_loop_idx = torch.arange(num_valid_nodes, device=self.device)
                                edge_idx = torch.stack([self_loop_idx, self_loop_idx], dim=0)
                            else:
                                edge_idx = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)

                        valid_preds = self.model(x_valid_batch, edge_idx)
                        # 将预测结果放回到正确位置
                        batch_pred[valid_mask] = valid_preds.squeeze(1)

                    # 将当前时间步的预测结果放入输出张量
                    y_pred[step] = batch_pred

        return y_pred

    def fit_kfold(self, x, y, z=None, edge: list = None, k: int = 5, train_size=None, test_size=None,
                  collect: bool = False):
        from ..utils import from_pandas_to_list, split_dataset_by_index
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
            edge_train, edge_valid = split_dataset_by_index(edge, train_index, valid_index)
            if z is not None:
                z_train, z_valid = split_dataset_by_index(z_list, train_index, valid_index)
            else:
                z_train, z_valid = None, None
            print("fold: ", fold)
            self.fit(x_train, y_train, x_valid, y_valid, z_train=z_train, z_valid=z_valid, edge_train=edge_train,
                     edge_valid=edge_valid)
            if collect:
                for i in range(len(x_valid)):
                    self.output.append(Series(self.predict_(
                        x_valid[i],
                        edge_index=edge_valid[i],
                    ).view(-1, )))
        if collect:
            self.output = concat(self.output, axis=0)
            if isinstance(x, DataFrame):
                self.output.index = x.index[-len(self.output):]

    def predict_pandas(self, x: DataFrame, edge_index=None) -> Series:
        from ..utils import from_pandas_to_list
        x_tensor = from_pandas_to_list(x, fillna=self.fillna)
        result = []

        for i in range(len(x_tensor)):
            current_edge = edge_index[i] if edge_index is not None else None

            result.append(Series(self.predict_(
                x_tensor[i],
                edge_index=current_edge,
            ).view(-1, )))

        days = x.index.get_level_values(0).unique()[-len(result):]
        instrument = x.index.get_level_values(1).unique().to_list()
        name_0, name_1 = x.index.names[0], x.index.names[1]

        predict = []
        for i in range(len(result)):
            df = DataFrame(result[i])
            df[name_0] = days[i]
            df[name_1] = instrument
            predict.append(df.set_index([name_0, name_1]).iloc[:, 0])

        predict = concat(predict, axis=0)
        return predict[predict.index.isin(x.index)]


class MyGCN(Model):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, output_shape: int = 1,
                 jump_connection: bool = False, num_layers: int = 1, add_self_loops: bool = False,
                 fillna: bool = False, dropout: float = 0.2, device: str = None, batch_size: int = 32, *args, **kwargs):
        """
        MyGCN模型，支持多层卷积和[T, N, K]张量输入
        """
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_shape = output_shape
        self.jump_connection = jump_connection
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size  # 添加batch_size参数

        # 保存原始dropout值，用于init_model方法
        self.dropout_p = dropout  # 保存原始dropout概率值

        # 输入层线性变换
        self.input_linear = torch.nn.Linear(input_channels, hidden_channels)
        self.input_bn = torch.nn.BatchNorm1d(hidden_channels)

        # 多层GCN卷积参数
        self.convs_weights = torch.nn.ParameterList()
        self.convs_biases = torch.nn.ParameterList()

        # 构建多层GCN
        in_channels = hidden_channels
        for i in range(num_layers):
            out_channels = output_channels if i == num_layers - 1 else hidden_channels
            self.convs_weights.append(torch.nn.Parameter(torch.randn(in_channels, out_channels)))
            self.convs_biases.append(torch.nn.Parameter(torch.zeros(out_channels)))
            in_channels = out_channels

        # 输出层
        self.output_linear = torch.nn.Linear(output_channels, 1)

        # 跳跃连接
        if jump_connection is True:
            self.jump_connection = torch.nn.Linear(input_channels, 1)

        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

        # 填充NaN值
        self.fillna = fillna

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化网络权重"""
        # 使用Xavier/Glorot初始化
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        # 初始化GCN卷积权重
        for weight in self.convs_weights:
            torch.nn.init.xavier_uniform_(weight)
        for bias in self.convs_biases:
            torch.nn.init.zeros_(bias)

    def forward(self, x, edge_index):
        """
        前向传播，保持使用mm和spmm手动实现GCN卷积算法
        """
        # 处理稀疏边索引
        if edge_index is None:
            # 创建全零邻接矩阵，表示没有边连接
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            if self.add_self_loops:
                edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        else:
            if self.add_self_loops:
                edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # 输入线性变换
        x_input = x

        if self.jump_connection:
            jump = self.jump_connection(x_input)


        x = self.input_linear(x)  # [N, hidden_channels]
        x = self.input_bn(x)   # 归一化hidden_channels维度

        # 多层GCN卷积，使用mm和spmm手动实现
        for i in range(self.num_layers):
            weight = self.convs_weights[i]
            bias = self.convs_biases[i]

            # GCN卷积: XW + b
            x = torch.mm(x, weight) + bias  # [N, hidden_channels]
            x = torch.spmm(edge_index, x)  # 使用spmm进行稀疏矩阵乘法
            x = torch.relu(x)
            x = self.dropout(x)

        # 输出层
        x = self.output_linear(x)

        if self.jump_connection:
            x = x + jump

        return x

    def init_model(self):
        self.model = MyGCN(
            input_channels=self.input_channels,
            hidden_channels=self.hidden_channels,
            output_channels=self.output_channels,
            output_shape=self.output_shape,
            jump_connection=self.jump_connection,
            num_layers=self.num_layers,
            add_self_loops=self.add_self_loops,
            fillna=self.fillna,
            dropout=self.dropout_p,
            epochs=self.epochs,
            loss=self.loss,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            device=self.device
        ).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, z_train=None, z_valid=None, edge_train: list = None,
            edge_valid: list = None, **kwargs):
        """
        训练模型，支持DataFrame和张量输入
        """
        # 检查是否为张量输入
        if isinstance(x_train, torch.Tensor):
            return self._fit_tensor(
                x_train, y_train, x_valid, y_valid, z_train, z_valid,
                edge_train, edge_valid, **kwargs
            )
        else:
            # 保持原有的DataFrame输入处理逻辑
            from ..utils import transform_data, split_dataset_by_index
            x_train, y_train, x_valid, y_valid, z_train, z_valid = transform_data(
                x_train, y_train, x_valid, y_valid,
                z_train, z_valid,
                for_cnn=self.for_cnn,
                for_rnn=self.for_rnn,
                fillna=self.fillna
            )

            # 添加最佳模型跟踪变量
            best_loss_val = float('inf')
            best_val_ic = -float('inf')
            best_epoch = 0
            best_save_path = None

            for epoch in range(1, self.epochs + 1):
                total_loss_train = 0
                total_loss_val = 0
                val_ic = 0

                # 训练阶段
                self.model.train()
                for i in range(len(x_train)):
                    loss_train = self.get_loss(x=x_train[i], y=y_train[i], z=z_train[i] if z_train is not None else None,
                                               edge_index=edge_train[i] if edge_train is not None else None)
                    total_loss_train += loss_train

                # 验证阶段
                self.model.eval()
                with torch.no_grad():
                    for i in range(len(x_valid)):
                        loss_val = self.test(x=x_valid[i], y=y_valid[i], z=z_valid[i] if z_valid is not None else None,
                                             edge_index=edge_valid[i] if edge_valid is not None else None)
                        total_loss_val += loss_val
                        val_ic += float(calc_tensor_corr(
                            self.predict_(x_valid[i], edge_index=edge_valid[i] if edge_valid is not None else None),
                            y_valid[i]))

                avg_loss_train = total_loss_train / len(x_train)
                avg_loss_val = total_loss_val / len(x_valid)
                avg_val_ic = val_ic / len(x_valid)

                print("Epoch:", epoch, "loss:", avg_loss_train, "val_loss:", avg_loss_val, "val_ic:", avg_val_ic)

                # 早停
                if self.early_stopping > 0:
                    if avg_loss_val < best_loss_val - self.min_delta:
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        print(f"Early stopping counter: {self.early_stop_counter}/{self.early_stopping}")
                        if self.early_stop_counter >= self.early_stopping:
                            print("Early stopping triggered")
                            break

                # 学习率调度
                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(avg_loss_val)
                    else:
                        self.lr_scheduler.step()

                # 保存模型
                if self.auto_save:
                    save_path = f"{self.save_folder}/epoch_{epoch}.pth"
                    folder_path = os.path.dirname(save_path)

                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    torch.save(self.state_dict(), save_path)
                    print(f"Model saved to {save_path}")

                    if avg_loss_val < best_loss_val:
                        best_loss_val = avg_loss_val
                        best_epoch = epoch
                        best_save_path = f"{self.save_folder}/best_epoch.pth"
                        torch.save(self.state_dict(), best_save_path)

            if self.auto_save and best_save_path:
                torch.save(self.state_dict(), best_save_path)
                print(f"Best epoch:{best_epoch}, MSE:{best_loss_val}, Best Model saved to {best_save_path}")

            return self

    def _fit_tensor(self, x_train_tensor, y_train_tensor, x_valid=None, y_valid=None, z_train=None, z_valid=None,
                    edge_train: list = None, edge_valid: list = None, **kwargs):
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

                # 按时间步处理
                for step in range(t, end_t):
                    x_batch = x_train_tensor[step].to(self.device)  # [N, K]
                    y_batch = y_train_tensor[step].to(self.device)  # [N]

                    # 获取当前时间步的边索引
                    if edge_train is not None and step < len(edge_train):
                        current_edge = edge_train[step].to(self.device)
                        # 如果是稀疏张量，转换为密集张量
                        if current_edge.is_sparse:
                            adj_matrix = current_edge.to_dense()
                        elif current_edge.is_sparse_csr:
                            adj_matrix = current_edge.to_dense()
                    else:
                        # 如果没有提供边索引，创建一个没有边的图（只有自环）
                        adj_matrix = torch.zeros((x_batch.size(0), x_batch.size(0)), dtype=torch.float32, device=self.device)

                    # 过滤掉NaN值
                    valid_mask = ~(torch.isnan(x_batch).any(dim=1) | torch.isnan(y_batch))

                    if valid_mask.any():
                        x_valid_batch = x_batch[valid_mask]
                        y_valid_batch = y_batch[valid_mask].unsqueeze(1)  # [N, 1]
                        adj_matrix = adj_matrix[valid_mask][:, valid_mask]
                        adj_matrix = normalize_matrix_tensor(adj_matrix)

                        # 将邻接矩阵转换为稀疏格式
                        edge_idx = adj_matrix.to_sparse_csr()

                        # 前向传播
                        predictions = self.model(x_valid_batch, edge_idx)

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
                avg_val_loss = self.valid(x_valid, y_valid, edge_valid)

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

    def valid(self, x_valid, y_valid, edge_valid=None):
        """
        验证模型在验证集上的性能

        参数:
        x_val: 验证数据，形状为[时间, 股票, 因子]
        y_val: 验证标签，形状为[时间, 股票]
        edge_index: 边索引列表（可选）

        返回:
        验证损失（float）
        """
        self.model.eval()
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            # 处理验证集
            val_time_steps = x_valid.shape[0]
            for t in range(0, val_time_steps, self.batch_size):
                end_t = min(t + self.batch_size, val_time_steps)

                for step in range(t, end_t):
                    x_val_batch = x_valid[step].to(self.device)
                    y_val_batch = y_valid[step].to(self.device)

                    # 获取当前时间步的边索引
                    if edge_valid is not None and step < len(edge_valid):
                        current_edge = edge_valid[step].to(self.device)
                        # 如果是稀疏张量，转换为密集张量
                        if current_edge.is_sparse:
                            adj_matrix = current_edge.to_dense()
                        elif current_edge.is_sparse_csr:
                            adj_matrix = current_edge.to_dense()
                    else:
                        # 如果没有提供边索引，创建一个没有边的图（只有自环）
                        adj_matrix = torch.zeros((x_val_batch.size(0), x_val_batch.size(0)), dtype=torch.float32, device=self.device)

                    # 过滤掉NaN值
                    valid_mask = ~(torch.isnan(x_val_batch).any(dim=1) | torch.isnan(y_val_batch))

                    if valid_mask.any():
                        x_val_valid = x_val_batch[valid_mask]
                        y_val_valid = y_val_batch[valid_mask]
                        adj_matrix = adj_matrix[valid_mask][:, valid_mask]
                        adj_matrix = normalize_matrix_tensor(adj_matrix)

                        # 将邻接矩阵转换为稀疏格式
                        edge_idx = adj_matrix.to_sparse_csr()

                        # 前向传播
                        val_preds = self.model(x_val_valid, edge_idx)

                        # 计算损失
                        if isinstance(self.loss, str):
                            val_batch_loss = eval("f." + self.loss + "(val_preds, y_val_valid)")
                        else:
                            val_batch_loss = self.loss(val_preds, y_val_valid)

                        val_loss += val_batch_loss.item()
                        val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

        return avg_val_loss

    def predict(self, x_test, edge_index=None):
        """
        生成预测结果，支持DataFrame和PyTorch张量输入

        参数:
        x_test: 测试数据，可以是pd.DataFrame或torch.Tensor [时间, 股票, 因子]
        edge_index: 边索引列表（可选）

        返回:
        对于DataFrame输入返回list，对于张量输入返回torch.Tensor [时间, 股票]
        """
        # 处理PyTorch张量输入
        if isinstance(x_test, torch.Tensor):
            return self._predict_tensor(x_test, edge_index)
        # 保持原有DataFrame接口
        else:
            return super().predict(x_test)

    def _predict_tensor(self, x_test_tensor, edge_index=None):
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

                for step in range(t, end_t):
                    # 获取当前时间步的输入 [N, K]
                    x_batch = x_test_tensor[step].to(self.device)

                    # 获取当前时间步的边索引
                    if edge_index is not None and step < len(edge_index):
                        current_edge = edge_index[step].to(self.device)
                        # 如果是稀疏张量，转换为密集张量
                        if current_edge.is_sparse:
                            adj_matrix = current_edge.to_dense()
                        elif current_edge.is_sparse_csr:
                            adj_matrix = current_edge.to_dense()
                    else:
                        adj_matrix = torch.zeros((x_batch.size(0), x_batch.size(0)), dtype=torch.float32, device=self.device)

                    # 创建当前时间步的预测结果数组，初始化为NaN
                    batch_pred = torch.full((x_batch.shape[0],), np.nan, dtype=torch.float32, device=self.device)

                    # 找出有效数据的索引
                    valid_mask = ~torch.isnan(x_batch).any(dim=1)
                    if valid_mask.any():
                        # 对有效数据进行预测
                        x_valid_batch = x_batch[valid_mask]

                        # 使用current_edge并过滤掉NaN值对应的节点
                        if adj_matrix.size(1) > 0:  # 如果有边存在
                            adj_matrix = adj_matrix[valid_mask][:, valid_mask]

                        edge_idx = adj_matrix.to_sparse_csr()

                        valid_preds = self.model(x_valid_batch, edge_idx)
                        # 将预测结果放回到正确位置
                        batch_pred[valid_mask] = valid_preds.squeeze(1)

                    # 将当前时间步的预测结果放入输出张量
                    y_pred[step] = batch_pred

        return y_pred

    def predict_pandas(self, x: DataFrame, edge_index=None) -> Series:
        from ..utils import from_pandas_to_list
        x_tensor = from_pandas_to_list(x, fillna=self.fillna)
        result = []

        for i in range(len(x_tensor)):
            current_edge = edge_index[i] if edge_index is not None else None

            result.append(Series(self.predict_(
                x_tensor[i],
                edge_index=current_edge,
            ).view(-1, )))

        days = x.index.get_level_values(0).unique()[-len(result):]
        instrument = x.index.get_level_values(1).unique().to_list()
        name_0, name_1 = x.index.names[0], x.index.names[1]

        predict = []
        for i in range(len(result)):
            df = DataFrame(result[i])
            df[name_0] = days[i]
            df[name_1] = instrument
            predict.append(df.set_index([name_0, name_1]).iloc[:, 0])

        predict = concat(predict, axis=0)
        return predict[predict.index.isin(x.index)]


class MyGAT(Model):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, output_shape: int = 1,
                 heads: int = 1, add_self_loop: bool = False, fillna: bool = False, device: str = None,
                 jump_connection: bool = False, batch_size: int = 32, dropout: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_shape = output_shape
        self.jump_connection = jump_connection
        self.heads = heads
        self.add_self_loops = add_self_loop
        self.fillna = fillna
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size  # 添加batch_size参数
        self.dropout_p = dropout  # 保存原始dropout概率值

        # 注意力机制参数
        self.att_weight = torch.nn.Parameter(torch.Tensor(2 * hidden_channels, 1))
        self.weight = torch.nn.Parameter(torch.Tensor(input_channels, hidden_channels * heads))
        self.bias = torch.nn.Parameter(torch.Tensor(hidden_channels * heads))
        self.dropout = torch.nn.Dropout(dropout)
        self.output_linear = torch.nn.Linear(hidden_channels * heads, 1)

        # 跳跃连接
        if jump_connection is True:
            self.jump_connection = torch.nn.Linear(input_channels, 1)

        self.init_weights()

    def init_weights(self):
        """初始化网络权重"""
        # 使用Xavier/Glorot初始化
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        # 初始化GAT特定参数
        torch.nn.init.xavier_uniform_(self.att_weight)
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = torch.sparse_csr_tensor(
                crow_indices=torch.tensor([0, x.shape[0]]),
                col_indices=torch.tensor([]),
                values=torch.tensor([]),
                size=(x.shape[0], x.shape[0])
            )

        # 输入线性变换
        x_input = x

        if self.jump_connection:
            jump = self.jump_connection(x_input)

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
        out = self.dropout(out)
        out = self.output_linear(out)

        if self.jump_connection:
            out = out + jump

        return out

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
            weight_decay=self.weight_decay,
            device=self.device,
            batch_size=self.batch_size,
            dropout=self.dropout_p
        ).to(torch.float32)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = lr_scheduler(self.optimizer, self.lr_scheduler_kwargs)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, z_train=None, z_valid=None, edge_train: list = None,
            edge_valid: list = None, **kwargs):
        """
        训练模型，支持DataFrame和张量输入
        """
        # 检查是否为张量输入
        if isinstance(x_train, torch.Tensor):
            return self._fit_tensor(
                x_train, y_train, x_valid, y_valid, z_train, z_valid,
                edge_train, edge_valid, **kwargs
            )
        else:
            # 保持原有的DataFrame输入处理逻辑
            from ..utils import transform_data, split_dataset_by_index
            x_train, y_train, x_valid, y_valid, z_train, z_valid = transform_data(
                x_train, y_train, x_valid, y_valid,
                z_train, z_valid,
                for_cnn=self.for_cnn,
                for_rnn=self.for_rnn,
                fillna=self.fillna
            )

            # 添加最佳模型跟踪变量
            best_loss_val = float('inf')
            best_val_ic = -float('inf')
            best_epoch = 0
            best_save_path = None

            for epoch in range(1, self.epochs + 1):
                total_loss_train = 0
                total_loss_val = 0
                val_ic = 0

                # 训练阶段
                self.model.train()
                for i in range(len(x_train)):
                    loss_train = self.get_loss(x=x_train[i], y=y_train[i], z=z_train[i] if z_train is not None else None,
                                               edge_index=edge_train[i] if edge_train is not None else None)
                    total_loss_train += loss_train

                # 验证阶段
                self.model.eval()
                with torch.no_grad():
                    for i in range(len(x_valid)):
                        loss_val = self.test(x=x_valid[i], y=y_valid[i], z=z_valid[i] if z_valid is not None else None,
                                             edge_index=edge_valid[i] if edge_valid is not None else None)
                        total_loss_val += loss_val
                        val_ic += float(calc_tensor_corr(
                            self.predict_(x_valid[i], edge_index=edge_valid[i] if edge_valid is not None else None),
                            y_valid[i]))

                avg_loss_train = total_loss_train / len(x_train)
                avg_loss_val = total_loss_val / len(x_valid)
                avg_val_ic = val_ic / len(x_valid)

                print("Epoch:", epoch, "loss:", avg_loss_train, "val_loss:", avg_loss_val, "val_ic:", avg_val_ic)

                # 早停
                if self.early_stopping > 0:
                    if avg_loss_val < best_loss_val - self.min_delta:
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        print(f"Early stopping counter: {self.early_stop_counter}/{self.early_stopping}")
                        if self.early_stop_counter >= self.early_stopping:
                            print("Early stopping triggered")
                            break

                # 学习率调度
                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(avg_loss_val)
                    else:
                        self.lr_scheduler.step()

                # 保存模型
                if self.auto_save:
                    save_path = f"{self.save_folder}/epoch_{epoch}.pth"
                    folder_path = os.path.dirname(save_path)

                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    torch.save(self.state_dict(), save_path)
                    print(f"Model saved to {save_path}")

                    if avg_loss_val < best_loss_val:
                        best_loss_val = avg_loss_val
                        best_epoch = epoch
                        best_save_path = f"{self.save_folder}/best_epoch.pth"
                        torch.save(self.state_dict(), best_save_path)

            if self.auto_save and best_save_path:
                torch.save(self.state_dict(), best_save_path)
                print(f"Best epoch:{best_epoch}, MSE:{best_loss_val}, Best Model saved to {best_save_path}")

            return self

    def _fit_tensor(self, x_train_tensor, y_train_tensor, x_valid=None, y_valid=None, z_train=None, z_valid=None,
                    edge_train: list = None, edge_valid: list = None, **kwargs):
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

                # 按时间步处理
                for step in range(t, end_t):
                    x_batch = x_train_tensor[step].to(self.device)  # [N, K]
                    y_batch = y_train_tensor[step].to(self.device)  # [N]

                    # 获取当前时间步的边索引
                    if edge_train is not None and step < len(edge_train):
                        current_edge = edge_train[step].to(self.device)
                        # 如果是稀疏张量，转换为密集张量
                        if current_edge.is_sparse:
                            adj_matrix = current_edge.to_dense()
                        elif current_edge.is_sparse_csr:
                            adj_matrix = current_edge.to_dense()
                    else:
                        # 如果没有提供边索引，创建一个没有边的图（只有自环）
                        adj_matrix = torch.zeros((x_batch.size(0), x_batch.size(0)), dtype=torch.float32, device=self.device)

                    # 过滤掉NaN值
                    valid_mask = ~(torch.isnan(x_batch).any(dim=1) | torch.isnan(y_batch))

                    if valid_mask.any():
                        x_valid_batch = x_batch[valid_mask]
                        y_valid_batch = y_batch[valid_mask].unsqueeze(1)  # [N, 1]
                        adj_matrix = adj_matrix[valid_mask][:, valid_mask]
                        adj_matrix = normalize_matrix_tensor(adj_matrix)

                        # 将邻接矩阵转换为稀疏格式
                        edge_idx = adj_matrix.to_sparse_csr()

                        # 前向传播
                        predictions = self.model(x_valid_batch, edge_idx)

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
                avg_val_loss = self.valid(x_valid, y_valid, edge_valid)

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

    def valid(self, x_valid, y_valid, edge_valid=None):
        """
        验证模型在验证集上的性能

        参数:
        x_val: 验证数据，形状为[时间, 股票, 因子]
        y_val: 验证标签，形状为[时间, 股票]
        edge_index: 边索引列表（可选）

        返回:
        验证损失（float）
        """
        self.model.eval()
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            # 处理验证集
            val_time_steps = x_valid.shape[0]
            for t in range(0, val_time_steps, self.batch_size):
                end_t = min(t + self.batch_size, val_time_steps)

                for step in range(t, end_t):
                    x_val_batch = x_valid[step].to(self.device)
                    y_val_batch = y_valid[step].to(self.device)

                    # 获取当前时间步的边索引
                    if edge_valid is not None and step < len(edge_valid):
                        current_edge = edge_valid[step].to(self.device)
                        # 如果是稀疏张量，转换为密集张量
                        if current_edge.is_sparse:
                            adj_matrix = current_edge.to_dense()
                        elif current_edge.is_sparse_csr:
                            adj_matrix = current_edge.to_dense()
                    else:
                        # 如果没有提供边索引，创建一个没有边的图（只有自环）
                        adj_matrix = torch.zeros((x_val_batch.size(0), x_val_batch.size(0)), dtype=torch.float32, device=self.device)

                    # 过滤掉NaN值
                    valid_mask = ~(torch.isnan(x_val_batch).any(dim=1) | torch.isnan(y_val_batch))

                    if valid_mask.any():
                        x_val_valid = x_val_batch[valid_mask]
                        y_val_valid = y_val_batch[valid_mask]
                        adj_matrix = adj_matrix[valid_mask][:, valid_mask]
                        adj_matrix = normalize_matrix_tensor(adj_matrix)

                        # 将邻接矩阵转换为稀疏格式
                        edge_idx = adj_matrix.to_sparse_csr()

                        # 前向传播
                        val_preds = self.model(x_val_valid, edge_idx)

                        # 计算损失
                        if isinstance(self.loss, str):
                            val_batch_loss = eval("f." + self.loss + "(val_preds, y_val_valid)")
                        else:
                            val_batch_loss = self.loss(val_preds, y_val_valid)

                        val_loss += val_batch_loss.item()
                        val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

        return avg_val_loss

    def predict(self, x_test, edge_index=None):
        """
        生成预测结果，支持DataFrame和PyTorch张量输入

        参数:
        x_test: 测试数据，可以是pd.DataFrame或torch.Tensor [时间, 股票, 因子]
        edge_index: 边索引列表（可选）

        返回:
        对于DataFrame输入返回list，对于张量输入返回torch.Tensor [时间, 股票]
        """
        # 处理PyTorch张量输入
        if isinstance(x_test, torch.Tensor):
            return self._predict_tensor(x_test, edge_index)
        # 保持原有DataFrame接口
        else:
            return super().predict(x_test)

    def _predict_tensor(self, x_test_tensor, edge_index=None):
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

                for step in range(t, end_t):
                    # 获取当前时间步的输入 [N, K]
                    x_batch = x_test_tensor[step].to(self.device)

                    # 获取当前时间步的边索引
                    if edge_index is not None and step < len(edge_index):
                        current_edge = edge_index[step].to(self.device)
                        # 如果是稀疏张量，转换为密集张量
                        if current_edge.is_sparse:
                            adj_matrix = current_edge.to_dense()
                        elif current_edge.is_sparse_csr:
                            adj_matrix = current_edge.to_dense()
                    else:
                        adj_matrix = torch.zeros((x_batch.size(0), x_batch.size(0)), dtype=torch.float32, device=self.device)

                    # 创建当前时间步的预测结果数组，初始化为NaN
                    batch_pred = torch.full((x_batch.shape[0],), np.nan, dtype=torch.float32, device=self.device)

                    # 找出有效数据的索引
                    valid_mask = ~torch.isnan(x_batch).any(dim=1)
                    if valid_mask.any():
                        # 对有效数据进行预测
                        x_valid_batch = x_batch[valid_mask]

                        # 使用current_edge并过滤掉NaN值对应的节点
                        if adj_matrix.size(1) > 0:  # 如果有边存在
                            adj_matrix = adj_matrix[valid_mask][:, valid_mask]

                        edge_idx = adj_matrix.to_sparse_csr()

                        valid_preds = self.model(x_valid_batch, edge_idx)
                        # 将预测结果放回到正确位置
                        batch_pred[valid_mask] = valid_preds.squeeze(1)

                    # 将当前时间步的预测结果放入输出张量
                    y_pred[step] = batch_pred

        return y_pred

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


class GNNHAR(Model):
    '''
    Source Code: https://github.com/chaozhang-ox/GNNHAR
    '''
    def __init__(self, input_channels=3, hidden_channels=64, output_shape=1,
                 batch_size: int = 32, device: str = None, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 时序特征分支
        self.time_linear = torch.nn.Linear(input_channels, hidden_channels)

        # 图卷积分支
        self.gcn_weight = torch.nn.Parameter(torch.Tensor(input_channels, hidden_channels))
        self.gcn_bias = torch.nn.Parameter(torch.Tensor(hidden_channels))

        # 融合层
        self.fusion = torch.nn.Linear(hidden_channels*2, output_shape)

        self.for_rnn=True
        self.init_weights()

    def init_weights(self):
        """初始化网络权重"""
        torch.nn.init.xavier_uniform_(self.gcn_weight)
        torch.nn.init.zeros_(self.gcn_bias)
        torch.nn.init.xavier_uniform_(self.time_linear.weight)
        torch.nn.init.xavier_uniform_(self.fusion.weight)

    def forward(self, x, edge_index=None):
        # x: [batch_size, num_nodes, input_dim]

        # 时序特征提取
        time_feat = self.time_linear(x)  # [batch, N, hidden]

        # 图卷积处理
        x_gcn = torch.matmul(x, self.gcn_weight) + self.gcn_bias  # [batch, N, hidden]
        x_gcn = x_gcn.squeeze(1)  # 变成 [50, 64]
        x_gcn = torch.spmm(edge_index, x_gcn)
        # x_gcn = x_gcn.unsqueeze(1)  # 变回 [50, 1, 64]
        graph_feat = torch.relu(x_gcn)

        # 特征融合
        combined = torch.cat([time_feat, graph_feat], dim=-1)
        output = self.fusion(combined)  # [batch, N, 1]

        return output

    def init_model(self):
        self.model = GNNHAR(
            input_channels=self.input_channels,
            hidden_channels=self.hidden_channels,
            output_shape=self.output_shape,
            device=self.device,
            epochs=self.epochs,
            lr=self.lr
        ).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.lr_scheduler = lr_scheduler(self.optimizer,self.lr_scheduler_kwargs)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, z_train=None, z_valid=None, edge_train: list = None,
            edge_valid: list = None, **kwargs):
        # 检查是否为张量输入
        if isinstance(x_train, torch.Tensor):
            return self._fit_tensor(
                x_train, y_train, x_valid, y_valid, z_train, z_valid,
                edge_train, edge_valid, **kwargs
            )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            x_train, y_train, x_valid, y_valid, z_train, z_valid = transform_data(
                x_train, y_train, x_valid, y_valid,
                z_train, z_valid,
                for_cnn=self.for_cnn,
                for_rnn=self.for_rnn,
                fillna=True
            )

            # 新增数据转换部分
            def to_device(data, device):
                if isinstance(data, DataFrame):
                    return torch.tensor(data.values, device=device, dtype=torch.float32)
                return data.to(device) if isinstance(data, torch.Tensor) else data

            # 转换训练数据到设备
            x_train = to_device(x_train, device)
            y_train = to_device(y_train, device)
            x_valid = to_device(x_valid, device)
            y_valid = to_device(y_valid, device)

            # 转换边数据到设备
            edge_train = [edge.to(device) for edge in edge_train]
            edge_valid = [edge.to(device) for edge in edge_valid]

            # 训练逻辑与原有GNN类完全一致
            best_val_ic = -float('inf')
            best_loss_val = float('inf')
            best_epoch = 0
            best_save_path = None

            for epoch in range(1, self.epochs + 1):
                total_loss_train = 0

                # 训练循环
                for i in range(len(x_train)):
                    loss_train = self.get_loss(
                        x=x_train[:, i:i + 1, :],
                        y=y_train[:, i:i + 1, :],
                        z=z_train[:, i:i + 1, :] if z_train else None,
                        edge_index=edge_train[i] if edge_train else None
                    )
                    total_loss_train += loss_train

                # 使用新的valid方法进行验证
                avg_loss_val, avg_val_ic = self.valid(x_valid, y_valid, z_valid, edge_valid)

                # 输出训练信息（与原有GNN保持相同格式）
                avg_loss_train = total_loss_train / len(x_train)
                print(f"Epoch: {epoch} loss: {avg_loss_train:.4f} val_loss: {avg_loss_val:.4f} val_ic: {avg_val_ic:.4f}")

                # 早停
                if self.early_stopping > 0:
                    if avg_loss_val < best_loss_val - self.min_delta:
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        print(f"Early stopping counter: {self.early_stop_counter}/{self.early_stopping}")
                        if self.early_stop_counter >= self.early_stopping:
                            print("Early stopping triggered")
                            break

                # 学习率调度
                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(avg_loss_val)
                    else:
                        self.lr_scheduler.step()

                # 保存模型
                if self.auto_save:
                    save_path = f"{self.save_folder}/epoch_{epoch}.pth"
                    folder_path = os.path.dirname(save_path)

                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    torch.save(self.state_dict(), save_path)
                    print(f"Model saved to {save_path}")

                    if avg_loss_val < best_loss_val:
                        best_loss_val = avg_loss_val
                        best_epoch = epoch
                        best_save_path = f"{self.save_folder}/best_epoch.pth"
                        torch.save(self.state_dict(), best_save_path)
                        print(f"Best model so far saved to {best_save_path}")

    def _fit_tensor(self, x_train_tensor, y_train_tensor, x_valid=None, y_valid=None, z_train=None, z_valid=None,
                    edge_train: list = None, edge_valid: list = None, **kwargs):
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

                # 按时间步处理
                for step in range(t, end_t):
                    x_batch = x_train_tensor[step].to(self.device)  # [N, K]
                    y_batch = y_train_tensor[step].to(self.device)  # [N]

                    # 获取当前时间步的边索引
                    if edge_train is not None and step < len(edge_train):
                        current_edge = edge_train[step].to(self.device)
                        # 如果是稀疏张量，转换为密集张量
                        if current_edge.is_sparse:
                            adj_matrix = current_edge.to_dense()
                        elif current_edge.is_sparse_csr:
                            adj_matrix = current_edge.to_dense()
                    else:
                        # 如果没有提供边索引，创建一个没有边的图（只有自环）
                        adj_matrix = torch.zeros((x_batch.size(0), x_batch.size(0)), dtype=torch.float32, device=self.device)

                    # 过滤掉NaN值
                    valid_mask = ~(torch.isnan(x_batch).any(dim=1) | torch.isnan(y_batch))

                    if valid_mask.any():
                        x_valid_batch = x_batch[valid_mask]
                        y_valid_batch = y_batch[valid_mask].unsqueeze(1)  # [N, 1]
                        adj_matrix = adj_matrix[valid_mask][:, valid_mask]
                        adj_matrix = normalize_matrix_tensor(adj_matrix)

                        # 将邻接矩阵转换为稀疏格式
                        edge_idx = adj_matrix.to_sparse_csr()

                        # 前向传播
                        predictions = self.model(x_valid_batch, edge_idx)

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
                avg_val_loss = self.valid(x_valid, y_valid, edge_valid)

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

    def predict(self, x_test, edge_index=None):
        """
        生成预测结果，支持DataFrame和PyTorch张量输入

        参数:
        x_test: 测试数据，可以是pd.DataFrame或torch.Tensor [时间, 股票, 因子]
        edge_index: 边索引列表（可选）

        返回:
        对于DataFrame输入返回list，对于张量输入返回torch.Tensor [时间, 股票]
        """
        # 处理PyTorch张量输入
        if isinstance(x_test, torch.Tensor):
            return self._predict_tensor(x_test, edge_index)
        # 保持原有DataFrame接口
        else:
            return super().predict(x_test)

    def _predict_tensor(self, x_test_tensor, edge_index=None):
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

                for step in range(t, end_t):
                    # 获取当前时间步的输入 [N, K]
                    x_batch = x_test_tensor[step].to(self.device)

                    # 获取当前时间步的边索引
                    if edge_index is not None and step < len(edge_index):
                        current_edge = edge_index[step].to(self.device)
                        # 如果是稀疏张量，转换为密集张量
                        if current_edge.is_sparse:
                            adj_matrix = current_edge.to_dense()
                        elif current_edge.is_sparse_csr:
                            adj_matrix = current_edge.to_dense()
                    else:
                        adj_matrix = torch.zeros((x_batch.size(0), x_batch.size(0)), dtype=torch.float32, device=self.device)

                    # 创建当前时间步的预测结果数组，初始化为NaN
                    batch_pred = torch.full((x_batch.shape[0],), np.nan, dtype=torch.float32, device=self.device)

                    # 找出有效数据的索引
                    valid_mask = ~torch.isnan(x_batch).any(dim=1)
                    if valid_mask.any():
                        # 对有效数据进行预测
                        x_valid_batch = x_batch[valid_mask]

                        # 使用current_edge并过滤掉NaN值对应的节点
                        if adj_matrix.size(1) > 0:  # 如果有边存在
                            adj_matrix = adj_matrix[valid_mask][:, valid_mask]

                        edge_idx = adj_matrix.to_sparse_csr()

                        valid_preds = self.model(x_valid_batch, edge_idx)
                        # 将预测结果放回到正确位置
                        batch_pred[valid_mask] = valid_preds.squeeze(1)

                    # 将当前时间步的预测结果放入输出张量
                    y_pred[step] = batch_pred

        return y_pred

    def valid(self, x_valid, y_valid, z_valid=None, edge_valid=None):
        """
        验证模型性能
        """
        if self.model is None:
            raise ValueError("模型未初始化，请先调用fit方法或init_model方法")

        device = next(self.model.parameters()).device
        self.model.eval()  # 设置为评估模式

        x_valid = x_valid.to(device) if isinstance(x_valid, torch.Tensor) else x_valid
        y_valid = y_valid.to(device) if isinstance(y_valid, torch.Tensor) else y_valid
        edge_valid = [edge.to(device) for edge in edge_valid] if edge_valid else None

        total_loss_val = 0
        val_ic = 0

        with torch.no_grad():  # 禁用梯度计算
            for i in range(len(x_valid)):
                # 处理边索引
                current_edge = edge_valid[i] if edge_valid else None

                # 计算验证损失
                loss_val = self.test(
                    x=x_valid[:, i:i + 1, :],
                    y=y_valid[:, i:i + 1, :],
                    z=z_valid[:, i:i + 1, :] if z_valid else None,
                    edge_index=current_edge
                )
                total_loss_val += loss_val

                # 计算IC
                val_preds = self.predict_(x_valid[:, i:i + 1, :], edge_index=current_edge)
                val_ic_corr = float(calc_tensor_corr(val_preds, y_valid[:, i:i + 1, :]))
                val_ic += val_ic_corr

        avg_loss_val = total_loss_val / len(x_valid)
        avg_val_ic = val_ic / len(x_valid)

        return avg_loss_val, avg_val_ic

    def predict_pandas(self, x: DataFrame, edge_index=None) -> Series:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        x_tensor = from_pandas_to_rnn(x, fillna=True).to(device)
        edge_index = [edge.to(device) for edge in edge_index] if edge_index else None

        result = []

        for i in range(x_tensor.shape[1]):
            current_edge = edge_index[i] if edge_index is not None else None

            result.append(Series(self.predict_(
                x_tensor[:, i:i + 1, :],
                edge_index=current_edge,
            ).view(-1, ).cpu()))

        days = x.index.get_level_values(0).unique()[-len(result):]
        instrument = x.index.get_level_values(1).unique().to_list()
        name_0, name_1 = x.index.names[0], x.index.names[1]

        predict = []
        for i in range(len(result)):
            df = DataFrame(result[i])
            df[name_0] = days[i]
            df[name_1] = instrument
            predict.append(df.set_index([name_0, name_1]).iloc[:, 0])

        predict = concat(predict, axis=0)
        return predict[predict.index.isin(x.index)]
