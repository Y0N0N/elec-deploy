import torch
from torch import Tensor

def calc_tensor_corr(x: Tensor, y: Tensor):
    """计算两个张量间的相关系数"""
    if x.shape != y.shape:
        raise ValueError("The shapes of x and y must be the same.")
    mask = ~torch.isnan(y)
    mean_x = torch.mean(x[mask])
    mean_y = torch.mean(y[mask])
    std_x = torch.std(x[mask])
    std_y = torch.std(y[mask])
    return torch.mean((x[mask] - mean_x) * (y[mask] - mean_y)) / (std_x * std_y)


class style_mse(torch.nn.Module):
    """控制预测值与某个变量相关系数的MSE损失"""
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, inputs: tuple):
        x, y, z = inputs[0], inputs[1], inputs[2]
        mse_loss = self.mse(x, y)
        ic_loss = calc_tensor_corr(x, z)
        return mse_loss * (1 + 0.05 * torch.abs(ic_loss))


class ic_mse(torch.nn.Module):
    """在MSE基础上增加预测值与目标值之间的IC"""
    def __init__(self, ic_weight: float = 0.1):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.ic_weight = ic_weight

    def forward(self, x, y):
        mse_loss = self.mse(x, y)
        ic = calc_tensor_corr(x, y)
        ic_penalty = 1 - ic  # 鼓励IC为正值
        return mse_loss + self.ic_weight * ic_penalty


class huber_loss(torch.nn.Module):
    def __init__(self, delta: float = 1.0):
        """
        Huber损失函数，结合了MSE和MAE的优点，对异常值不那么敏感

        参数:
        delta: 阈值参数，控制MSE和MAE之间的切换点

        使用方法:
        self.loss = huber_loss(delta=1.0)
        self.loss(x, y)  # 普通用法
        self.loss((x, y, z))  # 带样式控制的用法
        """
        super().__init__()
        self.delta = delta

    def forward(self, x, y=None, z=None):
        # 检查是否是元组形式的输入（带样式控制）
        if isinstance(x, tuple) and len(x) == 3:
            # 带样式控制的用法
            x_val, y_val, z_val = x[0], x[1], x[2]
            huber = self._huber_loss(x_val, y_val)
            ic_loss = calc_tensor_corr(x_val, z_val)
            return huber * (1 + 0.05 * torch.abs(ic_loss))
        else:
            # 普通用法 - 直接接收x和y作为参数
            if y is None:
                raise ValueError("在普通用法中，必须提供y参数")
            return self._huber_loss(x, y)

    def _huber_loss(self, x, y):
        # 计算差值的绝对值
        abs_diff = torch.abs(x - y)
        # 应用Huber损失公式
        quadratic_term = 0.5 * (abs_diff ** 2)
        linear_term = self.delta * (abs_diff - 0.5 * self.delta)
        # 根据阈值选择使用二次项还是线性项
        loss = torch.where(abs_diff <= self.delta, quadratic_term, linear_term)
        # 返回平均损失
        return torch.mean(loss)


class qlike_loss(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-2):
        """
        QLIKE损失函数，适用于波动率预测等非负值预测任务

        参数:
        epsilon: 小正数，防止除以零或对数为负无穷

        使用方法:
        self.loss = qlike_loss(epsilon=1e-6)
        self.loss(x, y)  # 普通用法
        self.loss((x, y, z))  # 带样式控制的用法
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x, y=None, z=None):
        # 检查是否是元组形式的输入（带样式控制）
        if isinstance(x, tuple) and len(x) == 3:
            # 带样式控制的用法
            x_val, y_val, z_val = x[0], x[1], x[2]
            qlike = self._qlike_loss(x_val, y_val)
            ic_loss = calc_tensor_corr(x_val, z_val)
            return qlike * (1 + 0.05 * torch.abs(ic_loss))
        else:
            # 普通用法 - 直接接收x和y作为参数
            if y is None:
                raise ValueError("在普通用法中，必须提供y参数")
            return self._qlike_loss(x, y)

    def _qlike_loss(self, x, y):
        # 确保预测值为正
        x_positive = torch.clamp(x, min=self.epsilon)

        # 确保真实值为正
        y_positive = torch.clamp(y, min=self.epsilon)

        # 计算QLIKE损失: y/x + log(x)
        # 这个公式来自于波动率预测的QLIKE损失函数
        ratio = y_positive / x_positive
        log_pred = torch.log(x_positive)

        # 计算损失
        loss = ratio + log_pred

        # 返回平均损失
        return torch.mean(loss)
