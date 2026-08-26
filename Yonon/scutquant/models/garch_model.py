import torch
import torch.nn as nn
import torch.optim as optim

class GARCH11(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.8))

    def forward(self, returns):
        # 确保参数非负（GARCH模型要求）
        omega = nn.functional.relu(self.omega)
        alpha = nn.functional.relu(self.alpha)
        beta = nn.functional.relu(self.beta)

        T = len(returns)
        # 初始化方差序列
        sigma2 = torch.zeros_like(returns)
        sigma2[0] = returns.var()

        # 不使用原地操作，而是创建一个新的计算图
        for t in range(1, T):
            # 计算当前方差
            current_sigma2 = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            # 使用索引赋值，但在每次循环中创建一个新的sigma2
            sigma2 = sigma2 * 1  # 创建一个新张量
            sigma2[t] = current_sigma2
        return sigma2

def negative_log_likelihood(model, returns):
    sigma2 = model(returns)
    loglik = 0.5 * (torch.log(sigma2) + (returns**2) / sigma2)
    return loglik.sum()

# Example: Replace this with your own return series
# returns = torch.tensor([...], dtype=torch.float32)

# model = GARCH11()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(1000):
#     optimizer.zero_grad()
#     loss = negative_log_likelihood(model, returns)
#     loss.backward()
#     optimizer.step()
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}: NLL = {loss.item():.4f}")
