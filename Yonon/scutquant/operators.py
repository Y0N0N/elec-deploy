import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import numba

"""

与qlib的将数据简单加工后扔给ai模型找规律的思路不同, worldquant的思路是用精细的operators挖到有逻辑, 且回测表现良好的因子,
用因子值构造中性组合, 直接用于投资
即qlib的量化投资流程是: 数据 -> 因子(这里的因子更接近feature的概念) -> 模型 -> 策略 -> 收益
而worldquant的流程是: 数据 -> 因子 -> 收益, 策略就是根据因子值构建投资组合(参考scutquant.alpha.market_neutralize的注释)
其实可以将qlib中的模型预测值当作worldquant中的因子, 那么qlib其实用是一种单因子策略进行投资, 只不过因子是由ai模型挖掘的, 而且策略更加多样
而worldquant的每一个因子都代表某个策略, 一个portfolio manager会选择多个因子, 并分配不同资金给每一个因子, 最后所有因子收益加总得到portfolio
一言以蔽之, worldquant 模式是量化1.0时代的经典模式, 而qlib的模式则适用于量化2.0甚至3.0时代.
但这并不意味着两者是不兼容的. 事实上, 一个被精细加工过的feature能让模型的预测效果更好, 反过来模型的预测值也能作为一个很好的因子素材

scutquant的alpha模块用的是qlib的思路, 而为了让用户按照worldquant的方式构造自己的因子, 本模块应运而生
在本模块中, ts是对每个instrument在时序上计算, 而cs是在截面上计算, 所有返回结果都是pd.Series
该模块提供了更加丰富的算子, 且速度也在不断优化. 计划以后alpha只提供因子表达式, 而具体计算由operators的算子完成
未来这部分可能会合并到alpha模块中, 让整个架构看起来不那么臃肿, 但也要考虑到合并后是否方便维护的问题

example:

from operators import *

factor = cs_zscore(ts_rank(ts_corr(df["close"], df["volume"], 15), 15))

"""


def ts_delay(data, n_period: int, matrix = False) -> pd.Series:
    """
    Returns data n_period days ago
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.shift(n_period)
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.shift(n_period))
    else:
        res: pd.Series = data.transform(lambda x: x.shift(n_period))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_delta(data: pd.Series, n_period: int, matrix = False) -> pd.Series:
    """
    Returns data - ts_delay(data, n_period)
    """
    return data - ts_delay(data, n_period, matrix)


def ts_returns(data: pd.Series, n_period: int, matrix = False) -> pd.Series:
    """
    Returns the relative change of data .
    :return:
    """
    return ts_delta(data, n_period, matrix) / ts_delay(data, n_period, matrix)


def ts_sum(data, n_period: int, matrix = False) -> pd.Series:
    """
    Sum values of data for the past n_period days.
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.rolling(n_period).sum()
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).sum())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).sum())
        res.index.names = ["datetime", "instrument"]
        return res


def sumif(x,n,condition):
    return condition.where(condition, ts_sum(x,n)).where(~condition, 0)


@numba.jit(nopython=True)
def _ts_product(arr, window):
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window-1, n):
        window_data = arr[i-window+1:i+1]
        mask = ~np.isnan(window_data)
        if np.any(mask):
            result[i] = np.prod(window_data[mask])
    return result

def ts_product(data, n_period: int, matrix = False) -> pd.Series:
    """
    Returns product of data for the past n_period days
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        result = data.copy()
        for col in data.columns:
            result[col] = _ts_product(data[col].values, n_period)
        return result
        # return data.rolling(n_period).apply(lambda x: x.prod(), raw=True)
    if isinstance(data, pd.Series):
        prod = data.groupby(level=1).transform(lambda x: x.cumprod())
    else:
        prod = data.transform(lambda x: x.cumprod())
        prod.index.names = ["datetime", "instrument"]
    return prod / ts_delay(data, n_period)


def ts_max(data, n_period: int, min_periods: int = 1, matrix = False) -> pd.Series:
    """
    Returns max value of data for the past n_period days
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.rolling(n_period, min_periods=min_periods).max()
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period, min_periods=min_periods).max())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period, min_periods=min_periods).max())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_min(data, n_period: int, min_periods: int = 1, matrix = False) -> pd.Series:
    """
    Returns min value of data for the past n_period days
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.rolling(n_period, min_periods=min_periods).min()
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period, min_periods=min_periods).min())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period, min_periods=min_periods).min())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_mean(data, n_period: int, min_periods: int = 1, matrix = False) -> pd.Series:
    """
    Returns average value of data for the past n_period days.
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.rolling(n_period, min_periods=min_periods).mean()
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period, min_periods=min_periods).mean())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period, min_periods=min_periods).mean())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_ewma(data, a: float, matrix = False) -> pd.Series:
    """
    Returns exponentially weighted moving average of data.
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.ewm(alpha=a).mean()
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.ewm(alpha=a).mean())
    else:
        res: pd.Series = data.transform(lambda x: x.ewm(alpha=a).mean())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_std(data, n_period: int, min_periods: int = 2, matrix = False) -> pd.Series:
    """
    Returns standard deviation of data for the past n_period days
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.rolling(n_period, min_periods=min_periods).std()
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period, min_periods=min_periods).std())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period, min_periods=min_periods).std())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_dstd(data, n_period: int, matrix = False) -> pd.Series:
    """
    Returns downside standard deviation of data for the past n_period days
    """

    def downside_std(df: pd.Series):
        downside_data = df.where(df < 0, np.nan)
        return downside_data.rolling(n_period, min_periods=2).std()

    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.apply(lambda x: downside_std(x))
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: downside_std(x))
    else:
        res: pd.Series = data.transform(lambda x: downside_std(x))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_kurt(data, n_period: int, min_periods: int = 2, matrix = False) -> pd.Series:
    """
    Returns kurtosis of data for the last n_period days.
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.rolling(n_period, min_periods=min_periods).kurt()
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period, min_periods=min_periods).kurt())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period, min_periods=min_periods).kurt())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_skew(data, n_period: int, min_periods: int = 2, matrix = False) -> pd.Series:
    """
    Return skewness of data for the past n_period days.
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.rolling(n_period, min_periods=min_periods).skew()
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period, min_periods=min_periods).skew())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period, min_periods=min_periods).skew())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_median(data, n_period: int, min_periods: int = 1, matrix = False) -> pd.Series:
    """
    Returns median value of data for the past n_period days
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.rolling(n_period, min_periods=min_periods).median()
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period, min_periods=min_periods).median())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period, min_periods=min_periods).median())
        res.index.names = ["datetime", "instrument"]
        return res


@numba.jit(nopython=True)
def _rank_last_numba(window_data):
    """使用numba加速的排名计算"""
    n = len(window_data)
    if n == 0:
        return np.nan

    # 计算排名
    ranks = np.zeros(n)
    for i in range(n):
        count = 0
        for j in range(n):
            if window_data[j] <= window_data[i]:
                count += 1
        ranks[i] = count / n
    return ranks[-1]

def ts_rank(data, n_period: int, min_periods: int = 1, matrix=False):
    """
    Rank the values of data for each instrument over the past n_period days, then return the rank of the current value
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.rolling(n_period, min_periods=min_periods).apply(_rank_last_numba, raw=True)
    # ... 其他分支保持不变
    elif isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period, min_periods=min_periods).apply(lambda y: y.rank(pct=True).iloc[-1]))
    else:
        res = data.transform(lambda x: x.rolling(n_period, min_periods=min_periods).apply(lambda y: y.rank(pct=True).iloc[-1]))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_variance(data, n_period: int, min_periods: int = 2, matrix = False) -> pd.Series:
    """
    Returns variance of data for the past n_period days
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.rolling(n_period, min_periods=min_periods).var()
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period, min_periods=min_periods).var())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period, min_periods=min_periods).var())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_quantile_up(data, n_period: int, min_periods: int = 1, matrix = False) -> pd.Series:
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.rolling(n_period, min_periods=min_periods).quantile(0.75)
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period, min_periods=min_periods).quantile(0.75))
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period, min_periods=min_periods).quantile(0.75))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_quantile_down(data, n_period: int, min_periods: int = 1, matrix = False) -> pd.Series:
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.rolling(n_period, min_periods=min_periods).quantile(0.25)
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period, min_periods=min_periods).quantile(0.25))
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period, min_periods=min_periods).quantile(0.25))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_zscore(data: pd.Series, n_period: int, min_periods: int = 2, matrix = False) -> pd.Series:
    """
    Z-score is a numerical measurement that describes a value's relationship to the mean of a group of values.
    Z-score is measured in terms of standard deviations from the mean:
    (data - ts_mean(data,n_period)) / ts_std(data,n_period)
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return (data - ts_mean(data, n_period, min_periods, matrix)) / ts_std(data, n_period, min_periods, matrix)
    return (data - ts_mean(data, n_period, min_periods)) / ts_std(data, n_period, min_periods)


def ts_robust_zscore(data: pd.Series, n_period: int, min_periods: int = 2, matrix = False) -> pd.Series:
    med = ts_median(data, n_period, min_periods, matrix)
    return (data - med) / (ts_median(abs(data - med), n_period, min_periods, matrix) * 1.4826)


def ts_scale(data: pd.Series, n_period: int, min_periods: int = 2, matrix = False) -> pd.Series:
    return (data - ts_min(data, n_period, min_periods, matrix)) / (ts_max(data, n_period, min_periods, matrix) - ts_min(data, n_period, min_periods, matrix))


def ts_sharpe(data, n_period: int, min_periods: int = 2, matrix = False) -> pd.Series:
    """
    Return sharpe ratio ts_mean(data, n_period) / ts_std(data, n_period)
    """
    return ts_mean(data, n_period, min_periods, matrix) / ts_std(data, n_period, min_periods, matrix)


def ts_av_diff(data: pd.Series, n_period: int, min_periods: int = 1, matrix = False) -> pd.Series:
    """
    Returns data - ts_mean(data, n_period)
    """
    return data - ts_mean(data, n_period, min_periods, matrix)


def ts_max_diff(data: pd.Series, n_period: int, min_periods: int = 1, matrix = False) -> pd.Series:
    """
    Returns data - ts_max(data, n_period)
    """
    return data - ts_max(data, n_period, min_periods, matrix)


def ts_min_diff(data: pd.Series, n_period: int, min_periods: int = 1, matrix = False) -> pd.Series:
    """
    Returns data - ts_min(data, n_period)
    """
    return data - ts_min(data, n_period, min_periods, matrix)


def ts_corr(x1: pd.Series, x2: pd.Series, n_period: int, rank: bool = False, matrix = False) -> pd.Series:
    """
    Returns correlation of data[feature] and data[label] for the past n_period days
    """
    x1.name = "feature"
    x2.name = "label"
    if matrix | (isinstance(x1, pd.DataFrame) & (x1.index.nlevels == 1)):
        if rank:
            from scipy.stats import spearmanr
            # 使用斯皮尔曼相关性
            return x1.rolling(n_period).apply(
                lambda x: spearmanr(x, x2.loc[x.index])[0],
                raw=False
            )
        else:
            # 使用皮尔逊相关性
            return x1.rolling(n_period).corr(x2)

    elif rank:
        concat_df = pd.concat([x1, x2], axis=1)
        res = concat_df.groupby(level=1).apply(
            lambda x: x["feature"].rolling(n_period).corr(x["label"], method="spearman")).reset_index(0, drop=True)
    else:
        concat_df = pd.concat([x1, x2], axis=1)
        res = concat_df.groupby(level=1).apply(
            lambda x: x["feature"].rolling(n_period).corr(x["label"])).reset_index(0, drop=True)
    return res.sort_index()


def ts_cov(x1: pd.Series, x2: pd.Series, n_period: int, matrix = False) -> pd.Series:
    """
    Returns covariance of data[feature] and data[label] for the past n_period days
    """
    x1.name = "feature"
    x2.name = "label"

    if matrix | (isinstance(x1, pd.DataFrame) & (x1.index.nlevels == 1)):
        return x1.rolling(n_period).cov(x2)

    concat_df = pd.concat([x1, x2], axis=1)
    cov = concat_df.groupby(level=1).apply(lambda x:
                                           x["feature"].rolling(n_period).cov(x["label"])).reset_index(0, drop=True)
    return cov.sort_index()


def ts_beta(x1: pd.Series, x2: pd.Series, n_period: int, matrix = False) -> pd.Series:
    """
    Returns beta of data[feature] and data[label] for the past n_period days
    """
    cov = ts_cov(x1, x2, n_period, matrix = matrix)
    var = ts_variance(x1, n_period, matrix = matrix)
    return cov / var


@numba.jit(nopython=True)
def sequence(n):
    return np.arange(1, n + 1, dtype=np.float64)


@numba.jit(nopython=True)
def _rolling_regbeta(x, window):
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        x_window = x[i - window + 1:i + 1]
        mask = ~np.isnan(x_window)
        x_clean = x_window[mask]
        valid_count = len(x_clean)
        x_clean = x_window[mask]
        if valid_count >= 2:
            y = sequence(valid_count)
            cov_xy = np.cov(x_clean, y)[0, 1]
            var_x = np.var(x_clean)
            if var_x > 1e-12:
                result[i] = cov_xy / var_x
    return result

def regbeta(x, d):
    if not (isinstance(x, (pd.Series, pd.DataFrame))):
        raise ValueError("x must be Series or DataFrame")
    result = x.copy()
    for col in x.columns:
        result[col] = _rolling_regbeta(x[col].values, d)
    return result


def ts_regression(x1: pd.Series, x2: pd.Series, n_periods: int, rettype: int = 0, matrix = False) -> pd.Series:
    """
    Returns results of linear model y = beta * x + alpha + resid

    :param x1:
    :param x2:
    :param n_periods:
    :param rettype: 0 for resid, 1 for beta, 2 for alpha, 3 for y_hat, 4 for R^2
    :return:
    """
    if rettype == 0:
        beta = ts_beta(x1, x2, n_periods, matrix = matrix)
        alpha: pd.Series = ts_mean(x2, n_periods, matrix = matrix) - beta * ts_mean(x1, n_periods, matrix = matrix)
        predict: pd.Series = beta * x1 + alpha
        resid: pd.Series = x2 - predict
        return resid
    elif rettype == 1:
        beta = ts_beta(x1, x2, n_periods, matrix = matrix)
        return beta
    elif rettype == 2:
        beta = ts_beta(x1, x2, n_periods, matrix = matrix)
        alpha: pd.Series = ts_mean(x2, n_periods, matrix = matrix) - beta * ts_mean(x1, n_periods, matrix = matrix)
        return alpha
    elif rettype == 3:
        beta = ts_beta(x1, x2, n_periods, matrix = matrix)
        alpha: pd.Series = ts_mean(x2, n_periods, matrix = matrix) - beta * ts_mean(x1, n_periods, matrix = matrix)
        predict: pd.Series = beta * x1 + alpha
        return predict
    else:
        beta = ts_beta(x1, x2, n_periods, matrix=matrix)
        alpha: pd.Series = ts_mean(x2, n_periods, matrix=matrix) - beta * ts_mean(x1, n_periods, matrix=matrix)
        predict: pd.Series = beta * x1 + alpha
        return ts_corr(predict, x2, n_periods, matrix=matrix) ** 2


def ts_pos_count(data: pd.Series, n_period: int, matrix = False) -> pd.Series:
    """
    Returns the number of days when data is bigger than 0 for the past n_period days

    psy = ts_pos_count(ts_delta(close, 1), d) / d * 100  # 一个计算d日psy指标的例子, 比alpha模块的对应函数简洁了不少
    """
    data_copy = data.copy()
    data_copy[data_copy > 0] = 1
    data_copy[data_copy < 0] = 0
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data_copy.transform(lambda x: x.rolling(n_period).sum())
    else:
        return data_copy.groupby(level=1).transform(lambda x: x.rolling(n_period).sum())


def ts_neg_count(data: pd.Series, n_period: int, matrix = False) -> pd.Series:
    """
    Returns the number of days when data is smaller than 0 for the past n_period days
    """
    data_copy = data.copy()
    data_copy[data_copy > 0] = 0
    data_copy[data_copy < 0] = 1
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data_copy.transform(lambda x: x.rolling(n_period).sum())
    else:
        return data_copy.groupby(level=1).transform(lambda x: x.rolling(n_period).sum())


# def decay_n(x: pd.Series, n: int) -> pd.Series:
#     arr = np.arange(1, n + 1)
#     weights = arr / sum(arr)
#     return x.rolling(n).apply(lambda y: np.dot(y, weights), raw=True)


@numba.jit(nopython=True)
def decay_n(arr, window, reverse=False):
    n = len(arr)
    result = np.full(n, np.nan)
    if reverse:
        weights = np.arange(window, 0, -1, dtype=np.float64)
    else:
        weights = np.arange(1, window+1, dtype=np.float64)
    weights = weights / np.sum(weights)
    for i in range(window-1, n):
        window_data = arr[i-window+1:i+1]
        mask = ~np.isnan(window_data)
        if np.any(mask):
            valid_data = window_data[mask]
            valid_weights = weights[-len(valid_data):] if len(valid_data) < window else weights
            valid_weights = valid_weights / np.sum(valid_weights)
            result[i] = np.sum(valid_data * valid_weights)
    return result


def ts_decay_linear(data, n_period: int, matrix = False, reverse=False) -> pd.Series:
    """
    Returns the linear decay on data for the past n_period days.
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        result = data.copy()
        for col in data.columns:
            result[col] = decay_n(data[col].values, n_period, reverse)
        return result
        # return data.transform(lambda x: decay_n(x, n_period))
    elif isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: decay_n(x, n_period))
    else:
        res: pd.Series = data.transform(lambda x: decay_n(x, n_period))
        res.index.names = ["datetime", "instrument"]
        return res


@numba.jit(nopython=True)
def _ts_argmax(arr, window):
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window-1, n):
        window_data = arr[i-window+1:i+1]
        mask = ~np.isnan(window_data)
        if np.any(mask):
            valid_data = window_data[mask]
            result[i] = np.argmax(valid_data) + 1  # 从1开始计数
    return result


def ts_argmax(data, n_period: int, matrix = False) -> pd.Series:
    def argmax(feature: pd.Series) -> pd.Series:
        return feature.rolling(n_period).apply(lambda x: np.argmax(x))

    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        result = data.copy()
        for col in data.columns:
            result[col] = _ts_argmax(data[col].values, n_period)
        return result
        # return data.transform(lambda x: argmax(x))
    elif isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: argmax(x))
    else:
        res = data.transform(lambda x: argmax(x))
        res.index.names = ["datetime", "instrument"]
        return res


@numba.jit(nopython=True)
def _ts_argmin(arr, window):
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window-1, n):
        window_data = arr[i-window+1:i+1]
        mask = ~np.isnan(window_data)
        if np.any(mask):
            valid_data = window_data[mask]
            result[i] = np.argmin(valid_data) + 1  # 从1开始计数
    return result


def ts_argmin(data, n_period: int, matrix = False) -> pd.Series:
    def argmin(feature: pd.Series) -> pd.Series:
        return feature.rolling(n_period).apply(lambda x: np.argmin(x))

    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        result = data.copy()
        for col in data.columns:
            result[col] = _ts_argmin(data[col].values, n_period)
        return result
        # return data.transform(lambda x: argmin(x))
    elif isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: argmin(x))
    else:
        res = data.transform(lambda x: argmin(x))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_backfill(data, matrix = False) -> pd.Series:
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.transform(lambda x: x.fillna(method="bfill"))
    elif isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.fillna(method="bfill"))
    else:
        res = data.transform(lambda x: x.fillna(method="bfill"))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_ffill(data, matrix = False) -> pd.Series:
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.transform(lambda x: x.fillna(method="ffill"))
    elif isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.fillna(method="ffill"))
    else:
        res = data.transform(lambda x: x.fillna(method="ffill"))
        res.index.names = ["datetime", "instrument"]
        return res


def cs_rank(data, matrix = False):
    """
    Ranks the input among all the instruments and returns an equally distributed number between 0.0 and 1.0.
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.apply(lambda row: row.rank(pct=True), axis=1)
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.groupby(level=0).transform(lambda x: x.rank(pct=True))
    else:
        res: pd.Series = data.transform(lambda x: x.rank(pct=True))
        res.index.names = ["datetime", "instrument"]
        return res


def cs_zscore(data, matrix = False):
    """
    Z-score is a numerical measurement that describes a value's relationship to the mean of a group of values.
    Z-score is measured in terms of standard deviations from the mean
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.groupby(level=0).transform(lambda x: (x - x.mean()) / x.std())
    else:
        res: pd.Series = data.transform(lambda x: (x - x.mean()) / x.std())
        res.index.names = ["datetime", "instrument"]
        return res


def cs_robust_zscore(data, matrix = False):
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.apply(lambda row: (row - row.median()) / (abs(row - row.median()).median() / 1.4826), axis=1)
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.groupby(level=0).transform(lambda x: (x - x.median()) / (abs(x - x.median()).median() / 1.4826))
    else:
        res: pd.Series = data.transform(lambda x: (x - x.median()) / (abs(x - x.median()).median() / 1.4826))
        res.index.names = ["datetime", "instrument"]
        return res


def cs_scale(data, a=1, matrix = False):
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        abs_sum = np.abs(data).sum(axis=1)
        result = data.mul(a).div(abs_sum, axis=0).where(abs_sum != 0, 0)
        return result
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.groupby(level=0).transform(lambda x: (x - x.min()) * a / (x.max() - x.min()))
    else:
        res: pd.Series = data.transform(lambda x: (x - x.min()) * a / (x.max() - x.min()))
        res.index.names = ["datetime", "instrument"]
        return res


def cs_mean(data, matrix = False):
    """
    This function is not for regular alphas which have two index levels. It calculates the mean value of all instruments
    on a particular time tick. You may use this for calculating the relationship between single instrument and the index
    """
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.apply(lambda row: row.mean(), axis=1)
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.groupby(level=0).transform(lambda x: x.mean())
    else:
        res: pd.Series = data.transform(lambda x: x.mean())
        res.index.names = ["datetime", "instrument"]
        return res


def cs_std(data, matrix = False):
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.apply(lambda row: row.std(), axis=1)
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.groupby(level=0).transform(lambda x: x.std())
    else:
        res: pd.Series = data.transform(lambda x: x.std())
        res.index.names = ["datetime", "instrument"]
        return res


def cs_variance(data, matrix = False):
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return data.apply(lambda row: row.var(), axis=1)
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.groupby(level=0).transform(lambda x: x.var())
    else:
        res: pd.Series = data.transform(lambda x: x.var())
        res.index.names = ["datetime", "instrument"]
        return res


def cs_cov(x1: pd.Series, x2: pd.Series, matrix: bool = False) -> pd.Series:
    x1.name = "feature"
    x2.name = "label"

    if matrix | (isinstance(x1, pd.DataFrame) & (x1.index.nlevels == 1)):
        return x1.apply(
            lambda row: row.cov(x2.loc[row.name])
            if not row.empty else np.nan,
            axis=1
        )
    concat_df = pd.concat([x1, x2], axis=1)
    ones = pd.Series(1, index=x1.index)
    res = concat_df.groupby(level=0).apply(lambda x: x["feature"].cov(x["label"]))
    return res / ones


def cs_corr(x1: pd.Series, x2: pd.Series, rank: bool = False, matrix: bool = False) -> pd.Series:
    x1.name = "feature"
    x2.name = "label"

    if matrix | (isinstance(x1, pd.DataFrame) & (x1.index.nlevels == 1)):
        return x1.apply(
            lambda row: row.corr(x2.loc[row.name], method="spearman" if rank else "pearson")
            if not row.empty else np.nan,
            axis=1
        )
    concat_df = pd.concat([x1, x2], axis=1)
    ones = pd.Series(1, index=x1.index)
    if rank:
        res = concat_df.groupby(level=0).apply(lambda x: x["feature"].corr(x["label"], method="spearman"))
    else:
        res = concat_df.groupby(level=0).apply(lambda x: x["feature"].corr(x["label"]))
    return res / ones


def cs_beta(x1: pd.Series, x2: pd.Series, matrix: bool = False) -> pd.Series:
    cov = cs_cov(x1, x2, matrix)
    var = cs_variance(x1, matrix)
    return cov / var


def cs_alpha(x1: pd.Series, x2: pd.Series, matrix: bool = False) -> pd.Series:
    beta = cs_beta(x1, x2, matrix)
    return cs_mean(x2, matrix) - cs_mean(x1, matrix) * beta


def cs_resid(x1: pd.Series, x2: pd.Series, matrix: bool = False) -> pd.Series:
    beta = cs_beta(x1, x2, matrix)
    alpha = cs_mean(x2, matrix) - cs_mean(x1, matrix) * beta
    if matrix | (isinstance(x1, pd.DataFrame) & (x1.index.nlevels == 1)):
        return x2 - x1.apply(lambda x: x * beta + alpha)
    else:
        return x2 - x1 * beta - alpha


def cs_shrink(data, matrix: bool = False):
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        data = data.apply(lambda row: row.where(row <= 3, 3 + (row - 3).div(row.max() - 3) * 0.5), axis=1)
        return data.apply(lambda row: row.where(row >= -3, -3 + (row + 3).div(row.min() + 3) * 0.5), axis=1)
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.groupby(level=0).transform(lambda x: x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5))
        data = data.groupby(level=0).transform(lambda x: x.where(x >= -3, -3 + (x + 3).div(x.min() + 3) * 0.5))
        return data
    else:
        res = data.transform(lambda x: x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5))
        res = res.transform(lambda x: x.where(x >= -3, -3 + (x + 3).div(x.min() + 3) * 0.5))
        res.index.names = ["datetime", "instrument"]
        return res


def demean(data, matrix: bool = False):
    return data - cs_mean(data, matrix)


def mean(data1: pd.Series, data2: pd.Series, matrix: bool = False) -> pd.Series:
    return (data1 + data2) / 2


def max(data1, data2, matrix: bool = False):
    if matrix | (isinstance(data1, pd.DataFrame) & (data1.index.nlevels == 1)):
        return np.maximum(data1, data2)
    if isinstance(data1, (pd.Series, pd.DataFrame)):
        return pd.Series(np.maximum(data1.values, data2.values), index=data1.index)
    else:
        return np.maximum(data1, data2)


def min(data1, data2, matrix: bool = False):
    if matrix | (isinstance(data1, pd.DataFrame) & (data1.index.nlevels == 1)):
        return np.minimum(data1, data2)
    if isinstance(data1, (pd.Series, pd.DataFrame)):
        return pd.Series(np.minimum(data1.values, data2.values), index=data1.index)
    else:
        return np.minimum(data1, data2)


def sign(data, matrix: bool = False):
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return np.sign(data)
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return pd.Series(np.sign(data.values), index=data.index)
    else:
        return np.sign(data)


def sign_power(data, p: float, matrix: bool = False):
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return (np.abs(data)/data * (np.abs(data) ** p)).astype('float')
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return pd.Series(np.sign(data.values) * (abs(data.values) ** p), index=data.index)
    else:
        return np.sign(data) * (abs(data) ** p)


def log(data: pd.Series, matrix: bool = False) -> pd.Series:
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        return np.log(data.where(data > 0))
    return pd.Series(np.log(data.values), index=data.index)


def tanh(data: pd.Series, matrix: bool = False) -> pd.Series:
    if matrix:
        return data.apply(
            lambda row: pd.Series(np.tanh(row.values), index=data.columns),
            axis=1
        )
    return pd.Series(np.tanh(data.values), index=data.index)


def sigmoid(data: pd.Series, matrix: bool = False) -> pd.Series:
    if matrix:
        return data.apply(
            lambda row: pd.Series(1 / (1 + np.exp(-row.values)), index=data.columns),
            axis=1
        )
    return pd.Series(1 / (1 + np.exp(-data.values)), index=data.index)


def bigger(data1: pd.Series, data2: pd.Series, matrix: bool = False) -> pd.Series:
    """
    Returns the bigger value of data1 and data2
    """
    return data1.where(data1 < data2, data2)  # 若不满足data1 < data2, 则返回data1


def smaller(data1: pd.Series, data2: pd.Series, matrix: bool = False) -> pd.Series:
    """
    Returns the smaller value of data1 and data2
    """
    return data1.where(data1 > data2, data2)  # 若不满足data1 > data2, 则返回data2


def mad_winsor(data, matrix: bool = False):
    if matrix | (isinstance(data, pd.DataFrame) & (data.index.nlevels == 1)):
        med = data.apply(lambda row: pd.Series(row.median(), index=data.columns), axis=1)
        mad = abs((data - med)).apply(lambda row: pd.Series(row.median(), index=data.columns), axis=1)
        up = med + 3 * mad * 1.4826
        down = med - 3 * mad * 1.4826
        return data.apply(lambda row: pd.Series(np.clip(row.values, down.loc[row.name].values, up.loc[row.name].values), index=data.columns), axis=1)
    med = data.groupby(level=0).median()
    mad = abs((data - med)).groupby(level=0).median()
    up = med + 3 * mad * 1.4826
    down = med - 3 * mad * 1.4826
    return data.clip(upper=up, lower=down)


def inf_mask(data, matrix: bool = False):
    """
    Replace inf with nan
    """
    data = data.where(data != np.inf, np.nan)
    return data.where(data != -np.inf, np.nan)


def if_else(condition, x, y):
    if isinstance(condition, pd.DataFrame):
        if np.isscalar(x):
            x = pd.DataFrame(x, index=condition.index, columns=condition.columns)
        if np.isscalar(y):
            y = pd.DataFrame(y, index=condition.index, columns=condition.columns)
    return condition.where(condition, y).where(~condition, x).astype(float)


def neutralize(data, target: pd.Series, features = None,
               n_jobs=-1, matrix: bool = False):
    """
    在截面上对选定的features进行target中性化, 剩余因子不变

    example:

    # 使用补充数据data, 对factor_raw的RSI, MACD和KDJ_K因子进行市值中性化

    factor_neutralized = alpha.neutralize(factor_raw, target=data["ln_market_value"], features=["RSI", "MACD", "KDJ_K"])

    :param data: 需要中性化的因子集合
    :param target: 解释变量
    :param features: 需要中性化的因子名(列表), 因为不同因子可能需要不同的中性化手法, 故通过此参数控制进行中性化的因子
    :param n_jobs: 同时调用的cpu数
    :return: pd.DataFrame, 包括中性化后的因子和未中性化的其它因子
    """
    if matrix:
        return data.apply(
            lambda row: pd.Series(cs_resid(target, row), index=data.columns),
            axis=1
        )
    if isinstance(data, pd.Series):
        return cs_resid(target, data)
    else:
        features = data.columns if features is None else features
        other_cols = [c for c in data.columns if c not in features]
        factor_neu = Parallel(n_jobs=n_jobs)(delayed(cs_resid)(target, data[f]) for f in features)
        data_neu = pd.concat(factor_neu, axis=1)
        data_neu.columns = features
        return pd.concat([data_neu, data[other_cols]], axis=1)
