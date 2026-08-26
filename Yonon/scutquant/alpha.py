from .operators import *
from .report import calc_drawdown
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def factor_neutralize(factors: pd.DataFrame | pd.Series, target: pd.Series,
                      feature: list[str] = None) -> pd.DataFrame | pd.Series:
    return neutralize(factors, features=feature, target=target)


def market_neutralize(x: pd.Series, long_only: bool = False) -> pd.Series:
    """
    市场组合中性化:
    (1) 对所有股票减去其截面上的因子均值
    (2) 在(1)之后, 对每支股票除以截面上的因子值绝对值之和

    这样处理后每支股票会获得一个权重, 代表着资金的方向和数量(例如0.5代表半仓做多, -0.25代表1/4仓做空),
    且截面上的权重之和为0, 绝对值之和为1.
    """
    _mean = x.groupby(level=0).mean()
    x -= _mean
    abs_sum = abs(x).groupby(level=0).sum()
    x /= abs_sum
    if long_only:
        x[x < 0] = 0
        x *= 2
    return x


def calc_factor_turnover(x: pd.Series) -> pd.Series:
    factor_neu = market_neutralize(x, long_only=False)
    instrument_to = abs(factor_neu - ts_delay(factor_neu, 1).fillna(0))  # 今日权重 - 昨日权重，在单利回测情况下代表资金变动的百分比
    return instrument_to.groupby(level=0).sum()


def get_factor_portfolio(feature: pd.Series, label: pd.Series, long_only: bool = False, compound: bool = False) -> pd.Series:
    """
    :param feature: 因子值
    :param label: 收益率
    :param long_only: 是否只做多
    :param compound: 是否复利

    :return: 时序数据portfolio, 代表累计收益率
    """
    x_neu = market_neutralize(feature, long_only=long_only)
    X = pd.DataFrame({"feature": x_neu, "label": label})
    X.dropna(inplace=True)
    X["factor_return"] = X["feature"] * X["label"]
    daily_return = X["factor_return"].groupby("datetime").sum()
    if compound:
        portfolio = daily_return.cumprod() - 1
        daily_return += 1
    else:
        portfolio = daily_return
    portfolio.index = pd.to_datetime(portfolio.index)
    return portfolio


def calc_fitness(sharpe: float, returns: float, turnover: float) -> float:
    """
    参考 https://platform.worldquantbrain.com/learn/documentation/discover-brain/intermediate-pack-part-1
    """
    return sharpe * ((abs(returns) / max(turnover, 0.125)) ** 0.5)


def get_factor_metrics(factor: pd.Series, label: pd.Series, metrics=None, handle_nan: bool = True,
                       long_only: bool = False, compound: bool = False, plot: bool = True,
                       ic_freq: int = 30, to_df: bool = False) -> dict:
    """
    :param factor:
    :param label:
    :param metrics: list[str] = ["ic", "return", "turnover", "sharpe", "ir", "fitness"] 有些指标的计算必须依赖其它指标
    :param handle_nan:
    :param long_only: 是否只做多
    :return:
    """
    if metrics is None:
        metrics = ["ic", "return", "turnover", "sharpe", "ir", "fitness"]
    if handle_nan:
        label.dropna(inplace=True)
        factor = factor[factor.index.isin(label.index)]
        label = label[label.index.isin(factor.index)]
    result: dict = {}
    if "ic" in metrics:  # information coefficient
        result["ic"] = cs_corr(factor, label, rank=False).groupby(level=0).mean()
        result["accumulated_ic"] = result["ic"].cumsum()
        result["ic_mean"] = result["ic"].mean()
        result["icir"] = result["ic"].mean() / result["ic"].std()
        result["t-stat"] = result["icir"] * (len(result["ic"]) ** 0.5)
    if "return" in metrics:
        result["return"]: pd.Series = get_factor_portfolio(factor, label, long_only=long_only)
        benchmark: pd.Series = label.groupby(level=0).mean()

        if compound:
            result["return"] = (result["return"] + 1).cumprod() - 1
            result["benchmark"] = (benchmark + 1).cumprod() - 1
        else:
            result["return"] = result["return"].cumsum()
            result["benchmark"] = benchmark.cumsum()
        # result["benchmark"] = pd.concat([pd.Series([1], index=[result["benchmark"].index[0] - pd.DateOffset(days=1)]),
        #                                  result["benchmark"]], axis=0)

        # 年化收益率计算（区分单复利）
        days = (result["return"].index[-1] - result["return"].index[0]).days
        if compound:
            annual_return = (1 + result["return"].iloc[-1]) ** (252/days) - 1
        else:
            annual_return = result["return"].iloc[-1] * 252 / days
        result["annual_return"] = annual_return
        result["excess_return"] = result["return"] - result["benchmark"]

        result["drawdown"] = calc_drawdown(result["return"])
        result["excess_return_drawdown"] = calc_drawdown(result["excess_return"])
        result["max_drawdown"] = result["drawdown"].min()
        result["excess_return_max_drawdown"] = result["excess_return_drawdown"].min()

    if "turnover" in metrics:
        result["turnover"] = calc_factor_turnover(factor)
        result["daily_turnover"] = result["turnover"].mean()
    if "sharpe" in metrics:
        result["sharpe"] = result["return"].mean() / result["return"].std()
    if "ir" in metrics:  # information ratio
        result["ir"] = result["excess_return"].mean() / result["excess_return"].std()
    if "fitness" in metrics:
        result["fitness"] = calc_fitness(result["sharpe"], result["return"].mean(), result["turnover"].mean())
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(result["excess_return"], label='excessive return')
        plt.plot(result["return"], label='return', color='r')
        plt.plot(result["benchmark"], label='benchmark', color = 'gray', alpha = 0.8)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 100:.2f}%'))
        plt.ylabel('return')
        plt.title('Return')
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.clf()

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(result["ic"].index, result["ic"].rolling(ic_freq).mean(), label='IC', color='gray', alpha=0.5, width=0.5)
        ax1.set_ylabel('IC')  # 设置 y 轴标签
        ax1.set_title(f'IC Series (rolling ' + str(ic_freq) + ')')

        ax2 = ax1.twinx()
        ax2.plot(result["accumulated_ic"], label='Accumulated_IC', color='r')
        ax2.set_ylabel('Accumulated IC')  # 设置次 y 轴标签


        ax1.grid(True)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.show()

    if to_df:
        result_df = pd.DataFrame()
        result_summary = pd.DataFrame()

        for key, value in result.items():
            if isinstance(value, pd.Series):
                result_df[key] = value
            else:
                result_summary[key] = pd.Series(value)

        return result_df, result_summary

    return result


def Brinson_Fachler_analysis(portfolio_weight: pd.Series, returns_benchmark: pd.Series, returns_portfolio=None,
                             benchmark_weight=None):
    """
    单期Brinson模型的BF分解. 可以简单累加作为多期Brinson模型的分解
    当returns_portfolio为None时，默认在benchmark成分内选股，即选择收益SR=0，超额收益完全来自配置收益AR
    当benchmark_weight为None时，benchmark默认为成分股等权指数
    为了正确计算中性组合的收益分解，对原式做了调整
    """
    if returns_portfolio is None:
        returns_portfolio: pd.Series = returns_benchmark
    # R_p: pd.Series = (portfolio_weight * returns_portfolio).groupby(level=0).sum()
    if benchmark_weight is None:
        benchmark_weight: pd.Series = returns_benchmark.groupby(level=0).transform(lambda x: 1 / x.count())
    R_b: pd.Series = (benchmark_weight * returns_benchmark).groupby(level=0).sum()

    adj_returns_portfolio = sign(portfolio_weight) * returns_portfolio
    adj_portfolio_weight = abs(portfolio_weight)

    # AR: pd.Series = ((portfolio_weight - benchmark_weight) * (returns_benchmark - R_b)).groupby(level=0).sum()
    # SR: pd.Series = R_p - R_b - AR
    SR: pd.Series = (benchmark_weight * (adj_returns_portfolio - returns_benchmark)).groupby(level=0).sum()
    AR: pd.Series = ((portfolio_weight - benchmark_weight) * (returns_benchmark - R_b)).groupby(level=0).sum()
    IR: pd.Series = ((adj_returns_portfolio - returns_benchmark) * (adj_portfolio_weight - benchmark_weight)).groupby(
        level=0).sum()
    return AR, SR, IR


class Alpha:
    def __init__(self):
        self.data = None
        self.norm_method = None
        self.process_nan = None
        self.result = None

    def call(self):
        """
        Write your alpha formula here
        """
        pass

    def normalize(self):
        self.result = mad_winsor(inf_mask(self.result))
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method=self.process_nan).fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class MA(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False, factor_name: str = "ma"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        self.factor_name = factor_name
        if matrix:
            self.result = {}
        else:
            self.result = pd.DataFrame(dtype='float64') | pd.Series(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_mean(self.data, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                if self.factor_name:
                    self.result[self.factor_name + str(d)] = ts_mean(self.data, d, matrix=self.matrix)
                else:
                    self.result["ma" + str(d)] = ts_mean(self.data, d, matrix=self.matrix)


class STD(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False, factor_name: str = "std"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        self.factor_name = factor_name
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_std(self.data, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                if self.factor_name:
                    self.result[self.factor_name + str(d)] = ts_std(self.data, d, matrix=self.matrix)
                else:
                    self.result["std" + str(d)] = ts_std(self.data, d, matrix=self.matrix)


class KURT(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_kurt(self.data, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                self.result["kurt" + str(d)] = ts_kurt(self.data, d, matrix=self.matrix)


class SKEW(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_skew(self.data, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                self.result["skew" + str(d)] = ts_skew(self.data, d, matrix=self.matrix)


class DELAY(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False, factor_name = None):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        self.factor_name = factor_name
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_delay(self.data, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                if self.factor_name:
                    self.result[self.factor_name + str(d)] = ts_delay(self.data, d, matrix=self.matrix)
                else:
                    self.result["delay" + str(d)] = ts_delay(self.data, d, matrix=self.matrix)


class DELTA(Alpha):
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill", matrix: bool = False, factor_name = None):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        self.factor_name = factor_name
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_delta(self.data, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                if self.factor_name:
                    self.result[self.factor_name + str(d)] = ts_delta(self.data, d, matrix=self.matrix)
                else:
                    self.result["delta" + str(d)] = ts_delta(self.data, d, matrix=self.matrix)


class MAX(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_max(self.data, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                self.result["max" + str(d)] = ts_max(self.data, d, matrix=self.matrix)


class MIN(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_min(self.data, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                self.result["min" + str(d)] = ts_min(self.data, d, matrix=self.matrix)


class RANK(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if self.matrix:
            self.result = {}
        else:
            self.result = pd.DataFrame(dtype='float64') |pd.Series(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_rank(self.data, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                self.result["rank" + str(d)] = ts_rank(self.data, d, matrix=self.matrix)


class QTLU(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_quantile_up(self.data, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                self.result["qtlu" + str(d)] = ts_quantile_up(self.data, d, matrix=self.matrix)


class QTLD(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_quantile_down(self.data, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                self.result["qtld" + str(d)] = ts_quantile_down(self.data, d, matrix=self.matrix)


class CORR(Alpha):
    def __init__(self, feature: pd.Series, label: pd.Series, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.feature = feature
        self.label = label
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_corr(self.feature, self.label, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                self.result["corr" + str(d)] = ts_corr(self.feature, self.label, d, matrix=self.matrix)


class CORD(Alpha):
    # The correlation between feature change ratio and label change ratio
    def __init__(self, feature: pd.Series, label: pd.Series, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.feature = feature
        self.label = label
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        fd = self.feature / ts_delay(self.feature, 1, matrix=self.matrix)
        ld = self.label / ts_delay(self.label, 1, matrix=self.matrix)
        if isinstance(self.periods, int):
            self.result = ts_corr(fd, ld, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                self.result["cord" + str(d)] = ts_corr(fd, ld, d, matrix=self.matrix)


class COV(Alpha):
    def __init__(self, feature: pd.Series, label: pd.Series, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.feature = feature
        self.label = label
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_cov(self.feature, self.label, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                self.result["cov" + str(d)] = ts_cov(self.feature, self.label, d, matrix=self.matrix)


class BETA(Alpha):
    def __init__(self, feature: pd.Series, label: pd.Series, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix=False):
        super().__init__()
        self.feature = feature
        self.label = label
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_beta(self.feature, self.label, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                self.result["beta" + str(d)] = ts_beta(self.feature, self.label, d, matrix=self.matrix)


class REGRESSION(Alpha):
    def __init__(self, feature: pd.Series, label: pd.Series, periods: list[int] | int, rettype: int = 0,
                 normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False, factor_name = None):
        super().__init__()
        self.feature = feature
        self.label = label
        self.periods = periods
        self.rettype = rettype
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        self.factor_name = factor_name
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_regression(self.feature, self.label, self.periods, rettype=self.rettype, matrix=self.matrix)
        else:
            for d in self.periods:
                if self.factor_name:
                    self.result[self.factor_name + str(d)] = ts_regression(self.feature, self.label, d, self.rettype, matrix=self.matrix)
                else:
                    self.result["reg" + str(d)] = ts_regression(self.feature, self.label, d, self.rettype, matrix=self.matrix)


class PSY(Alpha):
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        diff = ts_delta(self.data, 1)
        if isinstance(self.periods, int):
            self.result = ts_pos_count(diff, self.periods, matrix=self.matrix) / self.periods * 100
        else:
            for d in self.periods:
                self.result["psy" + str(d)] = ts_pos_count(diff, d, matrix=self.matrix) / d * 100


class KBAR(Alpha):
    def __init__(self, data: pd.DataFrame, normalize: str = "zscore", nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.data = data
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if self.matrix:
            self.result = {}
        else:
            self.result = pd.DataFrame(dtype='float64')

    def call(self):
        self.result["kmid"] = self.data["close"] / self.data["open"] - 1
        self.result["klen"] = (self.data["high"] - self.data["low"]) / self.data["open"]
        self.result["kmid2"] = (self.data["close"] - self.data["open"]) / (self.data["high"] - self.data["low"])
        self.result["kup"] = (self.data["high"] - bigger(self.data["open"], self.data["close"])) / self.data["open"]
        self.result["kup2"] = (self.data["high"] - bigger(self.data["open"], self.data["close"])) / (
                self.data["high"] - self.data["low"])
        self.result["klow"] = (smaller(self.data["open"], self.data["close"]) - self.data["low"]) / self.data["open"]
        self.result["klow2"] = (smaller(self.data["open"], self.data["close"]) - self.data["low"]) / (
                self.data["high"] - self.data["low"])
        self.result["ksft"] = (2 * self.data["close"] - self.data["high"] - self.data["low"]) / self.data["open"]
        self.result["ksft2"] = (2 * self.data["close"] - self.data["high"] - self.data["low"]) / (
                self.data["high"] - self.data["low"])


class RSV(Alpha):
    def __init__(self, data: pd.DataFrame, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if self.matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            lowest = ts_min(self.data["low"], self.periods, matrix=self.matrix)
            self.result = (self.data["close"] - lowest) / (ts_max(self.data["high"], self.periods, matrix=self.matrix) - lowest)
        else:
            for d in self.periods:
                lowest_d = ts_min(self.data["low"], d, matrix=self.matrix)
                self.result["rsv" + str(d)] = (self.data["close"] - lowest_d) / (
                        ts_max(self.data["high"], d, matrix=self.matrix) - lowest_d)


class CNTP(Alpha):
    # The percentage of days in past d days that price go up.
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if self.matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        diff = ts_delta(self.data, 1, matrix=self.matrix)
        if isinstance(self.periods, int):
            self.result = ts_pos_count(diff, self.periods, matrix=self.matrix) / self.periods
        else:
            for d in self.periods:
                self.result["cntp" + str(d)] = ts_pos_count(diff, d, matrix=self.matrix) / d


class CNTN(Alpha):
    # The percentage of days in past d days that price go down.
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if self.matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        diff = ts_delta(self.data, 1, matrix=self.matrix)
        if isinstance(self.periods, int):
            self.result = ts_neg_count(diff, self.periods, matrix=self.matrix) / self.periods
        else:
            for d in self.periods:
                self.result["cntn" + str(d)] = ts_neg_count(diff, d, matrix=self.matrix) / d


class SUMP(Alpha):
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill", matrix: bool = False, factor_name: str = "sump"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        self.factor_name = factor_name
        if self.matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        zeros = self.data - self.data
        diff = ts_delta(self.data, 1, matrix=self.matrix)
        if isinstance(self.periods, int):
            self.result = ts_sum(bigger(diff, zeros, matrix=self.matrix), self.periods, matrix=self.matrix) \
                    / ts_sum(abs(diff), self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                if self.factor_name:
                    self.result[self.factor_name + str(d)] = ts_sum(bigger(diff, zeros, matrix=self.matrix), d, matrix=self.matrix) \
                        / ts_sum(abs(diff), d, matrix=self.matrix)
                else:
                    self.result["sump" + str(d)] = ts_sum(bigger(diff, zeros, matrix=self.matrix), d, matrix=self.matrix) \
                        / ts_sum(abs(diff), d, matrix=self.matrix)


class SUMN(Alpha):
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill", matrix: bool = False, factor_name: str = "sumn"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        self.factor_name = factor_name
        if self.matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        zeros = self.data - self.data
        diff = ts_delta(self.data, 1, matrix=self.matrix)
        if isinstance(self.periods, int):
            self.result = ts_sum(bigger(-diff, zeros, matrix=self.matrix), self.periods, matrix=self.matrix) \
                            / ts_sum(abs(diff), self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                if self.factor_name:
                    self.result[self.factor_name + str(d)] = ts_sum(bigger(-diff, zeros, matrix=self.matrix), d, matrix=self.matrix) \
                                                / ts_sum(abs(diff), d, matrix=self.matrix)
                else:
                    self.result["sumn" + str(d)] = ts_sum(bigger(-diff, zeros, matrix=self.matrix), d, matrix=self.matrix) \
                                                / ts_sum(abs(diff), d, matrix=self.matrix)


class WVMA(Alpha):
    def __init__(self, price: pd.Series, volume: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill", matrix: bool = False):
        super().__init__()
        self.price = price
        self.volume = volume
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.matrix = matrix
        if self.matrix:
            self.result = {}
        else:
            self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        weight: pd.Series = abs(ts_returns(self.price, 1, matrix=self.matrix))
        weighted_vol: pd.Series = weight * self.volume
        if isinstance(self.periods, int):
            self.result = ts_std(weighted_vol, self.periods, matrix=self.matrix) / ts_mean(weighted_vol, self.periods, matrix=self.matrix)
        else:
            for d in self.periods:
                self.result["wvma" + str(d)] = ts_std(weighted_vol, d, matrix=self.matrix) / ts_mean(weighted_vol, d, matrix=self.matrix)


class CustomizedAlpha(Alpha):
    def __init__(self, data: pd.Series | pd.DataFrame, expression: list[str] | str, name: list[str] | str = None,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        """
        eg:
        factor = CustomizedAlpha(data=df, expression=[f"ts_std(data['{x}'], 5)" for x in df.columns]).get_factor_value()
        """
        super().__init__()
        self.data = data
        self.expression = expression
        self.name = name
        self.norma_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.expression, list):
            factors = []
            for e in self.expression:
                factors.append(eval(e.replace("data", "self.data")))
            self.result: pd.DataFrame = pd.concat(factors, axis=1)
            if self.name is not None:
                self.result.columns = self.name
        else:
            self.result: pd.Series = eval(self.expression.replace("data", "self.data"))
            if self.name is not None:
                self.result.name = self.name


def qlib360(data: pd.DataFrame, normalize=False, fill=False, windows=None) -> pd.DataFrame:
    """
    复现qlib的alpha 360.
    将qlib源代码中的vwap替换成amount, 因为按照qlib的workflow, vwap全是空值, 则vwap类的因子是没有意义的

    :param data: 包括以下几列: open, close, high, low, volume, amount
    :param normalize: 是否进行cs zscore标准化
    :param fill: 是否向后填充缺失值
    :param windows: 列表, 默认为[0-59]
    :return:
    """
    if windows is None:
        windows = [i for i in range(1, 60)]
    o_group = data["open"].groupby(level=1)
    c_group = data["close"].groupby(level=1)
    h_group = data["high"].groupby(level=1)
    l_group = data["low"].groupby(level=1)
    v_group = data["volume"].groupby(level=1)
    a_group = data["amount"].groupby(level=1)

    price = data["close"]
    volume = data["volume"]

    OPEN = DELAY(o_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    OPEN.columns = ["open" + str(w) for w in windows]
    for c in OPEN.columns:
        OPEN[c] /= price

    CLOSE = DELAY(c_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    CLOSE.columns = ["close" + str(w) for w in windows]
    for c in CLOSE.columns:
        CLOSE[c] /= price

    HIGH = DELAY(h_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    HIGH.columns = ["high" + str(w) for w in windows]
    for c in HIGH.columns:
        HIGH[c] /= price

    LOW = DELAY(l_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    LOW.columns = ["low" + str(w) for w in windows]
    for c in LOW.columns:
        LOW[c] /= price

    VOLUME = DELAY(v_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    VOLUME.columns = ["volume" + str(w) for w in windows]
    for c in VOLUME.columns:
        VOLUME[c] /= volume

    AMOUNT = DELAY(a_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    AMOUNT.columns = ["amount" + str(w) for w in windows]
    for c in AMOUNT.columns:
        AMOUNT[c] /= price * volume
    features = pd.concat([data[["open", "close", "high", "low", "volume", "amount"]], OPEN, CLOSE, HIGH, LOW, VOLUME,
                          AMOUNT], axis=1)
    return features


def qlib158_matrix(data: pd.DataFrame, normalize: bool = False, fill: bool = False, windows=None,
            n_jobs: int = -1, deunit: bool = True) -> pd.DataFrame:
    if windows is None:
        windows = [7, 14, 21, 28, 42, 56, 84]

    tasks = [(KBAR(data, matrix=True).get_factor_value, (normalize, fill)),
             (BETA(data["open"], data["close"], windows, matrix=True).get_factor_value, (normalize, fill)),
             (RANK(data["close"], windows, matrix=True).get_factor_value, (normalize, fill)),
             (RSV(data, windows, matrix=True).get_factor_value, (normalize, fill)),
             (CORR(data["close"], log(data["volume"], matrix=True), windows, matrix=True).get_factor_value, (normalize, fill)),
             (CORD(data["close"], data["volume"], windows, matrix=True).get_factor_value, (normalize, fill)),
             (CNTP(data["close"], windows, matrix=True).get_factor_value, (normalize, fill)),
             (CNTN(data["close"], windows, matrix=True).get_factor_value, (normalize, fill)),
             (SUMP(data["close"], windows, matrix=True).get_factor_value, (normalize, fill)),
             (SUMN(data["close"], windows, matrix=True).get_factor_value, (normalize, fill)),
             (WVMA(data["close"], data["volume"], windows, matrix=True).get_factor_value, (normalize, fill))]

    parallel_result1 = Parallel(n_jobs=n_jobs)(delayed(func)(*args) for func, args in tasks)
    final_result_1 = {}
    for result_dict in parallel_result1:
        final_result_1.update(result_dict)

    task2 = [(MA(data["close"], windows, matrix=True).get_factor_value, (normalize, fill)),
             (STD(data["close"], windows, matrix=True).get_factor_value, (normalize, fill)),
             (MAX(data["close"], windows, matrix=True).get_factor_value, (normalize, fill)),
             (MIN(data["close"], windows, matrix=True).get_factor_value, (normalize, fill)),
             (QTLU(data["close"], windows, matrix=True).get_factor_value, (normalize, fill)),
             (QTLD(data["close"], windows, matrix=True).get_factor_value, (normalize, fill))]

    parallel_result2 = Parallel(n_jobs=n_jobs)(delayed(func)(*args) for func, args in task2)
    final_result_2 = {}
    for result_dict in parallel_result2:
        final_result_2.update(result_dict)

    # 价格延迟因子: 从 d7 开始, 7 的倍数, 避免 d1~d4 导致误差叠加
    DELAY_PERIODS = [7, 14, 21, 28]
    OPEN = DELAY(data["open"], periods=DELAY_PERIODS, matrix=True, factor_name="open").get_factor_value(normalize=normalize, handle_nan=fill)
    CLOSE = DELAY(data["close"], periods=DELAY_PERIODS, matrix=True, factor_name="close").get_factor_value(normalize=normalize, handle_nan=fill)
    HIGH = DELAY(data["high"], periods=DELAY_PERIODS, matrix=True, factor_name="high").get_factor_value(normalize=normalize, handle_nan=fill)
    LOW = DELAY(data["low"], periods=DELAY_PERIODS, matrix=True, factor_name="low").get_factor_value(normalize=normalize, handle_nan=fill)
    VOLUME = DELAY(data["volume"], periods=DELAY_PERIODS, matrix=True, factor_name="volume").get_factor_value(normalize=normalize, handle_nan=fill)
    AMOUNT = DELAY(data["amount"], periods=DELAY_PERIODS, matrix=True, factor_name="amount").get_factor_value(normalize=normalize, handle_nan=fill)

    basedata_price = {**OPEN, **CLOSE, **HIGH, **LOW}

    roc = DELTA(data["close"], periods=windows, matrix=True, factor_name="roc").get_factor_value(normalize=normalize, handle_nan=fill)

    r2 = REGRESSION(ts_returns(data["close"], 1, matrix=True),
                    ts_delay(ts_returns(data["close"], 1, matrix=True),
                             1, matrix=True),
                             windows,
                             matrix=True,
                             factor_name="rsqr",
                             rettype=4).get_factor_value(normalize=normalize, handle_nan=fill)

    resi = REGRESSION(ts_returns(data["close"], 1, matrix=True),
                      ts_delay(ts_returns(data["close"], 1, matrix=True),
                               1, matrix=True),
                               windows,
                               matrix=True,
                               factor_name="resi",
                               rettype=0).get_factor_value(normalize=normalize, handle_nan=fill)

    vma = MA(data["volume"], windows, matrix=True, factor_name="vma").get_factor_value(normalize=normalize, handle_nan=fill)
    vstd = STD(data["volume"], windows, matrix=True, factor_name="vstd").get_factor_value(normalize=normalize, handle_nan=fill)
    vsump = SUMP(data["volume"], windows, matrix=True, factor_name="vsump").get_factor_value(normalize=normalize, handle_nan=fill)
    vsumn = SUMN(data["volume"], windows, matrix=True, factor_name="vsumn").get_factor_value(normalize=normalize, handle_nan=fill)


    return {**data,**final_result_1, **final_result_2, **basedata_price, **VOLUME, **AMOUNT, **roc, **r2,
            **resi, **vma, **vstd, **vsump, **vsumn}


# design example
# pvd10 ic_mean:0.015864
def PVD(data: pd.DataFrame, normalize: bool = False, fill: bool = False, windows=None,
            n_jobs: int = -1, deunit: bool = True) -> pd.DataFrame:

    if windows is None:
        windows = [10]

    pvd = CustomizedAlpha(data=data,
                          expression=[f" - ts_corr(data['amount'] / data['volume'], data['volume'], {w})" for w in windows],
                          name=[f"pvd{w}" for w in windows],
                          ).get_factor_value()

    log_pvd = CustomizedAlpha(data=data,
                          expression=[f" - ts_corr(data['amount'] / data['volume'], log(data['volume']), {w})" for w in windows],
                          name=[f"log_pvd{w}" for w in windows],
                          ).get_factor_value()

    features = pd.concat([pvd,log_pvd],axis=1)

    return features

# ic_mean: -0.007898
def DBCD(data: pd.DataFrame, normalize: bool = False, fill: bool = False, windows=None,
            n_jobs: int = -1, deunit: bool = True) -> pd.DataFrame:

    # if windows is None:
    #     windows = [5, 10, 15, 20]

    dbcd = CustomizedAlpha(data=data,
                          expression=[f"ts_decay(ts_delta(data['close'] / ts_decay(data['close'], 5) - 1, 16), 17)"],
                          name=[f"dbcd"],
                          ).get_factor_value()
    return dbcd


def alpha191(data: pd.DataFrame, normalize: bool = False, fill: bool = False, windows=None,
            n_jobs: int = -1, deunit: bool = True) -> pd.DataFrame:

    if windows is None:
        windows = [5, 10, 20, 30]

    # 价量背离
    alpha1 = CustomizedAlpha(data=data,
                          expression=[f" - ts_corr(cs_rank(ts_delta(log(data['volume']),1)), cs_rank((data['close']-data['open'])/data['open']), 6)"],
                          name=[f"alpha1"],
                          ).get_factor_value()

    # 价格位置变化
    alpha2 = CustomizedAlpha(data=data,
                          expression=[f" - ts_delta(((data['close']-data['low'])-(data['high']-data['close']))/(data['high']-data['low']), 1)"],
                          name=[f"alpha2"],
                          ).get_factor_value()

    # 累积方向波幅
    alpha3 = CustomizedAlpha(
        data=data,
        expression=[
            "ts_sum( \
                if_else( \
                    data['close'] == ts_delay(data['close'], 1), \
                    pd.Series(0, index=data.index), \
                    data['close'] - if_else( \
                        (data['close'] > ts_delay(data['close'], 1)).values, \
                        np.minimum(data['low'], ts_delay(data['close'], 1)), \
                        np.maximum(data['high'], ts_delay(data['close'], 1)) \
                    ) \
                ), \
                6 \
            )"
        ],
        name=["alpha3"]
    ).get_factor_value()


    alpha5 = CustomizedAlpha(data=data,
                        expression=[f" - ts_max(ts_corr(ts_rank(data['volume'],5),ts_rank(data['high'],5),5),3)"],
                        name=[f"alpha5"],
                        ).get_factor_value()

    alpha6 = CustomizedAlpha(data=data,
                        expression=[f"cs_rank(- sign(ts_delta((data['open']*0.85+data['high']*0.15),4)))"],
                        name=[f"alpha6"],
                        ).get_factor_value()

    alpha7 = CustomizedAlpha(data=data,
                        expression=[f"(cs_rank(max(data['amount'] / data['volume'] - data['close'], 3)) \
                                    + cs_rank(min(data['amount'] / data['volume'] - data['close'], 3))) \
                                    * cs_rank(ts_delta(data['volume'], 3))"],
                        name=[f"alpha7"],
                        ).get_factor_value()


    features = pd.concat([alpha1, alpha2, alpha3, alpha5, alpha6, alpha7], axis=1)

    return features


def qlib360_san(data: pd.DataFrame, normalize=False, fill=False, windows=None) -> pd.DataFrame:
    """
    适配单层日期索引的 qlib360 因子复现。
    数据列必须包含：'TD', 'BD', 'AD', 'SD', 'RD', 'WD', 'ABWS', '申报容量', 'P省加权'
    'P省加权' 作为基准价格（原close），其余列按价格类因子处理（滞后除以基准价格）。

    :param data: 仅含日期索引的 DataFrame，列名为上述字段
    :param normalize: 是否对每个滞后因子进行横截面 zscore 标准化（此处为单序列，标准化按该列自身）
    :param fill: 是否向后填充滞后产生的缺失值（使用 bfill + ffill）
    :param windows: 滞后期列表，默认为 [1, 2, ..., 59]
    :return: 原始列 + 各窗口延迟因子
    """
    if windows is None:
        windows = list(range(1, 60))

    cols = ['TD', 'BD', 'AD', 'SD', 'RD', 'ABWS', '申报容量', 'price']
    base = data['price']  # 基准价格

    delay_dfs = []

    for col in cols:
        series = data[col]
        # 生成各滞后期的因子
        delay_dict = {}
        for w in windows:
            shifted = series.shift(w)          # 滞后 w 期
            shifted /= base                    # 除以当期基准价格（索引对齐）
            # 处理缺失值（根据 fill 参数）
            if fill:
                shifted = shifted.fillna(method='bfill').fillna(method='ffill')
            # 标准化（如果启用）
            if normalize:
                # 对单列进行标准化（减去均值除以标准差）
                shifted = (shifted - shifted.mean()) / shifted.std()
            delay_dict[f"{col}{w}"] = shifted
        delay_df = pd.DataFrame(delay_dict, index=data.index)
        delay_dfs.append(delay_df)

    delays = pd.concat(delay_dfs, axis=1)
    # 返回原始列 + 延迟因子
    return pd.concat([data[cols], delays], axis=1)