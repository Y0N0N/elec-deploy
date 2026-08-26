import pandas as pd
import tushare as ts
from datetime import datetime
import os


def get_adj_hfq(price: pd.Series, pre_close: pd.Series) -> pd.Series:
    """
    计算后复权因子
    """
    price_ratio = (price / pre_close).groupby(level=1).transform(lambda x: x.cumprod())
    adj = price_ratio.groupby(level=1).transform(lambda x: x / x[0])
    return adj


def tus_init(tus_token: str = ""):
    token = tus_token
    ts.set_token(token)
    pro = ts.pro_api()
    return pro


def get_index_cons(pro, index_code: str = "000905.SH",
                   start: int = 20100101, end: int = 20101231,
                   output_folder: str = "./data/"):
    """
    获取指数成分股权重，start/end 直接给 int 格式（如 20100101）
    """
    # int → datetime，方便后面按年分段
    start_date = datetime.strptime(str(start), "%Y%m%d")
    end_date   = datetime.strptime(str(end), "%Y%m%d")

    ranges = []
    cur = start_date
    while cur <= end_date:
        y_end = datetime(cur.year, 12, 31)
        if y_end > end_date:
            y_end = end_date
        # 仍用 int 格式去调 tushare 接口
        ranges.append((int(cur.strftime("%Y%m%d")),
                       int(y_end.strftime("%Y%m%d"))))
        cur = datetime(cur.year + 1, 1, 1)

    df = pd.DataFrame()
    for s, e in ranges:
        # pro.index_weight 要求 int 型日期，现在直接满足
        tmp = pro.index_weight(index_code=index_code,
                               start_date=s,
                               end_date=e)
        df = pd.concat([df, tmp], ignore_index=True)

    # 后续处理不变
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = (df.set_index('trade_date')
          .rename_axis('datetime')
          .sort_index())

    output_path = os.path.join(output_folder, 'index_weight.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    return df

def get_trading_days(pro, start_date: str = "20100101", end_date: str = "20101231", return_type: str = "datetime", folder_path="./data/") -> pd.DataFrame:
    """
    获取交易日期
    """
    trade_cal1 = pro.trade_cal()
    trading_days = trade_cal1\
        .query(f'(cal_date>="{start_date}")&(cal_date<="{end_date}")&(is_open==1)')\
        ['cal_date'].tolist()
    trading_days = trading_days[::-1]
    if return_type == "datetime":
        for i in range(len(trading_days)):
            trading_days[i] = pd.to_datetime(trading_days[i])
    return trading_days


def process_index_cons(from_file_path="index_weight.csv", folder_path="./data/"):
    idx_cons = pd.DataFrame()

    filepath = folder_path + from_file_path
    sub_df = pd.read_csv(filepath)
    sub_df.set_index("datetime", inplace=True)
    code_list = pd.DataFrame()
    codes = sub_df["con_code"].groupby(level=0).apply(lambda x: ','.join(x.astype(str)))
    code_list["ts_code"] = codes
    code_list["days"] = code_list.index.get_level_values(0)
    code_list["days"] = code_list["days"].astype(str)
    code_list["days"] = pd.to_datetime(code_list["days"], format="%Y-%m-%d")
    # print(code_list)
    code_list.reset_index(inplace=True)
    code_list.set_index("days", inplace=True)
    new_index = pd.date_range(start=code_list.index.min(), end=code_list.index.max(), freq='D')
    code_list = code_list.reindex(new_index)
    idx_cons = pd.concat([idx_cons, code_list], axis=0)
    idx_cons.sort_index(inplace=True)
    idx_cons.index.name = "days"
    idx_cons["datetime"] = idx_cons.index.get_level_values(0).strftime("%Y%m%d").astype(int)
    idx_cons.fillna(method="ffill", inplace=True)
    idx_cons.to_csv(folder_path+"instrument_list.csv")

    return idx_cons


def get_stock_data(pro, file_path="instrument_list.csv", folder_path="./data/", adjust_price: bool = False) -> pd.DataFrame:
    instrument_data = pd.DataFrame()
    # 读取code_list后，按照list获取每支股票的数据
    df1 = pd.read_csv(folder_path+file_path)
    df1.fillna(method='ffill', inplace=True)

    date = df1['datetime'].unique()
    day = []
    for i in range(len(date)):
        day.append(str(date[i]))

    for i in range(len(date)):
        df = pd.DataFrame(pro.daily(ts_code=str(df1['ts_code'].values[i]), start_date=day[i], end_date=day[i]))  # 行情数据

        # 处理非交易日
        if df.empty:
            print(f"No data returned for date range {day[i]}.")
            continue  # 跳过这一轮循环
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index(['trade_date'], inplace=True)
        df.index.names = ['datetime']
        df = df.sort_index()
        df = df.dropna()
        instrument_data = pd.concat([instrument_data, df], axis=0).sort_index()
    instrument_data = instrument_data.reset_index()
    instrument_data.set_index(["datetime", "ts_code"], inplace=True)
    instrument_data.index.names = ["datetime", "instrument"]
    if adjust_price:
        adj = get_adj_hfq(instrument_data["close"], instrument_data["pre_close"])
        # fixme: 增加调整volume的功能
        prices = ["open", "close", "high", "low"]
        for p in prices:
            instrument_data[p] *= adj
    instrument_data.to_csv(folder_path+'stock_data.csv')
    return instrument_data

def get_stock_data_mins(pro, file_path="instrument_list.csv", folder_path="./data/", adjust_price: bool = False) -> pd.DataFrame:
    instrument_data = pd.DataFrame()
    # 读取code_list后，按照list获取每支股票的数据
    df1 = pd.read_csv(folder_path+file_path)
    df1.fillna(method='ffill', inplace=True)

    date = df1['datetime'].unique()
    day = []
    for i in range(len(date)):
        day.append(str(date[i]))

    for i in range(len(date)):
        df = pd.DataFrame(pro.stk_mins(ts_code=str(df1['ts_code'].values[i]),
                          freq='5min',
                          start_date=day[i]+ ' 09:30:00',
                          end_date=day[i]+ ' 15:00:00'))  # 行情数据

        # 处理非交易日
        if df.empty:
            print(f"No data returned for date range {day[i]}.")
            continue  # 跳过这一轮循环
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df['datetime'] = df['trade_time'].dt.date
        df['time'] = df['trade_time'].dt.time

        # 使用日期列作为索引
        df.set_index(['datetime'], inplace=True)
        df.index.names = ['datetime']
        df = df.sort_index()
        df = df.dropna()
        instrument_data = pd.concat([instrument_data, df], axis=0).sort_index()
    instrument_data = instrument_data.reset_index()
    instrument_data.set_index(["datetime", "ts_code"], inplace=True)
    instrument_data.index.names = ["datetime", "instrument"]
    if adjust_price:
        adj = get_adj_hfq(instrument_data["close"], instrument_data["pre_close"])
        # fixme: 增加调整volume的功能
        prices = ["open", "close", "high", "low"]
        for p in prices:
            instrument_data[p] *= adj
    instrument_data.to_csv(folder_path+'stock_data_mins.csv')
    return instrument_data
