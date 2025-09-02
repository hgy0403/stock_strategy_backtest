import numpy as np
import talib
from tqdm import tqdm
from data_solve.efinance_price import get_price
# from utils.tongdaxin import *
import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np
import math


def add_dongliang_score(df, window_size=20):
    """
    为DataFrame新增score列，计算方式参考get_rank函数中的逻辑

    参数:
    df: 输入的DataFrame，包含日期和收盘价等数据
    window_size: 计算score使用的历史数据窗口大小

    返回:
    添加了score列的新DataFrame
    """
    # 初始化score列
    df['score'] = np.nan
    # 对每一行计算score(跳过前window_size-1行，因为没有足够历史数据)
    for i in range(window_size - 1, len(df)):
        # 获取当前行及前window_size-1行的数据
        window_data = df.iloc[i - window_size + 1:i + 1]
        # 计算对数收盘价
        y = np.log(window_data['close'].values)
        x = np.arange(len(y))
        # 线性回归拟合
        slope, intercept = np.polyfit(x, y, 1)
        # 计算年化收益率
        annualized_returns = math.pow(math.exp(slope), 250) - 1
        # 计算R平方
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        # 计算score
        score = annualized_returns * r_squared
        # 将score赋值给当前行
        df.at[df.index[i], 'score'] = score
    return df
