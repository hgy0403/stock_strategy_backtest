from strategys.strategy import BuyStrategy, SellStrategy, FeatureStrategy
import exchange_calendars as trade_date
from datetime import datetime
import pandas as pd


def is_first_trading_day_of_month_cn(target_date):
    """
    判断指定日期是否是当月的第一个交易日（中国A股市场）
    参数:
        target_date (str/datetime/date): 日期，支持格式:
            - 字符串: 'YYYY-MM-DD'
            - datetime对象
            - date对象

    返回:
        bool: 如果是当月第一个交易日返回True，否则False

    示例:
        >>> is_first_trading_day_of_month_cn('2023-10-09')  # 国庆节后第一个交易日
        True
        >>> is_first_trading_day_of_month_cn('2023-07-03')  # 7月第一个交易日
        True
    """
    # 转换为datetime.date对象
    if isinstance(target_date, str):
        dt = datetime.strptime(target_date, '%Y-%m-%d').date()
    elif isinstance(target_date, datetime):
        dt = target_date.date()
    else:
        dt = target_date

    # 获取当月所有交易日
    year, month = dt.year, dt.month
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{pd.Timestamp(year=year, month=month, day=1).days_in_month:02d}"

    # 获取交易日历
    cal = trade_date.get_calendar("XSHG")
    trading_days = cal.sessions_in_range(start, end)

    # 检查是否是第一个交易日
    if len(trading_days) == 0:
        return False
    return dt == trading_days[0].date()


# 示例策略实现
class BalanceBuyStrategy(BuyStrategy):
    def should_buy(self, row):
        current_date = row['trade_date']
        buy_signal = is_first_trading_day_of_month_cn(current_date)
        return buy_signal

    def score(self, row):
        return row['score']

    def calculate_position_ratios(self, buy_candidates):
        """
        根据选出的股票计算仓位比例
        :param buy_candidates: 选出的股票数据（DataFrame）
        :return: 仓位比例列表（总和为 1）
        """
        # 默认平均分配
        # num_candidates = len(buy_candidates) #0.2, 0.4, 0.25, 0.15
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        # weights = [0.4, 0.2, 0.25, 0.15]
        return weights


class BalanceSellStrategy(SellStrategy):
    def should_sell(self, row, positions, buy_candidates):
        # print(buy_candidates)
        position_data = positions.get(row['stock_symbol'], {})
        if not position_data: return False
        # 获取当前日期和交易日期
        # 处理 trade_date 和 current_date
        if isinstance(position_data['trade_date'], str):
            # 如果 positions['trade_date'] 是字符串，解析为 datetime
            trade_date = datetime.strptime(position_data['trade_date'], '%Y-%m-%d')
        else:
            # 如果 positions['trade_date'] 是 Timestamp，直接使用
            trade_date = position_data['trade_date']
        # 如果 row['trade_date'] 是 Timestamp，直接使用
        current_date = row['trade_date']
        result = is_first_trading_day_of_month_cn(current_date)
        holding_days = (current_date - trade_date).days
        sell_condition = holding_days > 10
        sell_signal = (sell_condition & result)
        return sell_signal


class ReaseachFeatureStrategy(FeatureStrategy):

    def __init__(self, model_version='V1'):
        super().__init__()
        self.model_version = model_version

    def add_features(self, stock_df):
        """
        自定义特征计算逻辑
        :param stock_df: 股票数据
        :return: 添加指标后的股票数据
        """
        # 没有买入信号将buy_signal设置为1
        stock_df['buy_signal'] = 1
        return stock_df

    def add_scores(self, stock_df, symbol_id):
        """没有排序叫score设置为1"""
        stock_df['score'] = 1
        return stock_df
