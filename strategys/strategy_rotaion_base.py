from strategys.strategy import BuyStrategy, SellStrategy, FeatureStrategy
from factors.factor_test import *
from datetime import datetime


# 示例策略实现
class RotationBuyStrategy(BuyStrategy):
    def should_buy(self, row):
        buy_signal = (row['buy_signal'] == 1)
        return buy_signal

    def score(self, row):
        return row['score']


class RotationSellStrategy(SellStrategy):
    def should_sell(self, row, positions, buy_candidates):
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
        holding_days = (current_date - trade_date).days
        sell_condition = holding_days > 0
        position_symbol = list(positions.keys())[0]
        sorted_df = buy_candidates.sort_values(by='score', ascending=False)
        stock_symbols = sorted_df['stock_symbol'].tolist()
        sell_etf = False
        if position_symbol != stock_symbols[0]:
            sell_etf = True
        sell_signal = (sell_condition & sell_etf)
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
        stock_df = add_dongliang_score(stock_df, window_size=25)
        stock_df['buy_signal'] = 1
        return stock_df

    def add_scores(self, stock_df, symbol_id):
        test_data = stock_df
        return test_data
