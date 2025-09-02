# !/usr/bin/python3
"""
Authors: hanguangyao
策略父类
"""


class FeatureStrategy:

    def __init__(self, model_version='V1'):
        self.model_version = model_version

    def add_features(self, stock_df):
        """
        添加自定义指标计算逻辑
        :param stock_df: 股票数据
        :return: 添加指标后的股票数据
        """
        return stock_df

    def add_scores(self, stock_df, symbol_id):
        """
        添加自定义评分逻辑
        :param stock_df: 股票数据
        :return: 添加评分后的股票数据
        """
        return stock_df


class BuyStrategy:
    def filter_stocks(self, row):
        """
        自定义选股逻辑
        :param stock_df: 股票数据（包含历史数据和基础信息）
        :return: 过滤后的股票数据
        """
        return True

    def should_buy(self, row):
        """买入信号判断"""
        return False

    def score(self, row):
        """评分排序规则"""
        return 0

    def calculate_position_ratios(self, buy_candidates):
        """
        根据选出的股票计算仓位比例
        :param buy_candidates: 选出的股票数据（DataFrame）
        :return: 仓位比例列表（总和为 1）
        """
        # 默认平均分配
        num_candidates = len(buy_candidates)
        return [1 / num_candidates] * num_candidates


class SellStrategy:
    def should_sell(self, row, positions, buy_candidates):
        """卖出信号判断"""
        return False
