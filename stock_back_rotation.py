# !/usr/bin/python3
"""
量化回测
"""
import pandas as pd
from backtests.backtest import StockBacktest
from strategys.strategy_rotaion_base import RotationBuyStrategy, RotationSellStrategy, ReaseachFeatureStrategy

pd.set_option('display.max_rows', None)


def back_return(stock_pools: list, start_date: str, end_date: str, strategy_name: str):
    # 初始化策略
    buy_strat = RotationBuyStrategy()
    sell_strat = RotationSellStrategy()
    feature_strategy = ReaseachFeatureStrategy()

    # 创建回测实例
    backtester = StockBacktest(
        stock_symbols=stock_pools,
        start_date=start_date,
        end_date=end_date,
        top_n=1,
        initial_cash=100000,
        buy_strategy=buy_strat,
        sell_strategy=sell_strat,
        feature_strategy=feature_strategy,
        strategy_name=strategy_name
    )
    # 运行回测
    backtester.run_backtest()

    # 获取结果
    performance_df = pd.DataFrame(backtester.daily_performance)
    trades_df = pd.DataFrame(backtester.trade_list)
    print("Performance：\n")
    print(performance_df.tail())
    print("Trades：\n")
    print(trades_df)
    trades_df.to_csv(f'./backtest_results/trades_list_{strategy_name}_{start_date}_{end_date}.csv')


if __name__ == '__main__':
    strategy_name = '轮动策略'
    stock_pools = ['513100', '518880', '159915', '510300']
    start_date = '2018-01-01'
    end_date = '2025-09-02'
    back_return(stock_pools, start_date, end_date, strategy_name=strategy_name)
