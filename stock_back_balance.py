# !/usr/bin/python3
"""
量化回测
"""
import pandas as pd
from backtests.backtest import StockBacktest
from strategys.strategy_balance_base import ReaseachFeatureStrategy, BalanceBuyStrategy, BalanceSellStrategy

pd.set_option('display.max_rows', None)


def back_return(stock_pools: list, start_date: str, end_date: str, strategy_name: str):
    # 初始化策略
    buy_strat = BalanceBuyStrategy()
    sell_strat = BalanceSellStrategy()
    feature_strategy = ReaseachFeatureStrategy()

    # 创建回测实例
    backtester = StockBacktest(
        stock_symbols=stock_pools,
        start_date=start_date,
        end_date=end_date,
        top_n=5,
        initial_cash=50000,
        buy_strategy=buy_strat,
        sell_strategy=sell_strat,
        feature_strategy=feature_strategy,
        position_allocation_method='custom_weight',
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
    """
    512890 红利低波ETF
    511090 30年国债ETF
    518880 黄金ETF
    513100 纳指ETF
    515080 中证红利ETF
    510880 红利ETF
    513500 标普500ETF
    511010 国债ETF
    511260 十年国债ETF
    511270 10年地方债ETF
    513030 德国ETF
    513530 港股通红利
    513820 港股红利ETF基金
    517180 中国国企ETF
    """
    strategy_name = '大类资产均衡策略'
    # stock_pools = ['511260', '518880', '513100', '515080']
    # stock_pools = ['512890', '518880', '513100', '513500', '515080']
    stock_pools = ['511270', '518880', '513100', '513500', '510880']
    start_date = '2018-02-01'
    end_date = '2025-08-31'
    back_return(stock_pools, start_date, end_date, strategy_name=strategy_name)
