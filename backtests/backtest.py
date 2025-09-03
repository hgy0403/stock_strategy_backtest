# !/usr/bin/python3
"""
我的量化回测引擎
"""
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from strategys.strategy import BuyStrategy, SellStrategy, FeatureStrategy
from data_solve.efinance_price import get_price
from utils.logger import setup_logger
from pyecharts import options as opts
from pyecharts.charts import Line
import exchange_calendars as trade_date


class StockBacktest:
    def __init__(
            self,
            stock_symbols,
            start_date,
            end_date,
            initial_cash=10000,
            top_n=5,
            buy_strategy=None,
            sell_strategy=None,
            feature_strategy=None,
            strategy_name=None,
            load_data=False,
            outstanding_share=False,
            index_data=False,
            slippage=0.0,  # 滑点比例，例如 0.01 表示 1%
            commission=0.0003,  # 手续费比例，例如 0.0003 表示 0.03%
            position_allocation_method='equal_weight',
            benchmark_symbol='沪深300'
    ):
        """
        初始化回测类
        :param stock_symbols: 股票代码列表
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        :param initial_cash: 初始资金
        :param top_n: 最多同时持有的股票数量
        :param buy_strategy: 自定义买入策略实例
        :param sell_strategy: 自定义卖出策略实例
        """
        self.stock_symbols = stock_symbols
        self.start_date = start_date
        self.end_date = end_date
        self.strategy_name = strategy_name
        self.load_data = load_data
        self.initial_cash = initial_cash
        self.top_n = top_n
        self.slippage = slippage
        self.commission = commission
        self.position_allocation_method = position_allocation_method
        self.benchmark_symbol = benchmark_symbol
        self.outstanding_share = outstanding_share
        self.index_data = index_data
        # 获取中国上海交易所的交易日历
        self.trade_dates = trade_date.get_calendar("XSHG").sessions_in_range(start_date, end_date).astype(str).tolist()
        self.position = {}  # 当前持仓 {股票代码: {'amount': 数量, 'buy_price': 成本价}}
        self.trade_list = []  # 交易记录
        self.current_cash = initial_cash  # 当前现金
        self.daily_data = None  # 当日数据，初始化为 None
        self.daily_performance = []  # 每日收益情况
        self.buy_strategy = buy_strategy if buy_strategy else BuyStrategy()
        self.sell_strategy = sell_strategy if sell_strategy else SellStrategy()
        self.feature_strategy = feature_strategy if feature_strategy else FeatureStrategy()
        # 初始化峰值资产和最大回撤
        self.peak_assets = initial_cash  # 初始峰值资产
        self.buy_candidates = pd.DataFrame()
        self.max_drawdown = 0  # 最大回撤
        # 初始化交易统计变量
        self.total_trades = 0  # 总交易次数
        self.winning_trades = 0  # 盈利交易次数
        # 初始化 logger
        self.logger = setup_logger()
        # 初始化基准数据
        self.benchmark_data = self._get_benchmark_data()

    def _get_benchmark_data(self):
        """获取基准指数数据"""
        benchmark_data = get_price(self.benchmark_symbol, self.start_date, self.end_date, klt=101)
        benchmark_data = benchmark_data[['open', 'high', 'low', 'close']]
        benchmark_data['trade_date'] = benchmark_data.index
        benchmark_data['return'] = (benchmark_data['close'] / benchmark_data['close'].iloc[0] - 1) * 100
        return benchmark_data

    def _get_benchmark_return(self, trade_date):
        """获取基准指数的累计收益率"""
        if self.benchmark_data.empty:
            return 0
        # 找到基准数据中当前交易日或之前的最新数据
        benchmark_row = self.benchmark_data[self.benchmark_data['trade_date'] <= trade_date].iloc[-1]
        return benchmark_row['return'] / 100  # 转换为小数形式

    def get_backtest_data(self, trade_date, klt=101, stock_data_num=400):
        """
        获取当天股票池中所有股票的指标数据
        :param trade_date: 当前交易日
        :param klt: K线类型
        :param stock_data_num: 需要的历史数据条数
        :return: 包含所有股票最后一个交易日数据的DataFrame
        """
        data_beg = (pd.to_datetime(trade_date) - pd.Timedelta(days=stock_data_num)).strftime('%Y-%m-%d')
        data_end = trade_date
        # 判断是否满仓
        # if len(self.position) >= self.top_n:
        #     # 满仓时，只计算持仓股票的特征
        #     stock_symbols = list(self.position.keys())
        # else:
        #     # 仓位不足时，计算整个股票池的特征
        #     stock_symbols = self.stock_symbols
        stock_datas = []
        for stock_symbol in self.stock_symbols:
            stock_df = get_price(stock_symbol, data_beg, data_end, klt=klt, index_data=self.index_data,
                                 outstanding_share=self.outstanding_share)
            data_cols = ['symbol_name', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'quote_rate']
            if self.index_data:
                data_cols.extend(['INDEXO', 'INDEXC', 'INDEXH', 'INDEXL'])
            if self.outstanding_share:
                data_cols.append('outstanding_share')
            stock_df = stock_df[data_cols]
            stock_df['stock_symbol'] = str(stock_symbol)
            stock_df['trade_date'] = stock_df.index  # 将索引作为普通列
            stock_df.reset_index(drop=True, inplace=True)
            # 计算指标
            stock_df = self.feature_strategy.add_features(stock_df)
            stock_df = self.feature_strategy.add_scores(stock_df, stock_symbol)
            # 取最后一个交易日的数据
            last_day_data = stock_df.iloc[[-1]]
            stock_datas.append(last_day_data)
        result_df = pd.DataFrame()
        if len(stock_datas) > 0:
            # 合并所有股票的最后一个交易日数据
            result_df = pd.concat(stock_datas)
            # 将 stock_symbol 和 trade_date 列放在最前面
            column_order = ['stock_symbol', 'trade_date'] + [col for col in result_df.columns if
                                                             col not in ['stock_symbol', 'trade_date']]
            result_df = result_df[column_order]
        return result_df

    def get_backtest_data_batch(self, klt=101, stock_data_num=250):
        """为提高回测效率，采用了批量获取数据进行回测的方式"""
        import os
        cache_dir = "./predict_results"  # You can change this to your preferred directory
        filename = f"stock_{self.strategy_name}_{self.start_date}_{self.end_date}.csv"
        filepath = os.path.join(cache_dir, filename)
        # Check if cached file exists
        if os.path.exists(filepath) and self.load_data:
            print(f"Loading cached data from {filepath}")
            result_df = pd.read_csv(filepath, parse_dates=['trade_date'])
            return result_df
        stock_datas = []
        data_beg = (pd.to_datetime(self.start_date) - pd.Timedelta(days=stock_data_num)).strftime('%Y-%m-%d')
        for stock_symbol in tqdm(self.stock_symbols, desc='Processing stocks'):
            stock_df = get_price(stock_symbol, start_date=data_beg, end_date=self.end_date, klt=klt,
                                 index_data=self.index_data,
                                 outstanding_share=self.outstanding_share)
            data_cols = ['symbol_name', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'quote_rate']
            if self.index_data:
                data_cols.extend(['INDEXO', 'INDEXC', 'INDEXH', 'INDEXL'])
            if self.outstanding_share:
                data_cols.append('outstanding_share')
            stock_df = stock_df[data_cols]
            stock_df['stock_symbol'] = str(stock_symbol)
            stock_df['trade_date'] = stock_df.index  # 将索引作为普通列
            stock_df.reset_index(drop=True, inplace=True)
            stock_df = self.feature_strategy.add_features(stock_df)
            stock_df = self.feature_strategy.add_scores(stock_df, stock_symbol)
            stock_df = stock_df.dropna()
            # 截取从 start_date 到 end_date 的数据
            stock_df = stock_df[
                (stock_df['trade_date'] >= pd.to_datetime(self.start_date)) & (
                        stock_df['trade_date'] <= pd.to_datetime(self.end_date))]
            stock_datas.append(stock_df)
        result_df = pd.DataFrame()
        if len(stock_datas) > 0:
            # 合并所有股票的最后一个交易日数据
            result_df = pd.concat(stock_datas)
            # 将 stock_symbol 和 trade_date 列放在最前面
            column_order = ['stock_symbol', 'trade_date'] + [col for col in result_df.columns if
                                                             col not in ['stock_symbol', 'trade_date']]
            result_df = result_df[column_order]
            result_df.to_csv(filepath)
        return result_df

    def run_backtest(self):
        """执行回测主逻辑"""
        prev_total_assets = self.initial_cash
        result_df = self.get_backtest_data_batch()
        for trade_date in tqdm(self.trade_dates, desc='Backtesting'):
            # 如果股票池合约数量比较少可以用这个函数，但是主要用这个函数需要注释掉：result_df = self.get_backtest_data_batch()
            # self.daily_data = self.get_backtest_data(trade_date)
            self.daily_data = result_df[result_df['trade_date'] == trade_date]
            if self.daily_data.empty:
                self.logger.warning(f"{trade_date} 无数据，跳过")
                continue
            # 卖出逻辑
            portfolio_symbols = list(self.position.keys())
            portfolio_data = self.daily_data[self.daily_data['stock_symbol'].isin(portfolio_symbols)]
            sell_signals = portfolio_data[portfolio_data.apply(
                lambda row: self.sell_strategy.should_sell(row, self.position, self.buy_candidates),
                axis=1
            )]
            for _, row in sell_signals.iterrows():
                self._execute_sell(row, trade_date)

            # 股票过滤
            daily_data_filter = self.daily_data[
                self.daily_data.apply(lambda row: self.buy_strategy.filter_stocks(row), axis=1)]
            # 买入逻辑
            self.buy_candidates = daily_data_filter[
                daily_data_filter.apply(lambda row: self.buy_strategy.should_buy(row), axis=1)]
            # 排序
            if not self.buy_candidates.empty:
                # 自定义评分排序
                buy_candidates_sort = self.buy_candidates.assign(
                    score=self.buy_candidates.apply(self.buy_strategy.score, axis=1)
                ).sort_values('score', ascending=False).head(self.top_n - len(self.position))

                for _, row in buy_candidates_sort.iterrows():
                    self._execute_buy(row, trade_date)

            # 记录当日表现
            self._record_daily_performance(trade_date, prev_total_assets)
            prev_total_assets = self.total_assets
        # 统计日胜率
        total_days = len(self.daily_performance)
        profitable_days = sum(1 for day in self.daily_performance if day['is_profit'])
        daily_win_rate = (profitable_days / total_days) * 100 if total_days > 0 else 0
        # 回测结束后打印总体结果
        total_return = (self.total_assets - self.initial_cash) / self.initial_cash
        benchmark_return = self._get_benchmark_return(self.end_date)
        excess_return = total_return - benchmark_return
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        # 计算年化收益率（新增部分）
        total_days = len(self.daily_performance)
        if total_days > 0:
            # 计算总收益率
            total_return = (self.total_assets - self.initial_cash) / self.initial_cash
            # 计算年化收益率（假设252个交易日/年）
            annualized_return = (1 + total_return) ** (252 / total_days) - 1
        else:
            annualized_return = 0

        self.logger.info(
            f"\n回测结束\n"
            f"总资金：{self.total_assets:.2f}\n"
            f"总收益：{total_return:.2%}\n"
            f"年化收益率：{annualized_return:.2%}\n"
            f"基准收益：{benchmark_return:.2%}\n"
            f"超额收益：{excess_return:.2%}\n"
            f"总体最大回撤：{self.max_drawdown:.2%}\n"
            f"总交易次数：{self.total_trades}\n"
            f"总胜率：{win_rate:.2f}%\n"
            f"日胜率：{daily_win_rate:.2f}%"
        )
        # 绘制收益曲线
        self.plot_return_curve()

    def _execute_sell(self, row, trade_date):
        """执行卖出操作"""
        position = self.position.pop(row['stock_symbol'])
        # 应用滑点调整卖出价格
        sell_price = row['close'] * (1 - self.slippage)
        sell_amount = position['amount'] * sell_price

        # 应用手续费
        commission_fee = sell_amount * self.commission
        net_sell_amount = sell_amount - commission_fee

        self.current_cash += net_sell_amount

        # 计算收益率
        trade_return = (sell_price - position['buy_price']) / position['buy_price']

        # 记录交易
        self.total_trades += 1
        if trade_return > 0:
            self.winning_trades += 1

        self.trade_list.append({
            'date': trade_date,
            'symbol': row['stock_symbol'],
            'symbol_name': row['symbol_name'],
            'action': 'sell',
            'price': sell_price,
            'return': trade_return,
            'commission': commission_fee
        })
        self.logger.info(
            f"{trade_date} 卖出 {row['stock_symbol']}，价格：{sell_price:.2f}，收益率：{trade_return:.2%}，手续费：{commission_fee:.2f}"
        )

    def _execute_buy(self, row, trade_date):
        """执行买入操作"""
        if row['stock_symbol'] in self.position:
            self.logger.warning(f"{trade_date} {row['stock_symbol']} 已持仓，跳过买入")
            return
        # 计算可用资金
        available_cash = self.current_cash

        if self.position_allocation_method == 'equal_weight':
            # 等权重分配
            cash_for_stock = available_cash / (self.top_n - len(self.position))
        elif self.position_allocation_method == 'custom_weight':
            # 自定义权重分配
            position_ratios = self.buy_strategy.calculate_position_ratios(self.buy_candidates)
            candidate_symbols = self.buy_candidates['stock_symbol'].tolist()
            try:
                index = candidate_symbols.index(row['stock_symbol'])
                position_ratio = position_ratios[index]
            except (ValueError, IndexError):
                self.logger.warning(f"{trade_date} {row['stock_symbol']} 未找到仓位比例，跳过")
                return
            cash_for_stock = available_cash * position_ratio
        else:
            raise ValueError(f"未知的仓位分配方式: {self.position_allocation_method}")

        # 应用滑点调整买入价格
        buy_price = row['close'] * (1 + self.slippage)

        # 计算可买入的最大手数
        max_shares = int(cash_for_stock // (row['close'] * 100))  # 1 手 = 100 股
        if max_shares < 1:
            self.logger.warning(f"{trade_date} 可用资金不足，无法买入 {row['stock_symbol']} 的 1 手")
            return

        # 确保买入数量是 1 手的整数倍
        buy_shares = max_shares * 100  # 1 手 = 100 股
        buy_amount = buy_shares * buy_price

        # 应用手续费
        commission_fee = buy_amount * self.commission
        total_cost = buy_amount + commission_fee

        # 如果没有持仓，直接添加
        self.position[row['stock_symbol']] = {
            'trade_date': trade_date,
            'amount': buy_shares,
            'buy_price': buy_price
        }

        self.current_cash -= total_cost

        self.trade_list.append({
            'date': trade_date,
            'symbol': row['stock_symbol'],
            'symbol_name': row['symbol_name'],
            'action': 'buy',
            'price': buy_price,
            'shares': buy_shares,
            'commission': commission_fee
        })
        self.logger.info(
            f"{trade_date} 买入 {row['stock_symbol']}，价格：{row['close']:.2f}，数量：{buy_shares} 股，手续费：{commission_fee:.2f}"
        )

    def _record_daily_performance(self, trade_date, prev_total_assets):
        """记录每日业绩"""
        # 计算持仓价值
        # holding_value = sum(
        #     pos['amount'] * self._get_last_price(symbol, trade_date)  # 获取当前交易日的最新价格
        #     for symbol, pos in self.position.items()
        # )
        holding_value = 0
        for symbol, pos in self.position.items():
            last_price = self._get_last_price(symbol, trade_date)
            holding_value += pos['amount'] * last_price
        # 计算总资产
        self.total_assets = self.current_cash + holding_value

        # 更新峰值资产
        if self.total_assets > self.peak_assets:
            self.peak_assets = self.total_assets

        # 计算当前回撤
        current_drawdown = (self.peak_assets - self.total_assets) / self.peak_assets

        # 更新最大回撤
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        # 计算日收益率
        if prev_total_assets == 0:
            daily_return = 0  # 如果前一日总资产为 0，日收益率为 0
        else:
            daily_return = (self.total_assets / prev_total_assets) - 1

        # 计算总收益率
        total_return = (self.total_assets - self.initial_cash) / self.initial_cash
        # 计算基准收益
        benchmark_return = self._get_benchmark_return(trade_date)

        # 计算超额收益
        excess_return = total_return - benchmark_return
        # 判断当日是否盈利
        is_profit = daily_return > 0

        # 记录每日表现
        self.daily_performance.append({
            'date': trade_date,
            'total_assets': self.total_assets,
            'daily_return': daily_return,
            'total_return': total_return,  # 添加总收益率
            'benchmark_return': benchmark_return,  # 基准总收益率
            'excess_return': excess_return,  # 超额收益
            'cash': self.current_cash,
            'holdings': holding_value,
            'current_drawdown': current_drawdown,
            'is_profit': is_profit  # 记录当日是否盈利
        })
        # 打印当日收益和总收益率
        self.logger.info(
            f"{datetime.strptime(trade_date, '%Y-%m-%d'):%Y-%m-%d} | "
            f"总收益率：{total_return:.2%} |"
            f"日收益率：{daily_return:.2%} |"
            f"总资产：{self.total_assets:.2f} | "
            f"当前回撤：{current_drawdown:.2%} |"
            f"现金：{self.current_cash:.2f} |"
            f"持仓价值：{holding_value:.2f} |"
            f"基准收益率：{benchmark_return:.2%} | "
            f"超额收益：{excess_return:.2%} |"
            f"当前持仓股票名称：{', '.join(self.position.keys())}"
        )

    def _get_last_price(self, symbol, trade_date):
        """获取股票最新价格"""
        # 获取当前交易日的股票数据
        stock_data = self.daily_data[
            (self.daily_data['stock_symbol'] == symbol) &
            (self.daily_data['trade_date'] == trade_date)
            ]
        if not stock_data.empty:
            return stock_data['close'].iloc[-1]  # 返回当前交易日的收盘价
        else:
            data_beg = (pd.to_datetime(trade_date) - pd.Timedelta(days=20)).strftime('%Y-%m-%d')
            stock_data = get_price(symbol, data_beg, trade_date, klt=101)
            stock_data = stock_data[['open', 'high', 'low', 'close']]
            return stock_data['close'].iloc[0]

    def plot_return_curve(self):
        """
        绘制收益曲线
        """
        # 提取策略收益
        strategy_returns = [performance['total_return'] * 100 for performance in self.daily_performance]
        benchmark_returns = [performance['benchmark_return'] * 100 for performance in self.daily_performance]
        dates = [performance['date'] for performance in self.daily_performance]

        # 创建 Line 图对象
        line_chart = Line()

        # 添加 x 轴数据
        line_chart.add_xaxis(xaxis_data=dates)

        # 添加 y 轴数据
        line_chart.add_yaxis(
            series_name='策略收益',
            y_axis=strategy_returns,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color="red")
        )
        line_chart.add_yaxis(
            series_name='基准收益',
            y_axis=benchmark_returns,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color="blue")
        )

        # 设置全局配置项
        line_chart.set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(type_="category", name="日期"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}%"),  # Y 轴显示百分比
                name="收益率"
            ),
            title_opts=opts.TitleOpts(title="策略收益 vs 基准收益")
        )

        # 渲染图表
        line_chart.render(
            f"./backtest_results/return_curve_{self.strategy_name}_{self.start_date}_{self.end_date}.html")
        self.logger.info(
            f"收益曲线已保存至 ./backtest_results/return_curve_{self.strategy_name}_{self.start_date}_{self.end_date}.html")
