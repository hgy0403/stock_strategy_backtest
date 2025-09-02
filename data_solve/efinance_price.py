# !/usr/bin/python3
"""
Authors: hanguangyao
基于efinance获取股票数据
"""
import efinance as ef
import pandas as pd
from datetime import datetime
from efinance.utils import MarketType
import akshare as ak
import time
pd.set_option('display.max_columns', None)


def get_symbol_id(company_name, symbol_type='stock'):
    """
    获取股票代码
    :param company_name:
    :return:股票代码
    """
    if symbol_type == 'stock':
        security_data = pd.read_csv('./stockdata/security_info_df.csv',
                                    index_col='display_name')
        symbol_id = security_data.loc[company_name, :]['symbol'].split('.')[0]
    elif symbol_type == 'ETF':
        # security_data = pd.read_csv('/Users/hanguangyao/stock_quant/stockdata/etf_security_info_df.csv',
        #                             index_col='name')
        # symbol_id = security_data.loc[company_name, :]['symbol_id']
        symbol_id = company_name
    else:
        symbol_id = company_name

    return symbol_id


def get_symbol_name(symbol_id_ef):
    """
    获取股票名称
    :param symbol_id:
    :return:
    """
    global security_data
    symbol_name = security_data.loc[symbol_id_ef, :]['display_name']
    symbol_id = security_data.loc[symbol_id_ef, :]['symbol']
    return symbol_id, symbol_name


def get_price(symbol_id, start_date='20000101', end_date='20300101', klt=101, market_type=None, symbol_type='stock',
              outstanding_share=False, index_data=False):
    """
    获取股票历史数据： 被封ip后就点开这个网址，确定自己是人就会被解封。否则就无法下载数据。
    https://quote.eastmoney.com/center/gridlist.html#hs_a_board
    :param symbol_id:
    :param start_date:
    :param end_date:
    :return:
    """
    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')
    if symbol_type == 'stock':
        stock_df = ef.stock.get_quote_history(symbol_id, beg=start_date, end=end_date, klt=klt, fqt=1,
                                              market_type=market_type)
        time.sleep(1)
        stock_df.rename(
            columns={'股票名称': 'symbol_name', '股票代码': 'symbol_id', '开盘': 'open', '收盘': 'close',
                     '最高': 'high', '最低': 'low',
                     '成交量': 'volume',
                     '成交额': 'turnover',
                     '振幅': 'amp_rate', '涨跌幅': 'quote_rate', '换手率': 'turnover_rate'}, inplace=True)
    else:
        stock_df = ef.futures.get_quote_history(symbol_id, beg=start_date, end=end_date, klt=klt, fqt=1)
        stock_df.rename(
            columns={'期货名称': 'symbol_name', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
                     '成交量': 'volume',
                     '成交额': 'turnover',
                     '振幅': 'amp_rate', '涨跌幅': 'quote_rate', '换手率': 'turnover_rate'}, inplace=True)
        stock_df[['open', 'close', 'high', 'low']] = stock_df[['open', 'close', 'high', 'low']].astype(float)
    stock_df.index = pd.to_datetime(stock_df['日期'])
    stock_df.index.name = None
    stock_df = stock_df[
        ['symbol_name', 'symbol_id', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'turnover_rate',
         'quote_rate']]

    if outstanding_share or index_data:
        base_info = ef.stock.get_base_info(symbol_id)
        time.sleep(0.5)
        symbol_code = base_info['股票代码']
    if outstanding_share:
        # 判断市场并添加前缀
        if symbol_code.startswith('6'):
            symbol = f"sh{symbol_code}"  # 上海股票
        elif symbol_code.startswith(('0', '3')):
            symbol = f"sz{symbol_code}"  # 深圳股票（主板/创业板）
        else:
            symbol = symbol_code  # 其他情况（如北交所股票以8开头）
        stock_df.reset_index(inplace=True)
        stock_df.rename(columns={'index': 'date'}, inplace=True)
        stock_df["date"] = pd.to_datetime(stock_df["date"])
        df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, end_date=end_date, adjust="")[
            ['date', 'outstanding_share']]
        df["date"] = pd.to_datetime(df["date"])
        stock_df = pd.merge(stock_df, df, on='date', how='left')
        stock_df.set_index('date', inplace=True)
        stock_df.index.name = None
    if index_data:
        # 判断市场并添加前缀
        if symbol_code.startswith('6'):
            index_symbol = "上证指数"
        elif symbol_code.startswith(('0', '3')):
            index_symbol = "深证成指"  # 深圳股票（主板/创业板）
        else:
            index_symbol = "创业板指"  # 其他情况（如北交所股票以8开头）
        index_df = ef.stock.get_quote_history(index_symbol, beg=start_date, end=end_date, klt=klt, fqt=1,
                                              market_type=market_type)
        index_df.rename(
            columns={'开盘': 'INDEXO', '收盘': 'INDEXC', '最高': 'INDEXH', '最低': 'INDEXL', }, inplace=True)
        index_df[['INDEXO', 'INDEXC', 'INDEXH', 'INDEXL']] = index_df[['INDEXO', 'INDEXC', 'INDEXH', 'INDEXL']].astype(
            float)
        index_df.index = pd.to_datetime(index_df['日期'])
        index_df.index.name = None
        stock_df.reset_index(inplace=True)
        stock_df.rename(columns={'index': 'date'}, inplace=True)
        index_df = index_df[['INDEXO', 'INDEXC', 'INDEXH', 'INDEXL']]
        index_df.reset_index(inplace=True)
        index_df.rename(columns={'index': 'date'}, inplace=True)
        stock_df = pd.merge(stock_df, index_df, on='date', how='left')
        stock_df.set_index('date', inplace=True)
        stock_df.index.name = None
    return stock_df

def get_price_ak(symbol_code, start_date='20000101', end_date='20300101', klt=101, market_type=None, symbol_type='stock',
              outstanding_share=False, index_data=False):
    if symbol_code.startswith('6'):
        symbol = f"sh{symbol_code}"  # 上海股票
    elif symbol_code.startswith(('0', '3')):
        symbol = f"sz{symbol_code}"  # 深圳股票（主板/创业板）
    else:
        symbol = symbol_code  # 其他情况（如北交所股票以8开头）
    df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")
    # print(df)




if __name__ == '__main__':
    data = get_price('沪深300', '2023-04-01', end_date='2024-05-20', klt=101)
    print(data)

    # data = get_price_ak('000333', '2024-04-01', end_date='2025-06-06')
    # print(data)

