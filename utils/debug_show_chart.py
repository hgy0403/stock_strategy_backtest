import numpy as np
# from utils.time_choose_util import log
from pyecharts.charts import Line, Scatter, Kline, Bar, Grid
from pyecharts import options as opts
import pandas as pd
from pyecharts.commons.utils import JsCode

signal_points = {
    '韩氏反弹': {'symbol': 'triangle', 'symbol_size': 9, 'color': '#7F7F7F'},  # 中灰色圆形
    # '大行情Spring': {'symbol': 'pentagon', 'symbol_size': 9, 'color': '#5F9EA0'},  # 晨雾蓝正三角
    # '韩氏抄底': {'symbol': 'triangle', 'symbol_size': 9, 'color': '#708090'},  # 石板灰菱形
    # '韩氏抄底起始': {'symbol': 'circle', 'symbol_size': 9, 'color': '#2F4F4F'},  # 深石板灰正方形
    # 'macd超短线卖出': {'symbol': 'pentagon', 'symbol_size': 9, 'color': 'black'},  # 板岩蓝倒三角
    # '动力线建仓': {'symbol': 'circle', 'symbol_size': 8, 'color': '#B0C4DE'},  # 淡钢蓝圆形
    # '极品抄底_看底': {'symbol': 'diamond', 'symbol_size': 8, 'color': '#778899'},  # 灰蓝色左三角
    # '年顶': {'symbol': 'diamond', 'symbol_size': 8, 'color': 'black'},  # 皇家蓝正三角
    # '抄底XG': {'symbol': 'pentagon', 'symbol_size': 12, 'color': '#6495ED'},  # 矢车菊蓝五边形
    # '老介抄底低位': {'symbol': 'circle', 'symbol_size': 12, 'color': '#778899'},  # 灰蓝色左三角
    # '股神黄金坑': {'symbol': 'triangle', 'symbol_size': 12, 'color': 'black'},  # 皇家蓝正三角
    # '股神黄金坑': {'symbol': 'pentagon', 'symbol_size': 8, 'color': '#6495ED'},  # 矢车菊蓝五边形
    # '极品抄底': {'symbol': 'triangle', 'symbol_size': 12, 'color': 'black'},  # 皇家蓝正三角
}


def draw_kline_with_volume_wyckoff(SZ_index_data, start_date, end_date, company_name, signal_points=signal_points):
    SZ_index_data.sort_index(inplace=True)
    # SZ_index_data = SZ_index_data.loc[start_date:end_date]
    # 获取日期和K线图所需的数据
    dates = SZ_index_data.index.tolist()
    open_values = SZ_index_data['open'].tolist()
    close_values = SZ_index_data['close'].tolist()
    low_values = SZ_index_data['low'].tolist()
    high_values = SZ_index_data['high'].tolist()
    volume_values = SZ_index_data['volume'].tolist()  # 假设DataFrame中有'volume'列

    # 创建K线图对象
    kline_chart = Kline()
    kline_chart.add_xaxis(dates)
    kline_chart.add_yaxis(
        "Kline",
        [[i, j, k, l, m] for i, j, k, l, m in zip(open_values, close_values, low_values, high_values, volume_values)],
        itemstyle_opts=opts.ItemStyleOpts(color="#ec0000", color0="#00da3c"),
    )

    # 创建淡色折线图对象
    line_chart = Line()
    line_chart.add_xaxis(dates)
    line_chart.add_yaxis(
        "Close",
        close_values,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(color="#a6a6a6"),
        symbol="none"
    )
    kline_chart.overlap(line_chart)

    # 添加 ice_line 和 supply_line
    # 确保必要的均线数据存在于数据集中
    if all(col in SZ_index_data.columns for col in ['ema_30']):
        # 获取均线数据
        ema_lines = {
            # "EMA 10": {"values": SZ_index_data['ema_10'].tolist(), "color": "blue"},
            "EMA 30": {"values": SZ_index_data['ema_30'].tolist(), "color": "#ff4500"},
            "EMA 60": {"values": SZ_index_data['ema_60'].tolist(), "color": "#8b00ff"},
            "EMA 200": {"values": SZ_index_data['ema_200'].tolist(), "color": "#32cd32"},
            # "ice_line": {"values": SZ_index_data['ice_line'].tolist(), "color": "#red"},
            # "supply_line": {"values": SZ_index_data['supply_line'].tolist(), "color": "green"}
        }

        # 遍历绘制均线
        for name, line_data in ema_lines.items():
            line = Line()
            line.add_xaxis(dates)
            line.add_yaxis(
                name,
                line_data["values"],
                linestyle_opts=opts.LineStyleOpts(color=line_data["color"], width=2, type_="dashed"),
                label_opts=opts.LabelOpts(is_show=False),
                symbol="none",
                is_smooth=True
            )
            kline_chart.overlap(line)
    # 循环添加信号点
    for signal, style in signal_points.items():
        signal_indices = SZ_index_data.index[SZ_index_data[signal] >= 1].tolist()
        signal_values = SZ_index_data.loc[signal_indices, 'close'].tolist()
        signal_indices = [SZ_index_data.index.get_loc(timestamp) for timestamp in signal_indices]

        scatter = Scatter()
        scatter.add_xaxis([dates[i] for i in signal_indices])
        scatter.add_yaxis(
            f"{signal} Signal",
            signal_values,
            symbol=style['symbol'],
            symbol_size=style['symbol_size'],
            color=style['color']
        )
        kline_chart.overlap(scatter)

    # 创建成交量的柱状图对象，根据价格涨跌变色
    volume_bar = Bar()
    volume_colors = ["#ec0000" if close > open else "#00da3c" for open, close in zip(open_values, close_values)]

    # Using JsCode to dynamically set color based on volume_colors array
    volume_bar.add_xaxis(dates)
    volume_bar.add_yaxis(
        "Volume",
        volume_values,
        gap="0%",
        itemstyle_opts=opts.ItemStyleOpts(
            color=JsCode(f"function(params) {{ return {volume_colors}[params.dataIndex]; }}")
        ),
        label_opts=opts.LabelOpts(is_show=False)
    )

    datazoom_opts = opts.DataZoomOpts(type_="slider", range_start=0, range_end=100, xaxis_index=[0])

    # 设置全局选项
    kline_chart.set_global_opts(
        datazoom_opts=datazoom_opts,
        yaxis_opts=opts.AxisOpts(is_scale=True),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=False,
            axisline_opts=opts.AxisLineOpts(is_on_zero=False),
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
    )

    volume_bar.set_global_opts(yaxis_opts=opts.AxisOpts(is_scale=True), datazoom_opts=datazoom_opts)

    # 合并K线图和成交量柱状图
    grid_chart = Grid(init_opts=opts.InitOpts(width="1500px", height="700px"))
    grid_chart.add(
        kline_chart,
        grid_opts=opts.GridOpts(pos_left="10%", pos_right="10%", height="50%"),
    )
    grid_chart.add(
        volume_bar,
        grid_opts=opts.GridOpts(pos_left="10%", pos_right="10%", pos_top="65%", height="20%"),
    )

    # 渲染图表
    grid_chart.render(
        f"/Users/hanguangyao/Downloads//predict_results/wyckoff_kline_{company_name}_{start_date}_{end_date}.html")
