# !/usr/bin/python3
"""
Authors: hanguangyao
日志
"""
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import datetime

def setup_logger(log_dir="logs", log_level=logging.INFO):
    """
    设置并返回一个全局的 logger
    :param log_dir: 日志文件保存目录
    :param log_level: 日志级别（默认为 INFO）
    :return: logger 实例
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建 logger
    logger = logging.getLogger("BacktestLogger")

    logger.setLevel(log_level)

    # 创建日志格式
    log_format = logging.Formatter(
        "%(message)s"
    )

    # 创建控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # 创建文件 handler（按天轮换日志文件）
    log_file = os.path.join(log_dir, f"backtest_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=7
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger