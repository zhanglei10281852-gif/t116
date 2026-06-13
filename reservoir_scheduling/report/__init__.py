# -*- coding: utf-8 -*-
"""
水库调度报告模块
提供文本和HTML格式的调度结果报告生成功能
"""

from .generator import (
    ReportGenerator,
    format_flow,
    format_level,
    format_storage,
    get_level_color,
)

__all__ = [
    'ReportGenerator',
    'format_flow',
    'format_level',
    'format_storage',
    'get_level_color',
]
