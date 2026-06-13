# -*- coding: utf-8 -*-
"""
梯级水库联合调度系统
====================

一个用于梯级水库防洪调度的命令行工具，支持：
- simulate: 给定下泄方案进行梯级演算
- optimize: 自动优化下泄方案（滚动时域优化）
- check: 校验方案合规性
- report: 生成调度报告

用法示例：
    python -m reservoir_scheduling simulate --reservoirs examples/reservoirs.json \\
        --inflow examples/inflow_forecast.json \\
        --release examples/initial_release.json \\
        --output output/simulate_result.json
"""

__version__ = "1.0.0"
__author__ = "Reservoir Scheduling Team"

from .core import (
    MuskingumRouter,
    Reservoir,
    ConstraintChecker,
    Violation,
    CascadeSimulator,
)
from .optimize import CascadeOptimizer
from .report import ReportGenerator
from .utils import (
    load_reservoir_config,
    load_inflow_forecast,
    load_release_scheme,
    save_result,
    interpolate_storage_to_level,
    interpolate_level_to_storage,
)

__all__ = [
    "MuskingumRouter",
    "Reservoir",
    "ConstraintChecker",
    "Violation",
    "CascadeSimulator",
    "CascadeOptimizer",
    "ReportGenerator",
    "load_reservoir_config",
    "load_inflow_forecast",
    "load_release_scheme",
    "save_result",
    "interpolate_storage_to_level",
    "interpolate_level_to_storage",
]
