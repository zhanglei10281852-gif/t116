# -*- coding: utf-8 -*-
"""
插值计算模块
实现水位-蓄水量关系曲线的线性插值计算
"""

import numpy as np
from typing import List, Dict, Union, Any


def _validate_capacity_curve(capacity_curve: List[Dict[str, float]]) -> None:
    """
    内部函数：校验库容曲线格式

    Parameters
    ----------
    capacity_curve : List[Dict[str, float]]
        库容曲线，格式为 [{"level": x, "storage": y}, ...]

    Raises
    ------
    ValueError
        曲线格式错误时抛出
    """
    if not isinstance(capacity_curve, list):
        raise ValueError(
            f"库容曲线格式错误: 应为列表类型，实际为 {type(capacity_curve).__name__}"
        )

    if len(capacity_curve) < 2:
        raise ValueError(
            f"库容曲线点数不足: 至少需要2个点，实际为 {len(capacity_curve)} 个点"
        )

    levels = []
    storages = []

    for idx, point in enumerate(capacity_curve):
        if not isinstance(point, dict):
            raise ValueError(
                f"库容曲线第 {idx + 1} 个点格式错误: 应为字典类型，实际为 {type(point).__name__}"
            )

        if "level" not in point:
            raise ValueError(f"库容曲线第 {idx + 1} 个点缺少 'level' 字段")

        if "storage" not in point:
            raise ValueError(f"库容曲线第 {idx + 1} 个点缺少 'storage' 字段")

        try:
            level = float(point["level"])
            storage = float(point["storage"])
        except (ValueError, TypeError):
            raise ValueError(
                f"库容曲线第 {idx + 1} 个点数值无效: level={point['level']}, storage={point['storage']}"
            )

        if storage < 0:
            raise ValueError(
                f"库容曲线第 {idx + 1} 个点蓄水量不能为负: {storage}"
            )

        levels.append(level)
        storages.append(storage)

    level_diff = np.diff(levels)
    if np.any(level_diff <= 0):
        raise ValueError(
            "库容曲线的水位必须按严格递增顺序排列"
        )

    storage_diff = np.diff(storages)
    if np.any(storage_diff <= 0):
        raise ValueError(
            "库容曲线的蓄水量必须按严格递增顺序排列"
        )


def _extract_curve_arrays(
    capacity_curve: List[Dict[str, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """
    内部函数：从库容曲线中提取水位和蓄水量数组

    Parameters
    ----------
    capacity_curve : List[Dict[str, float]]
        库容曲线，格式为 [{"level": x, "storage": y}, ...]

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (水位数组, 蓄水量数组)
    """
    levels = np.array([float(p["level"]) for p in capacity_curve], dtype=np.float64)
    storages = np.array([float(p["storage"]) for p in capacity_curve], dtype=np.float64)
    return levels, storages


def interpolate_storage_to_level(
    capacity_curve: List[Dict[str, float]],
    storage: Union[float, List[float], np.ndarray]
) -> Union[float, np.ndarray]:
    """
    根据蓄水量查水位（线性插值）

    Parameters
    ----------
    capacity_curve : List[Dict[str, float]]
        库容曲线，格式为 [{"level": x, "storage": y}, ...]
    storage : Union[float, List[float], np.ndarray]
        蓄水量值(m³)，可以是单个值或数组

    Returns
    -------
    Union[float, np.ndarray]
        插值得到的水位值(m)，返回类型与输入一致

    Raises
    ------
    ValueError
        库容曲线格式错误或蓄水量超出范围时抛出
    """
    _validate_capacity_curve(capacity_curve)
    levels, storages = _extract_curve_arrays(capacity_curve)

    storage_arr = np.asarray(storage, dtype=np.float64)

    if np.any(storage_arr < storages[0]):
        raise ValueError(
            f"蓄水量小于库容曲线最小值: 最小值={storages[0]}, 输入={storage_arr.min()}"
        )

    if np.any(storage_arr > storages[-1]):
        raise ValueError(
            f"蓄水量大于库容曲线最大值: 最大值={storages[-1]}, 输入={storage_arr.max()}"
        )

    result = np.interp(storage_arr, storages, levels)

    if isinstance(storage, (int, float)):
        return float(result)
    return result


def interpolate_level_to_storage(
    capacity_curve: List[Dict[str, float]],
    level: Union[float, List[float], np.ndarray]
) -> Union[float, np.ndarray]:
    """
    根据水位查蓄水量（线性插值）

    Parameters
    ----------
    capacity_curve : List[Dict[str, float]]
        库容曲线，格式为 [{"level": x, "storage": y}, ...]
    level : Union[float, List[float], np.ndarray]
        水位值(m)，可以是单个值或数组

    Returns
    -------
    Union[float, np.ndarray]
        插值得到的蓄水量值(m³)，返回类型与输入一致

    Raises
    ------
    ValueError
        库容曲线格式错误或水位超出范围时抛出
    """
    _validate_capacity_curve(capacity_curve)
    levels, storages = _extract_curve_arrays(capacity_curve)

    level_arr = np.asarray(level, dtype=np.float64)

    if np.any(level_arr < levels[0]):
        raise ValueError(
            f"水位小于库容曲线最小值: 最小值={levels[0]}, 输入={level_arr.min()}"
        )

    if np.any(level_arr > levels[-1]):
        raise ValueError(
            f"水位大于库容曲线最大值: 最大值={levels[-1]}, 输入={level_arr.max()}"
        )

    result = np.interp(level_arr, levels, storages)

    if isinstance(level, (int, float)):
        return float(result)
    return result
