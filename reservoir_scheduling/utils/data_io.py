# -*- coding: utf-8 -*-
"""
数据IO模块
实现JSON格式的水库配置、入库预报、下泄方案的读写功能
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np


def _load_json(filepath: str) -> Any:
    """
    内部函数：加载JSON文件

    Parameters
    ----------
    filepath : str
        JSON文件路径

    Returns
    -------
    Any
        解析后的JSON数据

    Raises
    ------
    FileNotFoundError
        文件不存在时抛出
    ValueError
        文件为空或JSON格式错误时抛出
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    if os.path.getsize(filepath) == 0:
        raise ValueError(f"文件为空: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误: {filepath}, 错误信息: {str(e)}")


def _interp_level_to_storage(curve_points: List[List[float]], level: float) -> float:
    """
    从库容曲线根据水位插值计算蓄水量

    Parameters
    ----------
    curve_points : list
        库容曲线点列表 [[level1, storage1], [level2, storage2], ...]
    level : float
        水位值(m)

    Returns
    -------
    float
        蓄水量(m³)
    """
    levels = np.array([p[0] for p in curve_points], dtype=np.float64)
    storages = np.array([p[1] for p in curve_points], dtype=np.float64)
    return float(np.interp(level, levels, storages))


def load_reservoir_config(filepath: str) -> List[Dict[str, Any]]:
    """
    加载水库配置，返回水库列表

    支持两种配置格式：
    1. 简化格式（推荐）：
       {
         "reservoirs": [
           {
             "name": "水库A",
             "capacity_curve": [{"level": 100, "storage": 0}, {"level": 120, "storage": 1e7}],
             "flood_limit_level": 115.0,
             "dead_level": 105.0,
             "max_release": 800.0,
             "ramp_rate": 100.0,
             "min_release": 20.0,
             "initial_level": 113.0,
             "downstream_reservoir": "水库B",
             "travel_time_hours": 3,
             "muskingum_K": 3.0,
             "muskingum_x": 0.25
           }
         ]
       }

    2. 原始格式：
       [
         {
           "name": "水库A",
           "level_storage_curve": [[100, 0], [120, 1e7]],
           "dead_storage": 1e6,
           "max_storage": 1e7,
           ...
         }
       ]

    Parameters
    ----------
    filepath : str
        水库配置JSON文件路径

    Returns
    -------
    List[Dict[str, Any]]
        水库配置列表，每个元素为一个水库的配置字典，
        已转换为 Reservoir 类可直接使用的格式

    Raises
    ------
    ValueError
        数据不完整或格式错误时抛出
    """
    import numpy as np

    data = _load_json(filepath)

    if isinstance(data, dict) and "reservoirs" in data:
        reservoir_list = data["reservoirs"]
    elif isinstance(data, list):
        reservoir_list = data
    else:
        raise ValueError(
            f"水库配置格式错误: 顶层应为数组或包含'reservoirs'键的字典，"
            f"实际为 {type(data).__name__}"
        )

    if len(reservoir_list) == 0:
        raise ValueError("水库配置为空: 至少需要配置一个水库")

    reservoirs = []

    for idx, reservoir in enumerate(reservoir_list):
        if not isinstance(reservoir, dict):
            raise ValueError(f"第 {idx + 1} 个水库配置格式错误: 应为字典类型")

        if "name" not in reservoir:
            raise ValueError(f"第 {idx + 1} 个水库配置缺少必填字段: name")

        name = reservoir["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"第 {idx + 1} 个水库名称无效: {name}")

        converted = {"name": name}

        if "capacity_curve" in reservoir:
            cap_curve = reservoir["capacity_curve"]
            if not isinstance(cap_curve, list) or len(cap_curve) < 2:
                raise ValueError(
                    f"水库 {name} 的库容曲线无效: 至少需要2个点，"
                    f"实际为 {len(cap_curve) if isinstance(cap_curve, list) else type(cap_curve).__name__}"
                )
            ls_curve = []
            for i, point in enumerate(cap_curve):
                if not isinstance(point, dict) or "level" not in point or "storage" not in point:
                    raise ValueError(
                        f"水库 {name} 的库容曲线第 {i + 1} 个点格式错误: "
                        f"应为 {{'level': x, 'storage': y}} 格式"
                    )
                try:
                    lv = float(point["level"])
                    st = float(point["storage"])
                    ls_curve.append([lv, st])
                except (ValueError, TypeError):
                    raise ValueError(
                        f"水库 {name} 的库容曲线第 {i + 1} 个点数值无效: {point}"
                    )
            converted["level_storage_curve"] = ls_curve
        elif "level_storage_curve" in reservoir:
            curve = reservoir["level_storage_curve"]
            if not isinstance(curve, list) or len(curve) < 2:
                raise ValueError(
                    f"水库 {name} 的水位-蓄水量曲线无效: 至少需要2个点"
                )
            ls_curve = []
            for i, point in enumerate(curve):
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError(
                        f"水库 {name} 的水位-蓄水量曲线第 {i + 1} 个点格式错误: "
                        f"应为 [level, storage] 格式"
                    )
                try:
                    ls_curve.append([float(point[0]), float(point[1])])
                except (ValueError, TypeError):
                    raise ValueError(
                        f"水库 {name} 的水位-蓄水量曲线第 {i + 1} 个点数值无效: {point}"
                    )
            converted["level_storage_curve"] = ls_curve
        else:
            raise ValueError(
                f"水库 {name} 缺少库容曲线: 需要 capacity_curve 或 level_storage_curve"
            )

        ls_curve = converted["level_storage_curve"]
        for i in range(1, len(ls_curve)):
            if ls_curve[i][0] <= ls_curve[i-1][0]:
                raise ValueError(
                    f"水库 {name} 的库容曲线水位必须严格递增: "
                    f"第{i}点 {ls_curve[i-1][0]} -> 第{i+1}点 {ls_curve[i][0]}"
                )
            if ls_curve[i][1] <= ls_curve[i-1][1]:
                raise ValueError(
                    f"水库 {name} 的库容曲线蓄水量必须严格递增"
                )

        if "dead_storage" in reservoir and "max_storage" in reservoir:
            try:
                converted["dead_storage"] = float(reservoir["dead_storage"])
                converted["max_storage"] = float(reservoir["max_storage"])
            except (ValueError, TypeError):
                raise ValueError(
                    f"水库 {name} 的库容参数无效: "
                    f"dead_storage={reservoir.get('dead_storage')}, "
                    f"max_storage={reservoir.get('max_storage')}"
                )
        elif "dead_level" in reservoir:
            try:
                dead_level = float(reservoir["dead_level"])
                max_level = ls_curve[-1][0]
                converted["dead_storage"] = _interp_level_to_storage(ls_curve, dead_level)
                converted["max_storage"] = ls_curve[-1][1]
                converted["dead_level"] = dead_level
                converted["max_level"] = max_level
            except (ValueError, TypeError) as e:
                raise ValueError(f"水库 {name} 的死水位参数无效: {e}")
        else:
            raise ValueError(
                f"水库 {name} 缺少库容参数: 需要 dead_storage/max_storage 或 dead_level"
            )

        if "flood_limit_level" in reservoir:
            try:
                flood_limit = float(reservoir["flood_limit_level"])
                converted["flood_limit_level"] = flood_limit
                converted["flood_limit_storage"] = _interp_level_to_storage(ls_curve, flood_limit)
            except (ValueError, TypeError):
                raise ValueError(
                    f"水库 {name} 的汛限水位参数无效: {reservoir.get('flood_limit_level')}"
                )

        if "initial_level" in reservoir:
            try:
                initial_level = float(reservoir["initial_level"])
                converted["initial_level"] = initial_level
                converted["initial_storage"] = _interp_level_to_storage(ls_curve, initial_level)
            except (ValueError, TypeError):
                raise ValueError(
                    f"水库 {name} 的初始水位参数无效: {reservoir.get('initial_level')}"
                )

        if "initial_storage" in reservoir:
            try:
                converted["initial_storage"] = float(reservoir["initial_storage"])
            except (ValueError, TypeError):
                raise ValueError(
                    f"水库 {name} 的初始蓄水量参数无效: {reservoir.get('initial_storage')}"
                )

        if "max_release" in reservoir:
            converted["max_release"] = float(reservoir["max_release"])
        if "min_release" in reservoir:
            converted["min_release"] = float(reservoir["min_release"])
        if "ramp_rate" in reservoir:
            converted["ramp_rate"] = float(reservoir["ramp_rate"])
        if "order" in reservoir:
            converted["order"] = int(reservoir["order"])

        if "downstream_reservoir" in reservoir:
            converted["downstream"] = reservoir["downstream_reservoir"]
        elif "downstream" in reservoir:
            converted["downstream"] = reservoir["downstream"]

        if "muskingum_K" in reservoir:
            converted["muskingum_K"] = float(reservoir["muskingum_K"])
        if "muskingum_x" in reservoir:
            converted["muskingum_x"] = float(reservoir["muskingum_x"])
        if "travel_time_hours" in reservoir:
            converted["travel_time_hours"] = float(reservoir["travel_time_hours"])

        if converted["dead_storage"] < 0:
            raise ValueError(f"水库 {name} 的死库容不能为负: {converted['dead_storage']}")
        if converted["max_storage"] <= converted["dead_storage"]:
            raise ValueError(
                f"水库 {name} 的最大库容必须大于死库容: "
                f"max={converted['max_storage']}, dead={converted['dead_storage']}"
            )

        reservoirs.append(converted)

    return reservoirs


def load_inflow_forecast(filepath: str) -> Dict[str, Any]:
    """
    加载入库预报，校验时间连续性和缺失数据

    Parameters
    ----------
    filepath : str
        入库预报JSON文件路径

    Returns
    -------
    Dict[str, Any]
        入库预报数据，包含:
        - start_time: 开始时间(datetime对象)
        - time_step_hours: 时间步长(小时)
        - n_steps: 预报时段数
        - inflow: 各水库入库流量字典 {水库名: [流量数组]}
        - time_series: 时间序列列表

    Raises
    ------
    ValueError
        数据不完整、时间不连续或缺失数据时抛出
    """
    data = _load_json(filepath)

    if not isinstance(data, dict):
        raise ValueError(f"入库预报格式错误: 顶层应为字典，实际为 {type(data).__name__}")

    required_fields = ["start_time", "time_step_hours", "n_steps", "inflow"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"入库预报缺少必填字段: {field}")

    try:
        start_time = datetime.strptime(data["start_time"], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"开始时间格式错误: {data['start_time']}, 正确格式应为 'YYYY-MM-DD HH:MM:SS', 错误信息: {str(e)}"
        )

    try:
        time_step_hours = float(data["time_step_hours"])
    except (ValueError, TypeError):
        raise ValueError(f"时间步长无效: {data['time_step_hours']}")

    if time_step_hours <= 0:
        raise ValueError(f"时间步长必须为正数: {time_step_hours}")

    try:
        n_steps = int(data["n_steps"])
    except (ValueError, TypeError):
        raise ValueError(f"预报时段数无效: {data['n_steps']}")

    if n_steps <= 0:
        raise ValueError(f"预报时段数必须为正整数: {n_steps}")

    inflow = data["inflow"]
    if not isinstance(inflow, dict):
        raise ValueError(f"入库流量数据格式错误: 应为字典，实际为 {type(inflow).__name__}")

    if len(inflow) == 0:
        raise ValueError("入库流量数据为空: 至少需要一个水库的入库流量")

    inflow_data = {}
    for reservoir_name, flows in inflow.items():
        if not isinstance(flows, list):
            raise ValueError(f"水库 {reservoir_name} 的入库流量格式错误: 应为数组")

        if len(flows) != n_steps:
            raise ValueError(
                f"水库 {reservoir_name} 的入库流量长度与预报时段数不匹配: "
                f"实际长度 {len(flows)}, 期望长度 {n_steps}"
            )

        for i, flow in enumerate(flows):
            if flow is None:
                raise ValueError(
                    f"水库 {reservoir_name} 的入库流量存在缺失数据: 第 {i + 1} 个时段"
                )
            try:
                flow_val = float(flow)
            except (ValueError, TypeError):
                raise ValueError(
                    f"水库 {reservoir_name} 的入库流量数值无效: 第 {i + 1} 个时段, 值为 {flow}"
                )
            if flow_val < 0:
                raise ValueError(
                    f"水库 {reservoir_name} 的入库流量不能为负: 第 {i + 1} 个时段, 值为 {flow_val}"
                )

        inflow_data[reservoir_name] = [float(f) for f in flows]

    time_series = [start_time + timedelta(hours=i * time_step_hours) for i in range(n_steps)]

    for i in range(1, len(time_series)):
        expected_time = time_series[i - 1] + timedelta(hours=time_step_hours)
        if time_series[i] != expected_time:
            raise ValueError(
                f"时间序列不连续: 第 {i} 个时间点应为 {expected_time}, 实际为 {time_series[i]}"
            )

    return {
        "start_time": start_time,
        "time_step_hours": time_step_hours,
        "n_steps": n_steps,
        "inflow": inflow_data,
        "time_series": time_series,
    }


def load_release_scheme(filepath: str) -> Dict[str, List[float]]:
    """
    加载下泄方案

    Parameters
    ----------
    filepath : str
        下泄方案JSON文件路径

    Returns
    -------
    Dict[str, List[float]]
        各水库下泄流量字典 {水库名: [下泄流量数组]}

    Raises
    ------
    ValueError
        数据不完整或格式错误时抛出
    """
    data = _load_json(filepath)

    if not isinstance(data, dict):
        raise ValueError(f"下泄方案格式错误: 顶层应为字典，实际为 {type(data).__name__}")

    if "release" not in data:
        raise ValueError("下泄方案缺少必填字段: release")

    release = data["release"]
    if not isinstance(release, dict):
        raise ValueError(f"下泄流量数据格式错误: 应为字典，实际为 {type(release).__name__}")

    if len(release) == 0:
        raise ValueError("下泄方案为空: 至少需要一个水库的下泄流量")

    release_data = {}
    for reservoir_name, flows in release.items():
        if not isinstance(flows, list):
            raise ValueError(f"水库 {reservoir_name} 的下泄流量格式错误: 应为数组")

        if len(flows) == 0:
            raise ValueError(f"水库 {reservoir_name} 的下泄流量数组为空")

        for i, flow in enumerate(flows):
            if flow is None:
                raise ValueError(
                    f"水库 {reservoir_name} 的下泄流量存在缺失数据: 第 {i + 1} 个时段"
                )
            try:
                flow_val = float(flow)
            except (ValueError, TypeError):
                raise ValueError(
                    f"水库 {reservoir_name} 的下泄流量数值无效: 第 {i + 1} 个时段, 值为 {flow}"
                )
            if flow_val < 0:
                raise ValueError(
                    f"水库 {reservoir_name} 的下泄流量不能为负: 第 {i + 1} 个时段, 值为 {flow_val}"
                )

        release_data[reservoir_name] = [float(f) for f in flows]

    return release_data


def save_result(filepath: str, result: Dict[str, Any]) -> None:
    """
    保存演算结果为JSON

    Parameters
    ----------
    filepath : str
        结果保存路径
    result : Dict[str, Any]
        演算结果字典

    Raises
    ------
    ValueError
        结果格式错误时抛出
    """
    if not isinstance(result, dict):
        raise ValueError(f"演算结果格式错误: 应为字典，实际为 {type(result).__name__}")

    dir_path = os.path.dirname(filepath)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    def _convert_to_serializable(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, (list, tuple)):
            return [_convert_to_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {key: _convert_to_serializable(value) for key, value in obj.items()}
        return obj

    serializable_result = _convert_to_serializable(result)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
