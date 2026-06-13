# -*- coding: utf-8 -*-
"""
约束检查模块
实现水库调度中的各种约束条件检查，包括水位、下泄流量、闸门爬升等约束
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Union
import numpy as np


@dataclass
class Violation:
    """
    约束违规记录类
    用于存储违反约束的详细信息
    """
    reservoir_name: str
    step_idx: int
    time_hours: int
    type: str
    value: float
    limit: float
    message: str


class ConstraintChecker:
    """
    约束检查器类
    对水库调度过程中的各种约束条件进行检查
    """

    def __init__(self, reservoirs: List, time_step_hours: int = 1):
        """
        初始化约束检查器

        Parameters
        ----------
        reservoirs : list
            水库对象列表，每个水库对象应包含以下属性:
            - name: 水库名称
            - dead_storage: 死库容(m³)
            - max_storage: 最大库容(m³)
            - max_release: 最大下泄能力(m³/s)
            - min_release: 最低生态基流(m³/s)
            - ramp_rate: 闸门爬升率(m³/s/h)
        time_step_hours : int, optional
            计算时间步长(小时)，默认为1小时
        """
        self.reservoirs = {r.name: r for r in reservoirs}
        self.time_step_hours = time_step_hours

    def _get_reservoir(self, reservoir_name: str):
        """
        根据名称获取水库对象

        Parameters
        ----------
        reservoir_name : str
            水库名称

        Returns
        -------
        Reservoir
            水库对象

        Raises
        ------
        ValueError
            当水库不存在时抛出异常
        """
        if reservoir_name not in self.reservoirs:
            raise ValueError(f"水库 '{reservoir_name}' 不存在")
        return self.reservoirs[reservoir_name]

    def check_water_level(self, reservoir_name: str, level: float,
                          step_idx: int) -> Optional[Violation]:
        """
        检查水位是否在[死水位, 最高水位]之间，以及是否超过汛限水位

        Parameters
        ----------
        reservoir_name : str
            水库名称
        level : float
            水位值(m)
        step_idx : int
            时段索引

        Returns
        -------
        Violation or None
            如果违反约束返回Violation对象，否则返回None
        """
        reservoir = self._get_reservoir(reservoir_name)

        dead_level = reservoir.storage_to_level(reservoir.dead_storage)
        max_level = reservoir.storage_to_level(reservoir.max_storage)
        flood_limit_level = getattr(reservoir, 'flood_limit_level', None)

        if level > max_level:
            return Violation(
                reservoir_name=reservoir_name,
                step_idx=step_idx,
                time_hours=step_idx * self.time_step_hours,
                type='water_level_high',
                value=level,
                limit=max_level,
                message=f"水库 {reservoir_name} 在时段 {step_idx} 水位 {level:.2f} m "
                        f"超过最高水位 {max_level:.2f} m，可能发生漫坝"
            )

        if level < dead_level:
            return Violation(
                reservoir_name=reservoir_name,
                step_idx=step_idx,
                time_hours=step_idx * self.time_step_hours,
                type='water_level_low',
                value=level,
                limit=dead_level,
                message=f"水库 {reservoir_name} 在时段 {step_idx} 水位 {level:.2f} m "
                        f"低于死水位 {dead_level:.2f} m"
            )

        if flood_limit_level is not None and level > flood_limit_level:
            return Violation(
                reservoir_name=reservoir_name,
                step_idx=step_idx,
                time_hours=step_idx * self.time_step_hours,
                type='flood_limit_exceeded',
                value=level,
                limit=flood_limit_level,
                message=f"水库 {reservoir_name} 在时段 {step_idx} 水位 {level:.2f} m "
                        f"超过汛限水位 {flood_limit_level:.2f} m"
            )

        return None

    def check_max_release(self, reservoir_name: str, release: float,
                          step_idx: int) -> Optional[Violation]:
        """
        检查下泄是否超过最大下泄能力

        Parameters
        ----------
        reservoir_name : str
            水库名称
        release : float
            下泄流量(m³/s)
        step_idx : int
            时段索引

        Returns
        -------
        Violation or None
            如果违反约束返回Violation对象，否则返回None
        """
        reservoir = self._get_reservoir(reservoir_name)
        max_release = getattr(reservoir, 'max_release', np.inf)

        if release > max_release:
            return Violation(
                reservoir_name=reservoir_name,
                step_idx=step_idx,
                time_hours=step_idx * self.time_step_hours,
                type='max_release',
                value=release,
                limit=max_release,
                message=f"水库 {reservoir_name} 在时段 {step_idx} 下泄流量 "
                        f"{release:.2f} m³/s 超过最大下泄能力 {max_release:.2f} m³/s"
            )

        return None

    def check_min_release(self, reservoir_name: str, release: float,
                          step_idx: int) -> Optional[Violation]:
        """
        检查是否满足最低生态基流

        Parameters
        ----------
        reservoir_name : str
            水库名称
        release : float
            下泄流量(m³/s)
        step_idx : int
            时段索引

        Returns
        -------
        Violation or None
            如果违反约束返回Violation对象，否则返回None
        """
        reservoir = self._get_reservoir(reservoir_name)
        min_release = getattr(reservoir, 'min_release', 0.0)

        if release < min_release:
            return Violation(
                reservoir_name=reservoir_name,
                step_idx=step_idx,
                time_hours=step_idx * self.time_step_hours,
                type='min_release',
                value=release,
                limit=min_release,
                message=f"水库 {reservoir_name} 在时段 {step_idx} 下泄流量 "
                        f"{release:.2f} m³/s 低于最低生态基流 {min_release:.2f} m³/s"
            )

        return None

    def check_ramp_rate(self, reservoir_name: str, release_prev: float,
                        release_curr: float, step_idx: int) -> Optional[Violation]:
        """
        检查闸门爬升约束：|release_curr - release_prev| <= ramp_rate * dt

        Parameters
        ----------
        reservoir_name : str
            水库名称
        release_prev : float
            上一时段下泄流量(m³/s)
        release_curr : float
            当前时段下泄流量(m³/s)
        step_idx : int
            时段索引

        Returns
        -------
        Violation or None
            如果违反约束返回Violation对象，否则返回None
        """
        reservoir = self._get_reservoir(reservoir_name)
        ramp_rate = getattr(reservoir, 'ramp_rate', np.inf)

        max_change = ramp_rate * self.time_step_hours
        actual_change = abs(release_curr - release_prev)

        if actual_change > max_change:
            return Violation(
                reservoir_name=reservoir_name,
                step_idx=step_idx,
                time_hours=step_idx * self.time_step_hours,
                type='ramp_rate',
                value=actual_change,
                limit=max_change,
                message=f"水库 {reservoir_name} 在时段 {step_idx} 闸门变幅 "
                        f"{actual_change:.2f} m³/s 超过最大允许变幅 {max_change:.2f} m³/s"
            )

        return None

    def check_all(self, result: Dict) -> List[Violation]:
        """
        对整个演算结果做全面检查

        Parameters
        ----------
        result : dict
            演算结果字典，包含各水库的水位、下泄等序列

        Returns
        -------
        list
            所有违反约束的Violation对象列表
        """
        violations = []

        time_steps = result.get('time_steps', [])
        if len(time_steps) == 0:
            return violations

        n_steps = len(time_steps)

        for res_name, reservoir in self.reservoirs.items():
            if res_name not in result:
                continue

            res_data = result[res_name]
            levels = res_data.get('level', [])
            releases = res_data.get('release', [])

            for i in range(n_steps):
                if i < len(levels):
                    violation = self.check_water_level(res_name, levels[i], i)
                    if violation:
                        violations.append(violation)

                if i < len(releases):
                    violation = self.check_max_release(res_name, releases[i], i)
                    if violation:
                        violations.append(violation)

                    violation = self.check_min_release(res_name, releases[i], i)
                    if violation:
                        violations.append(violation)

                    if i > 0 and i < len(releases):
                        violation = self.check_ramp_rate(
                            res_name, releases[i - 1], releases[i], i
                        )
                        if violation:
                            violations.append(violation)

        return violations
