# -*- coding: utf-8 -*-
"""
水库水量平衡计算模块
实现单个水库的水位-蓄水量转换和时段水量平衡计算
"""

import numpy as np


class Reservoir:
    """
    水库类
    实现水库的基本属性和水量平衡计算
    """

    def __init__(self, config):
        """
        从配置初始化水库

        Parameters
        ----------
        config : dict
            水库配置字典，包含以下键:
            - name: 水库名称
            - level_storage_curve: 水位-蓄水量关系曲线
              格式: [[level1, storage1], [level2, storage2], ...]
              其中level单位为米(m)，storage单位为立方米(m³)
            - dead_storage: 死库容(m³)
            - max_storage: 最大库容(m³)
            - max_release: 最大下泄能力(m³/s)，可选
            - min_release: 最低生态基流(m³/s)，可选
            - ramp_rate: 闸门爬升率(m³/s/h)，可选
            - order: 梯级顺序，可选
            - downstream: 下游水库名称，可选
            - muskingum_K: 马斯京根参数K(小时)，可选
            - muskingum_x: 马斯京根参数x，可选
        """
        self.name = config.get("name", "Unnamed Reservoir")
        self.dead_storage = config.get("dead_storage", 0.0)
        self.max_storage = config.get("max_storage", np.inf)

        # 约束参数
        self.max_release = config.get("max_release", np.inf)
        self.min_release = config.get("min_release", 0.0)
        self.ramp_rate = config.get("ramp_rate", np.inf)

        # 梯级关系参数
        self.order = config.get("order", 0)
        self.downstream = config.get("downstream", None)
        self.travel_time_hours = config.get("travel_time_hours", 1.0)

        # 马斯京根参数
        self.muskingum_K = config.get("muskingum_K", 1.0)
        self.muskingum_x = config.get("muskingum_x", 0.25)

        # 特征水位
        self.flood_limit_level = config.get("flood_limit_level", None)
        self.flood_limit_storage = config.get("flood_limit_storage", None)
        self.dead_level = config.get("dead_level", None)
        self.max_level = config.get("max_level", None)
        self.initial_level = config.get("initial_level", None)
        self.initial_storage = config.get("initial_storage", None)

        # 水位-蓄水量关系曲线
        ls_curve = np.asarray(config["level_storage_curve"], dtype=np.float64)
        self._levels = ls_curve[:, 0]
        self._storages = ls_curve[:, 1]

        # 校验曲线数据
        assert len(self._levels) == len(self._storages), \
            "水位和蓄水量数组长度不一致"
        assert len(self._levels) >= 2, \
            "水位-蓄水量曲线至少需要2个点"
        assert np.all(np.diff(self._levels) > 0), \
            "水位必须按递增顺序排列"
        assert np.all(np.diff(self._storages) > 0), \
            "蓄水量必须按递增顺序排列"

    def level_to_storage(self, level):
        """
        水位转换为蓄水量

        Parameters
        ----------
        level : float or numpy.ndarray
            水位值(m)

        Returns
        -------
        float or numpy.ndarray
            蓄水量(m³)
        """
        return np.interp(level, self._levels, self._storages)

    def storage_to_level(self, storage):
        """
        蓄水量转换为水位

        Parameters
        ----------
        storage : float or numpy.ndarray
            蓄水量(m³)

        Returns
        -------
        float or numpy.ndarray
            水位值(m)
        """
        return np.interp(storage, self._storages, self._levels)

    def step(self, current_storage, total_inflow, release, dt_hours=1):
        """
        单时段水量平衡计算

        水量平衡公式:
            V_next = V_current + (I - O) * 3600 * dt

        其中:
            V_next - 下一时段蓄水量(m³)
            V_current - 当前时段蓄水量(m³)
            I - 时段平均入库流量(m³/s)
            O - 时段平均出库流量(m³/s)
            dt - 计算时段长(h)
            3600 - 小时转秒的换算系数

        Parameters
        ----------
        current_storage : float
            当前蓄水量(m³)
        total_inflow : float
            时段平均入库流量(m³/s)
        release : float
            时段平均出库流量(m³/s)
        dt_hours : float, optional
            计算时间步长(小时)，默认为1小时

        Returns
        -------
        tuple
            (next_storage, next_level)
            next_storage: 下一时段蓄水量(m³)
            next_level: 下一时段水位(m)

        Raises
        ------
        ValueError
            当计算出的蓄水量为负时抛出异常
        """
        # 计算净入流量乘以时间（转换为秒）
        # dt_hours * 3600 将小时转换为秒
        inflow_volume = (total_inflow - release) * 3600 * dt_hours

        # 计算下一时段蓄水量
        next_storage = current_storage + inflow_volume

        # 计算下一时段水位
        next_level = self.storage_to_level(next_storage)

        return next_storage, next_level
