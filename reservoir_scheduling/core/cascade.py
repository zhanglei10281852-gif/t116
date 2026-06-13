# -*- coding: utf-8 -*-
"""
梯级水库联合演算模块
实现多水库梯级调度的逐时段滚动演算，包括河道演进和水量平衡计算
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from .reservoir import Reservoir
from .muskingum import MuskingumRouter


class CascadeSimulator:
    """
    梯级水库联合演算类
    实现梯级水库群的逐时段滚动演算
    """

    def __init__(self, reservoirs: List[Reservoir], inflow_forecast: Dict,
                 time_step_hours: int = 1, initial_storage: Optional[Dict] = None,
                 initial_release: Optional[Dict] = None):
        """
        初始化梯级水库演算器

        Parameters
        ----------
        reservoirs : list
            水库对象列表，每个水库对象应包含以下属性:
            - name: 水库名称
            - order: 水库在梯级中的顺序（从小到大，上游在前）
            - downstream: 下游水库名称（最下游水库为None）
            - muskingum_K: 马斯京根参数K（小时）
            - muskingum_x: 马斯京根参数x
        inflow_forecast : dict
            入库流量预报字典，格式为:
            {
                "time_step_hours": 1,
                "n_steps": 48,
                "inflow": {
                    "水库A": [inflow1, inflow2, ...],
                    "水库B": [inflow1, inflow2, ...],
                    ...
                }
            }
            其中inflow为区间入流序列(m³/s)
        time_step_hours : int, optional
            计算时间步长(小时)，默认为1小时
        initial_storage : dict, optional
            初始蓄水量字典，格式为 {"水库A": storage1, ...} (m³)
            如为None则使用各水库的死库容作为初始蓄水量
        initial_release : dict, optional
            初始下泄流量字典，格式为 {"水库A": release1, ...} (m³/s)
            如为None则使用各水库的区间入流作为初始下泄
        """
        self.time_step_hours = time_step_hours
        self.n_steps = inflow_forecast.get('n_steps', 0)
        self.inflow_forecast = inflow_forecast.get('inflow', {})

        self.reservoirs = sorted(reservoirs, key=lambda r: getattr(r, 'order', 0))
        self.reservoir_dict = {r.name: r for r in self.reservoirs}

        self._init_muskingum_routers()
        self._init_initial_conditions(initial_storage, initial_release)

    def _init_muskingum_routers(self):
        """
        初始化各水库的马斯京根演进器
        """
        self.muskingum_routers = {}
        for reservoir in self.reservoirs:
            K = getattr(reservoir, 'muskingum_K', 1.0)
            x = getattr(reservoir, 'muskingum_x', 0.25)
            self.muskingum_routers[reservoir.name] = MuskingumRouter(
                K=K, x=x, time_step_hours=self.time_step_hours
            )

    def _init_initial_conditions(self, initial_storage: Optional[Dict],
                                 initial_release: Optional[Dict]):
        """
        初始化初始条件（初始蓄水量和初始下泄）

        Parameters
        ----------
        initial_storage : dict or None
            初始蓄水量字典
        initial_release : dict or None
            初始下泄流量字典
        """
        self.initial_storage = {}
        self.initial_release = {}

        for reservoir in self.reservoirs:
            name = reservoir.name

            if initial_storage and name in initial_storage:
                self.initial_storage[name] = initial_storage[name]
            elif reservoir.initial_storage is not None:
                self.initial_storage[name] = reservoir.initial_storage
            else:
                self.initial_storage[name] = reservoir.dead_storage

            if initial_release and name in initial_release:
                self.initial_release[name] = initial_release[name]
            elif name in self.inflow_forecast and len(self.inflow_forecast[name]) > 0:
                self.initial_release[name] = self.inflow_forecast[name][0]
            else:
                self.initial_release[name] = getattr(reservoir, 'min_release', 0.0)

    def _get_upstream_reservoirs(self, reservoir_name: str) -> List[str]:
        """
        获取指定水库的所有上游水库名称列表

        Parameters
        ----------
        reservoir_name : str
            水库名称

        Returns
        -------
        list
            上游水库名称列表（按顺序排列）
        """
        upstream = []
        for reservoir in self.reservoirs:
            if getattr(reservoir, 'downstream', None) == reservoir_name:
                upstream.append(reservoir.name)
        return upstream

    def _route_single_step(self, router: MuskingumRouter, inflow_curr: float,
                           inflow_prev: float, outflow_prev: float) -> float:
        """
        单时段马斯京根演进计算

        Parameters
        ----------
        router : MuskingumRouter
            马斯京根演进器
        inflow_curr : float
            当前时段入流量(m³/s)
        inflow_prev : float
            上一时段入流量(m³/s)
        outflow_prev : float
            上一时段出流量(m³/s)

        Returns
        -------
        float
            当前时段出流量(m³/s)
        """
        outflow_curr = (router.C0 * inflow_curr +
                        router.C1 * inflow_prev +
                        router.C2 * outflow_prev)
        return outflow_curr

    def simulate(self, release_scheme: Dict, start_step: int = 0,
                 n_steps: Optional[int] = None) -> Dict:
        """
        核心逐时段滚动演算

        演算流程：
        1. 对每个时段t：
           a. 对每座水库（按上游到下游顺序）：
              i. 计算总入库 = 区间来水 + 上游演进后的来水
              ii. 进行水量平衡计算，得到下一时刻蓄水量和水位
              iii. 对下泄流量进行马斯京根演进，存入下游水库的入流序列

        Parameters
        ----------
        release_scheme : dict
            下泄方案字典，格式为:
            {
                "水库A": [release1, release2, ...],
                "水库B": [release1, release2, ...],
                ...
            }
            其中release为各时段下泄流量(m³/s)
        start_step : int, optional
            起始时段索引，默认为0
        n_steps : int, optional
            演算时段数，如为None则使用预报数据的总时段数

        Returns
        -------
        dict
            完整的演算结果字典，包含:
            - time_steps: 时段索引数组
            - {reservoir_name}: {
                'inflow_total': 总入流序列,
                'inflow_local': 区间入流序列,
                'inflow_upstream': 上游演进入流序列,
                'release': 下泄序列,
                'storage': 蓄水量序列,
                'level': 水位序列
              }
            - outlet_flow: 流域出口断面流量序列（最下游水库下泄）
        """
        if n_steps is None:
            n_steps = self.n_steps - start_step

        end_step = start_step + n_steps
        time_steps = np.arange(start_step, end_step)

        result = {
            'time_steps': time_steps,
            'time_step_hours': self.time_step_hours,
        }

        reservoir_data = {}
        current_storage = {}
        current_level = {}
        for reservoir in self.reservoirs:
            name = reservoir.name
            reservoir_data[name] = {
                'inflow_total': np.zeros(n_steps, dtype=np.float64),
                'inflow_local': np.zeros(n_steps, dtype=np.float64),
                'inflow_upstream': np.zeros(n_steps, dtype=np.float64),
                'release': np.zeros(n_steps, dtype=np.float64),
                'storage': np.zeros(n_steps, dtype=np.float64),
                'level': np.zeros(n_steps, dtype=np.float64),
            }
            current_storage[name] = self.initial_storage[name]
            current_level[name] = reservoir.storage_to_level(current_storage[name])

        routed_flow = {}
        for reservoir in self.reservoirs:
            name = reservoir.name
            routed_flow[name] = {
                'prev_inflow': self.initial_release[name],
                'prev_outflow': self.initial_release[name],
            }

        for t in range(n_steps):
            step_idx = start_step + t

            for reservoir in self.reservoirs:
                name = reservoir.name
                data = reservoir_data[name]
                local_inflows = self.inflow_forecast.get(name, [])
                releases = release_scheme.get(name, [])

                if step_idx < len(local_inflows):
                    data['inflow_local'][t] = local_inflows[step_idx]

                if step_idx < len(releases):
                    data['release'][t] = releases[step_idx]
                elif t > 0:
                    data['release'][t] = data['release'][t - 1]
                else:
                    data['release'][t] = self.initial_release[name]

                upstream_reservoirs = self._get_upstream_reservoirs(name)
                data['inflow_upstream'][t] = 0.0
                for up_res_name in upstream_reservoirs:
                    up_routed = routed_flow[up_res_name]

                    if up_res_name in release_scheme:
                        up_releases = release_scheme[up_res_name]
                        if step_idx < len(up_releases):
                            up_inflow_curr = up_releases[step_idx]
                        elif t > 0 and up_res_name in reservoir_data:
                            up_inflow_curr = reservoir_data[up_res_name]['release'][t - 1]
                        else:
                            up_inflow_curr = up_routed['prev_inflow']
                    else:
                        up_inflow_curr = up_routed['prev_inflow']

                    router = self.muskingum_routers[up_res_name]
                    up_outflow_curr = self._route_single_step(
                        router, up_inflow_curr,
                        up_routed['prev_inflow'], up_routed['prev_outflow']
                    )

                    data['inflow_upstream'][t] += up_outflow_curr
                    up_routed['prev_inflow'] = up_inflow_curr
                    up_routed['prev_outflow'] = up_outflow_curr

                data['inflow_total'][t] = data['inflow_local'][t] + data['inflow_upstream'][t]

                data['storage'][t] = current_storage[name]
                data['level'][t] = current_level[name]

                current_storage[name], current_level[name] = reservoir.step(
                    current_storage[name], data['inflow_total'][t],
                    data['release'][t], self.time_step_hours
                )

                curr_routed = routed_flow[name]
                curr_routed['prev_inflow'] = data['release'][t]
                router = self.muskingum_routers[name]
                curr_routed['prev_outflow'] = self._route_single_step(
                    router, data['release'][t],
                    curr_routed['prev_inflow'], curr_routed['prev_outflow']
                )

        for name, data in reservoir_data.items():
            result[name] = data

        downstream_reservoir = self.reservoirs[-1].name
        result['outlet_flow'] = result[downstream_reservoir]['release'].copy()

        return result
