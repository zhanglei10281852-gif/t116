# -*- coding: utf-8 -*-
"""
梯级水库滚动时域优化模块
实现基于滚动时域控制（Receding Horizon Control）的梯级水库优化调度
采用坐标下降法求解优化问题
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from ..core import Reservoir, CascadeSimulator


class CascadeOptimizer:
    """
    梯级水库滚动时域优化器
    基于滚动时域控制策略，在每个时段对未来预测时域内的下泄决策进行优化，
    仅执行第一步决策，然后滚动到下一时段重复优化过程
    """

    def __init__(self, reservoirs: List[Reservoir], inflow_forecast: Dict,
                 config: Optional[Dict] = None):
        """
        初始化优化器

        Parameters
        ----------
        reservoirs : list
            水库对象列表
        inflow_forecast : dict
            入库流量预报字典，格式参考 CascadeSimulator
        config : dict, optional
            优化配置参数，可包含:
            - prediction_horizon: 预测时域（小时），默认12小时
            - control_horizon: 控制时域（小时），默认3小时
            - alpha: 水位超限惩罚权重，默认1.0
            - beta: 出口洪峰惩罚权重，默认1.0
            - gamma: 上游过度拦蓄惩罚权重，默认0.3
            - flood_limit: 各水库汛限水位字典，格式 {"水库名": 水位(m)}，
              如未提供则使用最高水位的90%作为默认值
            - max_iterations: 坐标下降最大迭代次数，默认20
            - convergence_tol: 收敛容忍度，默认1e-6
            - search_density: 单变量搜索密度，默认20
        """
        self.reservoirs = sorted(reservoirs, key=lambda r: getattr(r, 'order', 0))
        self.reservoir_dict = {r.name: r for r in self.reservoirs}
        self.inflow_forecast = inflow_forecast
        self.time_step_hours = inflow_forecast.get('time_step_hours', 1)
        self.n_steps_total = inflow_forecast.get('n_steps', 0)

        default_config = {
            'prediction_horizon': 12,
            'control_horizon': 3,
            'alpha': 1.0,
            'beta': 1.0,
            'gamma': 0.3,
            'flood_limit': {},
            'max_iterations': 20,
            'convergence_tol': 1e-6,
            'search_density': 20,
        }

        if config is not None:
            default_config.update(config)

        self.prediction_horizon = default_config['prediction_horizon']
        self.control_horizon = default_config['control_horizon']
        self.alpha = default_config['alpha']
        self.beta = default_config['beta']
        self.gamma = default_config['gamma']
        self.flood_limit = default_config['flood_limit']
        self.max_iterations = default_config['max_iterations']
        self.convergence_tol = default_config['convergence_tol']
        self.search_density = default_config['search_density']

        self._init_flood_limits()
        self._init_simulator()

    def _init_flood_limits(self):
        """
        初始化各水库的汛限水位
        如未显式指定，则使用最高水位的90%作为默认值
        """
        for reservoir in self.reservoirs:
            name = reservoir.name
            if name not in self.flood_limit:
                max_level = reservoir.storage_to_level(reservoir.max_storage)
                dead_level = reservoir.storage_to_level(reservoir.dead_storage)
                self.flood_limit[name] = dead_level + 0.9 * (max_level - dead_level)

    def _init_simulator(self):
        """
        初始化梯级演算器，用于预测时域内的模拟
        """
        self.simulator = CascadeSimulator(
            reservoirs=self.reservoirs,
            inflow_forecast=self.inflow_forecast,
            time_step_hours=self.time_step_hours
        )

    def _objective(self, release_decisions: Dict[str, np.ndarray],
                   current_state: Dict[str, float], start_step: int,
                   horizon: int) -> float:
        """
        计算目标函数值

        目标函数 J 由三部分组成:
        J = α * Σ(max(level - flood_limit, 0)²)   # 水位超汛限惩罚
          + β * max(outlet_flow)²                  # 出口洪峰惩罚
          + γ * Σ(max(flood_limit - level, 0)²)    # 上游过度拦蓄惩罚

        Parameters
        ----------
        release_decisions : dict
            下泄决策字典，格式为 {"水库名": [r1, r2, ...]}，长度为horizon
        current_state : dict
            当前状态字典，包含各水库当前蓄水量 {"水库名": storage}
        start_step : int
            起始时段索引
        horizon : int
            优化时域长度（时段数）

        Returns
        -------
        float
            目标函数值（损失值）
        """
        self.simulator.initial_storage = current_state.copy()

        initial_release = {}
        for name in current_state:
            if name in release_decisions and len(release_decisions[name]) > 0:
                initial_release[name] = release_decisions[name][0]
            else:
                inflow = self.inflow_forecast.get('inflow', {}).get(name, [])
                if start_step < len(inflow):
                    initial_release[name] = inflow[start_step]
                else:
                    initial_release[name] = 0.0
        self.simulator.initial_release = initial_release

        result = self.simulator.simulate(
            release_scheme=release_decisions,
            start_step=start_step,
            n_steps=horizon
        )

        J1 = 0.0
        J3 = 0.0

        for reservoir in self.reservoirs:
            name = reservoir.name
            if name not in result:
                continue
            levels = result[name].get('level', np.array([]))
            flood_limit = self.flood_limit.get(name, 0.0)

            for level in levels:
                overflow = max(level - flood_limit, 0.0)
                J1 += overflow ** 2

                underflow = max(flood_limit - level, 0.0)
                J3 += underflow ** 2

        outlet_flow = result.get('outlet_flow', np.array([]))
        if len(outlet_flow) > 0:
            max_outlet = np.max(outlet_flow)
            J2 = max_outlet ** 2
        else:
            J2 = 0.0

        J = self.alpha * J1 + self.beta * J2 + self.gamma * J3

        return J

    def _project_to_feasible(self, release_prev: float, release_candidate: float,
                             reservoir: Reservoir) -> float:
        """
        将候选下泄值投影到可行域

        约束条件:
        1. min_release ≤ release ≤ max_release
        2. |release - release_prev| ≤ ramp_rate * dt

        Parameters
        ----------
        release_prev : float
            上一时段下泄流量(m³/s)
        release_candidate : float
            候选下泄流量(m³/s)
        reservoir : Reservoir
            水库对象

        Returns
        -------
        float
            投影后的可行下泄值(m³/s)
        """
        min_release = getattr(reservoir, 'min_release', 0.0)
        max_release = getattr(reservoir, 'max_release', np.inf)
        ramp_rate = getattr(reservoir, 'ramp_rate', np.inf)

        release = release_candidate

        release = max(min_release, min(release, max_release))

        max_change = ramp_rate * self.time_step_hours
        lower_bound = release_prev - max_change
        upper_bound = release_prev + max_change
        release = max(lower_bound, min(release, upper_bound))

        release = max(min_release, min(release, max_release))

        return release

    def _simulate_horizon(self, release_decisions: Dict[str, np.ndarray],
                          current_state: Dict[str, float], start_step: int,
                          horizon: int) -> Dict:
        """
        在预测时域内进行模拟，返回完整的演算结果

        Parameters
        ----------
        release_decisions : dict
            下泄决策字典
        current_state : dict
            当前状态字典
        start_step : int
            起始时段索引
        horizon : int
            时域长度

        Returns
        -------
        dict
            演算结果
        """
        self.simulator.initial_storage = current_state.copy()

        initial_release = {}
        for name in current_state:
            if name in release_decisions and len(release_decisions[name]) > 0:
                initial_release[name] = release_decisions[name][0]
            else:
                inflow = self.inflow_forecast.get('inflow', {}).get(name, [])
                if start_step < len(inflow):
                    initial_release[name] = inflow[start_step]
                else:
                    initial_release[name] = 0.0
        self.simulator.initial_release = initial_release

        return self.simulator.simulate(
            release_scheme=release_decisions,
            start_step=start_step,
            n_steps=horizon
        )

    def optimize_step(self, current_state: Dict[str, float],
                      start_step: int) -> Dict[str, float]:
        """
        单步优化：在预测时域内用坐标下降法找最优下泄决策

        坐标下降法流程:
        1. 初始化下泄决策（使用区间入流作为初始值）
        2. 迭代直至收敛:
           a. 对每个水库（按上游到下游顺序）:
              i. 固定其他水库的决策
              ii. 在控制时域内对该水库的每个时段进行一维搜索
              iii. 找到使目标函数最小的下泄值
           b. 检查目标函数变化是否小于收敛容忍度
        3. 返回第一步的最优下泄决策

        Parameters
        ----------
        current_state : dict
            当前状态字典，包含各水库当前蓄水量 {"水库名": storage}
        start_step : int
            起始时段索引

        Returns
        -------
        dict
            第一步的最优下泄决策 {"水库名": release}
        """
        horizon_steps = min(
            int(self.prediction_horizon / self.time_step_hours),
            self.n_steps_total - start_step
        )
        control_steps = min(
            int(self.control_horizon / self.time_step_hours),
            horizon_steps
        )

        if horizon_steps <= 0:
            return {r.name: 0.0 for r in self.reservoirs}

        release_decisions = {}
        prev_release = {}
        for reservoir in self.reservoirs:
            name = reservoir.name
            inflow = self.inflow_forecast.get('inflow', {}).get(name, [])

            decisions = np.zeros(horizon_steps, dtype=np.float64)
            for t in range(horizon_steps):
                step_idx = start_step + t
                if step_idx < len(inflow):
                    decisions[t] = inflow[step_idx]
                elif t > 0:
                    decisions[t] = decisions[t - 1]
                else:
                    decisions[t] = getattr(reservoir, 'min_release', 0.0)

            release_decisions[name] = decisions

            if start_step > 0:
                prev_inflow = inflow[start_step - 1] if start_step - 1 < len(inflow) else decisions[0]
                prev_release[name] = prev_inflow
            else:
                prev_release[name] = decisions[0]

        for t in range(1, horizon_steps):
            for reservoir in self.reservoirs:
                name = reservoir.name
                release_decisions[name][t] = self._project_to_feasible(
                    release_decisions[name][t - 1],
                    release_decisions[name][t],
                    reservoir
                )

        J_prev = self._objective(release_decisions, current_state, start_step, horizon_steps)

        for iteration in range(self.max_iterations):
            J_start = J_prev

            for reservoir in self.reservoirs:
                name = reservoir.name
                res = self.reservoir_dict[name]

                for t in range(control_steps):
                    if t == 0:
                        release_prev_t = prev_release[name]
                    else:
                        release_prev_t = release_decisions[name][t - 1]

                    min_release = getattr(res, 'min_release', 0.0)
                    max_release = getattr(res, 'max_release', np.inf)
                    ramp_rate = getattr(res, 'ramp_rate', np.inf)
                    max_change = ramp_rate * self.time_step_hours

                    search_lower = max(min_release, release_prev_t - max_change)
                    search_upper = min(max_release, release_prev_t + max_change)

                    if search_upper <= search_lower:
                        continue

                    best_release = release_decisions[name][t]
                    best_J = J_prev

                    search_points = np.linspace(search_lower, search_upper, self.search_density)

                    for candidate in search_points:
                        candidate = self._project_to_feasible(
                            release_prev_t, candidate, res
                        )

                        original_value = release_decisions[name][t]
                        release_decisions[name][t] = candidate

                        for t2 in range(t + 1, horizon_steps):
                            release_prev_t2 = release_decisions[name][t2 - 1]
                            release_decisions[name][t2] = self._project_to_feasible(
                                release_prev_t2, release_decisions[name][t2], res
                            )

                        J_curr = self._objective(
                            release_decisions, current_state, start_step, horizon_steps
                        )

                        if J_curr < best_J:
                            best_J = J_curr
                            best_release = candidate
                        else:
                            release_decisions[name][t] = original_value
                            for t2 in range(t + 1, horizon_steps):
                                release_prev_t2 = release_decisions[name][t2 - 1]
                                release_decisions[name][t2] = self._project_to_feasible(
                                    release_prev_t2, release_decisions[name][t2], res
                                )

                    release_decisions[name][t] = best_release
                    J_prev = best_J

            J_change = abs(J_start - J_prev)
            if J_change < self.convergence_tol * (1 + abs(J_start)):
                break

        first_step_decision = {}
        for reservoir in self.reservoirs:
            name = reservoir.name
            first_step_decision[name] = release_decisions[name][0]

        return first_step_decision

    def optimize(self, initial_release_scheme: Optional[Dict] = None) -> Dict:
        """
        全过程滚动优化

        从t=0开始，每步调用optimize_step得到当前时段最优下泄，
        执行该下泄更新状态，滚动到下一时段，重复直到结束。

        Parameters
        ----------
        initial_release_scheme : dict, optional
            初始下泄方案，作为优化的起点参考，格式为:
            {"水库名": [release1, release2, ...]}

        Returns
        -------
        dict
            优化结果字典，包含:
            - release_scheme: 最优下泄方案字典
            - simulation_result: 完整的演算结果
            - objective_value: 最终目标函数值
        """
        n_steps = self.n_steps_total

        opt_release_scheme = {}
        for reservoir in self.reservoirs:
            name = reservoir.name
            opt_release_scheme[name] = np.zeros(n_steps, dtype=np.float64)

        current_state = {}
        prev_release = {}
        for reservoir in self.reservoirs:
            name = reservoir.name
            current_state[name] = getattr(reservoir, 'initial_storage', reservoir.dead_storage)
            inflow = self.inflow_forecast.get('inflow', {}).get(name, [])
            if len(inflow) > 0:
                prev_release[name] = inflow[0]
            else:
                prev_release[name] = getattr(reservoir, 'min_release', 0.0)

        self.simulator.initial_storage = current_state.copy()
        self.simulator.initial_release = prev_release.copy()

        for step in range(n_steps):
            optimal_release = self.optimize_step(current_state, step)

            for reservoir in self.reservoirs:
                name = reservoir.name
                release = optimal_release.get(name, prev_release[name])
                release = self._project_to_feasible(
                    prev_release[name], release, self.reservoir_dict[name]
                )
                opt_release_scheme[name][step] = release
                prev_release[name] = release

            step_scheme = {name: [opt_release_scheme[name][step]] for name in opt_release_scheme}
            self.simulator.initial_storage = current_state.copy()
            self.simulator.initial_release = {
                name: opt_release_scheme[name][step] for name in opt_release_scheme
            }

            step_result = self.simulator.simulate(
                release_scheme=step_scheme,
                start_step=step,
                n_steps=1
            )

            for reservoir in self.reservoirs:
                name = reservoir.name
                if name in step_result and 'storage' in step_result[name]:
                    storages = step_result[name]['storage']
                    levels = step_result[name]['level']
                    if len(storages) > 0:
                        if step + 1 < n_steps:
                            total_inflow = step_result[name]['inflow_total'][0]
                            release = opt_release_scheme[name][step]
                            next_storage, _ = self.reservoir_dict[name].step(
                                current_state[name], total_inflow, release, self.time_step_hours
                            )
                            current_state[name] = next_storage
                        else:
                            current_state[name] = storages[-1]

        release_scheme_list = {}
        for name in opt_release_scheme:
            release_scheme_list[name] = opt_release_scheme[name].tolist()

        self.simulator.initial_storage = {
            r.name: getattr(r, 'initial_storage', r.dead_storage)
            for r in self.reservoirs
        }
        self.simulator.initial_release = {
            r.name: self.inflow_forecast.get('inflow', {}).get(r.name, [0])[0]
            for r in self.reservoirs
        }

        simulation_result = self.simulator.simulate(
            release_scheme=release_scheme_list,
            start_step=0,
            n_steps=n_steps
        )

        J_final = self._objective(
            opt_release_scheme,
            {r.name: r.dead_storage for r in self.reservoirs},
            0,
            n_steps
        )

        result = {
            'release_scheme': release_scheme_list,
            'simulation_result': simulation_result,
            'objective_value': J_final,
            'time_step_hours': self.time_step_hours,
            'n_steps': n_steps,
        }

        return result
