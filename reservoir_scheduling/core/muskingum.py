# -*- coding: utf-8 -*-
"""
马斯京根河道演进算法实现
采用numpy进行数值计算
"""

import numpy as np


class MuskingumRouter:
    """
    马斯京根河道演进类
    用于模拟河道洪水演进过程
    """

    def __init__(self, K, x, time_step_hours=1):
        """
        初始化马斯京根参数

        Parameters
        ----------
        K : float
            蓄量常数（小时），反映洪水波在河段中的传播时间
        x : float
            流量比重因素，无量纲，取值范围一般为0~0.5
        time_step_hours : float, optional
            计算时间步长（小时），默认为1小时
        """
        self.K = K
        self.x = x
        self.dt = time_step_hours
        self._compute_coefficients()

    def _compute_coefficients(self):
        """
        计算马斯京根演进系数C0, C1, C2
        保证C0 + C1 + C2 = 1

        系数公式:
            C0 = (0.5*dt - K*x) / (K - K*x + 0.5*dt)
            C1 = (0.5*dt + K*x) / (K - K*x + 0.5*dt)
            C2 = (K - K*x - 0.5*dt) / (K - K*x + 0.5*dt)
        """
        denominator = self.K - self.K * self.x + 0.5 * self.dt

        self.C0 = (0.5 * self.dt - self.K * self.x) / denominator
        self.C1 = (0.5 * self.dt + self.K * self.x) / denominator
        self.C2 = (self.K - self.K * self.x - 0.5 * self.dt) / denominator

        # 校验系数和为1
        coeff_sum = self.C0 + self.C1 + self.C2
        assert abs(coeff_sum - 1.0) < 1e-10, \
            f"系数和校验失败: C0+C1+C2 = {coeff_sum}，应为1.0"

    def route(self, inflow_series, initial_outflow=None):
        """
        对入流序列进行河道演进，返回出流序列

        演进公式:
            Q2 = C0*I2 + C1*I1 + C2*Q1

        其中:
            Q2 - 本时段出流量
            I2 - 本时段入流量
            I1 - 上时段入流量
            Q1 - 上时段出流量

        Parameters
        ----------
        inflow_series : numpy.ndarray
            入流序列数组，长度为n
        initial_outflow : float, optional
            初始出流量，如为None则使用inflow_series[0]初始化

        Returns
        -------
        numpy.ndarray
            出流序列数组，长度与入流序列相同
        """
        inflow = np.asarray(inflow_series, dtype=np.float64)
        n = len(inflow)

        if n == 0:
            return np.array([], dtype=np.float64)

        # 初始化出流序列
        outflow = np.zeros(n, dtype=np.float64)

        # 设置初始出流量
        if initial_outflow is None:
            outflow[0] = inflow[0]
        else:
            outflow[0] = initial_outflow

        # 逐时段演进计算
        for i in range(1, n):
            outflow[i] = (self.C0 * inflow[i] +
                          self.C1 * inflow[i - 1] +
                          self.C2 * outflow[i - 1])

        return outflow
