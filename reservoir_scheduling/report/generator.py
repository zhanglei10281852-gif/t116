# -*- coding: utf-8 -*-
"""
报告生成器模块
实现水库调度结果的文本和HTML格式报告生成
"""

from typing import List, Dict, Optional
import numpy as np


def format_flow(q: float) -> str:
    """
    格式化流量

    Parameters
    ----------
    q : float
        流量值(m³/s)

    Returns
    -------
    str
        格式化后的流量字符串
    """
    return "%.1f m³/s" % q


def format_level(z: float) -> str:
    """
    格式化水位

    Parameters
    ----------
    z : float
        水位值(m)

    Returns
    -------
    str
        格式化后的水位字符串
    """
    return "%.2f m" % z


def format_storage(v: float) -> str:
    """
    格式化蓄水量

    Parameters
    ----------
    v : float
        蓄水量值(m³)

    Returns
    -------
    str
        格式化后的蓄水量字符串（单位：万m³）
    """
    return "%.1f 万m³" % (v / 10000.0)


def get_level_color(level: float, flood_limit: float, dead_level: float) -> str:
    """
    获取水位对应的CSS颜色类

    Parameters
    ----------
    level : float
        当前水位(m)
    flood_limit : float
        汛限水位(m)
    dead_level : float
        死水位(m)

    Returns
    -------
    str
        CSS颜色类名
    """
    if level > flood_limit:
        return 'level-danger'
    elif level < dead_level:
        return 'level-warning'
    else:
        return 'level-normal'


class ReportGenerator:
    """
    水库调度报告生成器
    生成文本和HTML格式的调度结果报告
    """

    def __init__(self, result: Dict, violations: Optional[List] = None,
                 reservoirs: Optional[List] = None):
        """
        初始化报告生成器

        Parameters
        ----------
        result : dict
            演算结果字典，包含：
            - time_steps: 时段索引数组
            - time_step_hours: 时间步长(小时)
            - {reservoir_name}: 各水库数据(inflow_total, release, storage, level等)
            - outlet_flow: 出口断面流量序列
        violations : list, optional
            违规记录列表，每个元素为Violation对象
        reservoirs : list, optional
            水库对象列表，每个元素为Reservoir对象
        """
        self.result = result
        self.violations = violations or []
        self.reservoirs = reservoirs or []
        self.reservoir_dict = {r.name: r for r in self.reservoirs}

        self.time_steps = result.get('time_steps', np.array([]))
        self.time_step_hours = result.get('time_step_hours', 1)
        self.n_steps = len(self.time_steps)

        self.statistics = self._compute_statistics()

    def _compute_statistics(self) -> Dict:
        """
        计算关键统计指标

        Returns
        -------
        dict
            统计字典，包含：
            - reservoirs: 各水库统计指标(最高水位、最低水位、超汛限时长、最大下泄)
            - outlet: 出口断面统计(洪峰流量、峰现时间、天然洪峰、削峰率)
        """
        stats = {
            'reservoirs': {},
            'outlet': {},
        }

        for res_name, reservoir in self.reservoir_dict.items():
            if res_name not in self.result:
                continue

            res_data = self.result[res_name]
            levels = res_data.get('level', np.array([]))
            releases = res_data.get('release', np.array([]))

            dead_level = reservoir.storage_to_level(reservoir.dead_storage)
            max_level = reservoir.storage_to_level(reservoir.max_storage)
            flood_limit = getattr(reservoir, 'flood_limit_level', max_level)

            if len(levels) > 0:
                max_level_val = np.max(levels)
                min_level_val = np.min(levels)
                max_level_step = int(np.argmax(levels))
                min_level_step = int(np.argmin(levels))
            else:
                max_level_val = 0.0
                min_level_val = 0.0
                max_level_step = 0
                min_level_step = 0

            over_flood_duration = 0
            for level in levels:
                if level > flood_limit:
                    over_flood_duration += self.time_step_hours

            if len(releases) > 0:
                max_release_val = np.max(releases)
                max_release_step = int(np.argmax(releases))
            else:
                max_release_val = 0.0
                max_release_step = 0

            stats['reservoirs'][res_name] = {
                'max_level': max_level_val,
                'max_level_time': max_level_step * self.time_step_hours,
                'max_level_step': max_level_step,
                'min_level': min_level_val,
                'min_level_time': min_level_step * self.time_step_hours,
                'min_level_step': min_level_step,
                'over_flood_duration': over_flood_duration,
                'max_release': max_release_val,
                'max_release_time': max_release_step * self.time_step_hours,
                'max_release_step': max_release_step,
                'flood_limit_level': flood_limit,
                'dead_level': dead_level,
            }

        outlet_flow = self.result.get('outlet_flow', np.array([]))
        if len(outlet_flow) > 0:
            peak_flow = np.max(outlet_flow)
            peak_step = int(np.argmax(outlet_flow))
            peak_time = peak_step * self.time_step_hours
        else:
            peak_flow = 0.0
            peak_step = 0
            peak_time = 0

        natural_peak = self._compute_natural_peak()

        if natural_peak > 0:
            flood_reduction_rate = (natural_peak - peak_flow) / natural_peak * 100.0
        else:
            flood_reduction_rate = 0.0

        stats['outlet'] = {
            'peak_flow': peak_flow,
            'peak_time': peak_time,
            'peak_step': peak_step,
            'natural_peak': natural_peak,
            'flood_reduction_rate': flood_reduction_rate,
        }

        return stats

    def _compute_natural_peak(self) -> float:
        """
        计算天然洪峰（无调控时的出口洪峰）

        近似方法：假设水库不调蓄，所有入库流量直接下泄，
        即 release = inflow_total，重新计算出口流量

        Returns
        -------
        float
            天然洪峰流量(m³/s)
        """
        if not self.reservoirs:
            outlet_flow = self.result.get('outlet_flow', np.array([]))
            return np.max(outlet_flow) if len(outlet_flow) > 0 else 0.0

        natural_outlet = np.zeros(self.n_steps, dtype=np.float64)

        downstream_res = self.reservoirs[-1]
        downstream_name = downstream_res.name

        for t in range(self.n_steps):
            total_inflow = 0.0
            for res_name, reservoir in self.reservoir_dict.items():
                if res_name not in self.result:
                    continue
                res_data = self.result[res_name]
                inflows = res_data.get('inflow_local', np.array([]))
                if t < len(inflows):
                    total_inflow += inflows[t]

            natural_outlet[t] = total_inflow

        return np.max(natural_outlet) if len(natural_outlet) > 0 else 0.0

    def generate_text(self, filepath: Optional[str] = None) -> str:
        """
        生成纯文本报告

        Parameters
        ----------
        filepath : str, optional
            输出文件路径，如为None则返回字符串

        Returns
        -------
        str
            纯文本报告内容
        """
        lines = []

        lines.append("=" * 70)
        lines.append("水库调度演算报告")
        lines.append("=" * 70)
        lines.append("")

        lines.append("一、调度概述")
        lines.append("-" * 70)
        lines.append(f"总时段数：{self.n_steps}")
        lines.append(f"时间步长：{self.time_step_hours} 小时")
        lines.append(f"总时长：{self.n_steps * self.time_step_hours} 小时")
        lines.append(f"水库数量：{len(self.reservoirs)}")
        lines.append("")

        lines.append("二、各水库水位过程表（每6小时）")
        lines.append("-" * 70)

        step_interval = max(1, int(6 // self.time_step_hours))

        for res_name, res_stats in self.statistics['reservoirs'].items():
            lines.append("")
            lines.append(f"水库：{res_name}")
            lines.append(f"  汛限水位：{format_level(res_stats['flood_limit_level'])}")
            lines.append(f"  死水位：{format_level(res_stats['dead_level'])}")
            lines.append("-" * 50)
            lines.append(f"{'时段':<8}{'时间(h)':<10}{'水位':<14}{'蓄水量':<16}{'下泄':<14}")
            lines.append("-" * 50)

            if res_name in self.result:
                res_data = self.result[res_name]
                levels = res_data.get('level', [])
                storages = res_data.get('storage', [])
                releases = res_data.get('release', [])

                for t in range(0, self.n_steps, step_interval):
                    time_h = t * self.time_step_hours
                    level = levels[t] if t < len(levels) else 0.0
                    storage = storages[t] if t < len(storages) else 0.0
                    release = releases[t] if t < len(releases) else 0.0

                    mark = ""
                    if level > res_stats['flood_limit_level']:
                        mark = " *"
                    elif level < res_stats['dead_level']:
                        mark = " #"

                    lines.append(
                        f"{t:<8}{time_h:<10}{format_level(level):<14}"
                        f"{format_storage(storage):<16}{format_flow(release):<14}{mark}"
                    )
            lines.append("")

        lines.append("三、关键时刻表")
        lines.append("-" * 70)
        lines.append(f"{'事件':<20}{'水库/断面':<15}{'时间(h)':<12}{'数值':<18}")
        lines.append("-" * 70)

        for res_name, res_stats in self.statistics['reservoirs'].items():
            lines.append(
                f"{'最高水位':<20}{res_name:<15}"
                f"{res_stats['max_level_time']:<12}"
                f"{format_level(res_stats['max_level']):<18}"
            )
            lines.append(
                f"{'最低水位':<20}{res_name:<15}"
                f"{res_stats['min_level_time']:<12}"
                f"{format_level(res_stats['min_level']):<18}"
            )
            lines.append(
                f"{'最大下泄':<20}{res_name:<15}"
                f"{res_stats['max_release_time']:<12}"
                f"{format_flow(res_stats['max_release']):<18}"
            )

        outlet_stats = self.statistics['outlet']
        lines.append(
            f"{'出口洪峰':<20}{'出口断面':<15}"
            f"{outlet_stats['peak_time']:<12}"
            f"{format_flow(outlet_stats['peak_flow']):<18}"
        )
        lines.append("")

        lines.append("四、削峰效果表")
        lines.append("-" * 70)
        lines.append(f"{'指标':<25}{'数值':<20}")
        lines.append("-" * 70)
        lines.append(f"{'天然洪峰流量':<25}{format_flow(outlet_stats['natural_peak']):<20}")
        lines.append(f"{'调度后洪峰流量':<25}{format_flow(outlet_stats['peak_flow']):<20}")
        lines.append(
            f"{'削峰率':<25}{'%.2f %%' % outlet_stats['flood_reduction_rate']:<20}"
        )
        lines.append("")

        if self.violations:
            lines.append("五、违规清单")
            lines.append("-" * 70)
            lines.append(f"{'序号':<6}{'水库':<12}{'时段':<8}{'时间(h)':<10}"
                         f"{'类型':<20}{'详情':<30}")
            lines.append("-" * 70)

            type_names = {
                'water_level_high': '漫坝风险',
                'water_level_low': '水位过低',
                'flood_limit_exceeded': '超汛限水位',
                'max_release': '超最大下泄',
                'min_release': '低于最小下泄',
                'ramp_rate': '超闸门变幅',
            }

            for i, v in enumerate(self.violations, 1):
                v_type = type_names.get(v.type, v.type)
                lines.append(
                    f"{i:<6}{v.reservoir_name:<12}{v.step_idx:<8}{v.time_hours:<10}"
                    f"{v_type:<20}{v.message:<30}"
                )
        else:
            lines.append("五、违规清单")
            lines.append("-" * 70)
            lines.append("本次调度无违规记录。")

        lines.append("")
        lines.append("=" * 70)
        lines.append("报告生成时间：")
        lines.append("说明：* 表示超汛限水位，# 表示低于死水位")
        lines.append("=" * 70)

        report_content = "\n".join(lines)

        if filepath is not None:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)

        return report_content

    def generate_html(self, filepath: Optional[str] = None) -> str:
        """
        生成HTML报告

        Parameters
        ----------
        filepath : str, optional
            输出文件路径，如为None则返回HTML字符串

        Returns
        -------
        str
            HTML报告内容
        """
        html_parts = []

        html_parts.append('<!DOCTYPE html>')
        html_parts.append('<html lang="zh-CN">')
        html_parts.append('<head>')
        html_parts.append('<meta charset="UTF-8">')
        html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html_parts.append('<title>水库调度演算报告</title>')
        html_parts.append('<style>')
        html_parts.append(self._get_css())
        html_parts.append('</style>')
        html_parts.append('</head>')
        html_parts.append('<body>')

        html_parts.append('<div class="container">')

        html_parts.append(self._generate_html_header())
        html_parts.append(self._generate_html_summary())
        html_parts.append(self._generate_html_reservoir_tables())
        html_parts.append(self._generate_html_outlet_table())
        html_parts.append(self._generate_html_key_metrics())
        html_parts.append(self._generate_html_violations())

        html_parts.append('</div>')

        html_parts.append('</body>')
        html_parts.append('</html>')

        html_content = "\n".join(html_parts)

        if filepath is not None:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

        return html_content

    def _get_css(self) -> str:
        """
        获取内联CSS样式

        Returns
        -------
        str
            CSS样式字符串
        """
        return """
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: "Microsoft YaHei", "SimHei", sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                color: #333;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: #fff;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: #fff;
                padding: 40px;
                text-align: center;
            }

            .header h1 {
                font-size: 32px;
                margin-bottom: 10px;
                letter-spacing: 2px;
            }

            .header p {
                font-size: 14px;
                opacity: 0.9;
            }

            .section {
                padding: 30px 40px;
                border-bottom: 1px solid #eee;
            }

            .section:last-child {
                border-bottom: none;
            }

            .section-title {
                font-size: 20px;
                font-weight: bold;
                color: #1e3c72;
                margin-bottom: 20px;
                padding-left: 15px;
                border-left: 4px solid #2a5298;
            }

            .summary-cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }

            .summary-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
            }

            .summary-card:hover {
                transform: translateY(-5px);
            }

            .summary-card .label {
                font-size: 13px;
                color: #666;
                margin-bottom: 8px;
            }

            .summary-card .value {
                font-size: 24px;
                font-weight: bold;
                color: #1e3c72;
            }

            .metrics-cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
            }

            .metric-card {
                background: #fff;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                padding: 25px;
                text-align: center;
                transition: all 0.3s ease;
            }

            .metric-card:hover {
                border-color: #2a5298;
                box-shadow: 0 8px 25px rgba(42, 82, 152, 0.2);
            }

            .metric-card.success {
                border-color: #4caf50;
                background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            }

            .metric-card.warning {
                border-color: #ff9800;
                background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            }

            .metric-card.danger {
                border-color: #f44336;
                background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            }

            .metric-card .metric-icon {
                font-size: 36px;
                margin-bottom: 10px;
            }

            .metric-card .metric-label {
                font-size: 14px;
                color: #666;
                margin-bottom: 8px;
            }

            .metric-card .metric-value {
                font-size: 28px;
                font-weight: bold;
                color: #333;
            }

            .metric-card .metric-sub {
                font-size: 12px;
                color: #999;
                margin-top: 5px;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                background: #fff;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            }

            thead {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: #fff;
            }

            th, td {
                padding: 12px 15px;
                text-align: center;
                border-bottom: 1px solid #eee;
            }

            th {
                font-weight: 600;
                font-size: 14px;
            }

            tbody tr:hover {
                background: #f5f7fa;
            }

            .level-normal {
                color: #4caf50;
                font-weight: 600;
            }

            .level-warning {
                color: #ff9800;
                font-weight: 600;
            }

            .level-danger {
                color: #f44336;
                font-weight: 600;
            }

            .reservoir-title {
                font-size: 16px;
                font-weight: bold;
                color: #2a5298;
                margin: 20px 0 10px 0;
                padding: 10px 15px;
                background: #e3f2fd;
                border-radius: 6px;
            }

            .reservoir-meta {
                display: flex;
                gap: 30px;
                margin-bottom: 15px;
                padding: 0 15px;
            }

            .reservoir-meta span {
                font-size: 13px;
                color: #666;
            }

            .reservoir-meta strong {
                color: #333;
            }

            .violations-section {
                background: #fff3e0;
            }

            .violation-card {
                background: #fff;
                border-left: 4px solid #ff9800;
                padding: 15px 20px;
                margin-bottom: 10px;
                border-radius: 4px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            }

            .violation-card.danger {
                border-left-color: #f44336;
                background: #ffebee;
            }

            .violation-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
            }

            .violation-type {
                font-weight: bold;
                color: #f44336;
            }

            .violation-time {
                font-size: 13px;
                color: #999;
            }

            .violation-message {
                font-size: 14px;
                color: #666;
            }

            .no-violations {
                text-align: center;
                padding: 30px;
                color: #4caf50;
                font-size: 16px;
            }

            .no-violations .icon {
                font-size: 48px;
                margin-bottom: 10px;
            }

            .legend {
                display: flex;
                gap: 20px;
                margin-top: 15px;
                padding: 10px 15px;
                background: #f5f5f5;
                border-radius: 6px;
            }

            .legend-item {
                display: flex;
                align-items: center;
                gap: 5px;
                font-size: 13px;
            }

            .legend-color {
                width: 16px;
                height: 16px;
                border-radius: 3px;
            }

            .legend-color.normal {
                background: #4caf50;
            }

            .legend-color.warning {
                background: #ff9800;
            }

            .legend-color.danger {
                background: #f44336;
            }

            .footer {
                text-align: center;
                padding: 20px;
                color: #999;
                font-size: 12px;
                background: #f5f5f5;
            }
        """

    def _generate_html_header(self) -> str:
        """
        生成HTML报告头部

        Returns
        -------
        str
            HTML头部内容
        """
        return """
            <div class="header">
                <h1>水库调度演算报告</h1>
                <p>Reservoir Operation Simulation Report</p>
            </div>
        """

    def _generate_html_summary(self) -> str:
        """
        生成HTML摘要卡片

        Returns
        -------
        str
            HTML摘要内容
        """
        parts = []
        parts.append('<div class="section">')
        parts.append('<div class="section-title">调度概述</div>')
        parts.append('<div class="summary-cards">')

        parts.append(f"""
            <div class="summary-card">
                <div class="label">总时段数</div>
                <div class="value">{self.n_steps}</div>
            </div>
        """)

        parts.append(f"""
            <div class="summary-card">
                <div class="label">时间步长</div>
                <div class="value">{self.time_step_hours} h</div>
            </div>
        """)

        parts.append(f"""
            <div class="summary-card">
                <div class="label">总时长</div>
                <div class="value">{self.n_steps * self.time_step_hours} h</div>
            </div>
        """)

        parts.append(f"""
            <div class="summary-card">
                <div class="label">水库数量</div>
                <div class="value">{len(self.reservoirs)}</div>
            </div>
        """)

        if self.violations:
            violation_count = len(self.violations)
            parts.append(f"""
                <div class="summary-card">
                    <div class="label">违规记录</div>
                    <div class="value" style="color: #f44336;">{violation_count} 条</div>
                </div>
            """)
        else:
            parts.append(f"""
                <div class="summary-card">
                    <div class="label">违规记录</div>
                    <div class="value" style="color: #4caf50;">无</div>
                </div>
            """)

        parts.append('</div>')
        parts.append('</div>')

        return "\n".join(parts)

    def _generate_html_reservoir_tables(self) -> str:
        """
        生成各水库水位过程表格

        Returns
        -------
        str
            HTML水库表格内容
        """
        parts = []
        parts.append('<div class="section">')
        parts.append('<div class="section-title">各水库水位过程（每6小时）</div>')

        step_interval = max(1, int(6 // self.time_step_hours))

        for res_name, res_stats in self.statistics['reservoirs'].items():
            flood_limit = res_stats['flood_limit_level']
            dead_level = res_stats['dead_level']

            parts.append(f'<div class="reservoir-title">{res_name}</div>')
            parts.append(f'''
                <div class="reservoir-meta">
                    <span>汛限水位：<strong>{format_level(flood_limit)}</strong></span>
                    <span>死水位：<strong>{format_level(dead_level)}</strong></span>
                </div>
            ''')

            parts.append('<table>')
            parts.append('<thead>')
            parts.append('<tr>')
            parts.append('<th>时段</th>')
            parts.append('<th>时间(h)</th>')
            parts.append('<th>水位</th>')
            parts.append('<th>蓄水量</th>')
            parts.append('<th>下泄</th>')
            parts.append('</tr>')
            parts.append('</thead>')
            parts.append('<tbody>')

            if res_name in self.result:
                res_data = self.result[res_name]
                levels = res_data.get('level', [])
                storages = res_data.get('storage', [])
                releases = res_data.get('release', [])

                for t in range(0, self.n_steps, step_interval):
                    time_h = t * self.time_step_hours
                    level = levels[t] if t < len(levels) else 0.0
                    storage = storages[t] if t < len(storages) else 0.0
                    release = releases[t] if t < len(releases) else 0.0

                    color_class = get_level_color(level, flood_limit, dead_level)

                    parts.append('<tr>')
                    parts.append(f'<td>{t}</td>')
                    parts.append(f'<td>{time_h}</td>')
                    parts.append(f'<td class="{color_class}">{format_level(level)}</td>')
                    parts.append(f'<td>{format_storage(storage)}</td>')
                    parts.append(f'<td>{format_flow(release)}</td>')
                    parts.append('</tr>')

            parts.append('</tbody>')
            parts.append('</table>')

        parts.append('''
            <div class="legend">
                <div class="legend-item">
                    <span class="legend-color normal"></span>
                    <span>正常水位</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color warning"></span>
                    <span>低于死水位</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color danger"></span>
                    <span>超汛限水位</span>
                </div>
            </div>
        ''')

        parts.append('</div>')

        return "\n".join(parts)

    def _generate_html_outlet_table(self) -> str:
        """
        生成出口流量过程表格

        Returns
        -------
        str
            HTML出口流量表格内容
        """
        outlet_flow = self.result.get('outlet_flow', np.array([]))
        if len(outlet_flow) == 0:
            return ""

        parts = []
        parts.append('<div class="section">')
        parts.append('<div class="section-title">出口断面流量过程（每6小时）</div>')

        step_interval = max(1, int(6 // self.time_step_hours))

        parts.append('<table>')
        parts.append('<thead>')
        parts.append('<tr>')
        parts.append('<th>时段</th>')
        parts.append('<th>时间(h)</th>')
        parts.append('<th>流量</th>')
        parts.append('</tr>')
        parts.append('</thead>')
        parts.append('<tbody>')

        peak_flow = self.statistics['outlet']['peak_flow']

        for t in range(0, self.n_steps, step_interval):
            time_h = t * self.time_step_hours
            flow = outlet_flow[t] if t < len(outlet_flow) else 0.0

            if abs(flow - peak_flow) < 0.01:
                flow_class = 'level-danger'
            else:
                flow_class = 'level-normal'

            parts.append('<tr>')
            parts.append(f'<td>{t}</td>')
            parts.append(f'<td>{time_h}</td>')
            parts.append(f'<td class="{flow_class}">{format_flow(flow)}</td>')
            parts.append('</tr>')

        parts.append('</tbody>')
        parts.append('</table>')
        parts.append('</div>')

        return "\n".join(parts)

    def _generate_html_key_metrics(self) -> str:
        """
        生成关键指标卡片

        Returns
        -------
        str
            HTML关键指标卡片内容
        """
        parts = []
        parts.append('<div class="section">')
        parts.append('<div class="section-title">关键指标</div>')
        parts.append('<div class="metrics-cards">')

        outlet_stats = self.statistics['outlet']

        parts.append(f'''
            <div class="metric-card">
                <div class="metric-icon">🌊</div>
                <div class="metric-label">天然洪峰流量</div>
                <div class="metric-value">{format_flow(outlet_stats['natural_peak'])}</div>
                <div class="metric-sub">无调控时的出口洪峰</div>
            </div>
        ''')

        parts.append(f'''
            <div class="metric-card">
                <div class="metric-icon">📊</div>
                <div class="metric-label">调度后洪峰流量</div>
                <div class="metric-value">{format_flow(outlet_stats['peak_flow'])}</div>
                <div class="metric-sub">峰现时间：{outlet_stats['peak_time']} h</div>
            </div>
        ''')

        reduction_rate = outlet_stats['flood_reduction_rate']
        if reduction_rate >= 30:
            card_class = 'success'
            icon = '✅'
        elif reduction_rate >= 10:
            card_class = 'warning'
            icon = '⚠️'
        else:
            card_class = 'danger'
            icon = '❌'

        parts.append(f'''
            <div class="metric-card {card_class}">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">削峰率</div>
                <div class="metric-value">{'%.2f %%' % reduction_rate}</div>
                <div class="metric-sub">削峰效果评价</div>
            </div>
        ''')

        for res_name, res_stats in self.statistics['reservoirs'].items():
            if res_stats['over_flood_duration'] > 0:
                card_class = 'danger'
            else:
                card_class = 'success'

            parts.append(f'''
                <div class="metric-card {card_class}">
                    <div class="metric-icon">🏞️</div>
                    <div class="metric-label">{res_name} 超汛限时长</div>
                    <div class="metric-value">{res_stats['over_flood_duration']} h</div>
                    <div class="metric-sub">最高水位：{format_level(res_stats['max_level'])}</div>
                </div>
            ''')

        parts.append('</div>')
        parts.append('</div>')

        return "\n".join(parts)

    def _generate_html_violations(self) -> str:
        """
        生成违规警告区域

        Returns
        -------
        str
            HTML违规警告内容
        """
        parts = []
        parts.append('<div class="section violations-section">')
        parts.append('<div class="section-title">违规警告</div>')

        if not self.violations:
            parts.append('''
                <div class="no-violations">
                    <div class="icon">✅</div>
                    <div>本次调度无违规记录</div>
                </div>
            ''')
        else:
            type_icons = {
                'water_level_high': '�',
                'water_level_low': '📉',
                'flood_limit_exceeded': '⚠️',
                'max_release': '💦',
                'min_release': '💧',
                'ramp_rate': '⚡',
            }

            type_names = {
                'water_level_high': '漫坝风险',
                'water_level_low': '水位过低',
                'flood_limit_exceeded': '超汛限水位',
                'max_release': '超最大下泄',
                'min_release': '低于最小下泄',
                'ramp_rate': '超闸门变幅',
            }

            for v in self.violations:
                icon = type_icons.get(v.type, '⚠️')
                v_type = type_names.get(v.type, v.type)
                card_class = 'danger' if 'high' in v.type or 'max' in v.type or 'flood' in v.type else ''

                parts.append(f'''
                    <div class="violation-card {card_class}">
                        <div class="violation-header">
                            <span class="violation-type">{icon} {v_type} - {v.reservoir_name}</span>
                            <span class="violation-time">时段 {v.step_idx} | {v.time_hours} h</span>
                        </div>
                        <div class="violation-message">{v.message}</div>
                    </div>
                ''')

        parts.append('</div>')

        parts.append('<div class="footer">')
        parts.append('水库调度演算报告系统 | Reservoir Operation Report System')
        parts.append('</div>')

        return "\n".join(parts)
