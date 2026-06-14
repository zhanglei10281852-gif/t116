# -*- coding: utf-8 -*-
"""
CLI命令实现模块
实现四个子命令的业务逻辑：simulate、optimize、check、report
"""

import sys
import os
import json
from typing import Dict, List, Any, Optional
import numpy as np

from ..core import (
    Reservoir,
    CascadeSimulator,
    ConstraintChecker,
    Violation,
)
from ..optimize import CascadeOptimizer
from ..report import ReportGenerator
from ..utils import (
    load_reservoir_config,
    load_inflow_forecast,
    load_release_scheme,
    save_result,
)


def _print_error(message: str) -> None:
    """
    打印错误信息到标准错误输出

    Parameters
    ----------
    message : str
        错误信息
    """
    print(f"[错误] {message}", file=sys.stderr)


def _print_warning(message: str) -> None:
    """
    打印警告信息

    Parameters
    ----------
    message : str
        警告信息
    """
    print(f"[警告] {message}")


def _print_info(message: str) -> None:
    """
    打印信息

    Parameters
    ----------
    message : str
        信息内容
    """
    print(f"[信息] {message}")


def _create_reservoirs(config_list: List[Dict]) -> List[Reservoir]:
    """
    从配置列表创建水库对象列表

    Parameters
    ----------
    config_list : list
        水库配置列表

    Returns
    -------
    list
        水库对象列表
    """
    reservoirs = []
    for config in config_list:
        reservoir = Reservoir(config)
        reservoirs.append(reservoir)
    return sorted(reservoirs, key=lambda r: getattr(r, 'order', 0))


def _result_to_serializable(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    将演算结果转换为可序列化的格式

    Parameters
    ----------
    result : dict
        演算结果字典

    Returns
    -------
    dict
        可序列化的结果字典
    """
    serializable = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, dict):
            serializable[key] = _result_to_serializable(value)
        else:
            serializable[key] = value
    return serializable


def _save_release_scheme(filepath: str, scheme: Dict[str, List[float]]) -> None:
    """
    保存下泄方案到JSON文件

    Parameters
    ----------
    filepath : str
        保存路径
    scheme : dict
        下泄方案字典
    """
    dir_path = os.path.dirname(filepath)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    data = {"release": scheme}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_result(filepath: str) -> Dict[str, Any]:
    """
    加载演算结果JSON文件

    Parameters
    ----------
    filepath : str
        结果文件路径

    Returns
    -------
    dict
        演算结果字典

    Raises
    ------
    FileNotFoundError
        文件不存在时抛出
    ValueError
        文件格式错误时抛出
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误: {filepath}, 错误信息: {str(e)}")

    return data


def _load_violations(filepath: str) -> List[Violation]:
    """
    加载违规列表JSON文件

    Parameters
    ----------
    filepath : str
        违规列表文件路径

    Returns
    -------
    list
        Violation对象列表

    Raises
    ------
    FileNotFoundError
        文件不存在时抛出
    ValueError
        文件格式错误时抛出
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误: {filepath}, 错误信息: {str(e)}")

    violations = []
    if isinstance(data, list):
        for item in data:
            violation = Violation(
                reservoir=item.get("reservoir", ""),
                step_idx=item.get("step_idx", 0),
                time_hours=item.get("time_hours", 0),
                type=item.get("type", ""),
                value=float(item.get("value", 0.0)),
                limit=float(item.get("limit", 0.0)),
                message=item.get("message", ""),
            )
            violations.append(violation)

    return violations


def cmd_simulate(args) -> int:
    """
    执行水库调度演算

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数

    Returns
    -------
    int
        退出码，0表示成功，非0表示失败
    """
    try:
        _print_info("开始水库调度演算...")

        _print_info(f"加载水库配置: {args.reservoirs}")
        reservoir_configs = load_reservoir_config(args.reservoirs)
        reservoirs = _create_reservoirs(reservoir_configs)
        _print_info(f"成功加载 {len(reservoirs)} 座水库配置")

        _print_info(f"加载入库预报: {args.inflow}")
        inflow_forecast = load_inflow_forecast(args.inflow)
        _print_info(f"成功加载入库预报，共 {inflow_forecast['n_steps']} 个时段")

        _print_info(f"加载下泄方案: {args.release}")
        release_scheme = load_release_scheme(args.release)
        _print_info(f"成功加载 {len(release_scheme)} 座水库的下泄方案")

        _print_info("创建梯级演算器...")
        simulator = CascadeSimulator(
            reservoirs=reservoirs,
            inflow_forecast=inflow_forecast,
            time_step_hours=inflow_forecast["time_step_hours"],
        )

        _print_info("执行逐时段演算...")
        result = simulator.simulate(release_scheme)
        _print_info("演算完成")

        _print_info("检查约束条件...")
        checker = ConstraintChecker(
            reservoirs=reservoirs,
            time_step_hours=inflow_forecast["time_step_hours"],
        )
        violations = checker.check_all(result)
        _print_info(f"约束检查完成，发现 {len(violations)} 条违规")

        serializable_result = _result_to_serializable(result)
        serializable_result["violations"] = [
            {
                "reservoir_name": v.reservoir_name,
                "step_idx": v.step_idx,
                "time_hours": v.time_hours,
                "type": v.type,
                "value": v.value,
                "limit": v.limit,
                "message": v.message,
            }
            for v in violations
        ]

        _print_info(f"保存结果到: {args.output}")
        save_result(args.output, serializable_result)
        _print_info("结果保存成功")

        print("\n" + "=" * 70)
        print("演算摘要")
        print("=" * 70)

        for reservoir in reservoirs:
            name = reservoir.name
            if name in result:
                levels = result[name]["level"]
                max_level = float(np.max(levels))
                print(f"  {name}: 最高水位 = {max_level:.2f} m")

        outlet_flow = result.get("outlet_flow", [])
        if len(outlet_flow) > 0:
            peak_flow = float(np.max(outlet_flow))
            print(f"\n  流域出口洪峰: {peak_flow:.2f} m³/s")

        print(f"  约束违规数量: {len(violations)} 条")

        if len(violations) > 0:
            _print_warning(f"存在 {len(violations)} 条约束违规，请查看结果文件获取详情")

        print("=" * 70 + "\n")
        _print_info("演算任务完成")

        return 0

    except FileNotFoundError as e:
        _print_error(str(e))
        return 1
    except ValueError as e:
        _print_error(f"数据格式错误: {e}")
        return 1
    except Exception as e:
        _print_error(f"演算失败: {str(e)}")
        return 1


def cmd_optimize(args) -> int:
    """
    执行水库调度优化（滚动时域优化）

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数

    Returns
    -------
    int
        退出码，0表示成功，非0表示失败
    """
    try:
        _print_info("开始水库调度优化（滚动时域优化）...")

        _print_info(f"加载水库配置: {args.reservoirs}")
        reservoir_configs = load_reservoir_config(args.reservoirs)
        reservoirs = _create_reservoirs(reservoir_configs)
        _print_info(f"成功加载 {len(reservoirs)} 座水库配置")

        _print_info(f"加载入库预报: {args.inflow}")
        inflow_forecast = load_inflow_forecast(args.inflow)
        _print_info(f"成功加载入库预报，共 {inflow_forecast['n_steps']} 个时段")

        initial_scheme = None
        if args.initial:
            _print_info(f"加载初始方案: {args.initial}")
            initial_scheme = load_release_scheme(args.initial)
            _print_info("成功加载初始下泄方案")
        else:
            _print_info("未提供初始方案，将使用默认方案（各库按最小下泄起步）")

        _print_info("创建滚动时域优化器...")
        opt_config = {
            'prediction_horizon': getattr(args, 'prediction_horizon', 12),
            'control_horizon': getattr(args, 'control_horizon', 3),
            'alpha': getattr(args, 'alpha', 1.0),
            'beta': getattr(args, 'beta', 1.0),
            'gamma': getattr(args, 'gamma', 0.3),
            'max_iterations': getattr(args, 'max_iterations', 20),
            'search_density': getattr(args, 'search_density', 10),
        }
        _print_info(f"  预测时域: {opt_config['prediction_horizon']} h")
        _print_info(f"  控制时域: {opt_config['control_horizon']} h")
        _print_info(f"  水位超限惩罚权重 α: {opt_config['alpha']}")
        _print_info(f"  出口洪峰惩罚权重 β: {opt_config['beta']}")
        _print_info(f"  过度拦蓄惩罚权重 γ: {opt_config['gamma']}")

        optimizer = CascadeOptimizer(
            reservoirs=reservoirs,
            inflow_forecast=inflow_forecast,
            config=opt_config,
        )

        _print_info("执行滚动时域优化...")
        _print_info("  （正在平衡当前削峰与未来风险，避免上游占满库容导致后期被迫集中下泄）")
        optimize_result = optimizer.optimize(initial_release_scheme=initial_scheme)
        _print_info("优化完成")

        best_scheme = optimize_result["release_scheme"]
        best_result = optimize_result["simulation_result"]

        _print_info("计算初始方案洪峰用于对比...")
        if initial_scheme is None:
            initial_simulator = CascadeSimulator(reservoirs, inflow_forecast)
            default_scheme = {}
            for r in reservoirs:
                inflow_list = inflow_forecast.get('inflow', {}).get(r.name, [])
                default_scheme[r.name] = [
                    inflow_list[i] if i < len(inflow_list) else getattr(r, 'min_release', 0.0)
                    for i in range(inflow_forecast['n_steps'])
                ]
            initial_result = initial_simulator.simulate(default_scheme)
        else:
            initial_simulator = CascadeSimulator(reservoirs, inflow_forecast)
            initial_result = initial_simulator.simulate(initial_scheme)
        initial_peak = float(np.max(initial_result['outlet_flow']))
        final_peak = float(np.max(best_result['outlet_flow']))
        peak_reduction = initial_peak - final_peak
        peak_reduction_rate = peak_reduction / initial_peak if initial_peak > 0 else 0.0

        _print_info("检查约束条件...")
        checker = ConstraintChecker(
            reservoirs=reservoirs,
            time_step_hours=inflow_forecast["time_step_hours"],
        )
        violations = checker.check_all(best_result)
        _print_info(f"约束检查完成，发现 {len(violations)} 条违规")

        _print_info(f"保存最优下泄方案到: {args.output_scheme}")
        _save_release_scheme(args.output_scheme, best_scheme)
        _print_info("下泄方案保存成功")

        serializable_result = _result_to_serializable(best_result)
        serializable_result["violations"] = [
            {
                "reservoir_name": v.reservoir_name,
                "step_idx": v.step_idx,
                "time_hours": v.time_hours,
                "type": v.type,
                "value": v.value,
                "limit": v.limit,
                "message": v.message,
            }
            for v in violations
        ]
        serializable_result["optimization_summary"] = {
            "prediction_horizon": opt_config['prediction_horizon'],
            "control_horizon": opt_config['control_horizon'],
            "alpha": opt_config['alpha'],
            "beta": opt_config['beta'],
            "gamma": opt_config['gamma'],
            "initial_peak": initial_peak,
            "final_peak": final_peak,
            "peak_reduction": peak_reduction,
            "peak_reduction_rate": peak_reduction_rate,
            "objective_value": float(optimize_result.get("objective_value", 0.0)),
        }

        _print_info(f"保存优化结果到: {args.output_result}")
        save_result(args.output_result, serializable_result)
        _print_info("优化结果保存成功")

        print("\n" + "=" * 70)
        print("优化摘要")
        print("=" * 70)
        print(f"  初始方案洪峰: {initial_peak:.2f} m³/s")
        print(f"  优化后洪峰: {final_peak:.2f} m³/s")
        print(f"  洪峰削减量: {peak_reduction:.2f} m³/s")
        print(f"  洪峰削减率: {peak_reduction_rate * 100:.2f}%")
        print(f"  目标函数值: {optimize_result.get('objective_value', 0.0):.2f}")

        print("\n  各水库最高水位:")
        for reservoir in reservoirs:
            name = reservoir.name
            if name in best_result:
                levels = best_result[name]["level"]
                max_level = float(np.max(levels))
                flood_limit = getattr(reservoir, 'flood_limit_level', None)
                mark = ""
                if flood_limit and max_level > flood_limit:
                    mark = " *"
                print(f"    {name}: {max_level:.2f} m (汛限: {flood_limit:.2f} m){mark}")

        print(f"\n  约束违规数量: {len(violations)} 条")

        if len(violations) > 0:
            _print_warning(f"存在 {len(violations)} 条约束违规，请查看结果文件获取详情")

        print("=" * 70 + "\n")
        _print_info("优化任务完成")

        return 0

    except FileNotFoundError as e:
        _print_error(str(e))
        return 1
    except ValueError as e:
        _print_error(f"数据格式错误: {e}")
        return 1
    except Exception as e:
        _print_error(f"优化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_check(args) -> int:
    """
    检查约束条件

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数

    Returns
    -------
    int
        退出码，0表示无违规，1表示有违规或错误
    """
    try:
        _print_info("开始约束检查...")

        _print_info(f"加载水库配置: {args.reservoirs}")
        reservoir_configs = load_reservoir_config(args.reservoirs)
        reservoirs = _create_reservoirs(reservoir_configs)
        _print_info(f"成功加载 {len(reservoirs)} 座水库配置")

        _print_info(f"加载入库预报: {args.inflow}")
        inflow_forecast = load_inflow_forecast(args.inflow)
        _print_info(f"成功加载入库预报，共 {inflow_forecast['n_steps']} 个时段")

        _print_info(f"加载下泄方案: {args.release}")
        release_scheme = load_release_scheme(args.release)
        _print_info(f"成功加载 {len(release_scheme)} 座水库的下泄方案")

        _print_info("创建梯级演算器...")
        simulator = CascadeSimulator(
            reservoirs=reservoirs,
            inflow_forecast=inflow_forecast,
            time_step_hours=inflow_forecast["time_step_hours"],
        )

        _print_info("执行逐时段演算...")
        result = simulator.simulate(release_scheme)
        _print_info("演算完成")

        _print_info("全面检查约束条件...")
        checker = ConstraintChecker(
            reservoirs=reservoirs,
            time_step_hours=inflow_forecast["time_step_hours"],
        )
        violations = checker.check_all(result)
        _print_info(f"约束检查完成，发现 {len(violations)} 条违规")

        print("\n" + "=" * 70)
        print("约束违规清单")
        print("=" * 70)

        if len(violations) == 0:
            print("  ✓ 无约束违规")
        else:
            print(f"  ✗ 共发现 {len(violations)} 条约束违规:\n")
            for i, v in enumerate(violations, 1):
                time_str = f"T+{v.time_hours}h (时段 {v.step_idx})"
                print(f"  {i}. [{time_str}] {v.reservoir_name}")
                print(f"     违规类型: {v.type}")
                print(f"     违规数值: {v.value:.2f}")
                print(f"     约束限值: {v.limit:.2f}")
                print(f"     详情: {v.message}")
                if i < len(violations):
                    print()

        print("\n" + "=" * 70)
        print("各项指标")
        print("=" * 70)

        inflow_data = inflow_forecast.get("inflow", {})
        downstream_reservoir = reservoirs[-1].name if len(reservoirs) > 0 else None

        for reservoir in reservoirs:
            name = reservoir.name
            if name in result:
                levels = result[name]["level"]
                max_level = float(np.max(levels))
                min_level = float(np.min(levels))
                releases = result[name]["release"]
                max_release = float(np.max(releases))
                min_release = float(np.min(releases))

                print(f"\n  【{name}】")
                print(f"    最高水位: {max_level:.2f} m")
                print(f"    最低水位: {min_level:.2f} m")
                print(f"    最大下泄: {max_release:.2f} m³/s")
                print(f"    最小下泄: {min_release:.2f} m³/s")

        outlet_flow = result.get("outlet_flow", [])
        if len(outlet_flow) > 0:
            peak_flow = float(np.max(outlet_flow))
            avg_flow = float(np.mean(outlet_flow))

            if downstream_reservoir and downstream_reservoir in inflow_data:
                downstream_inflows = inflow_data[downstream_reservoir]
                total_upstream_inflow = 0
                for res_name, inflows in inflow_data.items():
                    if res_name != downstream_reservoir:
                        total_upstream_inflow += sum(inflows)

                natural_peak = peak_flow
                if len(inflow_data) > 1:
                    combined_inflow = np.zeros(len(outlet_flow))
                    for res_name, inflows in inflow_data.items():
                        if len(inflows) == len(combined_inflow):
                            combined_inflow += np.array(inflows)
                    natural_peak = float(np.max(combined_inflow))

                peak_reduction_rate = (natural_peak - peak_flow) / natural_peak * 100 if natural_peak > 0 else 0

                print(f"\n  【流域出口】")
                print(f"    洪峰流量: {peak_flow:.2f} m³/s")
                print(f"    平均流量: {avg_flow:.2f} m³/s")
                print(f"    自然洪峰: {natural_peak:.2f} m³/s")
                print(f"    削峰率: {peak_reduction_rate:.2f}%")

        print("\n" + "=" * 70)

        if args.output:
            serializable_result = _result_to_serializable(result)
            serializable_result["violations"] = [
                {
                    "reservoir_name": v.reservoir_name,
                    "step_idx": v.step_idx,
                    "time_hours": v.time_hours,
                    "type": v.type,
                    "value": v.value,
                    "limit": v.limit,
                    "message": v.message,
                }
                for v in violations
            ]
            _print_info(f"保存检查结果到: {args.output}")
            save_result(args.output, serializable_result)
            _print_info("检查结果保存成功")

        if len(violations) > 0:
            _print_warning(f"检查完成，存在 {len(violations)} 条约束违规")
            return 1
        else:
            _print_info("检查完成，无约束违规")
            return 0

    except FileNotFoundError as e:
        _print_error(str(e))
        return 1
    except ValueError as e:
        _print_error(f"数据格式错误: {e}")
        return 1
    except Exception as e:
        _print_error(f"检查失败: {str(e)}")
        return 1


def cmd_report(args) -> int:
    """
    生成调度报告

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数

    Returns
    -------
    int
        退出码，0表示成功，非0表示失败
    """
    try:
        _print_info("开始生成报告...")

        _print_info(f"加载演算结果: {args.input}")
        result = _load_result(args.input)
        _print_info("成功加载演算结果")

        violations = None
        if args.violations:
            _print_info(f"加载违规列表: {args.violations}")
            violations = _load_violations(args.violations)
            _print_info(f"成功加载 {len(violations)} 条违规记录")

        if violations is None and "violations" in result:
            _print_info("从结果文件中提取违规记录...")
            violations_data = result["violations"]
            violations = []
            for v_data in violations_data:
                violation = Violation(
                    reservoir_name=v_data.get("reservoir_name", ""),
                    step_idx=v_data.get("step_idx", 0),
                    time_hours=v_data.get("time_hours", 0),
                    type=v_data.get("type", ""),
                    value=float(v_data.get("value", 0.0)),
                    limit=float(v_data.get("limit", 0.0)),
                    message=v_data.get("message", ""),
                )
                violations.append(violation)
            _print_info(f"提取到 {len(violations)} 条违规记录")

        reservoirs = None
        if "reservoirs" in result:
            reservoirs = result["reservoirs"]

        _print_info("创建报告生成器...")
        generator = ReportGenerator(
            result=result,
            violations=violations,
            reservoirs=reservoirs,
        )

        format_type = getattr(args, 'format', 'text')
        _print_info(f"生成 {format_type} 格式报告...")
        if format_type == 'html':
            report_content = generator.generate_html()
        else:
            report_content = generator.generate_text()
        _print_info("报告生成完成")

        if args.output:
            _print_info(f"保存报告到: {args.output}")
            dir_path = os.path.dirname(args.output)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report_content)
            _print_info("报告保存成功")
        else:
            _print_info("报告内容:")
            print("\n" + "=" * 70)
            print(report_content)
            print("=" * 70 + "\n")

        _print_info("报告生成任务完成")
        return 0

    except FileNotFoundError as e:
        _print_error(str(e))
        return 1
    except ValueError as e:
        _print_error(f"数据格式错误: {e}")
        return 1
    except Exception as e:
        _print_error(f"生成报告失败: {str(e)}")
        return 1
