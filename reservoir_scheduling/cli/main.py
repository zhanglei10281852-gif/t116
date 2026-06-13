# -*- coding: utf-8 -*-
"""
CLI主入口模块
使用argparse解析命令行参数，调度相应的命令处理函数
"""

import sys
import argparse
from typing import List

from .commands import cmd_simulate, cmd_optimize, cmd_check, cmd_report


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器

    Returns
    -------
    argparse.ArgumentParser
        配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        prog="reservoir-scheduling",
        description="梯级水库联合调度系统 - 命令行接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 演算调度
  python -m reservoir_scheduling.cli.main simulate \\
      --reservoirs examples/reservoirs.json \\
      --inflow examples/inflow_forecast.json \\
      --release examples/initial_release.json \\
      --output output/simulate_result.json

  # 优化调度
  python -m reservoir_scheduling.cli.main optimize \\
      --reservoirs examples/reservoirs.json \\
      --inflow examples/inflow_forecast.json \\
      --initial examples/initial_release.json \\
      --output-scheme output/optimal_scheme.json \\
      --output-result output/optimize_result.json

  # 约束检查
  python -m reservoir_scheduling.cli.main check \\
      --reservoirs examples/reservoirs.json \\
      --inflow examples/inflow_forecast.json \\
      --release examples/initial_release.json

  # 生成报告
  python -m reservoir_scheduling.cli.main report \\
      --input output/simulate_result.json \\
      --format html \\
      --output output/report.html
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="子命令",
        description="可用的子命令",
        help="子命令帮助",
    )

    # simulate 子命令
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="执行水库调度演算",
        description="根据给定的下泄方案执行水库调度演算",
    )
    simulate_parser.add_argument(
        "--reservoirs",
        required=True,
        help="水库配置JSON文件路径",
        metavar="FILE",
    )
    simulate_parser.add_argument(
        "--inflow",
        required=True,
        help="入库预报JSON文件路径",
        metavar="FILE",
    )
    simulate_parser.add_argument(
        "--release",
        required=True,
        help="下泄方案JSON文件路径",
        metavar="FILE",
    )
    simulate_parser.add_argument(
        "--output",
        required=True,
        help="演算结果输出JSON文件路径",
        metavar="FILE",
    )
    simulate_parser.set_defaults(func=cmd_simulate)

    # optimize 子命令
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="执行水库调度优化",
        description="通过优化算法寻找最优下泄方案以最小化出口洪峰",
    )
    optimize_parser.add_argument(
        "--reservoirs",
        required=True,
        help="水库配置JSON文件路径",
        metavar="FILE",
    )
    optimize_parser.add_argument(
        "--inflow",
        required=True,
        help="入库预报JSON文件路径",
        metavar="FILE",
    )
    optimize_parser.add_argument(
        "--initial",
        help="初始下泄方案JSON文件路径（可选，默认使用最小下泄方案）",
        metavar="FILE",
    )
    optimize_parser.add_argument(
        "--output-scheme",
        required=True,
        help="最优下泄方案输出JSON文件路径",
        metavar="FILE",
    )
    optimize_parser.add_argument(
        "--output-result",
        required=True,
        help="优化结果输出JSON文件路径",
        metavar="FILE",
    )
    optimize_parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=12,
        help="预测时域，单位小时（默认：12），用于前瞻未来降雨",
        metavar="N",
    )
    optimize_parser.add_argument(
        "--control-horizon",
        type=int,
        default=3,
        help="控制时域，单位小时（默认：3），每步优化的决策时段数",
        metavar="N",
    )
    optimize_parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="水位超汛限惩罚权重（默认：1.0）",
        metavar="ALPHA",
    )
    optimize_parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="出口洪峰惩罚权重（默认：1.0）",
        metavar="BETA",
    )
    optimize_parser.add_argument(
        "--gamma",
        type=float,
        default=0.3,
        help="上游过度拦蓄惩罚权重（默认：0.3），防止上游占满库容导致后期被迫集中下泄",
        metavar="GAMMA",
    )
    optimize_parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="坐标下降最大迭代次数（默认：20）",
        metavar="N",
    )
    optimize_parser.add_argument(
        "--search-density",
        type=int,
        default=10,
        help="单变量搜索密度（默认：10），值越大精度越高但速度越慢",
        metavar="N",
    )
    optimize_parser.set_defaults(func=cmd_optimize)

    # check 子命令
    check_parser = subparsers.add_parser(
        "check",
        help="检查约束条件",
        description="对给定的下泄方案进行全面的约束检查",
    )
    check_parser.add_argument(
        "--reservoirs",
        required=True,
        help="水库配置JSON文件路径",
        metavar="FILE",
    )
    check_parser.add_argument(
        "--inflow",
        required=True,
        help="入库预报JSON文件路径",
        metavar="FILE",
    )
    check_parser.add_argument(
        "--release",
        required=True,
        help="下泄方案JSON文件路径",
        metavar="FILE",
    )
    check_parser.add_argument(
        "--output",
        help="检查结果输出JSON文件路径（可选）",
        metavar="FILE",
    )
    check_parser.set_defaults(func=cmd_check)

    # report 子命令
    report_parser = subparsers.add_parser(
        "report",
        help="生成调度报告",
        description="根据演算结果生成格式化的调度报告",
    )
    report_parser.add_argument(
        "--input",
        required=True,
        help="演算结果JSON文件路径",
        metavar="FILE",
    )
    report_parser.add_argument(
        "--violations",
        help="违规列表JSON文件路径（可选，如未提供则从结果文件中提取）",
        metavar="FILE",
    )
    report_parser.add_argument(
        "--format",
        choices=["text", "html"],
        default="text",
        help="报告格式，可选值：text、html（默认：text）",
    )
    report_parser.add_argument(
        "--output",
        help="报告输出文件路径（可选，如未提供则打印到控制台）",
        metavar="FILE",
    )
    report_parser.set_defaults(func=cmd_report)

    return parser


def main(argv: List[str] = None) -> int:
    """
    主函数入口

    Parameters
    ----------
    argv : list, optional
        命令行参数列表，如为None则使用sys.argv

    Returns
    -------
    int
        退出码，0表示成功，非0表示失败
    """
    parser = build_parser()

    if argv is None:
        argv = sys.argv[1:]

    if len(argv) == 0:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
