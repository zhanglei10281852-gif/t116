# -*- coding: utf-8 -*-
"""
基础功能测试
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reservoir_scheduling import (
    load_reservoir_config,
    load_inflow_forecast,
    load_release_scheme,
    Reservoir,
    CascadeSimulator,
    ConstraintChecker,
    CascadeOptimizer,
    ReportGenerator,
)

print("=" * 60)
print("测试1: 加载水库配置")
print("=" * 60)
configs = load_reservoir_config('examples/reservoirs.json')
print(f"加载了 {len(configs)} 座水库")
for cfg in configs:
    print(f"  - {cfg['name']}: order={cfg['order']}")
    print(f"    dead_storage={cfg['dead_storage']:.2e}, max_storage={cfg['max_storage']:.2e}")
    print(f"    flood_limit={cfg.get('flood_limit_level')}, initial={cfg.get('initial_level')}")
    print(f"    downstream={cfg.get('downstream')}, K={cfg.get('muskingum_K')}, x={cfg.get('muskingum_x')}")

print("\n" + "=" * 60)
print("测试2: 加载入库预报")
print("=" * 60)
inflow = load_inflow_forecast('examples/inflow_forecast.json')
print(f"入库预报: {inflow['n_steps']} 时段, 时间步长 {inflow['time_step_hours']}h")
for name, flows in inflow['inflow'].items():
    print(f"  - {name}: {len(flows)} 个数据, 洪峰 {max(flows):.1f} m³/s")

print("\n" + "=" * 60)
print("测试3: 加载下泄方案")
print("=" * 60)
release = load_release_scheme('examples/initial_release.json')
print(f"下泄方案: {len(release)} 座水库")
for name, flows in release.items():
    print(f"  - {name}: {len(flows)} 个数据")

print("\n" + "=" * 60)
print("测试4: 创建水库对象")
print("=" * 60)
reservoirs = []
for cfg in configs:
    r = Reservoir(cfg)
    reservoirs.append(r)
    print(f"  - {r.name}: 死水位={r.dead_level:.2f}m, 最高水位={r.max_level:.2f}m")
    print(f"    汛限水位={r.flood_limit_level:.2f}m, 初始水位={r.initial_level:.2f}m")

print("\n" + "=" * 60)
print("测试5: 梯级演算")
print("=" * 60)
simulator = CascadeSimulator(reservoirs, inflow)
result = simulator.simulate(release)
print(f"演算完成: {len(result['time_steps'])} 个时段")
for res_name in ['水库A', '水库B', '水库C']:
    data = result[res_name]
    max_level = max(data['level'])
    min_level = min(data['level'])
    max_release = max(data['release'])
    print(f"  - {res_name}:")
    print(f"    水位范围: {min_level:.2f}m ~ {max_level:.2f}m")
    print(f"    最大下泄: {max_release:.1f} m³/s")

outlet_peak = max(result['outlet_flow'])
print(f"  出口洪峰: {outlet_peak:.1f} m³/s")

print("\n" + "=" * 60)
print("测试6: 约束检查")
print("=" * 60)
checker = ConstraintChecker(reservoirs, time_step_hours=inflow['time_step_hours'])
violations = checker.check_all(result)
print(f"发现 {len(violations)} 条违规记录")
for v in violations[:10]:
    print(f"  [{v.time_hours}h] {v.reservoir_name}: {v.type} - {v.message}")

print("\n" + "=" * 60)
print("测试7: 报告生成")
print("=" * 60)
generator = ReportGenerator(result, violations, reservoirs)
text_report = generator.generate_text()
print("文本报告前500字符:")
print(text_report[:500])

html_report = generator.generate_html()
print(f"\nHTML报告长度: {len(html_report)} 字符")

print("\n" + "=" * 60)
print("测试8: 滚动时域优化（简短测试）")
print("=" * 60)
optimizer = CascadeOptimizer(reservoirs, inflow, config={
    'prediction_horizon': 6,
    'control_horizon': 2,
    'search_density': 5,
    'max_iterations': 5,
})
opt_result = optimizer.optimize()
optimal_scheme = opt_result['release_scheme']
optimal_result = opt_result['simulation_result']
print(f"优化完成: 方案包含 {len(optimal_scheme)} 座水库")
peak_optimized = max(optimal_result['outlet_flow'])
print(f"优化后出口洪峰: {peak_optimized:.1f} m³/s")
print(f"削峰率: {(outlet_peak - peak_optimized) / outlet_peak * 100:.1f}%")

print("\n" + "=" * 60)
print("所有基础测试通过！")
print("=" * 60)
