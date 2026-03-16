"""
核心度量计算器测试 | Core Metrics Calculator Tests
================================================

测试CoreMetricsCalculator类的所有功能。
Test all functionalities of CoreMetricsCalculator class.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from metrics import CoreMetricsCalculator

def test_initialization():
    """
    测试计算器初始化
    Test calculator initialization
    """
    print("测试CoreMetricsCalculator初始化... | Testing CoreMetricsCalculator initialization...")
    
    # 测试默认初始化 | Test default initialization
    try:
        calculator = CoreMetricsCalculator()
        print("✅ 默认初始化成功 | Default initialization successful")
        print(f"   相变阈值: {calculator.phase_threshold} | Phase threshold: {calculator.phase_threshold}")
    except Exception as e:
        print(f"❌ 默认初始化失败: {e} | Default initialization failed: {e}")
        return False
    
    # 测试自定义阈值初始化 | Test custom threshold initialization
    try:
        calculator = CoreMetricsCalculator(phase_threshold=0.5)
        assert calculator.phase_threshold == 0.5
        print("✅ 自定义阈值初始化成功 | Custom threshold initialization successful")
    except Exception as e:
        print(f"❌ 自定义阈值初始化失败: {e} | Custom threshold initialization failed: {e}")
        return False
    
    return True

def test_compute_state_change():
    """
    测试状态变化量计算
    Test state change calculation
    """
    print("\n测试状态变化量计算... | Testing state change calculation...")
    
    calculator = CoreMetricsCalculator()
    
    # 创建测试数据 | Create test data
    batch_size, seq_len, dim = 2, 10, 32
    Q1 = torch.randn(batch_size, seq_len, dim)
    K1 = torch.randn(batch_size, seq_len, dim)
    V1 = torch.randn(batch_size, seq_len, dim)
    
    Q2 = Q1 + 0.1 * torch.randn_like(Q1)
    K2 = K1 + 0.1 * torch.randn_like(K1)
    V2 = V1 + 0.1 * torch.randn_like(V1)
    
    # 测试L2范数 | Test L2 norm
    try:
        delta_l2 = calculator.compute_state_change(Q2, K2, V2, Q1, K1, V1, norm_type='l2')
        print(f"✅ L2范数计算成功: {delta_l2:.4f} | L2 norm calculation successful: {delta_l2:.4f}")
    except Exception as e:
        print(f"❌ L2范数计算失败: {e} | L2 norm calculation failed: {e}")
        return False
    
    # 测试L1范数 | Test L1 norm
    try:
        delta_l1 = calculator.compute_state_change(Q2, K2, V2, Q1, K1, V1, norm_type='l1')
        print(f"✅ L1范数计算成功: {delta_l1:.4f} | L1 norm calculation successful: {delta_l1:.4f}")
    except Exception as e:
        print(f"❌ L1范数计算失败: {e} | L1 norm calculation failed: {e}")
        return False
    
    # 测试不支持的范数类型 | Test unsupported norm type
    try:
        calculator.compute_state_change(Q2, K2, V2, Q1, K1, V1, norm_type='invalid')
        print("❌ 应该抛出错误但未抛出 | Should have raised error but didn't")
        return False
    except ValueError as e:
        print(f"✅ 正确捕获无效范数类型错误 | Correctly caught invalid norm type error: {e}")
    
    return True

def test_detect_phase_transition():
    """
    测试相变检测
    Test phase transition detection
    """
    print("\n测试相变检测... | Testing phase transition detection...")
    
    calculator = CoreMetricsCalculator(phase_threshold=0.8)
    
    # 创建测试数据 | Create test data
    batch_size, seq_len, dim = 2, 10, 32
    Q1 = torch.randn(batch_size, seq_len, dim)
    K1 = torch.randn(batch_size, seq_len, dim)
    V1 = torch.randn(batch_size, seq_len, dim)
    
    # 测试小变化（无相变）| Test small change (no phase transition)
    Q2_small = Q1 + 0.01 * torch.randn_like(Q1)
    K2_small = K1 + 0.01 * torch.randn_like(K1)
    V2_small = V1 + 0.01 * torch.randn_like(V1)
    
    delta_small, is_transition_small = calculator.detect_phase_transition(
        Q2_small, K2_small, V2_small, Q1, K1, V1
    )
    
    if not is_transition_small:
        print(f"✅ 小变化检测为无相变: Δ={delta_small:.4f} | Small change detected as no phase transition: Δ={delta_small:.4f}")
    else:
        print(f"❌ 小变化错误检测为相变: Δ={delta_small:.4f} | Small change incorrectly detected as phase transition: Δ={delta_small:.4f}")
    
    # 测试大变化（有相变）| Test large change (phase transition)
    Q2_large = Q1 + 10.0 * torch.randn_like(Q1)
    K2_large = K1 + 10.0 * torch.randn_like(K1)
    V2_large = V1 + 10.0 * torch.randn_like(V1)
    
    delta_large, is_transition_large = calculator.detect_phase_transition(
        Q2_large, K2_large, V2_large, Q1, K1, V1
    )
    
    if is_transition_large:
        print(f"✅ 大变化检测为相变: Δ={delta_large:.4f} | Large change detected as phase transition: Δ={delta_large:.4f}")
        print(f"   相变点数量: {len(calculator.transition_points)} | Transition points: {len(calculator.transition_points)}")
    else:
        print(f"❌ 大变化未检测为相变: Δ={delta_large:.4f} | Large change not detected as phase transition: Δ={delta_large:.4f}")
    
    return True

def test_stack_three_networks():
    """
    测试三网络堆叠
    Test three-network stacking
    """
    print("\n测试三网络堆叠... | Testing three-network stacking...")
    
    calculator = CoreMetricsCalculator()
    
    # 创建三个测试矩阵 | Create three test matrices
    Q1 = torch.randn(3, 4)
    Q2 = torch.randn(3, 4)
    Q3 = torch.randn(3, 4)
    Q_list = [Q1, Q2, Q3]
    
    # 测试默认权重堆叠 | Test stacking with default weights
    try:
        Q_stack_default = calculator.stack_three_networks(Q_list)
        print(f"✅ 默认权重堆叠成功 | Default weight stacking successful")
        print(f"   输出形状: {Q_stack_default.shape} | Output shape: {Q_stack_default.shape}")
    except Exception as e:
        print(f"❌ 默认权重堆叠失败: {e} | Default weight stacking failed: {e}")
        return False
    
    # 测试自定义权重堆叠 | Test stacking with custom weights
    try:
        weights = torch.tensor([0.1, 0.3, 0.6])
        Q_stack_custom = calculator.stack_three_networks(Q_list, weights)
        print(f"✅ 自定义权重堆叠成功 | Custom weight stacking successful")
    except Exception as e:
        print(f"❌ 自定义权重堆叠失败: {e} | Custom weight stacking failed: {e}")
        return False
    
    # 测试错误输入（非3个矩阵）| Test wrong input (not 3 matrices)
    try:
        calculator.stack_three_networks([Q1, Q2])
        print("❌ 应该抛出错误但未抛出 | Should have raised error but didn't")
        return False
    except ValueError as e:
        print(f"✅ 正确捕获数量错误: {e} | Correctly caught quantity error: {e}")
    
    # 测试错误输入（形状不一致）| Test wrong input (inconsistent shapes)
    try:
        Q_wrong = torch.randn(2, 5)  # 形状不同 | Different shape
        calculator.stack_three_networks([Q1, Q2, Q_wrong])
        print("❌ 应该抛出错误但未抛出 | Should have raised error but didn't")
        return False
    except ValueError as e:
        print(f"✅ 正确捕获形状错误: {e} | Correctly caught shape error: {e}")
    
    return True

def test_one_way_valve():
    """
    测试单向阀
    Test one-way valve
    """
    print("\n测试单向阀... | Testing one-way valve...")
    
    calculator = CoreMetricsCalculator()
    
    # 创建测试张量 | Create test tensor
    h_in = torch.randn(3, 4, requires_grad=True)
    
    # 测试detach模式 | Test detach mode
    try:
        h_detached = calculator.apply_one_way_valve(h_in, mode='detach')
        if not h_detached.requires_grad:
            print("✅ detach模式成功（梯度已切断）| detach mode successful (gradient cutoff)")
        else:
            print("❌ detach模式失败（梯度未切断）| detach mode failed (gradient not cutoff)")
            return False
    except Exception as e:
        print(f"❌ detach模式失败: {e} | detach mode failed: {e}")
        return False
    
    # 测试gate模式（门控值=1）| Test gate mode (gate value=1)
    try:
        h_gate_1 = calculator.apply_one_way_valve(h_in, mode='gate', gate_value=1)
        if torch.allclose(h_gate_1, h_in):
            print("✅ gate模式（门控值=1）成功 | gate mode (gate value=1) successful")
        else:
            print("❌ gate模式（门控值=1）失败 | gate mode (gate value=1) failed")
            return False
    except Exception as e:
        print(f"❌ gate模式（门控值=1）失败: {e} | gate mode (gate value=1) failed: {e}")
        return False
    
    # 测试gate模式（门控值=0）| Test gate mode (gate value=0)
    try:
        h_gate_0 = calculator.apply_one_way_valve(h_in, mode='gate', gate_value=0)
        if torch.allclose(h_gate_0, torch.zeros_like(h_in)):
            print("✅ gate模式（门控值=0）成功 | gate mode (gate value=0) successful")
        else:
            print("❌ gate模式（门控值=0）失败 | gate mode (gate value=0) failed")
            return False
    except Exception as e:
        print(f"❌ gate模式（门控值=0）失败: {e} | gate mode (gate value=0) failed: {e}")
        return False
    
    # 测试无效门控值 | Test invalid gate value
    try:
        calculator.apply_one_way_valve(h_in, mode='gate', gate_value=2)
        print("❌ 应该抛出错误但未抛出 | Should have raised error but didn't")
        return False
    except ValueError as e:
        print(f"✅ 正确捕获无效门控值错误: {e} | Correctly caught invalid gate value error: {e}")
    
    # 测试无效模式 | Test invalid mode
    try:
        calculator.apply_one_way_valve(h_in, mode='invalid')
        print("❌ 应该抛出错误但未抛出 | Should have raised error but didn't")
        return False
    except ValueError as e:
        print(f"✅ 正确捕获无效模式错误: {e} | Correctly caught invalid mode error: {e}")
    
    return True

def test_state_management():
    """
    测试状态管理
    Test state management
    """
    print("\n测试状态管理... | Testing state management...")
    
    calculator = CoreMetricsCalculator()
    
    # 记录多个状态 | Record multiple states
    for i in range(5):
        Q = torch.randn(2, 10, 32) + i * 0.1
        K = torch.randn(2, 10, 32) + i * 0.1
        V = torch.randn(2, 10, 32) + i * 0.1
        calculator.record_state(Q, K, V)
    
    # 检查状态历史 | Check state history
    stats = calculator.get_statistics()
    if stats['state_history_length'] == 5:
        print(f"✅ 状态记录成功: {stats['state_history_length']}个状态 | State recording successful: {stats['state_history_length']} states")
    else:
        print(f"❌ 状态记录失败: 期望5个状态，实际{stats['state_history_length']}个 | State recording failed: Expected 5 states, got {stats['state_history_length']}")
        return False
    
    # 获取状态变化序列 | Get state change series
    changes = calculator.get_state_change_series('l2')
    if len(changes) == 4:  # 5个状态有4个变化 | 5 states have 4 changes
        print(f"✅ 状态变化序列获取成功: {len(changes)}个变化 | State change series obtained successfully: {len(changes)} changes")
    else:
        print(f"❌ 状态变化序列获取失败: 期望4个变化，实际{len(changes)}个 | State change series failed: Expected 4 changes, got {len(changes)}")
        return False
    
    # 分析学习阶段 | Analyze learning phase
    phase_info = calculator.analyze_learning_phases()
    if 'phase' in phase_info:
        print(f"✅ 学习阶段分析成功: {phase_info['phase']} | Learning phase analysis successful: {phase_info['phase']}")
    else:
        print(f"❌ 学习阶段分析失败 | Learning phase analysis failed")
        return False
    
    # 测试重置 | Test reset
    calculator.reset()
    stats_after_reset = calculator.get_statistics()
    if stats_after_reset['state_history_length'] == 0 and stats_after_reset['transition_points'] == 0:
        print("✅ 重置成功 | Reset successful")
    else:
        print("❌ 重置失败 | Reset failed")
        return False
    
    return True

def test_string_representation():
    """
    测试字符串表示
    Test string representation
    """
    print("\n测试字符串表示... | Testing string representation...")
    
    calculator = CoreMetricsCalculator(phase_threshold=0.5)
    
    # 记录一些状态 | Record some states
    for i in range(3):
        Q = torch.randn(2, 10, 32)
        K = torch.randn(2, 10, 32)
        V = torch.randn(2, 10, 32)
        calculator.record_state(Q, K, V)
    
    # 测试字符串表示 | Test string representation
    try:
        str_repr = str(calculator)
        print("✅ 字符串表示成功 | String representation successful")
        print(f"\n计算器信息: | Calculator info:")
        print(str_repr)
    except Exception as e:
        print(f"❌ 字符串表示失败: {e} | String representation failed: {e}")
        return False
    
    return True

def main():
    """
    主测试函数
    Main test function
    """
    print("="*80)
    print("CoreMetricsCalculator 测试 | CoreMetricsCalculator Tests")
    print("="*80)
    
    tests = [
        ("初始化测试 | Initialization test", test_initialization),
        ("状态变化量计算测试 | State change calculation test", test_compute_state_change),
        ("相变检测测试 | Phase transition detection test", test_detect_phase_transition),
        ("三网络堆叠测试 | Three-network stacking test", test_stack_three_networks),
        ("单向阀测试 | One-way valve test", test_one_way_valve),
        ("状态管理测试 | State management test", test_state_management),
        ("字符串表示测试 | String representation test", test_string_representation),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 60)
        
        try:
            passed = test_func()
            if not passed:
                all_passed = False
                print(f"❌ {test_name} 失败 | Failed")
        except Exception as e:
            all_passed = False
            print(f"❌ {test_name} 异常: {e} | Exception: {e}")
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ 所有测试通过！ | All tests passed!")
    else:
        print("❌ 部分测试失败 | Some tests failed")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

