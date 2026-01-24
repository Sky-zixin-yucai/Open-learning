#!/usr/bin/env python3
"""
core包功能测试 | core Package Functionality Test
============================================
快速测试core包的核心功能。
Quick test core package core functionality.
"""

import sys
import os
import torch

# 设置路径
current_script_dir = os.path.dirname(os.path.abspath(__file__))
openlearning_dir = os.path.dirname(current_script_dir)
sys.path.insert(0, openlearning_dir)

print(f"当前脚本目录: {current_script_dir}")
print(f"openlearning目录: {openlearning_dir}")

def quick_test():
    """快速功能测试 | Quick functionality test"""
    print("="*80)
    print("core包快速功能测试 | core Package Quick Functionality Test")
    print("="*80)
    
    try:
        # 1. 导入模块
        print("\n1. 导入模块 | Importing modules...")
        from core import RGAConfig, CoreMetricsCalculator
        print("✅ 模块导入成功 | Modules imported successfully")
        
        # 2. 测试配置
        print("\n2. 测试配置 | Testing configuration...")
        config = RGAConfig(vocab_size=20000, dim=512)
        print(f"   创建配置: vocab_size={config.vocab_size}, dim={config.dim}")
        print(f"   Created config: vocab_size={config.vocab_size}, dim={config.dim}")
        
        # 验证配置
        config.validate()
        print("   配置验证通过 | Config validation passed")
        
        # 3. 测试计算器
        print("\n3. 测试计算器 | Testing calculator...")
        calculator = CoreMetricsCalculator(phase_threshold=0.5)
        print(f"   创建计算器: phase_threshold={calculator.phase_threshold}")
        print(f"   Created calculator: phase_threshold={calculator.phase_threshold}")
        
        # 4. 创建测试数据
        print("\n4. 创建测试数据 | Creating test data...")
        batch_size, seq_len, dim = 2, 10, 32
        Q1 = torch.randn(batch_size, seq_len, dim)
        K1 = torch.randn(batch_size, seq_len, dim)
        V1 = torch.randn(batch_size, seq_len, dim)
        
        Q2 = Q1 + 0.1 * torch.randn_like(Q1)
        K2 = K1 + 0.1 * torch.randn_like(K1)
        V2 = V1 + 0.1 * torch.randn_like(V1)
        
        # 5. 测试状态变化计算
        print("\n5. 测试状态变化计算 | Testing state change calculation...")
        delta = calculator.compute_state_change(Q2, K2, V2, Q1, K1, V1)
        print(f"   状态变化量: {delta:.4f} | State change: {delta:.4f}")
        
        # 6. 测试相变检测
        print("\n6. 测试相变检测 | Testing phase transition detection...")
        delta_total, is_transition = calculator.detect_phase_transition(Q2, K2, V2, Q1, K1, V1)
        print(f"   总变化量: {delta_total:.4f}, 相变: {is_transition}")
        print(f"   Total change: {delta_total:.4f}, transition: {is_transition}")
        
        # 7. 记录状态
        print("\n7. 记录状态 | Recording state...")
        calculator.record_state(Q2, K2, V2)
        print(f"   状态已记录 | State recorded")
        
        # 8. 分析学习阶段
        print("\n8. 分析学习阶段 | Analyzing learning phase...")
        phase_info = calculator.analyze_learning_phases()
        print(f"   学习阶段: {phase_info.get('phase', '未知')}")
        print(f"   Learning phase: {phase_info.get('phase', 'Unknown')}")
        
        print("\n✅ 快速功能测试通过！ | Quick functionality test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e} | Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ 功能测试失败: {e} | Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数 | Main test function"""
    print("core包功能测试开始... | core Package Functionality Test Started...\n")
    
    if quick_test():
        print("\n" + "="*80)
        print("✅ 功能测试通过！ | Functionality test passed!")
        print("core包功能正常，可以开始使用。 | core package functions normally, ready to use.")
        return 0
    else:
        print("\n" + "="*80)
        print("❌ 功能测试失败 | Functionality test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())