#!/usr/bin/env python3
"""
core包导入测试 | core Package Import Test
======================================

测试core包能否正常导入和使用。
Test if core package can be imported and used normally.
"""

import sys
import os

# 获取当前脚本所在目录（ccss目录）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"当前脚本目录: {current_script_dir}")

# 获取openlearning目录（父目录）
openlearning_dir = os.path.dirname(current_script_dir)
print(f"openlearning目录: {openlearning_dir}")

# 检查core目录是否存在
core_dir = os.path.join(openlearning_dir, 'core')
print(f"core目录: {core_dir}")
print(f"core目录是否存在: {os.path.exists(core_dir)}")

# 显示core目录内容
if os.path.exists(core_dir):
    print(f"core目录内容:")
    for item in os.listdir(core_dir):
        print(f"  - {item}")

# 将openlearning目录添加到Python路径
sys.path.insert(0, openlearning_dir)
print(f"\n已添加目录到Python路径: {openlearning_dir}")
print(f"当前Python路径: {sys.path[:3]}")

def test_core_import():
    """测试core包导入 | Test core package import"""
    print("\n" + "="*80)
    print("测试core包导入 | Testing core Package Import")
    print("="*80)
    
    try:
        # 方法1：尝试导入整个包
        print("\n1. 尝试导入整个core包 | Trying to import whole core package...")
        try:
            import core
            print(f"✅ core包导入成功 | core package import successful")
            print(f"   版本: {core.__version__} | Version: {core.__version__}")
            print(f"   作者: {core.__author__} | Author: {core.__author__}")
            print(f"   导出列表: {core.__all__} | Export list: {core.__all__}")
            
            # 验证导出列表中包含的类
            for export in core.__all__:
                if not export.startswith('__'):
                    try:
                        cls = getattr(core, export)
                        print(f"   ✅ {export} 可访问 | {export} is accessible")
                    except AttributeError:
                        print(f"   ❌ {export} 不可访问 | {export} is not accessible")
                        
        except ImportError as e:
            print(f"❌ core包导入失败: {e} | core package import failed: {e}")
            return False
        
        # 方法2：尝试直接导入类
        print("\n2. 尝试直接导入类 | Trying to import classes directly...")
        try:
            from core import RGAConfig
            print("✅ RGAConfig 导入成功 | RGAConfig import successful")
        except ImportError as e:
            print(f"❌ RGAConfig 导入失败: {e} | RGAConfig import failed: {e}")
            return False
        
        try:
            from core import CoreMetricsCalculator
            print("✅ CoreMetricsCalculator 导入成功 | CoreMetricsCalculator import successful")
        except ImportError as e:
            print(f"❌ CoreMetricsCalculator 导入失败: {e} | CoreMetricsCalculator import failed: {e}")
            return False
        
        # 方法3：尝试从模块导入
        print("\n3. 尝试从模块导入 | Trying to import from modules...")
        try:
            from core.config import RGAConfig as RGAConfig2
            print("✅ core.config.RGAConfig 导入成功 | core.config.RGAConfig import successful")
        except ImportError as e:
            print(f"❌ core.config 导入失败: {e} | core.config import failed: {e}")
            return False
        
        try:
            from core.metrics import CoreMetricsCalculator as Calculator2
            print("✅ core.metrics.CoreMetricsCalculator 导入成功 | core.metrics.CoreMetricsCalculator import successful")
        except ImportError as e:
            print(f"❌ core.metrics 导入失败: {e} | core.metrics import failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 导入测试异常: {e} | Import test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_class_instantiation():
    """测试类实例化 | Test class instantiation"""
    print("\n" + "="*80)
    print("测试类实例化 | Testing Class Instantiation")
    print("="*80)
    
    try:
        # 导入类
        from core import RGAConfig, CoreMetricsCalculator
        
        print("1. 测试 RGAConfig 实例化 | Testing RGAConfig instantiation...")
        try:
            # 创建默认配置
            config = RGAConfig()
            print(f"   ✅ 默认配置创建成功 | Default config creation successful")
            print(f"      vocab_size: {config.vocab_size}")
            print(f"      dim: {config.dim}")
            print(f"      num_units: {config.num_units}")
            
            # 验证配置
            config.validate()
            print(f"   ✅ 配置验证通过 | Config validation passed")
            
            # 创建自定义配置
            custom_config = RGAConfig(vocab_size=20000, dim=512, phase_threshold=0.43)
            print(f"   ✅ 自定义配置创建成功 | Custom config creation successful")
            
        except Exception as e:
            print(f"   ❌ RGAConfig 实例化失败: {e} | RGAConfig instantiation failed: {e}")
            return False
        
        print("\n2. 测试 CoreMetricsCalculator 实例化 | Testing CoreMetricsCalculator instantiation...")
        try:
            # 创建默认计算器
            calculator = CoreMetricsCalculator()
            print(f"   ✅ 默认计算器创建成功 | Default calculator creation successful")
            print(f"      phase_threshold: {calculator.phase_threshold}")
            
            # 创建自定义计算器
            custom_calculator = CoreMetricsCalculator(phase_threshold=0.5)
            print(f"   ✅ 自定义计算器创建成功 | Custom calculator creation successful")
            
        except Exception as e:
            print(f"   ❌ CoreMetricsCalculator 实例化失败: {e} | CoreMetricsCalculator instantiation failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e} | Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ 实例化测试异常: {e} | Instantiation test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_package_methods():
    """测试包方法 | Test package methods"""
    print("\n" + "="*80)
    print("测试包方法 | Testing Package Methods")
    print("="*80)
    
    try:
        # 导入类
        from core import RGAConfig, CoreMetricsCalculator
        
        print("1. 测试 RGAConfig 方法 | Testing RGAConfig methods...")
        config = RGAConfig(vocab_size=30000, dim=256)
        
        # 测试 to_dict
        config_dict = config.to_dict()
        print(f"   ✅ to_dict() 成功: {list(config_dict.keys())}")
        
        # 测试 from_dict
        new_config = RGAConfig.from_dict({'vocab_size': 40000, 'dim': 512})
        print(f"   ✅ from_dict() 成功: vocab_size={new_config.vocab_size}, dim={new_config.dim}")
        
        # 测试 update
        config.update(vocab_size=50000, dim=1024)
        print(f"   ✅ update() 成功: vocab_size={config.vocab_size}, dim={config.dim}")
        
        # 测试 __str__
        config_str = str(config)
        print(f"   ✅ __str__() 成功 (显示前100字符): {config_str[:100]}...")
        
        print("\n2. 测试 CoreMetricsCalculator 方法 | Testing CoreMetricsCalculator methods...")
        calculator = CoreMetricsCalculator()
        
        # 测试 reset
        calculator.reset()
        print(f"   ✅ reset() 成功")
        
        # 测试 get_statistics
        stats = calculator.get_statistics()
        print(f"   ✅ get_statistics() 成功: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 方法测试异常: {e} | Method test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数 | Main test function"""
    print("core包导入测试开始... | core Package Import Test Started...\n")
    
    # 运行所有测试
    tests = [
        ("导入测试 | Import Test", test_core_import),
        ("实例化测试 | Instantiation Test", test_class_instantiation),
        ("方法测试 | Method Test", test_package_methods),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 60)
        
        try:
            passed = test_func()
            if not passed:
                all_passed = False
        except Exception as e:
            all_passed = False
            print(f"❌ 测试异常: {e} | Test exception: {e}")
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ 所有导入测试通过！ | All import tests passed!")
        return 0
    else:
        print("❌ 部分导入测试失败 | Some import tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())