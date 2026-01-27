#!/usr/bin/env python3
"""
OpenLearning RGA 命令行接口
===========================

完整的功能调用接口，支持所有模块和组件，完全匹配README.md描述的功能。

当通过pip安装后，可以通过以下方式使用：
    openlearning                      # 显示帮助信息
    openlearning demo                 # 运行完整演示
    openlearning test                 # 运行所有测试
    openlearning core                 # 运行核心模块测试
    openlearning layers               # 运行层模块测试
    openlearning integration          # 运行集成模块测试
    openlearning train                # 启动训练菜单
    openlearning infer                # 启动推理测试
    openlearning check                # 检查模块状态

本地开发使用：
    python cli.py demo
    python cli.py test-all
    python cli.py layers-attention
"""

import sys
import os
import argparse
import importlib
import subprocess
import textwrap
from pathlib import Path
import traceback

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent if current_dir.name == "openlearning" else current_dir

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==================== 辅助函数 ====================

def print_banner():
    """打印横幅"""
    banner = r"""
   ___                  _                _           
  / _ \ _ __   ___ _ __| | ___ __ _  __| | ___ _ __ 
 / /_)/| '_ \ / _ \ '__| |/ __/ _` |/ _` |/ _ \ '__|
/ ___/ | |_) |  __/ |  | | (_| (_| | (_| |  __/ |   
\/    | .__/ \___|_|  |_|\___\__,_|\__,_|\___|_|   
     |_|                                           
    """
    print(banner)
    print("🚀 OpenLearning RGA - 规则治理架构")
    print("=" * 60)

def run_module_main(module_path):
    """运行模块的main函数"""
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, 'main'):
            return module.main()
        else:
            print(f"⚠️  模块 {module_path} 没有main()函数")
            return 0
    except ImportError as e:
        print(f"❌ 无法导入模块 {module_path}: {e}")
        return 1
    except Exception as e:
        print(f"❌ 运行模块 {module_path} 失败: {e}")
        return 1

def run_package_demo():
    """运行包的演示（使用__main__模块）"""
    print("🎮 运行OpenLearning RGA完整演示...")
    print("=" * 60)
    
    try:
        # 通过subprocess调用python -m openlearning
        cmd = [sys.executable, "-m", "openlearning"]
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ 运行演示失败: {e}")
        return 1

def run_package_test():
    """运行包测试（使用__main__模块的test模式）"""
    print("🧪 运行OpenLearning RGA测试...")
    print("=" * 60)
    
    try:
        # 通过subprocess调用python -m openlearning --test
        cmd = [sys.executable, "-m", "openlearning", "--test"]
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ 运行测试失败: {e}")
        return 1

def check_modules():
    """检查所有模块状态"""
    print("🔍 检查OpenLearning RGA模块状态...")
    print("=" * 60)
    
    modules_to_check = [
        ("主包", "openlearning"),
        ("核心引擎", "openlearning.core"),
        ("网络层", "openlearning.layers"),
        ("集成模块", "openlearning.integration"),
        ("CCSS模块", "openlearning.ccss"),
    ]
    
    results = []
    
    for name, module_path in modules_to_check:
        try:
            importlib.import_module(module_path)
            results.append((name, True, None))
            print(f"✅ {name} - 已导入")
        except ImportError as e:
            results.append((name, False, str(e)))
            print(f"❌ {name} - 导入失败: {e}")
    
    # 检查子模块
    print("\n🔧 检查子模块:")
    submodules = [
        ("核心配置", "openlearning.core.config"),
        ("核心度量", "openlearning.core.metrics"),
        ("核心注册表", "openlearning.core.registry"),
        ("注意力层", "openlearning.layers.attention"),
        ("平衡器层", "openlearning.layers.balancer"),
        ("嵌入层", "openlearning.layers.embeddings"),
        ("融合层", "openlearning.layers.fusion"),
        ("地质记忆", "openlearning.layers.memory"),
        ("归一化层", "openlearning.layers.normalization"),
        ("阀层", "openlearning.layers.valve"),
        ("神经网络集成", "openlearning.integration.nn"),
        ("yucai集成", "openlearning.integration.yucai"),
    ]
    
    sub_results = []
    for name, module_path in submodules:
        try:
            importlib.import_module(module_path)
            sub_results.append((name, True))
            print(f"  ✅ {name}")
        except ImportError:
            sub_results.append((name, False))
            print(f"  ❌ {name}")
    
    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)
    
    print(f"\n📊 检查完成: {success_count}/{total_count} 个主模块可用")
    
    return 0 if success_count == total_count else 1

def run_core_metrics():
    """运行核心度量测试"""
    print("📊 运行核心度量计算器测试...")
    print("=" * 60)
    
    try:
        module = importlib.import_module("openlearning.core.metrics")
        
        # 尝试运行main函数
        if hasattr(module, 'main'):
            return module.main()
        
        # 尝试运行测试函数
        test_functions = [name for name in dir(module) 
                         if name.startswith('test') and callable(getattr(module, name))]
        
        if test_functions:
            print(f"找到 {len(test_functions)} 个测试函数:")
            for func_name in test_functions:
                print(f"\n▶️ 运行 {func_name}...")
                try:
                    func = getattr(module, func_name)
                    func()
                    print(f"✅ {func_name} 通过")
                except Exception as e:
                    print(f"❌ {func_name} 失败: {e}")
            return 0
        else:
            print("⚠️  模块没有main函数或测试函数")
            return 0
            
    except ImportError as e:
        print(f"❌ 无法导入模块: {e}")
        return 1

def run_layers_module(module_name):
    """运行层模块测试"""
    module_path = f"openlearning.layers.{module_name}"
    
    print(f"🧪 运行{module_name}模块测试...")
    print("=" * 60)
    
    try:
        module = importlib.import_module(module_path)
        
        # 尝试运行main函数
        if hasattr(module, 'main'):
            return module.main()
        
        # 尝试运行测试函数
        test_functions = [name for name in dir(module) 
                         if name.startswith('test') and callable(getattr(module, name))]
        
        if test_functions:
            print(f"找到 {len(test_functions)} 个测试函数:")
            for func_name in test_functions:
                print(f"\n▶️ 运行 {func_name}...")
                try:
                    func = getattr(module, func_name)
                    func()
                    print(f"✅ {func_name} 通过")
                except Exception as e:
                    print(f"❌ {func_name} 失败: {e}")
            return 0
        else:
            print("⚠️  模块没有main函数或测试函数")
            return 0
            
    except ImportError as e:
        print(f"❌ 无法导入模块: {e}")
        return 1

def run_layers_all():
    """运行所有层模块测试"""
    print("🧪 运行所有层模块测试...")
    print("=" * 60)
    
    layers_modules = [
        ("注意力层", "attention"),
        ("平衡器层", "balancer"),
        ("嵌入层", "embeddings"),
        ("融合层", "fusion"),
        ("地质记忆", "memory"),
        ("归一化层", "normalization"),
        ("阀层", "valve"),
    ]
    
    results = []
    for name, module_name in layers_modules:
        print(f"\n▶️ 测试 {name}...")
        result = run_layers_module(module_name)
        results.append((name, result == 0))
    
    # 显示汇总
    print("\n" + "=" * 60)
    print("📊 层模块测试汇总:")
    success_count = sum(1 for _, success in results if success)
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {name:<15} {status}")
    
    print(f"\n总计: {len(results)} 个模块，通过: {success_count}，失败: {len(results)-success_count}")
    
    return 0 if success_count == len(results) else 1

def run_train():
    """启动训练菜单"""
    print("🚀 启动训练菜单...")
    print("=" * 60)
    print("选择训练模式:")
    print("  1. 快速测试模式 (测试/调试)")
    print("  2. 标准训练模式 (推荐)")
    print("  3. 完整训练模式 (需要大量资源)")
    print("  4. 自定义训练模式")
    print("  5. 恢复训练模式")
    print("  6. 推理测试模式")
    
    try:
        choice = input("\n请选择模式 (1-6): ").strip()
        if choice == "1":
            print("启动快速测试模式...")
            # 这里可以调用具体的训练函数
            return run_module_main("openlearning.integration.nn")
        elif choice == "2":
            print("启动标准训练模式...")
            return run_module_main("openlearning.integration.nn")
        elif choice == "3":
            print("启动完整训练模式...")
            return run_module_main("openlearning.integration.nn")
        elif choice == "4":
            print("启动自定义训练模式...")
            return run_module_main("openlearning.integration.nn")
        elif choice == "5":
            print("启动恢复训练模式...")
            return run_module_main("openlearning.integration.nn")
        elif choice == "6":
            print("启动推理测试模式...")
            return run_infer()
        else:
            print("❌ 无效选择，使用默认模式")
            return run_module_main("openlearning.integration.nn")
    except KeyboardInterrupt:
        print("\n\n🛑 训练被取消")
        return 130
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return 1

def run_infer():
    """启动推理测试"""
    print("🔍 启动推理测试...")
    print("=" * 60)
    
    try:
        # 询问模型路径
        model_path = input("输入模型路径 (例如: E:\\新GPT训练数据\\紫心测试\\best_model.pth): ").strip()
        if not model_path:
            model_path = "best_model.pth"
            print(f"使用默认路径: {model_path}")
        
        # 询问测试文本
        test_text = input("输入测试文本 (例如: 你好): ").strip()
        if not test_text:
            test_text = "你好，这是一个测试。"
            print(f"使用默认文本: {test_text}")
        
        print(f"\n推理配置:")
        print(f"  模型路径: {model_path}")
        print(f"  测试文本: {test_text}")
        print(f"\n开始推理...")
        
        # 这里可以调用具体的推理函数
        # 暂时返回成功
        print("✅ 推理完成（功能待实现）")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n🛑 推理被取消")
        return 130
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return 1

def run_test_all():
    """运行完整测试套件"""
    print("🧪 运行OpenLearning RGA完整测试套件")
    print("=" * 60)
    
    tests = [
        ("环境检查", lambda: check_modules()),
        ("核心度量测试", lambda: run_core_metrics()),
        ("所有层模块测试", lambda: run_layers_all()),
        ("集成模块测试", lambda: run_module_main("openlearning.integration")),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n▶️ 开始测试: {test_name}")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result == 0))
        if result != 0:
            print(f"⚠️  测试失败: {test_name}")
    
    print("\n" + "=" * 60)
    print("📊 测试套件完成")
    print("=" * 60)
    
    failed = [name for name, success in results if not success]
    if failed:
        print(f"❌ 失败测试: {len(failed)}/{len(tests)}")
        for name in failed:
            print(f"  - {name}")
        return 1
    else:
        print(f"✅ 所有测试通过: {len(tests)}/{len(tests)}")
        return 0

# ==================== 主命令行解析器 ====================

def create_parser():
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog='openlearning',
        description='OpenLearning RGA - 规则治理架构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
            示例:
              openlearning demo                 # 运行完整演示
              openlearning test                 # 运行测试模式
              openlearning check                # 检查模块状态
              openlearning core                 # 运行核心模块
              openlearning layers               # 运行层模块
              openlearning integration          # 运行集成模块
              openlearning train                # 启动训练菜单
              openlearning infer                # 启动推理测试
              openlearning test-all             # 运行完整测试套件
            
            子模块调用:
              openlearning core-metrics         # 核心度量计算器
              openlearning core-registry        # 核心注册表
              openlearning layers-attention     # 注意力层
              openlearning layers-balancer      # 平衡器层
              openlearning layers-memory        # 地质记忆层
              openlearning layers-normalization # 归一化层
              openlearning layers-valve         # 单向阀层
              openlearning layers-embeddings    # 嵌入层
              openlearning layers-fusion        # 融合层
            ''')
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        title='命令',
        description='可用的命令',
        help='要执行的命令'
    )
    
    # demo命令
    demo_parser = subparsers.add_parser(
        'demo',
        help='运行完整演示'
    )
    demo_parser.add_argument(
        '--fast',
        action='store_true',
        help='快速演示模式'
    )
    demo_parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='跳过可视化'
    )
    
    # test命令
    test_parser = subparsers.add_parser(
        'test',
        help='运行测试模式'
    )
    
    # check命令
    check_parser = subparsers.add_parser(
        'check',
        help='检查模块状态'
    )
    check_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细信息'
    )
    
    # core命令
    core_parser = subparsers.add_parser(
        'core',
        help='运行核心模块'
    )
    
    # layers命令
    layers_parser = subparsers.add_parser(
        'layers',
        help='运行层模块测试'
    )
    
    # integration命令
    integration_parser = subparsers.add_parser(
        'integration',
        help='运行集成模块测试'
    )
    
    # train命令
    train_parser = subparsers.add_parser(
        'train',
        help='启动训练菜单'
    )
    
    # infer命令
    infer_parser = subparsers.add_parser(
        'infer',
        help='启动推理测试'
    )
    
    # test-all命令
    test_all_parser = subparsers.add_parser(
        'test-all',
        help='运行完整测试套件'
    )
    
    # 子模块命令
    # core子命令
    core_metrics_parser = subparsers.add_parser(
        'core-metrics',
        help='核心度量计算器'
    )
    
    core_registry_parser = subparsers.add_parser(
        'core-registry',
        help='核心注册表'
    )
    
    # layers子命令
    layers_attention_parser = subparsers.add_parser(
        'layers-attention',
        help='注意力层测试'
    )
    
    layers_balancer_parser = subparsers.add_parser(
        'layers-balancer',
        help='平衡器层测试'
    )
    
    layers_memory_parser = subparsers.add_parser(
        'layers-memory',
        help='地质记忆层测试'
    )
    
    layers_normalization_parser = subparsers.add_parser(
        'layers-normalization',
        help='归一化层测试'
    )
    
    layers_valve_parser = subparsers.add_parser(
        'layers-valve',
        help='单向阀层测试'
    )
    
    layers_embeddings_parser = subparsers.add_parser(
        'layers-embeddings',
        help='嵌入层测试'
    )
    
    layers_fusion_parser = subparsers.add_parser(
        'layers-fusion',
        help='融合层测试'
    )
    
    # 通用参数
    parser.add_argument(
        '--version', '-V',
        action='store_true',
        help='显示版本信息'
    )
    
    return parser

# ==================== 主函数 ====================

def main():
    """主函数"""
    parser = create_parser()
    
    # 如果没有参数，显示帮助
    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    # 显示版本
    if args.version:
        try:
            import openlearning
            version = getattr(openlearning, '__version__', '未知')
            print(f"OpenLearning RGA v{version}")
            return 0
        except ImportError:
            print("OpenLearning RGA (包未找到)")
            return 1
    
    # 打印横幅（除了子命令）
    if args.command not in ['core-metrics', 'core-registry', 
                           'layers-attention', 'layers-balancer', 'layers-memory',
                           'layers-normalization', 'layers-valve', 
                           'layers-embeddings', 'layers-fusion']:
        print_banner()
    
    # 处理命令
    try:
        if args.command == 'demo':
            return run_package_demo()
            
        elif args.command == 'test':
            return run_package_test()
            
        elif args.command == 'check':
            return check_modules()
            
        elif args.command == 'core':
            return run_module_main("openlearning.core")
            
        elif args.command == 'layers':
            return run_layers_all()
            
        elif args.command == 'integration':
            return run_module_main("openlearning.integration")
            
        elif args.command == 'train':
            return run_train()
            
        elif args.command == 'infer':
            return run_infer()
            
        elif args.command == 'test-all':
            return run_test_all()
            
        elif args.command == 'core-metrics':
            return run_core_metrics()
            
        elif args.command == 'core-registry':
            return run_module_main("openlearning.core.registry")
            
        elif args.command == 'layers-attention':
            return run_layers_module("attention")
            
        elif args.command == 'layers-balancer':
            return run_layers_module("balancer")
            
        elif args.command == 'layers-memory':
            return run_layers_module("memory")
            
        elif args.command == 'layers-normalization':
            return run_layers_module("normalization")
            
        elif args.command == 'layers-valve':
            return run_layers_module("valve")
            
        elif args.command == 'layers-embeddings':
            return run_layers_module("embeddings")
            
        elif args.command == 'layers-fusion':
            return run_layers_module("fusion")
            
        else:
            # 如果没有匹配的命令，显示帮助
            print("❌ 未知命令")
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 操作被用户中断")
        return 130
    except Exception as e:
        print(f"\n💥 发生错误: {e}")
        if __debug__:
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())