"""
OpenLearning RGA 命令行接口
===========================

完整的功能调用接口，支持所有模块和组件。

使用方法：
    openlearning                      # 显示帮助信息
    openlearning demo                 # 运行完整演示
    openlearning test                 # 运行所有测试
    openlearning core                 # 运行核心模块测试
    openlearning layers               # 运行层模块测试
    openlearning integration          # 运行集成模块测试
    openlearning train                # 启动训练菜单
    openlearning infer                # 启动推理测试
    openlearning check                # 检查模块状态

子模块调用：
    openlearning core-metrics         # 运行核心度量测试
    openlearning core-registry        # 运行核心注册表
    openlearning layers-attention     # 运行注意力层测试
    openlearning layers-balancer      # 运行平衡器层测试
    openlearning layers-memory        # 运行地质记忆测试
    openlearning layers-normalization # 运行归一化测试
    openlearning layers-valve         # 运行阀层测试
    openlearning layers-embeddings    # 运行嵌入层测试
    openlearning layers-fusion        # 运行融合层测试
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(module_name, *args):
    """运行指定的Python模块"""
    cmd = [sys.executable, "-m", module_name] + list(args)
    print(f"运行命令: {' '.join(cmd)}")
    print("=" * 80)
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"错误: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="OpenLearning RGA 命令行接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  openlearning demo                 # 运行完整演示
  openlearning test                 # 运行所有测试
  openlearning core                 # 运行核心模块测试
  openlearning layers               # 运行层模块测试
  openlearning integration          # 运行集成模块测试
  openlearning train                # 启动训练菜单
  openlearning infer                # 启动推理测试
  
  # 子模块调用
  openlearning layers-attention     # 运行注意力层测试
  openlearning layers-memory        # 运行地质记忆测试
  openlearning core-metrics         # 运行核心度量测试
  
  # 完整测试流程
  openlearning test-all             # 运行完整测试套件
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        help="要执行的命令"
    )
    
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="传递给命令的参数"
    )
    
    args = parser.parse_args()
    
    commands = {
        # 主命令
        "demo": lambda: run_command("openlearning", "--demo"),
        "test": lambda: run_command("openlearning", "--test"),
        "check": lambda: run_command("openlearning", "--check-modules"),
        
        # 模块级命令
        "core": lambda: run_command("openlearning.core"),
        "layers": lambda: run_command("openlearning.layers"),
        "integration": lambda: run_command("openlearning.integration"),
        
        # 核心模块子命令
        "core-metrics": lambda: run_command("openlearning.core.metrics"),
        "core-registry": lambda: run_command("openlearning.core.registry"),
        
        # 层模块子命令
        "layers-attention": lambda: run_command("openlearning.layers.attention"),
        "layers-balancer": lambda: run_command("openlearning.layers.balancer"),
        "layers-embeddings": lambda: run_command("openlearning.layers.embeddings"),
        "layers-fusion": lambda: run_command("openlearning.layers.fusion"),
        "layers-memory": lambda: run_command("openlearning.layers.memory"),
        "layers-normalization": lambda: run_command("openlearning.layers.normalization"),
        "layers-valve": lambda: run_command("openlearning.layers.valve"),
        
        # 训练和推理
        "train": lambda: run_command("openlearning.integration.nn"),
        "infer": lambda: run_command("openlearning.integration.nn"),
        
        # 测试套件
        "test-all": lambda: run_test_suite(),
        
        # 帮助
        "help": lambda: print_help(parser),
    }
    
    command = args.command
    if command in commands:
        return commands[command]()
    else:
        print(f"错误: 未知命令 '{command}'")
        print_help(parser)
        return 1

def run_test_suite():
    """运行完整的测试套件"""
    print("=" * 80)
    print("运行 OpenLearning RGA 完整测试套件")
    print("=" * 80)
    
    tests = [
        ("检查模块状态", lambda: run_command("openlearning", "--check-modules")),
        ("核心模块测试", lambda: run_command("openlearning.core.metrics")),
        ("层模块测试", lambda: run_command("openlearning.layers")),
        ("注意力层测试", lambda: run_command("openlearning.layers.attention")),
        ("平衡器层测试", lambda: run_command("openlearning.layers.balancer")),
        ("地质记忆测试", lambda: run_command("openlearning.layers.memory")),
        ("集成模块测试", lambda: run_command("openlearning.integration")),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"开始测试: {test_name}")
        print(f"{'='*60}")
        result = test_func()
        results.append((test_name, result))
        if result != 0:
            print(f"⚠️  测试失败: {test_name}")
    
    print(f"\n{'='*80}")
    print("测试套件完成")
    print(f"{'='*80}")
    
    failed = [name for name, result in results if result != 0]
    if failed:
        print(f"❌ 失败测试: {len(failed)}/{len(tests)}")
        for name in failed:
            print(f"  - {name}")
        return 1
    else:
        print(f"✅ 所有测试通过: {len(tests)}/{len(tests)}")
        return 0

def print_help(parser):
    """打印帮助信息"""
    print("OpenLearning RGA - 规则治理架构")
    print("=" * 80)
    parser.print_help()
    print("\n快速参考:")
    print("  openlearning demo                 # 运行完整演示")
    print("  openlearning test                 # 运行测试模式")
    print("  openlearning check                # 检查模块状态")
    print("  openlearning core                 # 运行核心模块")
    print("  openlearning layers               # 运行层模块")
    print("  openlearning integration          # 运行集成模块")
    print("  openlearning train                # 启动训练菜单")
    print("  openlearning infer                # 启动推理测试")
    print("  openlearning test-all             # 运行完整测试套件")
    print()
    print("详细模块调用:")
    print("  核心模块:")
    print("    openlearning core-metrics       # 核心度量计算器")
    print("    openlearning core-registry      # 核心注册表")
    print("  层模块:")
    print("    openlearning layers-attention   # 注意力层")
    print("    openlearning layers-balancer    # 平衡器层")
    print("    openlearning layers-memory      # 地质记忆层")
    print("    openlearning layers-normalization # 归一化层")
    print("    openlearning layers-valve       # 单向阀层")
    print("    openlearning layers-embeddings  # 嵌入层")
    print("    openlearning layers-fusion      # 融合层")

if __name__ == "__main__":
    sys.exit(main())