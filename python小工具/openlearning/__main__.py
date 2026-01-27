"""
OpenLearning RGA - 主程序入口点
================================

当用户运行 `python -m openlearning` 时执行此文件。
提供完整的命令行界面和交互式环境。
"""

import sys
import os
import argparse
import textwrap
import time
from pathlib import Path
import subprocess
import json
import platform

# ==================== 环境检测和初始化 ====================

def check_environment():
    """检查运行环境"""
    print("🔍 环境检测中...")
    
    env_info = {
        "system": platform.system(),
        "python_version": platform.python_version(),
        "virtual_env": hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix),
        "current_dir": os.getcwd(),
        "script_dir": os.path.dirname(os.path.abspath(__file__)),
    }
    
    return env_info

def setup_environment():
    """设置运行环境"""
    # 添加项目根目录到sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"📁 项目根目录: {project_root}")
    return project_root

def check_dependencies():
    """检查依赖"""
    required_deps = ['numpy', 'torch']
    missing_deps = []
    
    print("📦 检查依赖...")
    for dep in required_deps:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            missing_deps.append(dep)
            print(f"   ❌ {dep}")
    
    if missing_deps:
        print(f"\n⚠️  缺少依赖: {', '.join(missing_deps)}")
        print("   运行以下命令安装: ")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    return True

# ==================== 核心功能模块 ====================

class RGACLI:
    """RGA 命令行主类"""
    
    def __init__(self):
        self.env = check_environment()
        self.project_root = setup_environment()
        self.banner = self._get_banner()
        
        # 尝试导入主包
        try:
            import openlearning
            self.rga = openlearning
            self.version = openlearning.__version__
            self.import_success = True
        except ImportError as e:
            print(f"❌ 导入失败: {e}")
            self.rga = None
            self.version = "未知"
            self.import_success = False
    
    def _get_banner(self):
        """获取ASCII艺术横幅"""
        return r"""
   ___                  _                _           
  / _ \ _ __   ___ _ __| | ___ __ _  __| | ___ _ __ 
 / /_)/| '_ \ / _ \ '__| |/ __/ _` |/ _` |/ _ \ '__|
/ ___/ | |_) |  __/ |  | | (_| (_| | (_| |  __/ |   
\/    | .__/ \___|_|  |_|\___\__,_|\__,_|\___|_|   
     |_|                                           
    """
    
    def print_banner(self):
        """打印横幅"""
        print(self.banner)
        print(f"🚀 OpenLearning RGA v{self.version}")
        print("=" * 60)
    
    def test_modules(self):
        """测试所有模块"""
        print("\n🧪 模块测试:")
        
        modules_to_test = [
            ("core", "核心引擎"),
            ("layers", "网络层"),
            ("integration", "集成模块"),
        ]
        
        results = []
        for module_name, description in modules_to_test:
            try:
                start_time = time.time()
                # 动态导入测试
                module = getattr(self.rga, module_name, None)
                duration = time.time() - start_time
                
                if module:
                    status = "✅"
                    message = f"{description} 加载成功 ({duration:.3f}s)"
                else:
                    status = "❌"
                    message = f"{description} 未找到"
                
                results.append((status, message))
                print(f"  {status} {message}")
                
            except Exception as e:
                results.append(("❌", f"{description} 失败: {str(e)}"))
                print(f"  ❌ {description} 失败: {str(e)}")
        
        return all(r[0] == "✅" for r in results)
    
    def run_demo(self):
        """运行演示示例"""
        print("\n🎮 运行演示...")
        
        if not self.import_success:
            print("  无法运行演示，导入失败")
            return
        
        try:
            # 运行一个简单的演示
            print("  1. 初始化RGA系统...")
            if hasattr(self.rga, 'initialize'):
                self.rga.initialize(verbose=True)
            
            print("  2. 检查导入状态...")
            if hasattr(self.rga, 'check_imports'):
                self.rga.check_imports()
            
            print("  3. 生成系统报告...")
            if hasattr(self.rga, 'generate_report'):
                report = self.rga.generate_report()
                print(report[:500] + "..." if len(report) > 500 else report)
            
            print("\n✅ 演示完成！")
            
        except Exception as e:
            print(f"❌ 演示失败: {e}")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
使用说明:

  1. 运行演示:
     python -m openlearning demo
  
  2. 测试模块:
     python -m openlearning test
  
  3. 进入交互模式:
     python -m openlearning interactive
  
  4. 显示系统信息:
     python -m openlearning info
  
  5. 检查依赖:
     python -m openlearning check

  6. 查看完整帮助:
     python -m openlearning help

示例:
  python -m openlearning demo --verbose
  python -m openlearning test --all
  python -m openlearning interactive --advanced
        """
        print(help_text)
    
    def show_info(self):
        """显示系统信息"""
        print("\n📊 系统信息:")
        print(f"  操作系统: {self.env['system']}")
        print(f"  Python版本: {self.env['python_version']}")
        print(f"  虚拟环境: {self.env['virtual_env']}")
        print(f"  当前目录: {self.env['current_dir']}")
        print(f"  RGA版本: {self.version}")
        print(f"  导入状态: {'✅ 成功' if self.import_success else '❌ 失败'}")
    
    def enter_interactive_mode(self, advanced=False):
        """进入交互模式"""
        print("\n💻 进入交互模式...")
        print("  输入 'help' 查看可用命令")
        print("  输入 'exit' 退出")
        
        if advanced and self.import_success:
            print("\n  高级功能已启用")
            print("  可以直接使用: rga.core, rga.layers, rga.integration")
        
        # 简单的交互循环
        while True:
            try:
                if advanced:
                    # 使用简单的输入（Windows兼容）
                    try:
                        import readline
                        command = input("\nRGA> ")
                    except ImportError:
                        # Windows环境，使用标准input
                        command = input("\nRGA> ")
                else:
                    command = input("\n命令> ").strip()
                
                if not command:
                    continue
                
                if command.lower() in ['exit', 'quit', 'q']:
                    print("👋 退出交互模式")
                    break
                
                elif command.lower() == 'help':
                    print("\n可用命令:")
                    print("  help     - 显示此帮助")
                    print("  info     - 显示系统信息")
                    print("  test     - 测试模块")
                    print("  demo     - 运行演示")
                    print("  modules  - 列出可用模块")
                    print("  exit     - 退出")
                    
                elif command.lower() == 'info':
                    self.show_info()
                
                elif command.lower() == 'test':
                    self.test_modules()
                
                elif command.lower() == 'demo':
                    self.run_demo()
                
                elif command.lower() == 'modules':
                    if self.import_success:
                        print("\n可用模块:")
                        for attr in dir(self.rga):
                            if not attr.startswith('_'):
                                print(f"  {attr}")
                    else:
                        print("模块未加载")
                
                elif command.startswith('!'):
                    # 执行系统命令
                    sys_command = command[1:].strip()
                    try:
                        result = subprocess.run(
                            sys_command, 
                            shell=True, 
                            capture_output=True, 
                            text=True
                        )
                        print(result.stdout)
                        if result.stderr:
                            print(f"错误: {result.stderr}")
                    except Exception as e:
                        print(f"执行失败: {e}")
                
                else:
                    print(f"未知命令: {command}")
                    print("输入 'help' 查看可用命令")
            
            except KeyboardInterrupt:
                print("\n\n中断操作，输入 'exit' 退出")
            except EOFError:
                print("\n\n退出交互模式")
                break
            except Exception as e:
                print(f"错误: {e}")

# ==================== 命令行解析器 ====================

def create_parser():
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        description='OpenLearning RGA - 规则治理架构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
            示例:
              %(prog)s demo              # 运行演示
              %(prog)s test --all        # 测试所有模块
              %(prog)s interactive       # 进入交互模式
              %(prog)s info              # 显示系统信息
            ''')
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='demo',
        choices=['demo', 'test', 'interactive', 'info', 'help', 'check'],
        help='要执行的命令'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细信息'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='测试所有模块（仅用于test命令）'
    )
    
    parser.add_argument(
        '--advanced', '-A',
        action='store_true',
        help='启用高级功能（仅用于interactive命令）'
    )
    
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
    args = parser.parse_args()
    
    # 初始化CLI
    cli = RGACLI()
    
    # 打印横幅
    cli.print_banner()
    
    # 检查依赖（如果命令需要）
    if args.command in ['demo', 'test', 'interactive']:
        if not check_dependencies():
            print("⚠️  某些功能可能无法正常工作")
    
    # 执行命令
    if args.version:
        print(f"版本: {cli.version}")
        return 0
    
    if args.command == 'demo':
        cli.run_demo()
    
    elif args.command == 'test':
        success = cli.test_modules()
        if args.all:
            print("\n📋 完整测试报告:")
            if success:
                print("✅ 所有测试通过！")
            else:
                print("❌ 某些测试失败")
        return 0 if success else 1
    
    elif args.command == 'interactive':
        cli.enter_interactive_mode(advanced=args.advanced)
    
    elif args.command == 'info':
        cli.show_info()
    
    elif args.command == 'check':
        check_dependencies()
    
    elif args.command == 'help':
        cli.show_help()
    
    else:
        # 默认执行演示
        cli.run_demo()
    
    print("\n✨ 感谢使用 OpenLearning RGA！")
    return 0

# ==================== 异常处理 ====================

def handle_exceptions(func):
    """异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\n\n🛑 操作被用户中断")
            return 130
        except Exception as e:
            print(f"\n💥 发生错误: {e}", file=sys.stderr)
            if __debug__:
                import traceback
                traceback.print_exc()
            return 1
    return wrapper

# ==================== 程序入口 ====================

if __name__ == "__main__":
    sys.exit(handle_exceptions(main)())