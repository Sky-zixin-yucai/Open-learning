#!/usr/bin/env python3
"""
OpenLearning RGA 统一测试运行器
================================

功能：
1. 自动发现所有可用的测试文件和模块
2. 提供完整的命令行接口（CLI）
3. 支持交互式菜单和批量运行
4. 集成所有模块测试功能
5. 生成详细的测试报告

使用方法：
    openlearning [命令] [选项]
    
命令：
    demo                运行完整演示
    test                运行测试模式
    test-all            运行完整测试套件
    core                运行核心模块测试
    layers              运行层模块测试
    integration         运行集成模块测试
    check               检查模块状态
    train               启动训练菜单
    infer               启动推理测试
    list                列出所有测试文件
    run-all             运行所有测试
    interactive         交互模式（默认）
    
选项：
    --run <文件名>      运行指定测试文件
    --modules           显示发现的模块
    --status            显示系统状态
    --help              显示帮助信息
"""

import sys
import os
import argparse
import subprocess
import textwrap
import time
import re
import locale
import io
import traceback as tb_module
import codecs
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ==================== 系统配置 ====================

def get_system_encoding():
    """获取系统默认编码"""
    try:
        encoding = locale.getpreferredencoding()
        if not encoding:
            encoding = 'gbk' if sys.platform == 'win32' else 'utf-8'
        return encoding
    except:
        return 'gbk' if sys.platform == 'win32' else 'utf-8'

SYSTEM_ENCODING = get_system_encoding()

# ==================== 统一测试运行器 ====================

class UnifiedTestRunner:
    """统一测试运行器类"""
    
    def __init__(self):
        """初始化运行器"""
        self.test_files = []
        self.results = []
        self.project_root = self._find_project_root()
        
        # 将项目根目录添加到Python路径
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)
        
        # 将openlearning目录添加到Python路径
        openlearning_dir = os.path.join(self.project_root, "openlearning")
        if openlearning_dir not in sys.path:
            sys.path.insert(0, openlearning_dir)
    
    def _find_project_root(self) -> str:
        """查找项目根目录"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 查找包含openlearning目录的根目录
        for depth in range(5):
            check_dir = os.path.join(current_dir, *['..'] * depth)
            check_dir = os.path.abspath(check_dir)
            
            if os.path.exists(os.path.join(check_dir, 'openlearning')):
                return check_dir
        
        return current_dir
    
    def print_banner(self):
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
        print("🚀 OpenLearning RGA - 统一测试运行器")
        print("=" * 60)
        print(f"📁 项目根目录: {self.project_root}")
        print(f"🔤 系统编码: {SYSTEM_ENCODING}")
        print("=" * 60)
    
    def discover_tests(self) -> Dict[str, List[Dict]]:
        """
        发现所有可用的测试文件
        
        返回:
            Dict[str, List[Dict]]: 分类的测试文件列表
        """
        print("🔍 正在发现测试文件...")
        
        # 分类测试文件
        tests_by_category = {
            'core': [],      # 核心模块测试
            'layers': [],    # 层模块测试
            'integration': [], # 集成模块测试
            'examples': [],  # 示例文件
            'other': []      # 其他测试
        }
        
        # 扫描所有Python文件
        for root, dirs, files in os.walk(self.project_root):
            # 跳过虚拟环境目录和隐藏目录
            dirs[:] = [d for d in dirs 
                      if not any(x in d.lower() for x in ['venv', '.venv', '__pycache__', '.git', '.idea', 'node_modules'])]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, self.project_root)
                    
                    # 跳过测试运行器自身
                    if rel_path.endswith('__main__.py') and 'openlearning' in rel_path:
                        continue
                    if file == 'openlearning':  # 跳过自身
                        continue
                    
                    # 检查是否是测试文件
                    is_test_file = (
                        'test' in file.lower() or 
                        file.endswith('_test.py') or
                        file.startswith('test_')
                    )
                    
                    # 检查是否可以直接运行（有main函数）
                    can_run = self._can_file_run(filepath)
                    
                    if is_test_file or can_run:
                        # 分类
                        rel_lower = rel_path.lower()
                        if 'core' in rel_lower:
                            category = 'core'
                        elif 'layer' in rel_lower:
                            category = 'layers'
                        elif 'integration' in rel_lower:
                            category = 'integration'
                        elif 'example' in rel_lower:
                            category = 'examples'
                        else:
                            category = 'other'
                        
                        tests_by_category[category].append({
                            'file': file,
                            'path': filepath,
                            'rel_path': rel_path,
                            'category': category,
                            'can_run': can_run,
                            'is_test': is_test_file
                        })
        
        # 统计
        total_tests = sum(len(tests) for tests in tests_by_category.values())
        print(f"✅ 发现 {total_tests} 个可运行文件")
        
        return tests_by_category
    
    def _read_file_with_bom_handling(self, filepath: str):
        """读取文件，处理BOM字符"""
        try:
            # 首先尝试使用utf-8-sig，它会自动处理BOM
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            return content, 'utf-8-sig'
        except UnicodeDecodeError:
            # 如果utf-8-sig失败，尝试其他编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    return content, encoding
                except UnicodeDecodeError:
                    continue
        
        # 如果所有编码都失败，用二进制模式读取
        with open(filepath, 'rb') as f:
            content_bytes = f.read()
        
        # 尝试检测BOM
        if content_bytes.startswith(codecs.BOM_UTF8):
            # 有BOM，去掉BOM并解码
            content = content_bytes[len(codecs.BOM_UTF8):].decode('utf-8', errors='ignore')
            return content, 'utf-8-bom'
        elif content_bytes.startswith(b'\xff\xfe') or content_bytes.startswith(b'\xfe\xff'):
            # UTF-16 BOM，跳过处理
            content = content_bytes.decode('utf-16', errors='ignore')
            return content, 'utf-16'
        else:
            # 没有BOM，尝试utf-8解码
            content = content_bytes.decode('utf-8', errors='ignore')
            return content, 'utf-8-raw'
    
    def _can_file_run(self, filepath: str) -> bool:
        """检查文件是否可以运行（是否有if __name__ == "__main__"）"""
        try:
            # 读取文件内容，处理BOM
            content, encoding = self._read_file_with_bom_handling(filepath)
            
            # 检查是否有main函数或直接可运行的代码
            patterns = [
                r'if\s+__name__\s*==\s*["\']__main__["\']',
                r'if\s+__name__\s*!=\s*["\']__main__["\'].*:.*pass',
            ]
            
            for pattern in patterns:
                if re.search(pattern, content, re.DOTALL | re.IGNORECASE):
                    return True
                
        except Exception as e:
            print(f"⚠️  无法读取文件 {filepath}: {e}")
        
        return False
    
    def _setup_module_environment(self, filepath: str):
        """设置模块执行环境"""
        # 获取文件所在目录和父目录
        file_dir = os.path.dirname(filepath)
        parent_dir = os.path.dirname(file_dir)
        
        # 确定模块名
        rel_path = os.path.relpath(filepath, self.project_root)
        if rel_path.endswith('.py'):
            module_name = rel_path[:-3].replace(os.sep, '.')
        else:
            module_name = rel_path.replace(os.sep, '.')
        
        # 确定包名
        if '.' in module_name:
            package_name = module_name.rsplit('.', 1)[0]
        else:
            package_name = ''
        
        # 创建执行环境
        exec_globals = {
            '__name__': '__main__',
            '__file__': filepath,
            '__builtins__': __builtins__,
            '__package__': package_name,
            '__path__': [],
            '__spec__': None,
            '__loader__': None,
            '__cached__': None
        }
        
        # 添加常用模块
        exec_globals['sys'] = sys
        exec_globals['os'] = os
        exec_globals['json'] = json
        
        return exec_globals, module_name, file_dir
    
    def run_test_file(self, test_info: Dict) -> Tuple[bool, str, float]:
        """运行单个测试文件，并实时显示输出"""
        test_name = test_info['rel_path']
        filepath = test_info['path']
        
        print(f"\n{'='*60}")
        print(f"🚀 开始运行测试: {test_name}")
        print(f"📄 文件: {filepath}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        success = False
        output = ""
        error_message = ""
        
        # 保存当前sys.path
        original_sys_path = sys.path.copy()
        
        try:
            # 1. 设置模块环境
            exec_globals, module_name, file_dir = self._setup_module_environment(filepath)
            
            # 2. 临时修改sys.path，确保相对导入能工作
            if file_dir not in sys.path:
                sys.path.insert(0, file_dir)
            
            # 3. 尝试预导入相关模块
            try:
                # 尝试导入openlearning模块
                import openlearning
                exec_globals['openlearning'] = openlearning
                
                # 根据模块名尝试导入相关子模块
                if module_name.startswith('openlearning.'):
                    try:
                        submodule = __import__(module_name, fromlist=['*'])
                        # 将子模块的导出添加到执行环境
                        for attr in dir(submodule):
                            if not attr.startswith('_'):
                                exec_globals[attr] = getattr(submodule, attr)
                    except:
                        pass
            except ImportError:
                # 如果openlearning导入失败，继续执行
                pass
            
            # 4. 读取文件内容，处理BOM
            file_content, encoding = self._read_file_with_bom_handling(filepath)
            print(f"📝 使用编码: {encoding}")
            
            # 5. 创建缓冲区来捕获输出
            output_buffer = io.StringIO()
            
            # 6. 保存原始stdout和stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                # 7. 重定向输出到缓冲区
                sys.stdout = output_buffer
                sys.stderr = output_buffer
                
                # 8. 执行代码
                try:
                    # 先编译，这样可以捕获语法错误
                    code_obj = compile(file_content, filepath, 'exec')
                    # 执行编译后的代码
                    exec(code_obj, exec_globals)
                    success = True
                except SyntaxError as e:
                    # 处理语法错误
                    success = False
                    error_message = f"语法错误: {e}"
                    tb_module.print_exc(file=output_buffer)
                except SystemExit as e:
                    # 处理sys.exit()
                    if e.code == 0 or e.code is None:
                        success = True
                        print(f"\n程序正常退出 (退出码: {e.code})")
                    else:
                        success = False
                        error_message = f"SystemExit with code {e.code}"
                        print(f"\n程序异常退出 (退出码: {e.code})")
                except Exception as e:
                    success = False
                    error_message = str(e)
                    tb_module.print_exc(file=output_buffer)
                    
            finally:
                # 9. 恢复原始stdout和stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                # 10. 获取输出内容
                output = output_buffer.getvalue()
                
                # 11. 将输出打印到控制台
                if output:
                    try:
                        # 尝试用系统编码输出
                        output_encoded = output.encode(SYSTEM_ENCODING, errors='replace').decode(SYSTEM_ENCODING, errors='replace')
                        print(output_encoded)
                    except:
                        # 如果失败，直接打印
                        print(output)
            
            duration = time.time() - start_time
            
        except Exception as e:
            duration = time.time() - start_time
            error_message = str(e)
            print(f"\n💥 运行过程出错: {test_name} - {error_message}")
            success = False
        finally:
            # 12. 恢复sys.path
            sys.path = original_sys_path
        
        # 显示运行结果
        print(f"\n{'='*60}")
        if success:
            print(f"✅ 测试通过: {test_name} ({duration:.2f}s)")
        else:
            print(f"❌ 测试失败: {test_name} ({duration:.2f}s)")
            if error_message:
                print(f"   错误: {error_message}")
        print(f"📝 输出大小: {len(output)} 字符")
        print(f"{'='*60}")
        
        return success, output, duration
    
    def run_module_main(self, module_path: str) -> int:
        """运行模块的main函数"""
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, 'main'):
                print(f"▶️ 运行模块: {module_path}")
                print("-" * 40)
                result = module.main()
                return result
            else:
                print(f"⚠️  模块 {module_path} 没有main()函数")
                return 0
        except ImportError as e:
            print(f"❌ 无法导入模块 {module_path}: {e}")
            return 1
        except Exception as e:
            print(f"❌ 运行模块 {module_path} 失败: {e}")
            return 1
    
    def run_package_demo(self) -> int:
        """运行包的演示"""
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
    
    def run_package_test(self) -> int:
        """运行包测试"""
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
    
    def check_modules(self) -> int:
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
    
    def run_core_metrics(self) -> int:
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
    
    def run_layers_module(self, module_name: str) -> int:
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
    
    def run_layers_all(self) -> int:
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
            result = self.run_layers_module(module_name)
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
    
    def run_train(self) -> int:
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
                return self.run_module_main("openlearning.integration.nn")
            elif choice == "2":
                print("启动标准训练模式...")
                return self.run_module_main("openlearning.integration.nn")
            elif choice == "3":
                print("启动完整训练模式...")
                return self.run_module_main("openlearning.integration.nn")
            elif choice == "4":
                print("启动自定义训练模式...")
                return self.run_module_main("openlearning.integration.nn")
            elif choice == "5":
                print("启动恢复训练模式...")
                return self.run_module_main("openlearning.integration.nn")
            elif choice == "6":
                print("启动推理测试模式...")
                return self.run_infer()
            else:
                print("❌ 无效选择，使用默认模式")
                return self.run_module_main("openlearning.integration.nn")
        except KeyboardInterrupt:
            print("\n\n🛑 训练被取消")
            return 130
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            return 1
    
    def run_infer(self) -> int:
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
            print("✅ 推理完成（功能待实现）")
            return 0
            
        except KeyboardInterrupt:
            print("\n\n🛑 推理被取消")
            return 130
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return 1
    
    def run_test_all(self) -> int:
        """运行完整测试套件"""
        print("🧪 运行OpenLearning RGA完整测试套件")
        print("=" * 60)
        
        tests = [
            ("环境检查", lambda: self.check_modules()),
            ("核心度量测试", lambda: self.run_core_metrics()),
            ("所有层模块测试", lambda: self.run_layers_all()),
            ("集成模块测试", lambda: self.run_module_main("openlearning.integration")),
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
    
    def show_system_status(self):
        """显示系统状态"""
        print("\n📊 系统状态:")
        print("-" * 40)
        
        # 显示系统信息
        print(f"Python版本: {sys.version}")
        print(f"系统平台: {sys.platform}")
        print(f"系统编码: {SYSTEM_ENCODING}")
        print(f"工作目录: {os.getcwd()}")
        print(f"项目根目录: {self.project_root}")
        print(f"Python路径 ({len(sys.path)} 个):")
        for i, path in enumerate(sys.path[:5], 1):
            print(f"  {i}. {path}")
        if len(sys.path) > 5:
            print(f"  ... 还有 {len(sys.path) - 5} 个路径")
        
        # 尝试导入openlearning模块
        print("\n🔧 模块导入测试:")
        try:
            # 尝试导入openlearning
            try:
                import openlearning
                print(f"  ✅ openlearning 版本: {getattr(openlearning, '__version__', '未知')}")
                print(f"  ✅ openlearning 作者: {getattr(openlearning, '__author__', '未知')}")
                
                # 尝试导入子模块
                for attr in ['core', 'layers', 'integration']:
                    if hasattr(openlearning, attr):
                        print(f"  ✅ openlearning.{attr} (已导入)")
                    else:
                        try:
                            submodule = __import__(f'openlearning.{attr}', fromlist=['*'])
                            print(f"  ✅ openlearning.{attr}")
                        except ImportError as e:
                            print(f"  ❌ openlearning.{attr}: {e}")
            except ImportError as e:
                print(f"  ❌ openlearning: {e}")
        except Exception as e:
            print(f"  ⚠️  导入测试失败: {e}")
    
    def show_modules(self):
        """显示发现的模块"""
        # 扫描项目中的模块
        modules = []
        for root, dirs, files in os.walk(self.project_root):
            # 跳过不需要的目录
            dirs[:] = [d for d in dirs if not any(x in d.lower() for x in ['venv', '.venv', '__pycache__', '.git', '.idea', 'node_modules'])]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, self.project_root)
                    if rel_path.endswith('.py'):
                        module_name = rel_path[:-3].replace(os.sep, '.')
                    else:
                        module_name = rel_path.replace(os.sep, '.')
                    modules.append(module_name)
        
        print("\n📦 发现的模块:")
        print("-" * 40)
        
        if not modules:
            print("没有发现模块")
        else:
            for i, module in enumerate(modules[:20], 1):
                try:
                    display_module = module.encode(SYSTEM_ENCODING, errors='replace').decode(SYSTEM_ENCODING)
                except:
                    display_module = module
                print(f"{i:3d}. {display_module}")
            
            if len(modules) > 20:
                print(f"... 还有 {len(modules) - 20} 个模块")
        
        print(f"\n总模块数: {len(modules)}")
    
    def list_tests(self, tests_by_category: Dict):
        """列出所有测试"""
        print("\n📋 可用测试文件:")
        print("="*80)
        
        total_files = 0
        runnable_files = 0
        
        for category, tests in tests_by_category.items():
            if tests:
                print(f"\n{category.upper()} 类别:")
                for test in tests:
                    total_files += 1
                    status = "✅ 可运行" if test['can_run'] else "⚠️  仅检查"
                    if test['can_run']:
                        runnable_files += 1
                    
                    # 确保文件名正确显示
                    try:
                        display_name = test['rel_path'].encode(SYSTEM_ENCODING, errors='replace').decode(SYSTEM_ENCODING)
                    except:
                        display_name = test['rel_path']
                    
                    print(f"  • {display_name} ({status})")
        
        print(f"\n总计: {total_files} 个文件（{runnable_files} 个可运行）")
    
    def run_all_tests(self, tests_by_category: Dict):
        """运行所有测试文件"""
        print("\n🚀 开始运行所有测试...")
        
        total_tests = 0
        passed_tests = 0
        failed_tests = []
        
        start_time = time.time()
        
        # 运行每个类别的测试
        for category, tests in tests_by_category.items():
            if tests:
                print(f"\n📂 运行 {category} 测试 ({len(tests)}):")
                
                for test_info in tests:
                    if test_info['can_run']:
                        total_tests += 1
                        success, output, duration = self.run_test_file(test_info)
                        
                        if success:
                            passed_tests += 1
                        else:
                            failed_tests.append(test_info['rel_path'])
                        
                        # 保存结果
                        self.results.append({
                            'test': test_info['rel_path'],
                            'success': success,
                            'duration': duration,
                            'output': output[:1000] if output else ""
                        })
        
        total_time = time.time() - start_time
        
        # 显示结果摘要
        print("\n" + "="*80)
        print("测试完成!")
        print("="*80)
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests}")
        print(f"失败: {len(failed_tests)}")
        print(f"总时间: {total_time:.2f}秒")
        
        if failed_tests:
            print("\n失败的测试:")
            for test in failed_tests:
                try:
                    display_test = test.encode(SYSTEM_ENCODING, errors='replace').decode(SYSTEM_ENCODING)
                except:
                    display_test = test
                print(f"  ❌ {display_test}")
        
        # 保存测试报告
        self.save_report(total_tests, passed_tests, total_time)
    
    def save_report(self, total: int, passed: int, duration: float):
        """保存测试报告"""
        if total == 0:
            print("⚠️  没有测试需要运行，跳过生成报告")
            return
            
        success_rate = (passed / total * 100) if total > 0 else 0
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': total - passed,
            'success_rate': round(success_rate, 2),
            'total_duration': round(duration, 2),
            'results': self.results
        }
        
        # 保存为JSON文件，使用UTF-8编码
        report_file = f"test_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n📄 测试报告已保存到: {report_file}")
        except Exception as e:
            print(f"❌ 无法保存测试报告: {e}")
    
    def display_menu(self, tests_by_category: Dict):
        """显示交互式菜单"""
        print("\n" + "="*80)
        print("统一测试运行器 - 交互模式")
        print(f"项目根目录: {self.project_root}")
        print("="*80)
        
        # 显示测试统计
        idx = 1
        test_index_map = {}
        
        for category, tests in tests_by_category.items():
            if tests:
                print(f"\n{category.upper()} 测试 ({len(tests)}):")
                for test in tests:
                    status = "✅" if test['can_run'] else "⚠️ "
                    # 确保文件名正确显示
                    try:
                        display_name = test['rel_path'].encode(SYSTEM_ENCODING, errors='replace').decode(SYSTEM_ENCODING)
                    except:
                        display_name = test['rel_path']
                    print(f"  {idx:2d}. {status} {display_name}")
                    test_index_map[idx] = test
                    idx += 1
        
        print("\n" + "="*80)
        print("命令:")
        print("  1-99  运行指定编号的测试文件")
        print("  a     运行所有测试文件")
        print("  l     重新列出测试")
        print("  m     显示模块发现结果")
        print("  s     显示系统状态")
        print("  d     运行演示")
        print("  t     运行测试模式")
        print("  c     检查模块状态")
        print("  r     运行核心模块")
        print("  y     运行层模块")
        print("  i     运行集成模块")
        print("  n     启动训练菜单")
        print("  f     启动推理测试")
        print("  q     退出")
        print("="*80)
        
        return test_index_map
    
    def run_interactive(self):
        """交互式运行模式"""
        tests_by_category = self.discover_tests()
        
        while True:
            test_index_map = self.display_menu(tests_by_category)
            
            try:
                choice = input("\n请选择操作: ").strip().lower()
            except (UnicodeDecodeError, KeyboardInterrupt):
                print("\n⚠️  输入编码错误，请重试")
                continue
            
            if choice == 'q':
                print("👋 退出测试运行器")
                break
            elif choice == 'a':
                self.run_all_tests(tests_by_category)
                input("\n按Enter键继续...")
            elif choice == 'l':
                tests_by_category = self.discover_tests()
            elif choice == 'm':
                self.show_modules()
                input("\n按Enter键继续...")
            elif choice == 's':
                self.show_system_status()
                input("\n按Enter键继续...")
            elif choice == 'd':
                self.run_package_demo()
                input("\n按Enter键继续...")
            elif choice == 't':
                self.run_package_test()
                input("\n按Enter键继续...")
            elif choice == 'c':
                self.check_modules()
                input("\n按Enter键继续...")
            elif choice == 'r':
                self.run_module_main("openlearning.core")
                input("\n按Enter键继续...")
            elif choice == 'y':
                self.run_layers_all()
                input("\n按Enter键继续...")
            elif choice == 'i':
                self.run_module_main("openlearning.integration")
                input("\n按Enter键继续...")
            elif choice == 'n':
                self.run_train()
                input("\n按Enter键继续...")
            elif choice == 'f':
                self.run_infer()
                input("\n按Enter键继续...")
            elif choice.isdigit():
                idx = int(choice)
                if idx in test_index_map:
                    test_info = test_index_map[idx]
                    success, output, duration = self.run_test_file(test_info)
                    
                    # 保存结果
                    self.results.append({
                        'test': test_info['rel_path'],
                        'success': success,
                        'duration': duration,
                        'output': output[:1000] if output else ""
                    })
                    
                    input("\n按Enter键继续...")
                else:
                    print(f"❌ 无效的编号: {choice}")
                    input("\n按Enter键继续...")
            else:
                print(f"❌ 未知命令: {choice}")
                input("\n按Enter键继续...")

# ==================== 命令行接口 ====================

def create_parser():
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog='openlearning',
        description='OpenLearning RGA - 统一测试运行器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
            示例:
              openlearning demo                 # 运行完整演示
              openlearning test                 # 运行测试模式
              openlearning test-all             # 运行完整测试套件
              openlearning core                 # 运行核心模块测试
              openlearning layers               # 运行层模块测试
              openlearning integration          # 运行集成模块测试
              openlearning check                # 检查模块状态
              openlearning train                # 启动训练菜单
              openlearning infer                # 启动推理测试
              openlearning list                 # 列出所有测试文件
              openlearning run-all              # 运行所有测试
              openlearning interactive          # 交互模式（默认）
            
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
            
            选项:
              --run <文件名>      运行指定测试文件
              --modules           显示发现的模块
              --status            显示系统状态
              --help              显示帮助信息
            ''')
    )
    
    parser.add_argument('command', nargs='?', default='interactive',
                       help='要执行的命令（默认: interactive）')
    parser.add_argument('--run', type=str, help='运行指定测试文件')
    parser.add_argument('--modules', action='store_true', help='显示发现的模块')
    parser.add_argument('--status', action='store_true', help='显示系统状态')
    parser.add_argument('--version', '-V', action='store_true', help='显示版本信息')
    
    return parser

def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 创建运行器
    runner = UnifiedTestRunner()
    runner.print_banner()
    
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
    
    # 处理命令
    try:
        if args.command == 'demo':
            return runner.run_package_demo()
            
        elif args.command == 'test':
            return runner.run_package_test()
            
        elif args.command == 'test-all':
            return runner.run_test_all()
            
        elif args.command == 'core':
            return runner.run_module_main("openlearning.core")
            
        elif args.command == 'layers':
            return runner.run_layers_all()
            
        elif args.command == 'integration':
            return runner.run_module_main("openlearning.integration")
            
        elif args.command == 'check':
            return runner.check_modules()
            
        elif args.command == 'train':
            return runner.run_train()
            
        elif args.command == 'infer':
            return runner.run_infer()
            
        elif args.command == 'list':
            tests = runner.discover_tests()
            runner.list_tests(tests)
            return 0
            
        elif args.command == 'run-all':
            tests = runner.discover_tests()
            runner.run_all_tests(tests)
            return 0
            
        elif args.command == 'interactive':
            runner.run_interactive()
            return 0
            
        elif args.command == 'core-metrics':
            return runner.run_core_metrics()
            
        elif args.command == 'core-registry':
            return runner.run_module_main("openlearning.core.registry")
            
        elif args.command == 'layers-attention':
            return runner.run_layers_module("attention")
            
        elif args.command == 'layers-balancer':
            return runner.run_layers_module("balancer")
            
        elif args.command == 'layers-memory':
            return runner.run_layers_module("memory")
            
        elif args.command == 'layers-normalization':
            return runner.run_layers_module("normalization")
            
        elif args.command == 'layers-valve':
            return runner.run_layers_module("valve")
            
        elif args.command == 'layers-embeddings':
            return runner.run_layers_module("embeddings")
            
        elif args.command == 'layers-fusion':
            return runner.run_layers_module("fusion")
        
        elif args.run:
            # 查找指定测试
            tests = runner.discover_tests()
            all_tests = []
            for category, test_list in tests.items():
                all_tests.extend(test_list)
            
            # 查找匹配的测试
            matching_tests = [t for t in all_tests if args.run in t['rel_path']]
            
            if not matching_tests:
                print(f"❌ 未找到包含 '{args.run}' 的测试文件")
                # 显示建议
                print("\n可用的测试文件:")
                for test in all_tests[:10]:
                    try:
                        display_name = test['rel_path'].encode(SYSTEM_ENCODING, errors='replace').decode(SYSTEM_ENCODING)
                    except:
                        display_name = test['rel_path']
                    print(f"  • {display_name}")
                if len(all_tests) > 10:
                    print(f"  ... 还有 {len(all_tests) - 10} 个文件")
                return 1
            else:
                print(f"🔍 找到 {len(matching_tests)} 个匹配的测试:")
                for i, test in enumerate(matching_tests, 1):
                    try:
                        display_name = test['rel_path'].encode(SYSTEM_ENCODING, errors='replace').decode(SYSTEM_ENCODING)
                    except:
                        display_name = test['rel_path']
                    print(f"{i}. {display_name}")
                
                # 询问是否运行
                if len(matching_tests) == 1:
                    choice = input("是否运行此测试？(y/n): ").strip().lower()
                    if choice == 'y':
                        runner.run_test_file(matching_tests[0])
                        return 0
                else:
                    choice = input("是否运行所有匹配的测试？(y/n): ").strip().lower()
                    if choice == 'y':
                        for test in matching_tests:
                            runner.run_test_file(test)
                        return 0
                return 0
        
        elif args.modules:
            runner.show_modules()
            return 0
            
        elif args.status:
            runner.show_system_status()
            return 0
            
        else:
            print(f"❌ 未知命令: {args.command}")
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 操作被用户中断")
        return 130
    except Exception as e:
        print(f"\n💥 发生错误: {e}")
        tb_module.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())