#!/usr/bin/env python3
"""
OpenLearning RGA 模块探查测试
==============================

这个脚本会实际测试每个模块是否能：
1. 正确导入
2. 运行main函数（如果有）
3. 运行测试函数（如果有）

直接在项目根目录运行：python test_all_modules.py
"""

import sys
import os
import importlib
import traceback
import inspect
from pathlib import Path
import time

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

print("🔬 OpenLearning RGA 模块深度探查")
print("=" * 80)

def print_color(text, color_code="\033[92m"):
    """彩色打印"""
    end_code = "\033[0m"
    if sys.platform == "win32":
        # Windows控制台可能不支持ANSI颜色
        print(text)
    else:
        print(f"{color_code}{text}{end_code}")

def test_import_module(module_path):
    """测试导入模块"""
    start_time = time.time()
    try:
        module = importlib.import_module(module_path)
        duration = time.time() - start_time
        
        # 检查模块属性
        module_info = {
            "name": module_path,
            "status": "✅ 导入成功",
            "duration": duration,
            "has_main": hasattr(module, "main"),
            "has_init": hasattr(module, "__init__") and callable(module.__init__),
            "functions": [],
            "classes": [],
            "attributes": [],
        }
        
        # 获取模块成员
        for name in dir(module):
            if name.startswith("_"):
                continue
                
            obj = getattr(module, name)
            
            if callable(obj):
                if name.startswith("test") or name.endswith("_test"):
                    module_info["functions"].append(f"🧪 {name}()")
                elif inspect.isclass(obj):
                    module_info["classes"].append(f"🏗️  {name}")
                else:
                    module_info["functions"].append(f"🔧 {name}()")
            else:
                module_info["attributes"].append(f"📝 {name}")
        
        return module_info, None
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            "name": module_path,
            "status": f"❌ 导入失败",
            "duration": duration,
            "error": str(e)
        }, str(e)

def test_module_main(module_path):
    """测试模块的main函数"""
    try:
        module = importlib.import_module(module_path)
        
        if hasattr(module, "main"):
            print(f"  🚀 尝试运行main()...")
            start_time = time.time()
            try:
                result = module.main()
                duration = time.time() - start_time
                return f"✅ main()运行成功 (返回: {result}, 耗时: {duration:.3f}s)"
            except Exception as e:
                return f"❌ main()运行失败: {str(e)[:100]}"
        else:
            return "⚠️  无main()函数"
            
    except Exception as e:
        return f"❌ 无法导入模块: {str(e)[:100]}"

def test_module_functions(module_path):
    """测试模块中的test函数"""
    try:
        module = importlib.import_module(module_path)
        
        test_functions = []
        for name in dir(module):
            if (name.startswith("test") or name.endswith("_test")) and callable(getattr(module, name)):
                test_functions.append(name)
        
        if not test_functions:
            return "⚠️  无test函数", []
        
        results = []
        for func_name in test_functions[:3]:  # 只测试前3个
            try:
                func = getattr(module, func_name)
                start_time = time.time()
                func()
                duration = time.time() - start_time
                results.append(f"✅ {func_name}() (耗时: {duration:.3f}s)")
            except Exception as e:
                results.append(f"❌ {func_name}() 失败: {str(e)[:80]}")
        
        return f"找到 {len(test_functions)} 个test函数", results
        
    except Exception as e:
        return f"❌ 无法导入模块: {str(e)[:100]}", []

def analyze_module_structure(module_path):
    """深度分析模块结构"""
    print(f"\n🔍 分析模块: {module_path}")
    print("-" * 60)
    
    # 1. 测试导入
    module_info, error = test_import_module(module_path)
    
    if error:
        print(f"  ❌ 无法导入: {error}")
        return False
    
    # 显示基本信息
    print(f"  {module_info['status']} (耗时: {module_info['duration']:.3f}s)")
    
    if module_info["has_main"]:
        print(f"  📌 有main()函数")
    
    # 2. 测试main函数
    if module_info["has_main"]:
        main_result = test_module_main(module_path)
        print(f"  {main_result}")
    
    # 3. 查找test函数
    test_result, test_details = test_module_functions(module_path)
    print(f"  {test_result}")
    
    if test_details:
        for detail in test_details:
            print(f"    {detail}")
    
    # 4. 显示模块内容摘要
    if module_info["classes"]:
        print(f"  🏗️  类: {', '.join(module_info['classes'][:5])}")
        if len(module_info["classes"]) > 5:
            print(f"    ... 还有 {len(module_info['classes']) - 5} 个类")
    
    if module_info["functions"]:
        # 过滤掉已经显示的test函数
        other_funcs = [f for f in module_info["functions"] if not f.startswith("🧪")]
        if other_funcs:
            print(f"  🔧 函数: {', '.join(other_funcs[:5])}")
            if len(other_funcs) > 5:
                print(f"    ... 还有 {len(other_funcs) - 5} 个函数")
    
    if module_info["attributes"]:
        print(f"  📝 属性: {', '.join(module_info['attributes'][:5])}")
        if len(module_info["attributes"]) > 5:
            print(f"    ... 还有 {len(module_info['attributes']) - 5} 个属性")
    
    return True

def check_file_exists(module_path):
    """检查模块文件是否存在"""
    # 将模块路径转换为文件路径
    parts = module_path.split(".")
    file_path = current_dir / Path(*parts)
    
    # 可能的文件扩展名
    possibilities = [
        file_path.with_suffix(".py"),
        file_path / "__init__.py",
    ]
    
    for possibility in possibilities:
        if possibility.exists():
            return True, possibility
    
    return False, None

def main():
    """主测试函数"""
    
    # 定义要测试的所有模块
    all_modules = [
        # 主模块
        "openlearning",
        "openlearning.__main__",
        
        # 核心模块
        "openlearning.core",
        "openlearning.core.config",
        "openlearning.core.metrics",
        "openlearning.core.registry",
        
        # 层模块
        "openlearning.layers",
        "openlearning.layers.attention",
        "openlearning.layers.balancer",
        "openlearning.layers.embeddings",
        "openlearning.layers.fusion",
        "openlearning.layers.memory",
        "openlearning.layers.normalization",
        "openlearning.layers.valve",
        
        # 集成模块
        "openlearning.integration",
        "openlearning.integration.nn",
        "openlearning.integration.yucai",
        
        # CCSS模块
        "openlearning.ccss",
        "openlearning.ccss.ccss",
        "openlearning.ccss.corec",
        "openlearning.ccss.cores",
        "openlearning.ccss.layersc",
        "openlearning.ccss.layerss",
        
        # CLI模块
        "openlearning.cli",
    ]
    
    print("📋 检查模块文件是否存在...")
    print("-" * 60)
    
    existing_modules = []
    missing_modules = []
    
    for module_path in all_modules:
        exists, file_path = check_file_exists(module_path)
        if exists:
            existing_modules.append((module_path, file_path))
            print(f"  ✅ {module_path}")
            print(f"     文件: {file_path}")
        else:
            missing_modules.append(module_path)
            print(f"  ❌ {module_path} (文件不存在)")
    
    print(f"\n📊 文件检查结果:")
    print(f"  存在: {len(existing_modules)} 个模块")
    print(f"  缺失: {len(missing_modules)} 个模块")
    
    if missing_modules:
        print("\n⚠️  缺失的模块:")
        for module in missing_modules[:10]:  # 只显示前10个
            print(f"  - {module}")
        if len(missing_modules) > 10:
            print(f"  ... 还有 {len(missing_modules) - 10} 个")
    
    # 实际测试导入和运行
    print("\n\n🔬 实际测试模块导入和运行...")
    print("=" * 80)
    
    successful_imports = 0
    failed_imports = 0
    
    for module_path, file_path in existing_modules:
        success = analyze_module_structure(module_path)
        if success:
            successful_imports += 1
        else:
            failed_imports += 1
    
    # 生成报告
    print("\n" + "=" * 80)
    print("📊 测试报告总结")
    print("=" * 80)
    
    print(f"📁 总模块数: {len(all_modules)}")
    print(f"📄 文件存在: {len(existing_modules)}")
    print(f"📄 文件缺失: {len(missing_modules)}")
    print(f"✅ 导入成功: {successful_imports}")
    print(f"❌ 导入失败: {failed_imports}")
    
    if successful_imports == 0:
        print("\n💥 严重问题: 没有任何模块可以导入！")
        print("可能的原因:")
        print("1. Python路径设置错误")
        print("2. 模块有语法错误")
        print("3. 缺少依赖包")
    elif successful_imports < len(existing_modules) / 2:
        print("\n⚠️  警告: 很多模块无法导入")
    else:
        print(f"\n🎉 好消息: {successful_imports}/{len(existing_modules)} 个模块可用")
    
    # 测试openlearning包的特殊功能
    print("\n\n🎯 测试openlearning包的高级功能...")
    print("-" * 60)
    
    try:
        import openlearning
        print(f"✅ openlearning包导入成功")
        print(f"   版本: {getattr(openlearning, '__version__', '未知')}")
        print(f"   作者: {getattr(openlearning, '__author__', '未知')}")
        
        # 测试__main__模块
        print("\n🔧 测试__main__模块...")
        try:
            import openlearning.__main__ as main_module
            if hasattr(main_module, 'main'):
                print("  ✅ __main__.main() 函数存在")
                
                # 可以尝试运行，但这里我们只检查
                print("  ⚠️  注意: 我们不实际运行main()，避免副作用")
            else:
                print("  ⚠️  __main__模块没有main()函数")
        except Exception as e:
            print(f"  ❌ 导入__main__失败: {e}")
        
        # 测试check_imports函数
        print("\n🔍 测试check_imports函数...")
        if hasattr(openlearning, 'check_imports'):
            try:
                results = openlearning.check_imports()
                print(f"  ✅ check_imports() 运行成功")
                print(f"     返回 {len(results)} 个结果")
                for module, result in results.items():
                    status = "成功" if result.get('success', False) else "失败"
                    print(f"     {module}: {status} ({result.get('duration', 0):.3f}s)")
            except Exception as e:
                print(f"  ❌ check_imports() 运行失败: {e}")
        else:
            print("  ⚠️  openlearning没有check_imports函数")
            
    except Exception as e:
        print(f"❌ 无法导入openlearning包: {e}")
    
    print("\n" + "=" * 80)
    print("🧪 测试完成！")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 测试被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 测试脚本出错: {e}")
        traceback.print_exc()
        sys.exit(1)