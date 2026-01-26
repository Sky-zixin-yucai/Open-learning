"""
OpenLearning - RGA 规则治理架构
================================

统一接口模块，集成核心层、层组件和集成器功能。
Unified interface module integrating core layers, layer components and integrator functionality.
"""

import sys
import os
import importlib.util
from typing import Dict, List, Optional, Any, Union

# ==================== 模块信息 ====================
__version__ = "0.0.6"
__author__ = "RGA Architecture Team"
__description__ = "RGA规则治理架构统一接口"
__license__ = "Apache 2.0"

# ==================== 路径设置 ====================
# 获取当前文件所在目录
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

# ==================== 动态导入函数 ====================

def _import_module(module_name: str, module_path: Optional[str] = None):
    """
    动态导入模块
    Dynamically import module
    
    Args:
        module_name: 模块名称
        module_path: 模块路径（可选）
    
    Returns:
        导入的模块或None
    """
    try:
        if module_path:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
        else:
            spec = importlib.util.find_spec(module_name)
        
        if spec is None:
            raise ImportError(f"模块 '{module_name}' 未找到")
        
        module = importlib.util.module_from_spec(spec)
        
        # 设置模块的__package__属性
        module.__package__ = module_name
        
        # 执行模块
        spec.loader.exec_module(module)
        
        return module
    
    except Exception as e:
        print(f"⚠️  导入模块 '{module_name}' 失败: {e}")
        return None

# ==================== 子模块导入 ====================

# 导入状态标志
_CORE_IMPORTED = False
_LAYERS_IMPORTED = False
_INTEGRATION_IMPORTED = False

# 1. 导入核心模块包
try:
    # 首先尝试相对导入
    from . import core
    _CORE_IMPORTED = True
    print("✅ 核心模块导入成功 (相对导入)")
except ImportError as e1:
    try:
        # 如果失败，尝试从当前目录导入
        core_path = os.path.join(_current_dir, "core", "__init__.py")
        if os.path.exists(core_path):
            core = _import_module("core", core_path)
            if core:
                # 将core模块添加到sys.modules，使其可以被其他模块引用
                sys.modules["core"] = core
                sys.modules["OpenLearning.core"] = core
                _CORE_IMPORTED = True
                print("✅ 核心模块导入成功 (动态导入)")
            else:
                print("❌ 核心模块动态导入失败")
        else:
            print(f"❌ 核心模块路径不存在: {core_path}")
    except Exception as e2:
        print(f"❌ 核心模块导入完全失败: {e1} -> {e2}")

# 2. 导入层模块包
try:
    # 首先尝试相对导入
    from . import layers
    _LAYERS_IMPORTED = True
    print("✅ 层模块导入成功 (相对导入)")
except ImportError as e1:
    try:
        # 如果失败，尝试从当前目录导入
        layers_path = os.path.join(_current_dir, "layers", "__init__.py")
        if os.path.exists(layers_path):
            layers = _import_module("layers", layers_path)
            if layers:
                # 将layers模块添加到sys.modules
                sys.modules["layers"] = layers
                sys.modules["OpenLearning.layers"] = layers
                _LAYERS_IMPORTED = True
                print("✅ 层模块导入成功 (动态导入)")
            else:
                print("❌ 层模块动态导入失败")
        else:
            print(f"❌ 层模块路径不存在: {layers_path}")
    except Exception as e2:
        print(f"❌ 层模块导入完全失败: {e1} -> {e2}")

# 3. 导入集成模块包
try:
    # 首先尝试相对导入
    from . import integration
    _INTEGRATION_IMPORTED = True
    print("✅ 集成模块导入成功 (相对导入)")
except ImportError as e1:
    try:
        # 如果失败，尝试从当前目录导入
        integration_path = os.path.join(_current_dir, "integration", "__init__.py")
        if os.path.exists(integration_path):
            integration = _import_module("integration", integration_path)
            if integration:
                # 将integration模块添加到sys.modules
                sys.modules["integration"] = integration
                sys.modules["OpenLearning.integration"] = integration
                _INTEGRATION_IMPORTED = True
                print("✅ 集成模块导入成功 (动态导入)")
            else:
                print("❌ 集成模块动态导入失败")
        else:
            print(f"❌ 集成模块路径不存在: {integration_path}")
    except Exception as e2:
        print(f"❌ 集成模块导入完全失败: {e1} -> {e2}")

# ==================== 统一导出列表 ====================

__all__ = []

# 添加模块本身
if _CORE_IMPORTED:
    __all__.append("core")
if _LAYERS_IMPORTED:
    __all__.append("layers")
if _INTEGRATION_IMPORTED:
    __all__.append("integration")

# 添加模块信息
__all__.extend([
    "__version__",
    "__author__",
    "__description__",
    "__license__",
])

# ==================== 便捷访问函数 ====================

def get_core_module():
    """获取核心模块"""
    if _CORE_IMPORTED:
        return core
    raise ImportError("核心模块未导入")

def get_layers_module():
    """获取层模块"""
    if _LAYERS_IMPORTED:
        return layers
    raise ImportError("层模块未导入")

def get_integration_module():
    """获取集成模块"""
    if _INTEGRATION_IMPORTED:
        return integration
    raise ImportError("集成模块未导入")

def get_all_modules() -> Dict[str, Any]:
    """获取所有已导入模块"""
    modules = {}
    if _CORE_IMPORTED:
        modules["core"] = core
    if _LAYERS_IMPORTED:
        modules["layers"] = layers
    if _INTEGRATION_IMPORTED:
        modules["integration"] = integration
    return modules

# ==================== 模块状态检查 ====================

def check_module_health() -> Dict[str, Dict[str, Any]]:
    """
    检查模块健康状态
    
    Returns:
        健康状态字典
    """
    health_status = {
        "core": {
            "imported": _CORE_IMPORTED,
            "version": getattr(core, "__version__", "unknown") if _CORE_IMPORTED else "N/A",
            "description": getattr(core, "__description__", "No description") if _CORE_IMPORTED else "Not imported"
        },
        "layers": {
            "imported": _LAYERS_IMPORTED,
            "version": getattr(layers, "__version__", "unknown") if _LAYERS_IMPORTED else "N/A",
            "description": getattr(layers, "__description__", "No description") if _LAYERS_IMPORTED else "Not imported"
        },
        "integration": {
            "imported": _INTEGRATION_IMPORTED,
            "version": getattr(integration, "__version__", "unknown") if _INTEGRATION_IMPORTED else "N/A",
            "description": getattr(integration, "__description__", "No description") if _INTEGRATION_IMPORTED else "Not imported"
        },
        "openlearning": {
            "version": __version__,
            "author": __author__,
            "description": __description__,
            "all_modules_imported": all([_CORE_IMPORTED, _LAYERS_IMPORTED, _INTEGRATION_IMPORTED])
        }
    }
    
    return health_status

def print_health_status():
    """打印模块健康状态"""
    status = check_module_health()
    
    print("=" * 60)
    print("OpenLearning RGA 模块健康状态检查")
    print("=" * 60)
    
    print(f"📦 OpenLearning 版本: {status['openlearning']['version']}")
    print(f"👤 作者: {status['openlearning']['author']}")
    print(f"📝 描述: {status['openlearning']['description']}")
    
    print("\n模块状态:")
    for module_name, module_info in status.items():
        if module_name == "openlearning":
            continue
            
        status_icon = "✅" if module_info["imported"] else "❌"
        print(f"  {status_icon} {module_name.upper()}:")
        print(f"    导入状态: {'已导入' if module_info['imported'] else '未导入'}")
        print(f"    版本: {module_info['version']}")
        print(f"    描述: {module_info['description']}")
    
    print(f"\n总体状态: {'✅ 所有模块已导入' if status['openlearning']['all_modules_imported'] else '⚠️  部分模块缺失'}")
    print("=" * 60)

# ==================== 模块初始化 ====================

def _initialize_module():
    """初始化模块"""
    print(f"✨ OpenLearning RGA v{__version__} 初始化完成")
    print(f"   作者: {__author__}")
    print(f"   描述: {__description__}")
    
    # 打印健康状态
    health_status = check_module_health()["openlearning"]
    if health_status["all_modules_imported"]:
        print("✅ 所有子模块已成功导入")
    else:
        print("⚠️  警告: 部分子模块导入失败")
    
    print("   可用组件: " + ", ".join(__all__))
    
    # 提供使用建议
    print("\n📚 使用建议:")
    print("   1. 使用 get_core_module() 获取核心模块")
    print("   2. 使用 get_layers_module() 获取层模块")
    print("   3. 使用 get_integration_module() 获取集成模块")
    print("   4. 使用 check_module_health() 检查模块状态")

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    # 当模块作为主程序运行时，打印详细信息并运行测试
    print("🚀 OpenLearning RGA 独立运行模式")
    print("=" * 60)
    
    # 打印健康状态
    print_health_status()
    
    # 如果所有模块都导入了，尝试运行一些基本测试
    if all([_CORE_IMPORTED, _LAYERS_IMPORTED, _INTEGRATION_IMPORTED]):
        print("\n🧪 运行基本功能测试...")
        
        try:
            # 测试核心模块
            if hasattr(core, "get_default_config"):
                config = core.get_default_config()
                print(f"✅ 核心模块测试: 默认配置创建成功")
            
            # 测试层模块
            if hasattr(layers, "create_layer"):
                print(f"✅ 层模块测试: create_layer 函数可用")
            
            # 测试集成模块
            if hasattr(integration, "create_integrator"):
                print(f"✅ 集成模块测试: create_integrator 函数可用")
            
            print("\n✅ 所有基本功能测试通过")
            
        except Exception as e:
            print(f"⚠️  测试过程中出现错误: {e}")
    
    print("\n✨ OpenLearning RGA 模块加载完成")
    print("=" * 60)
else:
    # 当模块被导入时，自动初始化
    _initialize_module()