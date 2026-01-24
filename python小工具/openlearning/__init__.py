"""
OpenLearning - RGA 规则治理架构
================================

统一接口模块，集成核心层、层组件和集成器功能。
Unified interface module integrating core layers, layer components and integrator functionality.
"""

import sys
import os
from typing import Dict, List, Optional, Any, Union

# ==================== 模块信息 ====================
__version__ = "0.0.3"
__author__ = "RGA Architecture Team"
__description__ = "RGA规则治理架构统一接口"

# ==================== 路径设置 ====================
# 获取当前文件所在目录
_current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前目录添加到系统路径
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# ==================== 子模块导入 ====================

# 1. 导入核心模块
try:
    import core
    from core import *
    _CORE_IMPORTED = True
except ImportError as e:
    # 尝试相对导入
    try:
        from . import core
        from .core import *
        _CORE_IMPORTED = True
    except ImportError as e2:
        print(f"⚠️  核心模块导入失败: {e2}")
        _CORE_IMPORTED = False

# 2. 导入层模块
try:
    import layers
    from layers import *
    _LAYERS_IMPORTED = True
except ImportError as e:
    # 尝试相对导入
    try:
        from . import layers
        from .layers import *
        _LAYERS_IMPORTED = True
    except ImportError as e2:
        print(f"⚠️  层模块导入失败: {e2}")
        _LAYERS_IMPORTED = False

# 3. 导入集成模块
try:
    import integration
    from integration import *
    _INTEGRATION_IMPORTED = True
except ImportError as e:
    # 尝试相对导入
    try:
        from . import integration
        from .integration import *
        _INTEGRATION_IMPORTED = True
    except ImportError as e2:
        print(f"⚠️  集成模块导入失败: {e2}")
        _INTEGRATION_IMPORTED = False

# ==================== 统一导出列表 ====================

__all__ = []

# 添加核心模块导出
if _CORE_IMPORTED:
    try:
        from core import __all__ as core_exports
        __all__.extend(core_exports)
    except ImportError:
        try:
            from .core import __all__ as core_exports
            __all__.extend(core_exports)
        except ImportError:
            pass

# 添加层模块导出
if _LAYERS_IMPORTED:
    try:
        from layers import __all__ as layers_exports
        __all__.extend(layers_exports)
    except ImportError:
        try:
            from .layers import __all__ as layers_exports
            __all__.extend(layers_exports)
        except ImportError:
            pass

# 添加集成模块导出
if _INTEGRATION_IMPORTED:
    try:
        from integration import __all__ as integration_exports
        __all__.extend(integration_exports)
    except ImportError:
        try:
            from .integration import __all__ as integration_exports
            __all__.extend(integration_exports)
        except ImportError:
            pass

# 添加模块信息到导出
__all__.extend([
    "__version__",
    "__author__",
    "__description__",
])

# ==================== 便捷函数 ====================

def get_package_status() -> Dict[str, bool]:
    """获取各模块导入状态"""
    return {
        "core": _CORE_IMPORTED,
        "layers": _LAYERS_IMPORTED,
        "integration": _INTEGRATION_IMPORTED
    }

def print_package_info():
    """打印包信息"""
    print("=" * 60)
    print(f"OpenLearning RGA 统一接口")
    print(f"版本: {__version__} | 作者: {__author__}")
    print("=" * 60)
    
    status = get_package_status()
    for module, loaded in status.items():
        status_icon = "✅" if loaded else "❌"
        print(f"{status_icon} {module:12} {'已加载' if loaded else '未加载'}")
    
    print("=" * 60)
    print(f"总导出项: {len(__all__)}")
    print("=" * 60)

# 添加到导出列表
__all__.extend([
    "get_package_status",
    "print_package_info",
])

# ==================== 模块初始化 ====================

if __name__ != "__main__":
    # 自动打印包信息
    print_package_info()
    
    # 检查依赖
    if not all(get_package_status().values()):
        print("⚠️  警告: 部分模块加载失败，功能可能不完整")
        print("   建议检查子模块安装和导入路径")
    else:
        print("✅ 所有模块加载成功")

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    print_package_info()
    
    # 如果有集成模块，运行基本测试
    if _INTEGRATION_IMPORTED:
        print("\n🔧 运行基本测试...")
        try:
            # 检查test_integrator是否存在
            if 'test_integrator' in globals():
                test_integrator()
            else:
                print("⚠️  测试函数不可用")
        except Exception as e:
            print(f"⚠️  测试失败: {e}")
    
    print("\n✨ OpenLearning RGA 模块加载完成")