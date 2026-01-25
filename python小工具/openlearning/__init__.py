"""
OpenLearning - RGA 规则治理架构
================================

统一接口模块，集成核心层、层组件和集成器功能。
Unified interface module integrating core layers, layer components and integrator functionality.
"""

import sys
import os

# ==================== 模块信息 ====================
__version__ = "0.0.5"
__author__ = "RGA Architecture Team"
__description__ = "RGA规则治理架构统一接口"

# ==================== 路径设置 ====================
# 获取当前文件所在目录
_current_dir = os.path.dirname(os.path.abspath(__file__))

# ==================== 子模块导入 ====================

# 1. 导入核心模块包
try:
    # 首先尝试从当前目录导入
    from . import core
    _CORE_IMPORTED = True
except ImportError as e:
    try:
        # 如果失败，尝试绝对导入
        import core
        _CORE_IMPORTED = True
    except ImportError as e2:
        print("⚠️  核心模块导入失败")
        _CORE_IMPORTED = False

# 2. 导入层模块包
try:
    from . import layers
    _LAYERS_IMPORTED = True
except ImportError as e:
    try:
        import layers
        _LAYERS_IMPORTED = True
    except ImportError as e2:
        print("⚠️  层模块导入失败")
        _LAYERS_IMPORTED = False

# 3. 导入集成模块包
try:
    from . import integration
    _INTEGRATION_IMPORTED = True
except ImportError as e:
    try:
        import integration
        _INTEGRATION_IMPORTED = True
    except ImportError as e2:
        print("⚠️  集成模块导入失败")
        _INTEGRATION_IMPORTED = False

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
])

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    print(f"✅ OpenLearning RGA v{__version__}")
    print(f"   core模块: {'✅' if _CORE_IMPORTED else '❌'}")
    print(f"   layers模块: {'✅' if _LAYERS_IMPORTED else '❌'}")
    print(f"   integration模块: {'✅' if _INTEGRATION_IMPORTED else '❌'}")
    
    if _INTEGRATION_IMPORTED:
        try:
            # 检查test_integrator是否存在
            if hasattr(integration, 'test_integrator'):
                integration.test_integrator()
        except Exception as e:
            print(f"⚠️  测试失败: {e}")
    
    print("✨ OpenLearning RGA 模块加载完成")