# ==================== 版本信息 ====================
# ==================== Version Information ====================

__version__ = "0.0.1"
__author__ = "yucai"
__license__ = "Apache-2.0"
__copyright__ = "Copyright (c) 2024 RGA Team"

# ==================== 子模块导入 ====================
# ==================== Submodule Imports ====================

# 注意：我们不直接导入所有子模块，以避免循环导入
# Note: We don't directly import all submodules to avoid circular imports

# ==================== 主要导出项 ====================
# ==================== Main Exports ====================

# 从集成模块导出主要接口 | Export main interfaces from integration module
__all__ = [
    # 只保留实际存在的导出
    "__version__",
    "__author__",
    "__license__",
    "__copyright__",
]

# ==================== 模块初始化 ====================
# ==================== Module Initialization ====================

def _init_module():
    """
    模块初始化函数
    Module initialization function
    """
    print(f"✅ Open-learning v{__version__} 已加载 | Open-learning v{__version__} loaded")
    print(f"   作者: {__author__} | Author: {__author__}")
    print(f"   许可证: {__license__} | License: {__license__}")
    print(f"   导出项: {len(__all__)} 个 | Exports: {len(__all__)} items")

# 自动初始化模块 | Automatically initialize module
_init_module()

# ==================== 主程序入口 ====================
# ==================== Main Program Entry ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Open-learning 包信息 | Open-learning Package Information")
    print("=" * 60)
    print(f"版本 | Version: {__version__}")
    print(f"作者 | Author: {__author__}")
    print(f"许可证 | License: {__license__}")
    print(f"描述 | Description: A Python package for rule-governed architecture in deep learning.")
    print()
    print("使用方法 | Usage:")
    print("   from rga import create_integrator")
    print("   integrator = create_integrator(vocab_size=20000, dim=512)")
    print()
    print("运行测试 | Run tests:")
    print("   python -c \"from rga import test_integrator; test_integrator()\"")
    print("=" * 60)