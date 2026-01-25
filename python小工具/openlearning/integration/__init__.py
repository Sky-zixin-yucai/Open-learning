"""
RGA集成模块 | RGA Integration Module
====================================

RGA规则治理架构集成模块包，提供完整的RGA集成器接口。
RGA Rule-Governed Architecture integration module package, providing complete RGA integrator interface.
"""

# ==================== 模块信息 ====================
# ==================== Module Information ====================

__version__ = "0.0.4"
__author__ = "RGA Architecture Team"
__description__ = "RGA规则治理架构集成模块"
__license__ = "Apache 2.0"

# ==================== 导入主模块 ====================
# ==================== Import Main Module ====================

# 首先定义导入状态变量
_IMPORT_SUCCESSFUL = False

try:
    # 直接导入当前目录下的yucai模块
    from .yucai import (
        IntegrationConfig,
        RGAIntegrator,
        create_integrator,
        save_disguised_model,
        load_disguised_model,
        get_default_integration_config,
        validate_integration_config,
        test_integrator,
    )
    
    _IMPORT_SUCCESSFUL = True
    print("✅ 集成模块导入成功")
    
except ImportError as e:
    try:
        # 如果直接导入失败，尝试相对导入
        from yucai import (
            IntegrationConfig,
            RGAIntegrator,
            create_integrator,
            save_disguised_model,
            load_disguised_model,
            get_default_integration_config,
            validate_integration_config,
            test_integrator,
        )
        
        _IMPORT_SUCCESSFUL = True
        print("✅ 集成模块导入成功（相对导入）")
        
    except ImportError as e2:
        print(f"❌ 集成模块导入失败: {e2}")

# ==================== 导出列表 ====================
# ==================== Export List ====================

__all__ = [
    # 配置类 | Configuration classes
    "IntegrationConfig",
    
    # 核心类 | Core classes
    "RGAIntegrator",
    
    # 工厂函数 | Factory functions
    "create_integrator",
    "save_disguised_model",
    "load_disguised_model",
    
    # 便捷函数 | Convenience functions
    "get_default_integration_config",
    "validate_integration_config",
    
    # 测试函数 | Test functions
    "test_integrator",
    
    # 模块信息 | Module information
    "__version__",
    "__author__",
    "__description__",
    "__license__",
]

# ==================== 模块初始化 ====================
# ==================== Module Initialization ====================

def _initialize_module():
    """初始化模块"""
    print(f"✅ RGA集成模块 v{__version__} 已加载")
    print(f"   作者: {__author__}")
    print(f"   导入状态: {'✅ 成功' if _IMPORT_SUCCESSFUL else '❌ 失败'}")

# 自动初始化模块
if __name__ != "__main__":
    _initialize_module()

# ==================== 主程序入口 ====================
# ==================== Main Entry Point ====================

if __name__ == "__main__":
    """
    模块独立运行时的主程序
    """
    print("🚀 RGA集成模块独立运行模式")
    print("=" * 60)
    
    # 打印模块信息
    _initialize_module()
    
    # 如果导入成功，运行测试
    if _IMPORT_SUCCESSFUL:
        print("\n🔧 运行快速测试...")
        try:
            # 创建一个最小配置的集成器进行测试
            import torch
            integrator = create_integrator({
                "vocab_size": 100,
                "dim": 16,
                "num_units": 1,
                "max_cycles": 1
            })
            
            input_ids = torch.randint(0, 100, (1, 8))
            output = integrator.forward(input_ids)
            
            print(f"✅ 前向传播成功")
            print(f"   Logits形状: {output['logits'].shape}")
            
        except Exception as e:
            print(f"❌ 快速测试失败: {e}")
    
    print("\n✨ 模块加载完成")
    print("=" * 60)