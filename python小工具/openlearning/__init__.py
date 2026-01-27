"""
OpenLearning - RGA 规则治理架构
================================

统一接口模块，直接暴露所有子包功能。
设计原则：简单、直接、无重复初始化。
"""

# ==================== 模块信息 ====================
__version__ = "0.0.7"
__author__ = "RGA Architecture Team"
__description__ = "RGA规则治理架构统一接口"
__license__ = "Apache 2.0"

# ==================== 静默导入子包 ====================
# 不打印任何导入信息，让子包自己处理初始化
try:
    from . import core
    CORE_IMPORTED = True
except ImportError:
    CORE_IMPORTED = False
    core = None

try:
    from . import layers
    LAYERS_IMPORTED = True
except ImportError:
    LAYERS_IMPORTED = False
    layers = None

try:
    from . import integration
    INTEGRATION_IMPORTED = True
except ImportError:
    INTEGRATION_IMPORTED = False
    integration = None

# ==================== 按类别暴露所有功能 ====================

# 1. 从 core 子包暴露功能
if CORE_IMPORTED:
    # 导入 core 的所有公开内容
    from .core import *
    print("📦 core 模块功能已暴露")

# 2. 从 layers 子包暴露功能
if LAYERS_IMPORTED:
    # 导入 layers 的所有公开内容
    from .layers import *
    print("🏗️  layers 模块功能已暴露")

# 3. 从 integration 子包暴露功能
if INTEGRATION_IMPORTED:
    # 导入 integration 的所有公开内容
    from .integration import *
    print("🔌 integration 模块功能已暴露")

# ==================== 构建导出列表 ====================
# 手动构建 __all__，确保包含所有重要导出

__all__ = [
    # 模块信息
    '__version__',
    '__author__',
    '__description__',
    '__license__',
    
    # 子包
    'core',
    'layers',
    'integration',
]

# 添加从 core 导出的重要类
if CORE_IMPORTED:
    __all__.extend([
        'RGAConfig',
        'CoreMetricsCalculator',
        'RGAEngine',
        'get_default_config',
        'calculate_state_change',
        'detect_phase_transition',
        'stack_three_networks',
        'apply_one_way_valve',
        'validate_config',
        'create_rga_engine',
    ])

# 添加从 layers 导出的重要类
if LAYERS_IMPORTED:
    __all__.extend([
        # 归一化层
        'FixedRMSNorm',
        'FixedGroupRMSNorm',
        'ScaledFixedRMSNorm',
        
        # 注意力层
        'VKQ_SubNet_WithFixedNorm',
        'QVK_SubNet_WithFixedNorm',
        'KQV_SubNet_WithFixedNorm',
        'ChainReactionUnit_Final',
        
        # 平衡器层
        'TriValueBalancer',
        'VDominantBalancer',
        'DensityDrivenBalancer',
        'AdaptiveStabilizer',
        
        # 嵌入层
        'EnhancedEmbeddingLayer',
        'ConceptAwareEmbedding',
        
        # 融合层
        'SandwichFusion',
        
        # 记忆层
        'GeologicalMemory',
        
        # 阀层
        'OneWayValve',
        'SimpleOneWayValve',
        
        # 工厂函数
        'create_fixed_norm',
        'create_attention_subnet',
        'create_chain_reaction_unit',
        'create_balancer_layer',
        'create_embedding_layer',
        'create_sandwich_fusion',
        'create_geological_memory',
        'create_one_way_valve',
        
        # 管理类
        'LayerRegistry',
        'LayerFactory',
        'LayerConfig',
        'LayerConfigManager',
        
        # 工具函数
        'get_layer_factory',
        'get_layer_registry',
        'create_layer',
        'list_available_layers',
    ])

# 添加从 integration 导出的重要类
if INTEGRATION_IMPORTED:
    __all__.extend([
        # 主类
        'RGAIntegrator',
        'create_integrator',
        
        # 数据相关
        'SmartTextDataset',
        
        # 训练相关
        'VisualTrainingProgress',
        'AdvancedConstrainedArchitectureTrainer',
        'RGAConfigManager',
        
        # 训练函数
        'train_zixin_complete_model',
        'quick_test_mode',
        'run_standard_training',
        'run_quick_training',
        'resume_training',
        
        # 推理和测试
        'test_model_inference',
        'interactive_training',
        'run_comprehensive_tests',
        'quick_diagnostic_test',
        'benchmark_performance',
        'example_usage',
    ])

# ==================== 最终初始化 ====================
def initialize():
    """初始化函数，显式调用时才会执行"""
    print(f"\n✨ OpenLearning RGA v{__version__} 初始化完成")
    print(f"   作者: {__author__}")
    print(f"   总导出项: {len([x for x in __all__ if not x.startswith('_')])} 个")
    
    # 打印导入状态
    print("\n📊 模块导入状态:")
    print(f"   ✅ core: {'已导入' if CORE_IMPORTED else '未导入'}")
    print(f"   ✅ layers: {'已导入' if LAYERS_IMPORTED else '未导入'}")
    print(f"   ✅ integration: {'已导入' if INTEGRATION_IMPORTED else '未导入'}")

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    print(f"🚀 OpenLearning RGA v{__version__} 独立运行模式")
    print("=" * 80)
    
    # 执行初始化
    initialize()
    
    print("\n✅ 所有功能已暴露到顶层")
    print("=" * 80)