"""
RGA集成模块 | RGA Integration Module
====================================

RGA规则治理架构集成模块包，提供完整的RGA集成器接口。
RGA Rule-Governed Architecture integration module package, providing complete RGA integrator interface.
"""

# ==================== 模块信息 ====================
# ==================== Module Information ====================

__version__ = "0.0.9"
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
        RGAIntegrator,           # RGA集成器主类
        RGAConfig,              # RGA配置类（从core模块导入）
        # 测试函数
        run_comprehensive_tests,
        quick_diagnostic_test,
        benchmark_performance,
        example_usage,
    )
    
    # 导入nn模块中的类
    from .nn import (
        SmartTextDataset,        # 智能文本数据集
        VisualTrainingProgress,  # 可视化训练进度
        AdvancedConstrainedArchitectureTrainer,  # 高级训练器
        RGAConfigManager,        # RGA配置管理器
        # 训练函数
        train_zixin_complete_model,
        quick_test_mode,
        run_standard_training,
        run_quick_training,
        resume_training,
        # 测试函数
        test_model_inference,
        interactive_training,
    )
    
    _IMPORT_SUCCESSFUL = True
    print("✅ 集成模块导入成功")
    
except ImportError as e:
    try:
        # 如果直接导入失败，尝试绝对导入
        from yucai import (
            RGAIntegrator,
            RGAConfig,
            run_comprehensive_tests,
            quick_diagnostic_test,
            benchmark_performance,
            example_usage,
        )
        
        from nn import (
            SmartTextDataset,
            VisualTrainingProgress,
            AdvancedConstrainedArchitectureTrainer,
            RGAConfigManager,
            train_zixin_complete_model,
            quick_test_mode,
            run_standard_training,
            run_quick_training,
            resume_training,
            test_model_inference,
            interactive_training,
        )
        
        _IMPORT_SUCCESSFUL = True
        print("✅ 集成模块导入成功（绝对导入）")
        
    except ImportError as e2:
        print(f"❌ 集成模块导入失败: {e2}")

# ==================== 创建集成器函数（用于测试兼容性）====================

def create_integrator(config):
    """
    创建RGA集成器实例（兼容性函数）
    
    Args:
        config: RGA配置对象
        
    Returns:
        RGAIntegrator: RGA集成器实例
    """
    if _IMPORT_SUCCESSFUL:
        return RGAIntegrator(config)
    else:
        raise ImportError("无法创建集成器，模块导入失败")

# ==================== 导出列表 ====================
# ==================== Export List ====================

__all__ = [
    # 模块信息
    '__version__',
    '__author__',
    '__description__',
    '__license__',
    
    # 主类和配置
    'RGAIntegrator',
    'RGAConfig',
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
]

# ==================== 模块初始化 ====================
# ==================== Module Initialization ====================

def _initialize_module():
    """初始化模块"""
    print(f"✅ RGA集成模块 v{__version__} 已加载")
    print(f"   作者: {__author__}")
    print(f"   导入状态: {'✅ 成功' if _IMPORT_SUCCESSFUL else '❌ 失败'}")
    print(f"   导出组件数: {len(__all__) - 4}")  # 减去4个模块信息字段

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
            
            # 创建配置对象
            config = RGAConfig(
                vocab_size=100,
                dim=16,
                num_units=3,  # 🆕 改为3，因为RGA架构需要3个单元
                max_cycles=1
            )
            
            # 使用配置对象创建集成器
            integrator = create_integrator(config)
            
            input_ids = torch.randint(0, 100, (1, 8))
            output = integrator.forward(input_ids)
            
            print(f"✅ 前向传播成功")
            print(f"   Logits形状: {output['logits'].shape}")
            if 'V_fused_mean' in output.get('V_stats', {}):
                print(f"   V_fused均值: {output['V_stats']['V_fused_mean']:.4f}")
            
            # 测试数据集创建
            print("\n🔧 测试数据集创建...")
            # 使用虚拟测试数据
            try:
                import json
                import tempfile
                import os
                
                # 创建临时测试文件
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
                test_data = [
                    ["你好，今天天气怎么样？", "天气很好，适合出门。"],
                    ["请问附近有餐厅吗？", "有的，前面有一家不错的餐厅。"],
                    ["谢谢你的帮助。", "不客气，很高兴能帮到你。"]
                ]
                json.dump(test_data, temp_file, ensure_ascii=False)
                temp_file.close()
                
                dataset = SmartTextDataset(
                    data_path=temp_file.name,
                    seq_length=32,
                    vocab_size=1000,
                    max_samples=10
                )
                print(f"✅ 数据集创建成功，大小: {len(dataset)}")
                print(f"   词汇表大小: {dataset.get_vocab_size()}")
                
                # 清理临时文件
                os.unlink(temp_file.name)
                
            except Exception as dataset_error:
                print(f"⚠️  数据集创建测试失败: {dataset_error}")
                print("   跳过数据集测试...")
            
        except Exception as e:
            print(f"❌ 快速测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✨ 模块加载完成")
    print("=" * 60)