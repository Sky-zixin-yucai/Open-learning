"""
测试 OpenLearning RGA 主包
"""

import sys
import os

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

print(f"📁 当前目录: {script_dir}")
print(f"📁 添加路径: {script_dir}")

print("\n🚀 开始导入 openlearning...")

try:
    import openlearning
    
    print(f"\n✅ 导入成功: OpenLearning v{openlearning.__version__}")
    
    # 检查重要功能是否可用
    print("\n🧪 检查重要功能:")
    
    # 1. 检查配置类
    try:
        config = openlearning.RGAConfig(vocab_size=1000, dim=64)
        print(f"   ✅ RGAConfig: 可用 (vocab_size={config.vocab_size})")
    except Exception as e:
        print(f"   ❌ RGAConfig: 不可用 - {e}")
    
    # 2. 检查层创建
    try:
        layer = openlearning.create_layer("FixedRMSNorm", dim=64)
        print(f"   ✅ create_layer: 可用 ({type(layer).__name__})")
    except Exception as e:
        print(f"   ❌ create_layer: 不可用 - {e}")
    
    # 3. 检查集成器创建
    try:
        if hasattr(openlearning, 'RGAConfig'):
            config = openlearning.RGAConfig(vocab_size=1000, dim=64)
            integrator = openlearning.create_integrator(config)
            print(f"   ✅ create_integrator: 可用 ({type(integrator).__name__})")
    except Exception as e:
        print(f"   ❌ create_integrator: 不可用 - {e}")
    
    # 4. 检查子包访问
    print("\n📦 检查子包访问:")
    for submodule in ['core', 'layers', 'integration']:
        if hasattr(openlearning, submodule) and getattr(openlearning, submodule) is not None:
            obj = getattr(openlearning, submodule)
            # 获取子包中的导出项数量
            exports = [x for x in dir(obj) if not x.startswith('_')]
            print(f"   ✅ {submodule}: {len(exports)} 个导出项")
        else:
            print(f"   ❌ {submodule}: 不可用")
    
    # 5. 列出一些重要的导出项
    print("\n📋 重要导出项示例:")
    
    # 按类别列出
    categories = {
        '配置类': ['RGAConfig', 'LayerConfig', 'RGAConfigManager'],
        '核心类': ['RGAEngine', 'CoreMetricsCalculator', 'RGAIntegrator'],
        '层类': ['FixedRMSNorm', 'OneWayValve', 'TriValueBalancer', 'VKQ_SubNet_WithFixedNorm'],
        '工具函数': ['get_default_config', 'create_layer', 'create_integrator'],
        '数据类': ['SmartTextDataset']
    }
    
    for category, items in categories.items():
        available = []
        for item in items:
            if hasattr(openlearning, item):
                available.append(item)
        if available:
            print(f"   📁 {category}: {', '.join(available)}")
    
    print(f"\n🎉 OpenLearning RGA 测试完成!")
    
except ImportError as e:
    print(f"\n❌ 导入失败: {e}")
    print("\n💡 建议:")
    print("  1. 确保当前目录包含 openlearning 文件夹")
    print("  2. 确保 openlearning 文件夹中有 __init__.py 文件")
    print("  3. 确保子包 core, layers, integration 存在")