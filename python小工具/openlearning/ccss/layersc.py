#!/usr/bin/env python3
"""
layers包导入测试 | layers Package Import Test
======================================

测试layers包能否正常导入和使用。
Test if layers package can be imported and used normally.
"""

import sys
import os

# 获取当前脚本所在目录（假设是tests目录）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"当前脚本目录: {current_script_dir}")

# 获取openlearning目录（父目录）
openlearning_dir = os.path.dirname(current_script_dir)
print(f"openlearning目录: {openlearning_dir}")

# 检查layers目录是否存在
layers_dir = os.path.join(openlearning_dir, 'layers')
print(f"layers目录: {layers_dir}")
print(f"layers目录是否存在: {os.path.exists(layers_dir)}")

# 显示layers目录内容
if os.path.exists(layers_dir):
    print(f"layers目录内容:")
    for item in os.listdir(layers_dir):
        print(f"  - {item}")

# 将openlearning目录添加到Python路径
sys.path.insert(0, openlearning_dir)
print(f"\n已添加目录到Python路径: {openlearning_dir}")
print(f"当前Python路径: {sys.path[:3]}")

def test_layers_import():
    """测试layers包导入 | Test layers package import"""
    print("\n" + "="*80)
    print("测试layers包导入 | Testing layers Package Import")
    print("="*80)
    
    try:
        # 方法1：尝试导入整个包
        print("\n1. 尝试导入整个layers包 | Trying to import whole layers package...")
        try:
            import layers
            print(f"✅ layers包导入成功 | layers package import successful")
            print(f"   版本: {layers.__version__} | Version: {layers.__version__}")
            print(f"   作者: {layers.__author__} | Author: {layers.__author__}")
            print(f"   导出列表: {layers.__all__} | Export list: {layers.__all__}")
            
            # 验证导出列表中包含的类
            for export in layers.__all__:
                if not export.startswith('__'):
                    try:
                        cls = getattr(layers, export)
                        print(f"   ✅ {export} 可访问 | {export} is accessible")
                    except AttributeError:
                        print(f"   ❌ {export} 不可访问 | {export} is not accessible")
                        
        except ImportError as e:
            print(f"❌ layers包导入失败: {e} | layers package import failed: {e}")
            return False
        
        # 方法2：尝试直接导入类
        print("\n2. 尝试直接导入类 | Trying to import classes directly...")
        try:
            from layers import EnhancedEmbeddingLayer
            print("✅ EnhancedEmbeddingLayer 导入成功 | EnhancedEmbeddingLayer import successful")
        except ImportError as e:
            print(f"❌ EnhancedEmbeddingLayer 导入失败: {e} | EnhancedEmbeddingLayer import failed: {e}")
            return False
        
        try:
            from layers import ConceptAwareEmbedding
            print("✅ ConceptAwareEmbedding 导入成功 | ConceptAwareEmbedding import successful")
        except ImportError as e:
            print(f"❌ ConceptAwareEmbedding 导入失败: {e} | ConceptAwareEmbedding import failed: {e}")
            return False
        
        try:
            from layers import GeologicalMemory
            print("✅ GeologicalMemory 导入成功 | GeologicalMemory import successful")
        except ImportError as e:
            print(f"❌ GeologicalMemory 导入失败: {e} | GeologicalMemory import failed: {e}")
            return False
        
        try:
            from layers import SandwichFusion
            print("✅ SandwichFusion 导入成功 | SandwichFusion import successful")
        except ImportError as e:
            print(f"❌ SandwichFusion 导入失败: {e} | SandwichFusion import failed: {e}")
            return False
        
        try:
            from layers import FixedRMSNorm
            print("✅ FixedRMSNorm 导入成功 | FixedRMSNorm import successful")
        except ImportError as e:
            print(f"❌ FixedRMSNorm 导入失败: {e} | FixedRMSNorm import failed: {e}")
            return False
        
        try:
            from layers import OneWayValve
            print("✅ OneWayValve 导入成功 | OneWayValve import successful")
        except ImportError as e:
            print(f"❌ OneWayValve 导入失败: {e} | OneWayValve import failed: {e}")
            return False
        
        try:
            from layers import TriValueBalancer
            print("✅ TriValueBalancer 导入成功 | TriValueBalancer import successful")
        except ImportError as e:
            print(f"❌ TriValueBalancer 导入失败: {e} | TriValueBalancer import failed: {e}")
            return False
        
        try:
            from layers import VKQ_SubNet_WithFixedNorm
            print("✅ VKQ_SubNet_WithFixedNorm 导入成功 | VKQ_SubNet_WithFixedNorm import successful")
        except ImportError as e:
            print(f"❌ VKQ_SubNet_WithFixedNorm 导入失败: {e} | VKQ_SubNet_WithFixedNorm import failed: {e}")
            return False
        
        try:
            from layers import ChainReactionUnit_Final
            print("✅ ChainReactionUnit_Final 导入成功 | ChainReactionUnit_Final import successful")
        except ImportError as e:
            print(f"❌ ChainReactionUnit_Final 导入失败: {e} | ChainReactionUnit_Final import failed: {e}")
            return False
        
        # 方法3：尝试导入工厂函数
        print("\n3. 尝试导入工厂函数 | Trying to import factory functions...")
        try:
            from layers import create_embedding_layer
            print("✅ create_embedding_layer 导入成功 | create_embedding_layer import successful")
        except ImportError as e:
            print(f"❌ create_embedding_layer 导入失败: {e} | create_embedding_layer import failed: {e}")
            return False
        
        try:
            from layers import create_sandwich_fusion
            print("✅ create_sandwich_fusion 导入成功 | create_sandwich_fusion import successful")
        except ImportError as e:
            print(f"❌ create_sandwich_fusion 导入失败: {e} | create_sandwich_fusion import failed: {e}")
            return False
        
        try:
            from layers import create_balancer_layer
            print("✅ create_balancer_layer 导入成功 | create_balancer_layer import successful")
        except ImportError as e:
            print(f"❌ create_balancer_layer 导入失败: {e} | create_balancer_layer import failed: {e}")
            return False
        
        try:
            from layers import create_chain_reaction_unit
            print("✅ create_chain_reaction_unit 导入成功 | create_chain_reaction_unit import successful")
        except ImportError as e:
            print(f"❌ create_chain_reaction_unit 导入失败: {e} | create_chain_reaction_unit import failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 导入测试异常: {e} | Import test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_class_instantiation():
    """测试类实例化 | Test class instantiation"""
    print("\n" + "="*80)
    print("测试类实例化 | Testing Class Instantiation")
    print("="*80)
    
    try:
        # 导入类
        from layers import EnhancedEmbeddingLayer, GeologicalMemory, SandwichFusion
        from layers import FixedRMSNorm, OneWayValve, TriValueBalancer
        from layers import VKQ_SubNet_WithFixedNorm, ChainReactionUnit_Final
        
        print("1. 测试 EnhancedEmbeddingLayer 实例化 | Testing EnhancedEmbeddingLayer instantiation...")
        try:
            # 创建默认配置
            embed_layer = EnhancedEmbeddingLayer(vocab_size=1000, embed_dim=128)
            print(f"   ✅ 默认配置创建成功 | Default config creation successful")
            print(f"      词汇表大小: {embed_layer.vocab_size}")
            print(f"      嵌入维度: {embed_layer.embed_dim}")
            print(f"      标记维度: {embed_layer.marker_dim}")
            
            # 创建自定义配置
            custom_embed_layer = EnhancedEmbeddingLayer(vocab_size=20000, embed_dim=512, marker_dim=64)
            print(f"   ✅ 自定义配置创建成功 | Custom config creation successful")
            
        except Exception as e:
            print(f"   ❌ EnhancedEmbeddingLayer 实例化失败: {e} | EnhancedEmbeddingLayer instantiation failed: {e}")
            return False
        
        print("\n2. 测试 GeologicalMemory 实例化 | Testing GeologicalMemory instantiation...")
        try:
            # 创建默认记忆层
            memory = GeologicalMemory(dim=64)
            print(f"   ✅ 默认记忆层创建成功 | Default memory layer creation successful")
            print(f"      维度: {memory.dim}")
            print(f"      衰退因子: {memory.decay_factor}")
            
            # 创建自定义记忆层
            custom_memory = GeologicalMemory(dim=128, decay_factor=0.8)
            print(f"   ✅ 自定义记忆层创建成功 | Custom memory layer creation successful")
            
        except Exception as e:
            print(f"   ❌ GeologicalMemory 实例化失败: {e} | GeologicalMemory instantiation failed: {e}")
            return False
        
        print("\n3. 测试 SandwichFusion 实例化 | Testing SandwichFusion instantiation...")
        try:
            # 创建融合层
            fusion = SandwichFusion()
            print(f"   ✅ 融合层创建成功 | Fusion layer creation successful")
            
        except Exception as e:
            print(f"   ❌ SandwichFusion 实例化失败: {e} | SandwichFusion instantiation failed: {e}")
            return False
        
        print("\n4. 测试 FixedRMSNorm 实例化 | Testing FixedRMSNorm instantiation...")
        try:
            # 创建归一化层
            norm = FixedRMSNorm(dim=128)
            print(f"   ✅ 归一化层创建成功 | Normalization layer creation successful")
            print(f"      维度: {norm.dim}")
            print(f"      epsilon: {norm.eps}")
            
        except Exception as e:
            print(f"   ❌ FixedRMSNorm 实例化失败: {e} | FixedRMSNorm instantiation failed: {e}")
            return False
        
        print("\n5. 测试 OneWayValve 实例化 | Testing OneWayValve instantiation...")
        try:
            # 创建单向阀
            valve = OneWayValve(dim=64)
            print(f"   ✅ 单向阀创建成功 | One-way valve creation successful")
            print(f"      维度: {valve.dim}")
            
        except Exception as e:
            print(f"   ❌ OneWayValve 实例化失败: {e} | OneWayValve instantiation failed: {e}")
            return False
        
        print("\n6. 测试 TriValueBalancer 实例化 | Testing TriValueBalancer instantiation...")
        try:
            # 创建平衡器
            balancer = TriValueBalancer(dim=64)
            print(f"   ✅ 平衡器创建成功 | Balancer creation successful")
            print(f"      维度: {balancer.dim}")
            
        except Exception as e:
            print(f"   ❌ TriValueBalancer 实例化失败: {e} | TriValueBalancer instantiation failed: {e}")
            return False
        
        print("\n7. 测试 VKQ_SubNet_WithFixedNorm 实例化 | Testing VKQ_SubNet_WithFixedNorm instantiation...")
        try:
            # 创建子网络
            subnet = VKQ_SubNet_WithFixedNorm(dim=64)
            print(f"   ✅ 子网络创建成功 | Subnet creation successful")
            print(f"      维度: {subnet.dim}")
            
        except Exception as e:
            print(f"   ❌ VKQ_SubNet_WithFixedNorm 实例化失败: {e} | VKQ_SubNet_WithFixedNorm instantiation failed: {e}")
            return False
        
        print("\n8. 测试 ChainReactionUnit_Final 实例化 | Testing ChainReactionUnit_Final instantiation...")
        try:
            # 创建链式反应单元，添加 unit_id 参数
            chain_reaction = ChainReactionUnit_Final(dim=64, unit_id=1)
            print(f"   ✅ 链式反应单元创建成功 | Chain reaction unit creation successful")
            print(f"      维度: {chain_reaction.dim}")
            print(f"      单元ID: {chain_reaction.unit_id}")
            
        except Exception as e:
            print(f"   ❌ ChainReactionUnit_Final 实例化失败: {e} | ChainReactionUnit_Final instantiation failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e} | Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ 实例化测试异常: {e} | Instantiation test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_package_methods():
    """测试包方法 | Test package methods"""
    print("\n" + "="*80)
    print("测试包方法 | Testing Package Methods")
    print("="*80)
    
    try:
        # 导入必要的库
        import torch
        
        # 导入类
        from layers import EnhancedEmbeddingLayer, GeologicalMemory
        
        print("1. 测试 EnhancedEmbeddingLayer 方法 | Testing EnhancedEmbeddingLayer methods...")
        embed_layer = EnhancedEmbeddingLayer(vocab_size=1000, embed_dim=128, marker_dim=32)
        
        # 测试获取配置
        config = embed_layer.get_config()
        print(f"   ✅ get_config() 成功: {list(config.keys())}")
        
        # 测试概念特征设置
        try:
            word_indices = torch.tensor([1, 2, 3, 4, 5])
            features = torch.randn(5, 3)
            embed_layer.set_concept_features(word_indices, features)
            print(f"   ✅ set_concept_features() 成功")
        except Exception as e:
            print(f"   ❌ set_concept_features() 失败: {e}")
        
        print("\n2. 测试 GeologicalMemory 方法 | Testing GeologicalMemory methods...")
        memory = GeologicalMemory(dim=64)
        
        # 测试重置
        memory.reset()
        print(f"   ✅ reset() 成功")
        
        # 测试获取能量统计
        energy_stats = memory.get_energy_stats()
        print(f"   ✅ get_energy_stats() 成功: 包含 {len(energy_stats)} 个统计项")
        
        # 测试获取记忆健康度
        health_stats = memory.get_memory_health()
        print(f"   ✅ get_memory_health() 成功: 总体健康度={health_stats.get('overall_health', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 方法测试异常: {e} | Method test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数 | Main test function"""
    print("layers包导入测试开始... | layers Package Import Test Started...\n")
    
    # 运行所有测试
    tests = [
        ("导入测试 | Import Test", test_layers_import),
        ("实例化测试 | Instantiation Test", test_class_instantiation),
        ("方法测试 | Method Test", test_package_methods),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 60)
        
        try:
            passed = test_func()
            if not passed:
                all_passed = False
        except Exception as e:
            all_passed = False
            print(f"❌ 测试异常: {e} | Test exception: {e}")
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ 所有导入测试通过！ | All import tests passed!")
        return 0
    else:
        print("❌ 部分导入测试失败 | Some import tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())