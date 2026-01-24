#!/usr/bin/env python3
"""
layers包功能测试 | layers Package Functionality Test
==================================================

测试layers包各模块的实际功能。
Test the actual functionality of each module in layers package.
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

# 将openlearning目录添加到Python路径
sys.path.insert(0, openlearning_dir)
print(f"\n已添加目录到Python路径: {openlearning_dir}")

def run_embeddings_tests():
    """运行embeddings模块测试 | Run embeddings module tests"""
    print("\n" + "="*80)
    print("测试embeddings模块 | Testing embeddings module")
    print("="*80)
    
    try:
        # 方法1：尝试导入整个模块
        print("\n1. 尝试导入embeddings模块 | Trying to import embeddings module...")
        try:
            # 由于embeddings模块在layers包内，我们需要通过layers导入
            from layers.embeddings import test_enhanced_embedding
            from layers.embeddings import test_concept_aware_embedding
            from layers.embeddings import test_embedding_factory
            print(f"✅ embeddings模块导入成功 | embeddings module import successful")
                
        except ImportError as e:
            print(f"❌ embeddings模块导入失败: {e} | embeddings module import failed: {e}")
            return False
        
        # 方法2：运行测试函数
        print("\n2. 运行测试函数 | Running test functions...")
        
        print("   运行 test_enhanced_embedding... | Running test_enhanced_embedding...")
        try:
            test_enhanced_embedding()
            print("   ✅ test_enhanced_embedding 通过 | test_enhanced_embedding passed")
        except Exception as e:
            print(f"   ❌ test_enhanced_embedding 失败: {e} | test_enhanced_embedding failed: {e}")
            return False
        
        print("   运行 test_concept_aware_embedding... | Running test_concept_aware_embedding...")
        try:
            test_concept_aware_embedding()
            print("   ✅ test_concept_aware_embedding 通过 | test_concept_aware_embedding passed")
        except Exception as e:
            print(f"   ❌ test_concept_aware_embedding 失败: {e} | test_concept_aware_embedding failed: {e}")
            return False
        
        print("   运行 test_embedding_factory... | Running test_embedding_factory...")
        try:
            test_embedding_factory()
            print("   ✅ test_embedding_factory 通过 | test_embedding_factory passed")
        except Exception as e:
            print(f"   ❌ test_embedding_factory 失败: {e} | test_embedding_factory failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ embeddings模块测试异常: {e} | embeddings module test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_fusion_tests():
    """运行fusion模块测试 | Run fusion module tests"""
    print("\n" + "="*80)
    print("测试fusion模块 | Testing fusion module")
    print("="*80)
    
    try:
        # 方法1：尝试导入整个模块
        print("\n1. 尝试导入fusion模块 | Trying to import fusion module...")
        try:
            # 由于fusion模块在layers包内，我们需要通过layers导入
            from layers.fusion import test_sandwich_fusion
            print(f"✅ fusion模块导入成功 | fusion module import successful")
                
        except ImportError as e:
            print(f"❌ fusion模块导入失败: {e} | fusion module import failed: {e}")
            return False
        
        # 方法2：运行测试函数
        print("\n2. 运行测试函数 | Running test functions...")
        
        print("   运行 test_sandwich_fusion... | Running test_sandwich_fusion...")
        try:
            test_sandwich_fusion()
            print("   ✅ test_sandwich_fusion 通过 | test_sandwich_fusion passed")
        except Exception as e:
            print(f"   ❌ test_sandwich_fusion 失败: {e} | test_sandwich_fusion failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ fusion模块测试异常: {e} | fusion module test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_memory_tests():
    """运行memory模块测试 | Run memory module tests"""
    print("\n" + "="*80)
    print("测试memory模块 | Testing memory module")
    print("="*80)
    
    try:
        # 方法1：尝试导入整个模块
        print("\n1. 尝试导入memory模块 | Trying to import memory module...")
        try:
            # 由于memory模块在layers包内，我们需要通过layers导入
            from layers.memory import test_geological_memory
            print(f"✅ memory模块导入成功 | memory module import successful")
                
        except ImportError as e:
            print(f"❌ memory模块导入失败: {e} | memory module import failed: {e}")
            return False
        
        # 方法2：运行测试函数
        print("\n2. 运行测试函数 | Running test functions...")
        
        print("   运行 test_geological_memory... | Running test_geological_memory...")
        try:
            test_geological_memory()
            print("   ✅ test_geological_memory 通过 | test_geological_memory passed")
        except Exception as e:
            print(f"   ❌ test_geological_memory 失败: {e} | test_geological_memory failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ memory模块测试异常: {e} | memory module test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_normalization_tests():
    """运行normalization模块测试 | Run normalization module tests"""
    print("\n" + "="*80)
    print("测试normalization模块 | Testing normalization module")
    print("="*80)
    
    try:
        # 方法1：尝试导入整个模块
        print("\n1. 尝试导入normalization模块 | Trying to import normalization module...")
        try:
            # 由于normalization模块在layers包内，我们需要通过layers导入
            from layers.normalization import test_fixed_rms_norm
            from layers.normalization import test_group_rms_norm
            from layers.normalization import test_scaled_rms_norm
            print(f"✅ normalization模块导入成功 | normalization module import successful")
                
        except ImportError as e:
            print(f"❌ normalization模块导入失败: {e} | normalization module import failed: {e}")
            return False
        
        # 方法2：运行测试函数
        print("\n2. 运行测试函数 | Running test functions...")
        
        print("   运行 test_fixed_rms_norm... | Running test_fixed_rms_norm...")
        try:
            test_fixed_rms_norm()
            print("   ✅ test_fixed_rms_norm 通过 | test_fixed_rms_norm passed")
        except Exception as e:
            print(f"   ❌ test_fixed_rms_norm 失败: {e} | test_fixed_rms_norm failed: {e}")
            return False
        
        print("   运行 test_group_rms_norm... | Running test_group_rms_norm...")
        try:
            test_group_rms_norm()
            print("   ✅ test_group_rms_norm 通过 | test_group_rms_norm passed")
        except Exception as e:
            print(f"   ❌ test_group_rms_norm 失败: {e} | test_group_rms_norm failed: {e}")
            return False
        
        print("   运行 test_scaled_rms_norm... | Running test_scaled_rms_norm...")
        try:
            test_scaled_rms_norm()
            print("   ✅ test_scaled_rms_norm 通过 | test_scaled_rms_norm passed")
        except Exception as e:
            print(f"   ❌ test_scaled_rms_norm 失败: {e} | test_scaled_rms_norm failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ normalization模块测试异常: {e} | normalization module test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_valve_tests():
    """运行valve模块测试 | Run valve module tests"""
    print("\n" + "="*80)
    print("测试valve模块 | Testing valve module")
    print("="*80)
    
    try:
        # 方法1：尝试导入整个模块
        print("\n1. 尝试导入valve模块 | Trying to import valve module...")
        try:
            # 由于valve模块在layers包内，我们需要通过layers导入
            from layers.valve import test_one_way_valve
            print(f"✅ valve模块导入成功 | valve module import successful")
                
        except ImportError as e:
            print(f"❌ valve模块导入失败: {e} | valve module import failed: {e}")
            return False
        
        # 方法2：运行测试函数
        print("\n2. 运行测试函数 | Running test functions...")
        
        print("   运行 test_one_way_valve... | Running test_one_way_valve...")
        try:
            test_one_way_valve()
            print("   ✅ test_one_way_valve 通过 | test_one_way_valve passed")
        except Exception as e:
            print(f"   ❌ test_one_way_valve 失败: {e} | test_one_way_valve failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ valve模块测试异常: {e} | valve module test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_balancer_tests():
    """运行balancer模块测试 | Run balancer module tests"""
    print("\n" + "="*80)
    print("测试balancer模块 | Testing balancer module")
    print("="*80)
    
    try:
        # 方法1：尝试导入整个模块
        print("\n1. 尝试导入balancer模块 | Trying to import balancer module...")
        try:
            # 由于balancer模块在layers包内，我们需要通过layers导入
            from layers.balancer import test_tri_value_balancer
            from layers.balancer import test_v_dominant_balancer
            from layers.balancer import test_density_driven_balancer
            from layers.balancer import test_adaptive_stabilizer
            from layers.balancer import test_balancer_factory
            print(f"✅ balancer模块导入成功 | balancer module import successful")
                
        except ImportError as e:
            print(f"❌ balancer模块导入失败: {e} | balancer module import failed: {e}")
            return False
        
        # 方法2：运行测试函数
        print("\n2. 运行测试函数 | Running test functions...")
        
        print("   运行 test_tri_value_balancer... | Running test_tri_value_balancer...")
        try:
            test_tri_value_balancer()
            print("   ✅ test_tri_value_balancer 通过 | test_tri_value_balancer passed")
        except Exception as e:
            print(f"   ❌ test_tri_value_balancer 失败: {e} | test_tri_value_balancer failed: {e}")
            return False
        
        print("   运行 test_v_dominant_balancer... | Running test_v_dominant_balancer...")
        try:
            test_v_dominant_balancer()
            print("   ✅ test_v_dominant_balancer 通过 | test_v_dominant_balancer passed")
        except Exception as e:
            print(f"   ❌ test_v_dominant_balancer 失败: {e} | test_v_dominant_balancer failed: {e}")
            return False
        
        print("   运行 test_density_driven_balancer... | Running test_density_driven_balancer...")
        try:
            test_density_driven_balancer()
            print("   ✅ test_density_driven_balancer 通过 | test_density_driven_balancer passed")
        except Exception as e:
            print(f"   ❌ test_density_driven_balancer 失败: {e} | test_density_driven_balancer failed: {e}")
            return False
        
        print("   运行 test_adaptive_stabilizer... | Running test_adaptive_stabilizer...")
        try:
            test_adaptive_stabilizer()
            print("   ✅ test_adaptive_stabilizer 通过 | test_adaptive_stabilizer passed")
        except Exception as e:
            print(f"   ❌ test_adaptive_stabilizer 失败: {e} | test_adaptive_stabilizer failed: {e}")
            return False
        
        print("   运行 test_balancer_factory... | Running test_balancer_factory...")
        try:
            test_balancer_factory()
            print("   ✅ test_balancer_factory 通过 | test_balancer_factory passed")
        except Exception as e:
            print(f"   ❌ test_balancer_factory 失败: {e} | test_balancer_factory failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ balancer模块测试异常: {e} | balancer module test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_attention_tests():
    """运行attention模块测试 | Run attention module tests"""
    print("\n" + "="*80)
    print("测试attention模块 | Testing attention module")
    print("="*80)
    
    try:
        # 方法1：尝试导入整个模块
        print("\n1. 尝试导入attention模块 | Trying to import attention module...")
        try:
            # 由于attention模块在layers包内，我们需要通过layers导入
            from layers.attention import test_vkq_subnet
            from layers.attention import test_qvk_subnet
            from layers.attention import test_kqv_subnet
            from layers.attention import test_chain_reaction_unit
            from layers.attention import test_attention_factory
            print(f"✅ attention模块导入成功 | attention module import successful")
                
        except ImportError as e:
            print(f"❌ attention模块导入失败: {e} | attention module import failed: {e}")
            return False
        
        # 方法2：运行测试函数
        print("\n2. 运行测试函数 | Running test functions...")
        
        print("   运行 test_vkq_subnet... | Running test_vkq_subnet...")
        try:
            test_vkq_subnet()
            print("   ✅ test_vkq_subnet 通过 | test_vkq_subnet passed")
        except Exception as e:
            print(f"   ❌ test_vkq_subnet 失败: {e} | test_vkq_subnet failed: {e}")
            return False
        
        print("   运行 test_qvk_subnet... | Running test_qvk_subnet...")
        try:
            test_qvk_subnet()
            print("   ✅ test_qvk_subnet 通过 | test_qvk_subnet passed")
        except Exception as e:
            print(f"   ❌ test_qvk_subnet 失败: {e} | test_qvk_subnet failed: {e}")
            return False
        
        print("   运行 test_kqv_subnet... | Running test_kqv_subnet...")
        try:
            test_kqv_subnet()
            print("   ✅ test_kqv_subnet 通过 | test_kqv_subnet passed")
        except Exception as e:
            print(f"   ❌ test_kqv_subnet 失败: {e} | test_kqv_subnet failed: {e}")
            return False
        
        print("   运行 test_chain_reaction_unit... | Running test_chain_reaction_unit...")
        try:
            test_chain_reaction_unit()
            print("   ✅ test_chain_reaction_unit 通过 | test_chain_reaction_unit passed")
        except Exception as e:
            print(f"   ❌ test_chain_reaction_unit 失败: {e} | test_chain_reaction_unit failed: {e}")
            return False
        
        print("   运行 test_attention_factory... | Running test_attention_factory...")
        try:
            test_attention_factory()
            print("   ✅ test_attention_factory 通过 | test_attention_factory passed")
        except Exception as e:
            print(f"   ❌ test_attention_factory 失败: {e} | test_attention_factory failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ attention模块测试异常: {e} | attention module test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数 | Main test function"""
    print("layers包功能测试开始... | layers Package Functionality Test Started...\n")
    
    # 运行所有模块测试
    modules_tests = [
        ("embeddings模块测试 | Embeddings Module Test", run_embeddings_tests),
        ("fusion模块测试 | Fusion Module Test", run_fusion_tests),
        ("memory模块测试 | Memory Module Test", run_memory_tests),
        ("normalization模块测试 | Normalization Module Test", run_normalization_tests),
        ("valve模块测试 | Valve Module Test", run_valve_tests),
        ("balancer模块测试 | Balancer Module Test", run_balancer_tests),
        ("attention模块测试 | Attention Module Test", run_attention_tests),
    ]
    
    all_passed = True
    for test_name, test_func in modules_tests:
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
        print("✅ 所有功能测试通过！ | All functionality tests passed!")
        return 0
    else:
        print("❌ 部分功能测试失败 | Some functionality tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())