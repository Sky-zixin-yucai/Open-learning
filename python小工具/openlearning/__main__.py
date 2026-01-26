"""
OpenLearning RGA - 主演示入口
==============================

通过以下方式运行：
    python -m openlearning           # 完整演示
    python -m openlearning --demo    # 运行演示
    python -m openlearning --test    # 运行测试
"""

import sys
import os
import argparse
import time
import warnings
import random
import string
from typing import Dict, List, Optional, Any, Tuple

# ==================== 路径设置 ====================
# 添加项目根目录到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ==================== 颜色输出 ====================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text: str, width: int = 60):
    """打印标题"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^{width}}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.END}")

def print_step(text: str, symbol: str = "✅"):
    """打印步骤"""
    print(f"{Colors.GREEN}{symbol} {text}{Colors.END}")

def print_info(text: str):
    """打印信息"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def print_warning(text: str):
    """打印警告"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text: str):
    """打印错误"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_success(text: str):
    """打印成功"""
    print(f"{Colors.GREEN}{Colors.BOLD}✨ {text}{Colors.END}")

def print_highlight(text: str):
    """高亮显示"""
    print(f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.END}")

# ==================== 模块检查 ====================

def check_modules() -> Tuple[bool, Dict[str, Any]]:
    """检查所有必需的模块"""
    print_header("模块检查")
    
    all_available = True
    module_status = {}
    
    # 检查核心模块
    try:
        import core
        module_status['core'] = {
            'loaded': True,
            'version': getattr(core, '__version__', '未知'),
            'exports': getattr(core, '__all__', [])[:5]  # 只显示前5个
        }
        print_step(f"✅ core模块已加载 (v{module_status['core']['version']})")
        
    except ImportError as e:
        print_error(f"❌ core模块导入失败: {e}")
        module_status['core'] = {'loaded': False, 'error': str(e)}
        all_available = False
    
    # 检查层模块
    try:
        import layers
        module_status['layers'] = {
            'loaded': True,
            'version': getattr(layers, '__version__', '未知'),
            'exports': getattr(layers, '__all__', [])[:5]
        }
        print_step(f"✅ layers模块已加载 (v{module_status['layers']['version']})")
        
    except ImportError as e:
        print_error(f"❌ layers模块导入失败: {e}")
        module_status['layers'] = {'loaded': False, 'error': str(e)}
        all_available = False
    
    # 检查集成模块
    try:
        import integration
        module_status['integration'] = {
            'loaded': True,
            'version': getattr(integration, '__version__', '未知'),
            'exports': getattr(integration, '__all__', [])[:5]
        }
        print_step(f"✅ integration模块已加载 (v{module_status['integration']['version']})")
        
    except ImportError as e:
        print_error(f"❌ integration模块导入失败: {e}")
        module_status['integration'] = {'loaded': False, 'error': str(e)}
        all_available = False
    
    return all_available, module_status

# ==================== 演示部分 ====================

class RGADemoRunner:
    """RGA演示运行器"""
    
    def __init__(self):
        self.models_created = []
        self.data_generated = {}
        self.results = {}
        self.module_available = {}
        
    def _safe_import(self, module_name: str, class_name: str):
        """安全导入类"""
        try:
            module = __import__(module_name)
            for sub in module_name.split('.')[1:]:
                module = getattr(module, sub)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            print_warning(f"导入 {module_name}.{class_name} 失败: {e}")
            return None
    
    def demo_1_basic_configuration(self):
        """演示1: 基础配置系统"""
        print_header("演示1: RGA基础配置系统")
        
        try:
            # 尝试导入核心模块
            RGAConfig = self._safe_import('core', 'RGAConfig')
            get_default_config = self._safe_import('core', 'get_default_config')
            validate_config = self._safe_import('core', 'validate_config')
            
            if not all([RGAConfig, get_default_config, validate_config]):
                print_error("核心模块功能不全，跳过此演示")
                return False
            
            # 1.1 创建默认配置
            print_info("1.1 创建默认配置")
            default_config = RGAConfig()
            print_step(f"默认配置创建成功:")
            print(f"  特征维度: {getattr(default_config, 'dim', 'N/A')}")
            print(f"  词汇表大小: {getattr(default_config, 'vocab_size', 'N/A')}")
            print(f"  相变阈值: {getattr(default_config, 'phase_threshold', 'N/A')}")
            
            # 1.2 创建自定义配置
            print_info("\n1.2 创建自定义配置")
            custom_config = RGAConfig(
                dim=64,
                vocab_size=1000,
                phase_threshold=0.8,
            )
            print_step(f"自定义配置创建成功:")
            print(f"  特征维度: {custom_config.dim}")
            print(f"  词汇表大小: {custom_config.vocab_size}")
            print(f"  相变阈值: {custom_config.phase_threshold}")
            
            # 1.3 验证配置
            print_info("\n1.3 配置验证")
            is_valid, message = validate_config(custom_config)
            if is_valid:
                print_step(f"配置验证通过: {message}")
            else:
                print_warning(f"配置验证警告: {message}")
            
            self.results['config'] = custom_config
            return True
            
        except Exception as e:
            print_error(f"配置演示失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def demo_2_layer_system(self):
        """演示2: 层系统"""
        print_header("演示2: RGA层系统")
        
        try:
            # 尝试导入层模块
            create_layer = self._safe_import('layers', 'create_layer')
            get_layer_factory = self._safe_import('layers', 'get_layer_factory')
            list_available_layers = self._safe_import('layers', 'list_available_layers')
            
            if not all([create_layer, get_layer_factory, list_available_layers]):
                print_error("层模块功能不全，跳过此演示")
                return False
            
            # 2.1 获取层工厂
            print_info("2.1 获取层工厂")
            layer_factory = get_layer_factory()
            print_step("层工厂获取成功")
            
            # 2.2 查看可用层
            print_info("\n2.2 可用层类型")
            layer_types = ['attention', 'balancer', 'normalization', 'valve']
            for layer_type in layer_types:
                available = list_available_layers(layer_type)
                if available:
                    print(f"  {layer_type}: {len(available)}种")
                    for layer in available[:2]:  # 只显示前两个
                        print(f"    - {layer}")
                    if len(available) > 2:
                        print(f"    ... 还有 {len(available)-2} 种")
            
            # 2.3 创建各种层
            print_info("\n2.3 创建各种RGA层")
            layers_to_create = [
                ('attention', 'VKQ注意力子网', {'subnet_type': 'vkq', 'dim': 64}),
                ('balancer', '三值平衡器', {'balancer_type': 'tri_value', 'dim': 64}),
                ('valve', '单向阀层', {'dim': 64, 'valve_type': 'learnable'}),
                ('FixedRMSNorm', '固定RMS归一化', {'dim': 64, 'eps': 1e-5}),
            ]
            
            created_layers = []
            for layer_type, layer_name, kwargs in layers_to_create:
                try:
                    print_info(f"创建{layer_name}...")
                    layer = create_layer(layer_type, **kwargs)
                    created_layers.append((layer_name, layer))
                    print_step(f"  {layer_name}创建成功: {layer.__class__.__name__}")
                except Exception as e:
                    print_warning(f"  创建{layer_name}失败: {e}")
            
            # 2.4 测试阀层功能
            print_info("\n2.4 测试单向阀功能")
            for name, layer in created_layers:
                if hasattr(layer, 'get_gate_values'):
                    try:
                        gate_values = layer.get_gate_values()
                        print_step(f"{name}门控值:")
                        print(f"  Q门: {gate_values.get('gate_Q', 0):.3f}")
                        print(f"  K门: {gate_values.get('gate_K', 0):.3f}")
                        print(f"  V门: {gate_values.get('gate_V', 0):.3f}")
                        if 'V_dominance' in gate_values:
                            print(f"  V主导性: {gate_values['V_dominance']}")
                    except Exception as e:
                        print_warning(f"  获取门控值失败: {e}")
            
            self.results['layers'] = created_layers
            return True
            
        except Exception as e:
            print_error(f"层系统演示失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def demo_3_engine_system(self):
        """演示3: RGA引擎系统"""
        print_header("演示3: RGA引擎系统")
        
        try:
            # 尝试导入引擎模块
            create_rga_engine = self._safe_import('core', 'create_rga_engine')
            
            if not create_rga_engine:
                print_error("引擎模块功能不全，跳过此演示")
                return False
            
            import torch
            
            # 3.1 创建引擎
            print_info("3.1 创建RGA引擎")
            engine_config = {
                "dim": 32,
                "vocab_size": 500,
                "phase_threshold": 0.5
            }
            
            engine = create_rga_engine(engine_config)
            print_step("RGA引擎创建成功")
            print(f"  配置: dim={engine_config['dim']}, 阈值={engine_config['phase_threshold']}")
            
            # 3.2 生成测试数据
            print_info("\n3.2 生成测试数据")
            batch_size = 2
            seq_len = 8
            dim = 32
            
            # 创建初始状态
            torch.manual_seed(42)  # 固定随机种子以便复现
            Q = torch.randn(batch_size, seq_len, dim)
            K = torch.randn(batch_size, seq_len, dim)
            V = torch.randn(batch_size, seq_len, dim)
            
            print_step(f"测试数据形状:")
            print(f"  Q: {Q.shape}")
            print(f"  K: {K.shape}")
            print(f"  V: {V.shape}")
            
            self.data_generated['engine'] = {'Q': Q, 'K': K, 'V': V}
            
            # 3.3 模拟状态变化和处理
            print_info("\n3.3 模拟状态变化处理")
            states_processed = []
            transitions_detected = 0
            
            for step in range(1, 11):  # 处理10个状态
                # 添加一些随机变化
                noise_scale = 0.2 if step % 3 == 0 else 0.05  # 每3步有一次大变化
                Q_new = Q + torch.randn_like(Q) * noise_scale
                K_new = K + torch.randn_like(K) * noise_scale
                V_new = V + torch.randn_like(V) * noise_scale
                
                # 处理状态
                result = engine.process_state(Q_new, K_new, V_new)
                states_processed.append(result)
                
                # 显示关键信息
                if step == 1 or step % 3 == 0:
                    print(f"  步骤{step:2d}: ", end="")
                    if 'state_change' in result:
                        print(f"变化={result['state_change']:.4f} ", end="")
                    if 'is_phase_transition' in result and result['is_phase_transition']:
                        print(f"🚨相变!", end="")
                        transitions_detected += 1
                    print()
                
                # 更新状态
                Q, K, V = Q_new, K_new, V_new
            
            # 3.4 获取分析报告
            print_info("\n3.4 获取分析报告")
            report = engine.get_analysis_report()
            
            print_step("分析报告摘要:")
            print(f"  处理状态数: {report.get('engine_info', {}).get('total_states_processed', 'N/A')}")
            
            transition_freq = report.get('performance_metrics', {}).get('transition_frequency', 0)
            print(f"  检测到相变: {transition_freq*100:.1f}%")
            
            learning_phase = report.get('learning_analysis', {}).get('phase', '未知')
            print(f"  当前学习阶段: {learning_phase}")
            
            # 3.5 显示优化建议
            recommendations = report.get('recommendations', [])
            if recommendations:
                print_info("优化建议:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec}")
            
            self.results['engine'] = engine
            return True
            
        except Exception as e:
            print_error(f"引擎系统演示失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def demo_4_integrator_system(self):
        """演示4: RGA集成器系统"""
        print_header("演示4: RGA集成器系统")
        
        try:
            # 尝试导入集成器模块
            create_integrator = self._safe_import('integration', 'create_integrator')
            RGAIntegrator = self._safe_import('integration', 'RGAIntegrator')
            RGAConfig = self._safe_import('core', 'RGAConfig')
            
            if not create_integrator:
                print_error("集成器模块功能不全，跳过此演示")
                return False
            
            import torch
            
            # 4.1 创建集成器
            print_info("4.1 创建RGA集成器")
            integrator_config = RGAConfig(
                vocab_size=100,
                dim=32,
                num_units=3,
                max_cycles=3,
            )
            
            integrator = create_integrator(integrator_config)
            print_step("RGA集成器创建成功")
            
            # 显示配置信息
            print(f"  词汇表大小: {integrator.config.vocab_size}")
            print(f"  特征维度: {integrator.config.dim}")
            print(f"  单元数量: {integrator.config.num_units}")
            print(f"  最大循环数: {integrator.config.max_cycles}")
            
            # 4.2 创建测试输入
            print_info("\n4.2 创建测试输入")
            torch.manual_seed(123)
            batch_size = 1
            seq_len = 16
            vocab_size = integrator.config.vocab_size
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            print_step(f"输入数据形状: {input_ids.shape}")
            print(f"  输入示例: {input_ids[0, :8].tolist()}...")  # 显示前8个token
            
            self.data_generated['integrator_input'] = input_ids
            
            # 4.3 执行前向传播
            print_info("\n4.3 执行前向传播")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = integrator.forward(input_ids)
            
            inference_time = time.time() - start_time
            
            print_step("推理完成")
            print(f"  推理时间: {inference_time:.4f}秒")
            print(f"  输出包含: {list(outputs.keys())}")
            
            # 4.4 分析输出
            print_info("\n4.4 分析模型输出")
            if 'logits' in outputs:
                logits = outputs['logits']
                print_step(f"Logits形状: {logits.shape}")
                if isinstance(logits, torch.Tensor):
                    print(f"  Logits范围: [{logits.min():.3f}, {logits.max():.3f}]")
            
            if 'attention_weights' in outputs:
                attn_weights = outputs['attention_weights']
                if isinstance(attn_weights, torch.Tensor):
                    print_step(f"注意力权重形状: {attn_weights.shape}")
                else:
                    print_step(f"注意力权重类型: {type(attn_weights)}")
            
            # 4.5 模型保存和加载演示
            print_info("\n4.5 模型保存和加载演示")
            try:
                import tempfile
                
                # 生成唯一的文件名
                random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                model_path = os.path.join(tempfile.gettempdir(), f"rga_demo_model_{random_str}.pth")
                
                print_info(f"保存模型到临时文件: {model_path}")
                
                # 保存模型状态
                torch.save({
                    'model_state_dict': integrator.state_dict(),
                    'config': integrator.config.to_dict(),
                }, model_path)
                print_step("模型保存成功")
                
                print_info("加载模型")
                # 创建新模型并加载状态
                loaded_integrator = create_integrator(integrator_config)
                checkpoint = torch.load(model_path)
                loaded_integrator.load_state_dict(checkpoint['model_state_dict'])
                print_step("模型加载成功")
                
                # 验证加载的模型
                print_info("验证加载的模型")
                test_outputs = loaded_integrator.forward(input_ids)
                print_step("加载模型推理成功")
                
                # 清理临时文件
                if os.path.exists(model_path):
                    os.unlink(model_path)
                    print_step("临时文件已清理")
                
            except Exception as e:
                print_warning(f"模型保存/加载演示跳过: {e}")
            
            self.results['integrator'] = integrator
            return True
            
        except Exception as e:
            print_error(f"集成器系统演示失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def demo_5_visualization(self):
        """演示5: 可视化功能"""
        print_header("演示5: RGA可视化功能")
        
        try:
            # 5.1 检查matplotlib
            try:
                import matplotlib
                import matplotlib.pyplot as plt
                matplotlib_available = True
                
                # 修复中文字体问题
                try:
                    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False
                except Exception as e:
                    print_warning(f"字体设置失败: {e}")
                    pass
                    
            except ImportError:
                print_warning("Matplotlib未安装，跳过可视化演示")
                print_info("安装命令: pip install matplotlib")
                return True  # 不是错误，只是跳过
            
            if not matplotlib_available:
                return True
            
            # 5.2 创建示例数据可视化
            print_info("5.1 创建状态变化图表")
            
            # 生成示例数据
            import numpy as np
            np.random.seed(42)
            
            # 模拟状态变化序列
            steps = 20
            state_changes = np.random.exponential(0.3, steps)
            state_changes[5] = 1.2  # 模拟相变
            state_changes[12] = 0.9  # 模拟相变
            
            # 创建图表 - 使用英文避免字体问题
            plt.figure(figsize=(10, 6))
            
            # 状态变化曲线
            plt.subplot(2, 1, 1)
            plt.plot(range(1, steps + 1), state_changes, 'b-o', linewidth=2, markersize=6)
            plt.axhline(y=0.8, color='r', linestyle='--', linewidth=1.5, label='Phase Threshold (0.8)')
            
            # 标记相变点
            transition_points = [5, 12]
            for point in transition_points:
                plt.scatter(point + 1, state_changes[point], color='red', s=150, 
                           zorder=5, marker='*', edgecolors='black')
            
            plt.xlabel('Step')
            plt.ylabel('State Change')
            plt.title('RGA State Change Monitoring', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 学习阶段分布
            plt.subplot(2, 1, 2)
            phases = ['Exploration', 'Learning', 'Stable', 'Convergence']
            phase_counts = [5, 8, 4, 3]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            bars = plt.bar(phases, phase_counts, color=colors, edgecolor='black', linewidth=1.5)
            
            # 添加数值标签
            for bar, count in zip(bars, phase_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.xlabel('Learning Phase', fontsize=12)
            plt.ylabel('Duration Steps', fontsize=12)
            plt.title('RGA Learning Phase Distribution', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # 保存图表
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                chart_path = tmp.name
            
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            
            # 显示图表
            plt.show()
            plt.close()
            
            print_step(f"可视化图表已生成: {chart_path}")
            print_info("图表显示:")
            print("  1. 状态变化趋势和相变检测")
            print("  2. 学习阶段分布")
            
            # 清理临时文件
            if os.path.exists(chart_path):
                os.unlink(chart_path)
                print_step("临时图表文件已清理")
            
            return True
            
        except Exception as e:
            print_warning(f"可视化演示遇到问题: {e}")
            return True  # 可视化不是核心功能，允许失败
    
    def run_complete_demo(self):
        """运行完整的演示流程"""
        print_header("OpenLearning RGA Complete Demo", width=80)
        print_highlight("体验完整的RGA规则治理架构!")
        print()
        
        # 检查环境
        print_info("检查环境...")
        modules_ok, module_status = check_modules()
        
        if not modules_ok:
            print_error("环境检查失败，无法继续演示")
            print_info("请确保所有子模块都已正确安装")
            return False
        
        print_step("环境检查通过，开始演示...")
        print()
        
        # 运行各个演示
        demos = [
            ("基础配置", self.demo_1_basic_configuration),
            ("层系统", self.demo_2_layer_system),
            ("引擎系统", self.demo_3_engine_system),
            ("集成器系统", self.demo_4_integrator_system),
            ("可视化", self.demo_5_visualization),
        ]
        
        results = {}
        total_demos = len(demos)
        successful_demos = 0
        
        for demo_name, demo_func in demos:
            try:
                print_info(f"开始演示: {demo_name}")
                success = demo_func()
                results[demo_name] = success
                
                if success:
                    successful_demos += 1
                    print_step(f"演示完成: {demo_name}")
                else:
                    print_warning(f"演示失败: {demo_name}")
                
                print()  # 空行分隔
                
            except Exception as e:
                print_error(f"演示异常: {demo_name} - {e}")
                import traceback
                traceback.print_exc()
                results[demo_name] = False
        
        # 演示总结
        print_header("演示总结", width=80)
        print_step(f"完成演示: {successful_demos}/{total_demos}")
        
        for demo_name, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            print(f"  {demo_name:15} {status}")
        
        print()
        
        if successful_demos == total_demos:
            print_success("🎉 所有演示均成功完成!")
            print()
            print_highlight("您已体验了完整的RGA规则治理架构:")
            print("  1. 灵活的RGA配置系统")
            print("  2. 丰富的专用层类型")
            print("  3. 智能状态监控和相变检测引擎")
            print("  4. 完整的模型集成能力")
            print("  5. 直观的可视化分析")
        else:
            print_warning(f"部分演示失败 ({total_demos - successful_demos}/{total_demos})")
            print_info("请检查模块导入和依赖项")
        
        return successful_demos == total_demos

# ==================== 主程序 ====================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="OpenLearning RGA - Main Demo Entry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m openlearning           # 运行完整演示
  python -m openlearning --demo    # 运行演示
  python -m openlearning --test    # 运行测试
        """
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="运行完整演示"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="运行测试"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="运行快速演示（跳过一些步骤）"
    )
    
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="跳过可视化演示"
    )
    
    parser.add_argument(
        "--check-modules",
        action="store_true",
        help="只检查模块状态"
    )
    
    args = parser.parse_args()
    
    # 显示欢迎信息
    print_header("OpenLearning RGA 规则治理架构", width=80)
    print_highlight("一站式体验脚本 - 无需额外安装")
    print()
    print("版本: 0.0.6 | 作者: RGA 架构团队")
    print("GitHub: https://github.com/Sky-zixin-yucai/Open-learning")
    print("="*80)
    
    # 如果只检查模块
    if args.check_modules:
        modules_ok, module_status = check_modules()
        return 0 if modules_ok else 1
    
    # 创建演示运行器
    runner = RGADemoRunner()
    
    # 如果没有指定任何参数，默认运行演示
    if not args.demo and not args.test:
        args.demo = True
    
    try:
        success = True
        
        if args.demo:
            success = runner.run_complete_demo() and success
        
        if args.test:
            print_header("运行测试")
            # 运行集成模块的测试
            try:
                import integration
                if hasattr(integration, 'test_integrator'):
                    integration.test_integrator()
                    print_step("测试完成")
                else:
                    print_warning("测试函数不可用")
                    success = False
            except ImportError:
                print_error("无法导入integration模块")
                success = False
            except Exception as e:
                print_error(f"测试失败: {e}")
                success = False
        
        if success:
            print()
            print_header("下一步", width=80)
            print("1. 查看项目文档以了解详细设计")
            print("2. 运行单元测试: python -m pytest tests/")
            print("3. 尝试修改配置参数以创建自定义模型")
            print("4. 在您自己的项目中导入和使用RGA组件")
            print("5. 为项目贡献: https://github.com/Sky-zixin-yucai/Open-learning")
            print()
            print_success("感谢使用 OpenLearning RGA! 🚀")
            return 0
        else:
            print_error("演示过程中遇到问题，请检查错误信息")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
        return 0
    except Exception as e:
        print_error(f"演示运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

# ==================== 入口点 ====================

if __name__ == "__main__":
    sys.exit(main())