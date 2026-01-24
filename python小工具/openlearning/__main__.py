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
from typing import Dict, List, Optional, Any

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

def check_modules():
    """检查所有必需的模块"""
    print_header("模块检查")
    
    all_available = True
    
    # 检查核心模块
    try:
        # 导入核心模块的__init__.py来检查
        from core import __init__ as core_module
        print_step(f"core模块已加载")
        
        # 检查核心模块是否有必要的类
        try:
            from core import RGAConfig, RGAEngine, create_rga_engine
            print_step(f"  - 核心类可用: RGAConfig, RGAEngine")
        except ImportError:
            print_warning(f"  - 部分核心类不可用")
            
    except ImportError as e:
        print_error(f"core模块导入失败: {e}")
        all_available = False
    
    # 检查层模块
    try:
        # 导入层模块的__init__.py来检查
        from layers import __init__ as layers_module
        print_step(f"layers模块已加载")
        
        # 检查层模块是否有必要的类
        try:
            from layers import create_layer, get_layer_factory
            print_step(f"  - 工厂函数可用")
        except ImportError:
            print_warning(f"  - 部分层函数不可用")
            
    except ImportError as e:
        print_error(f"layers模块导入失败: {e}")
        all_available = False
    
    # 检查集成模块
    try:
        # 导入集成模块的__init__.py来检查
        from integration import __init__ as integration_module
        print_step(f"integration模块已加载")
        
        # 检查集成模块是否有必要的类
        try:
            from integration import create_integrator, RGAIntegrator
            print_step(f"  - 集成器可用")
        except ImportError:
            print_warning(f"  - 部分集成器类不可用")
            
    except ImportError as e:
        print_error(f"integration模块导入失败: {e}")
        all_available = False
    
    return all_available

# ==================== 演示部分 ====================

class RGADemoRunner:
    """RGA演示运行器"""
    
    def __init__(self):
        self.models_created = []
        self.data_generated = {}
        self.results = {}
        
    def demo_1_basic_configuration(self):
        """演示1: 基础配置系统"""
        print_header("演示1: RGA基础配置系统")
        
        try:
            # 使用相对导入而不是绝对导入
            try:
                from .core import RGAConfig, get_default_config, validate_config
            except ImportError:
                from core import RGAConfig, get_default_config, validate_config
            
            # 1.1 创建默认配置
            print_info("1.1 创建默认配置")
            default_config = RGAConfig()
            print_step(f"默认配置创建成功:")
            print(f"  特征维度: {default_config.dim}")
            print(f"  词汇表大小: {default_config.vocab_size}")
            print(f"  相变阈值: {default_config.phase_threshold}")
            
            # 检查是否有额外的配置属性
            if hasattr(default_config, 'to_dict'):
                config_dict = default_config.to_dict()
                extra_info = False
                
                # 显示额外的配置信息（如果有的话）
                for key, value in config_dict.items():
                    if key not in ['dim', 'vocab_size', 'phase_threshold'] and value is not None:
                        if not extra_info:
                            print(f"  额外配置:")
                            extra_info = True
                        print(f"    {key}: {value}")
            
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
            return False
    
    def demo_2_layer_system(self):
        """演示2: 层系统"""
        print_header("演示2: RGA层系统")
        
        try:
            # 使用相对导入
            try:
                from .layers import create_layer, get_layer_factory, list_available_layers
            except ImportError:
                from layers import create_layer, get_layer_factory, list_available_layers
            
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
                    except:
                        pass
            
            self.results['layers'] = created_layers
            return True
            
        except Exception as e:
            print_error(f"层系统演示失败: {e}")
            return False
    
    def demo_3_engine_system(self):
        """演示3: RGA引擎系统"""
        print_header("演示3: RGA引擎系统")
        
        try:
            # 使用相对导入
            try:
                from .core import create_rga_engine, calculate_state_change
            except ImportError:
                from core import create_rga_engine, calculate_state_change
            
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
            print(f"  处理状态数: {report['engine_info']['total_states_processed']}")
            print(f"  检测到相变: {report['performance_metrics']['transition_frequency']*100:.1f}%")
            print(f"  当前学习阶段: {report['learning_analysis']['phase']}")
            
            # 3.5 显示优化建议
            if 'recommendations' in report and report['recommendations']:
                print_info("优化建议:")
                for i, rec in enumerate(report['recommendations'][:3], 1):
                    print(f"  {i}. {rec}")
            
            self.results['engine'] = engine
            return True
            
        except Exception as e:
            print_error(f"引擎系统演示失败: {e}")
            return False
    
    def demo_4_integrator_system(self):
        """演示4: RGA集成器系统"""
        print_header("演示4: RGA集成器系统")
        
        try:
            # 使用相对导入
            try:
                from .integration import create_integrator, save_disguised_model, load_disguised_model
            except ImportError:
                from integration import create_integrator, save_disguised_model, load_disguised_model
            
            import torch
            
            # 4.1 创建集成器
            print_info("4.1 创建RGA集成器")
            integrator_config = {
                "vocab_size": 100,
                "dim": 32,
                "num_units": 1,
                "max_cycles": 2,
                "enable_disguise": True
            }
            
            integrator = create_integrator(integrator_config)
            print_step("RGA集成器创建成功")
            
            # 通过config获取属性
            if hasattr(integrator, 'config'):
                print(f"  词汇表大小: {integrator.config.vocab_size}")
                print(f"  特征维度: {integrator.config.dim}")
                print(f"  单元数量: {integrator.config.num_units}")
            else:
                print(f"  配置: {integrator_config}")
            
            # 4.2 创建测试输入
            print_info("\n4.2 创建测试输入")
            torch.manual_seed(123)
            batch_size = 1
            seq_len = 16
            vocab_size = integrator_config['vocab_size']
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
            
            # 4.5 保存和加载模型（演示）
            print_info("\n4.5 模型保存和加载演示")
            try:
                # 创建临时文件路径
                import tempfile
                import random
                import string
                
                # 生成唯一的文件名
                random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                model_path = os.path.join(tempfile.gettempdir(), f"rga_demo_model_{random_str}.pth")
                
                print_info(f"保存模型到临时文件: {model_path}")
                save_disguised_model(integrator, model_path)
                print_step("模型保存成功")
                
                print_info("加载模型")
                loaded_model = load_disguised_model(model_path)
                print_step("模型加载成功")
                
                # 验证加载的模型
                print_info("验证加载的模型")
                test_outputs = loaded_model.forward(input_ids)
                print_step("加载模型推理成功")
                
                # 清理临时文件 - 更稳健的删除方式
                try:
                    import shutil
                    if os.path.exists(model_path):
                        if os.path.isfile(model_path):
                            os.unlink(model_path)
                        else:
                            shutil.rmtree(model_path)
                        print_step("临时文件已清理")
                except Exception as e:
                    print_warning(f"临时文件清理失败: {e}")
                
            except Exception as e:
                print_warning(f"模型保存/加载演示跳过: {e}")
            
            self.results['integrator'] = integrator
            return True
            
        except Exception as e:
            print_error(f"集成器系统演示失败: {e}")
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
                    # 尝试设置中文字体
                    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False
                except:
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
            plt.close()
            
            print_step(f"Visualization chart generated: {chart_path}")
            print_info("Chart shows:")
            print("  1. State change trend and phase transition detection")
            print("  2. Learning phase distribution")
            
            # 清理临时文件
            os.unlink(chart_path)
            
            return True
            
        except Exception as e:
            print_warning(f"Visualization demo encountered issues: {e}")
            return True  # 可视化不是核心功能，允许失败
    
    def run_complete_demo(self):
        """运行完整的演示流程"""
        print_header("OpenLearning RGA Complete Demo", width=80)
        print_highlight("Experience the complete RGA Rule-Governed Architecture!")
        print()
        
        # 检查环境
        print_info("Checking environment...")
        modules_ok = check_modules()
        
        if not modules_ok:
            print_error("Environment check failed, cannot continue demo")
            return False
        
        print_step("Environment check passed, starting demo...")
        print()
        
        # 运行各个演示
        demos = [
            ("Basic Configuration", self.demo_1_basic_configuration),
            ("Layer System", self.demo_2_layer_system),
            ("Engine System", self.demo_3_engine_system),
            ("Integrator System", self.demo_4_integrator_system),
            ("Visualization", self.demo_5_visualization),
        ]
        
        results = {}
        total_demos = len(demos)
        successful_demos = 0
        
        for demo_name, demo_func in demos:
            try:
                print_info(f"Starting: {demo_name}")
                success = demo_func()
                results[demo_name] = success
                
                if success:
                    successful_demos += 1
                    print_step(f"Demo completed: {demo_name}")
                else:
                    print_warning(f"Demo failed: {demo_name}")
                
                print()  # 空行分隔
                
            except Exception as e:
                print_error(f"Demo exception: {demo_name} - {e}")
                results[demo_name] = False
        
        # 演示总结
        print_header("Demo Summary", width=80)
        print_step(f"Completed demos: {successful_demos}/{total_demos}")
        
        for demo_name, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"  {demo_name:20} {status}")
        
        print()
        
        if successful_demos == total_demos:
            print_success("🎉 All demos completed successfully!")
            print()
            print_highlight("You've experienced the complete RGA Rule-Governed Architecture:")
            print("  1. Flexible RGA configuration system")
            print("  2. Rich variety of specialized layers")
            print("  3. Intelligent state monitoring and phase detection engine")
            print("  4. Complete model integration and disguise capability")
            print("  5. Intuitive visualization analysis")
        else:
            print_warning(f"Some demos failed ({total_demos - successful_demos}/{total_demos})")
            print_info("Check module imports and dependencies")
        
        return successful_demos == total_demos

# ==================== 主程序 ====================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="OpenLearning RGA - Main Demo Entry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m openlearning           # Run complete demo
  python -m openlearning --demo    # Run demo only
  python -m openlearning --test    # Run tests only
        """
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run complete demo"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast demo (skip some steps)"
    )
    
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip visualization demo"
    )
    
    args = parser.parse_args()
    
    # 显示欢迎信息
    print_header("OpenLearning RGA Rule-Governed Architecture", width=80)
    print_highlight("One-stop experience script - No installation required")
    print()
    print("Version: 0.0.3 | Author: RGA Architecture Team")
    print("GitHub: https://github.com/Sky-zixin-yucai/Open-learning")
    print("="*80)
    
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
            print_header("Running Tests")
            # 运行集成模块的测试
            try:
                from integration import test_integrator
                test_integrator()
                print_step("Tests completed")
            except ImportError:
                print_error("Test function not available")
                success = False
            except Exception as e:
                print_error(f"Test failed: {e}")
                success = False
        
        if success:
            print()
            print_header("Next Steps", width=80)
            print("1. View project documentation for detailed design")
            print("2. Run unit tests: python -m pytest tests/")
            print("3. Try modifying configuration parameters to create custom models")
            print("4. Import and use RGA components in your own projects")
            print("5. Contribute to the project: https://github.com/Sky-zixin-yucai/Open-learning")
            print()
            print_success("Thank you for using OpenLearning RGA! 🚀")
            return 0
        else:
            print_error("Issues encountered during demo, check error messages")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        return 0
    except Exception as e:
        print_error(f"Demo run failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

# ==================== 入口点 ====================

if __name__ == "__main__":
    sys.exit(main())