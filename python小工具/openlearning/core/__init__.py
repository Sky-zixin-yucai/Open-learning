"""
core包 | core Package
====================

RGA架构核心模块包。
RGA Architecture core module package.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import warnings

__version__ = "0.0.5"
__author__ = "RGA Architecture Team"

# 直接从当前目录导入模块
# Import modules directly from current directory
try:
    from config import RGAConfig
    from metrics import CoreMetricsCalculator
except ImportError:
    # 如果直接导入失败，尝试相对导入
    # If direct import fails, try relative import
    try:
        from .config import RGAConfig
        from .metrics import CoreMetricsCalculator
    except ImportError as e:
        raise ImportError(f"Failed to import core modules: {e}")

__all__ = ["RGAConfig", "CoreMetricsCalculator", "__version__", "__author__"]

# ==================== 便捷函数 | Convenience Functions ====================
def get_default_config(**kwargs) -> RGAConfig:
    """
    获取默认配置或自定义配置
    Get default configuration or customized configuration
    
    参数 | Parameters:
    ------------
    **kwargs : dict
        要覆盖的配置参数 | Configuration parameters to override
    
    返回 | Returns:
    ------------
    RGAConfig
        配置实例 | Configuration instance
    
    示例 | Example:
    ------------
        # 获取默认配置 | Get default configuration
        config = get_default_config()
        
        # 获取自定义配置 | Get customized configuration
        config = get_default_config(vocab_size=20000, dim=512)
    """
    return RGAConfig(**kwargs)


def validate_config(config: Union[RGAConfig, Dict]) -> Tuple[bool, str]:
    """
    验证配置的有效性
    Validate configuration validity
    
    参数 | Parameters:
    ------------
    config : Union[RGAConfig, Dict]
        配置实例或配置字典 | Configuration instance or dictionary
    
    返回 | Returns:
    ------------
    Tuple[bool, str]
        (是否有效, 验证消息) | (Whether valid, validation message)
    """
    try:
        if isinstance(config, dict):
            config = RGAConfig.from_dict(config)
        config.validate()
        return True, "配置验证通过 | Configuration validation passed"
    except Exception as e:
        return False, f"配置验证失败: {e} | Configuration validation failed: {e}"


def calculate_state_change(
    current_state: Dict[str, torch.Tensor],
    previous_state: Dict[str, torch.Tensor],
    norm_type: str = 'l2'
) -> float:
    """
    便捷计算状态变化量
    Conveniently calculate state change
    
    参数 | Parameters:
    ------------
    current_state : Dict[str, torch.Tensor]
        当前状态字典，包含Q、K、V | Current state dict with Q, K, V
    previous_state : Dict[str, torch.Tensor]
        上一状态字典，包含Q、K、V | Previous state dict with Q, K, V
    norm_type : str, optional
        范数类型 ('l1' 或 'l2') | Norm type ('l1' or 'l2')
    
    返回 | Returns:
    ------------
    float
        状态变化量 | State change amount
    """
    calculator = CoreMetricsCalculator()
    
    return calculator.compute_state_change(
        current_state.get('Q'), current_state.get('K'), current_state.get('V'),
        previous_state.get('Q'), previous_state.get('K'), previous_state.get('V'),
        norm_type
    )


def detect_phase_transition(
    current_state: Dict[str, torch.Tensor],
    previous_state: Dict[str, torch.Tensor],
    threshold: float = 1.0
) -> Tuple[float, bool, Dict[str, Any]]:
    """
    便捷检测相变
    Conveniently detect phase transition
    
    参数 | Parameters:
    ------------
    current_state : Dict[str, torch.Tensor]
        当前状态字典 | Current state dict
    previous_state : Dict[str, torch.Tensor]
        上一状态字典 | Previous state dict
    threshold : float, optional
        相变阈值 | Phase transition threshold
    
    返回 | Returns:
    ------------
    Tuple[float, bool, Dict[str, Any]]
        (变化量, 是否相变, 详细信息) | (Change amount, whether transition, details)
    """
    calculator = CoreMetricsCalculator(phase_threshold=threshold)
    
    delta, is_transition = calculator.detect_phase_transition(
        current_state.get('Q'), current_state.get('K'), current_state.get('V'),
        previous_state.get('Q'), previous_state.get('K'), previous_state.get('V')
    )
    
    return delta, is_transition, {
        "threshold": threshold,
        "delta": delta,
        "is_transition": is_transition
    }


def stack_three_networks(
    Q1: torch.Tensor,
    Q2: torch.Tensor,
    Q3: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    便捷堆叠三个网络
    Conveniently stack three networks
    
    参数 | Parameters:
    ------------
    Q1, Q2, Q3 : torch.Tensor
        三个Q矩阵 | Three Q matrices
    weights : Optional[torch.Tensor], optional
        权重张量 | Weight tensor
    
    返回 | Returns:
    ------------
    torch.Tensor
        堆叠后的Q矩阵 | Stacked Q matrix
    """
    calculator = CoreMetricsCalculator()
    return calculator.stack_three_networks([Q1, Q2, Q3], weights)


def apply_one_way_valve(
    tensor: torch.Tensor,
    mode: str = 'detach',
    gate_value: Optional[int] = None
) -> torch.Tensor:
    """
    便捷应用单向阀
    Conveniently apply one-way valve
    
    参数 | Parameters:
    ------------
    tensor : torch.Tensor
        输入张量 | Input tensor
    mode : str, optional
        操作模式 ('detach' 或 'gate') | Operation mode ('detach' or 'gate')
    gate_value : Optional[int], optional
        门控值 (0 或 1) | Gate value (0 or 1)
    
    返回 | Returns:
    ------------
    torch.Tensor
        处理后的张量 | Processed tensor
    """
    calculator = CoreMetricsCalculator()
    return calculator.apply_one_way_valve(tensor, mode, gate_value)


# ==================== RGA引擎 | RGA Engine ====================
class RGAEngine:
    """
    RGA统一引擎 - 提供完整的RGA功能
    RGA Unified Engine - Provides complete RGA functionality
    ============================================================
    
    功能: 集成配置管理、状态监控、相变检测和信息控制
    Purpose: Integrate configuration management, state monitoring, phase detection, and information control
    
    设计原则 | Design Principles:
    - 一站式解决RGA架构的所有需求 | One-stop solution for all RGA architecture needs
    - 配置驱动，灵活可扩展 | Configuration-driven, flexible and extensible
    - 实时监控，可视化状态 | Real-time monitoring, visualized state
    - 自动优化，智能调整 | Automatic optimization, intelligent adjustment
    
    使用示例 | Usage Example:
        # 创建引擎 | Create engine
        engine = RGAEngine()
        
        # 处理状态 | Process state
        engine.process_state(Q, K, V)
        
        # 获取分析报告 | Get analysis report
        report = engine.get_analysis_report()
        
        # 保存/加载状态 | Save/load state
        engine.save_state("state.pth")
        engine.load_state("state.pth")
    """
    
    def __init__(self, config: Optional[Union[RGAConfig, Dict]] = None):
        """
        初始化RGA引擎
        Initialize RGA Engine
        
        参数 | Parameters:
        ------------
        config : Optional[Union[RGAConfig, Dict]], optional
            配置实例或字典，None则使用默认配置 | Configuration instance or dict, None uses default config
        """
        # 初始化配置 | Initialize configuration
        if config is None:
            self.config = RGAConfig()
        elif isinstance(config, dict):
            self.config = RGAConfig.from_dict(config)
        else:
            self.config = config
        
        # 初始化计算器 | Initialize calculator
        self.calculator = CoreMetricsCalculator(
            phase_threshold=self.config.phase_threshold
        )
        
        # 引擎状态 | Engine state
        self.engine_state = {
            "total_states_processed": 0,
            "total_transitions": 0,
            "last_transition_step": -1,
            "learning_phase_history": [],
            "performance_metrics": {}
        }
        
        # 回调函数注册 | Callback function registration
        self.callbacks = {
            "on_state_record": [],
            "on_phase_transition": [],
            "on_learning_phase_change": [],
            "on_error": []
        }
        
        print(f"✅ RGA引擎初始化成功 | RGA Engine initialized successfully")
        print(f"   版本: {__version__} | Version: {__version__}")
        print(f"   配置: {self.config}")
    
    def register_callback(self, event: str, callback: Callable):
        """
        注册事件回调函数
        Register event callback function
        
        参数 | Parameters:
        ------------
        event : str
            事件名称 ('on_state_record', 'on_phase_transition', 'on_learning_phase_change', 'on_error')
        callback : Callable
            回调函数 | Callback function
        
        异常 | Raises:
        ------------
        ValueError
            当事件名称无效时抛出 | Raised when event name is invalid
        """
        if event not in self.callbacks:
            raise ValueError(f"无效的事件: {event} | Invalid event: {event}")
        
        self.callbacks[event].append(callback)
    
    def process_state(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Dict[str, Any]:
        """
        处理状态并记录
        Process state and record
        
        参数 | Parameters:
        ------------
        Q, K, V : torch.Tensor
            Q、K、V矩阵 | Q, K, V matrices
        
        返回 | Returns:
        ------------
        Dict[str, Any]
            处理结果 | Processing results
        """
        try:
            # 记录当前状态 | Record current state
            self.calculator.record_state(Q, K, V)
            self.engine_state["total_states_processed"] += 1
            
            # 触发状态记录回调 | Trigger state record callback
            for callback in self.callbacks["on_state_record"]:
                callback(self.engine_state["total_states_processed"], Q.shape)
            
            # 如果历史状态足够，计算变化并检测相变
            # If enough historical states, calculate change and detect phase transition
            results = {"state_recorded": True, "step": self.engine_state["total_states_processed"]}
            
            if len(self.calculator.state_history) >= 2:
                # 获取最近两个状态 | Get the last two states
                curr_state = self.calculator.state_history[-1]
                prev_state = self.calculator.state_history[-2]
                
                # 计算状态变化 | Calculate state change
                state_change = self.calculator.compute_state_change(
                    curr_state['Q'], curr_state['K'], curr_state['V'],
                    prev_state['Q'], prev_state['K'], prev_state['V']
                )
                
                # 检测相变 | Detect phase transition
                delta_total, is_transition = self.calculator.detect_phase_transition(
                    curr_state['Q'], curr_state['K'], curr_state['V'],
                    prev_state['Q'], prev_state['K'], prev_state['V']
                )
                
                results.update({
                    "state_change": state_change,
                    "delta_total": delta_total,
                    "is_phase_transition": is_transition
                })
                
                if is_transition:
                    self.engine_state["total_transitions"] += 1
                    self.engine_state["last_transition_step"] = self.engine_state["total_states_processed"]
                    
                    # 触发相变回调 | Trigger phase transition callback
                    for callback in self.callbacks["on_phase_transition"]:
                        callback(self.engine_state["total_states_processed"], delta_total)
            
            # 分析学习阶段 | Analyze learning phase
            phase_info = self.calculator.analyze_learning_phases()
            results["learning_phase"] = phase_info.get("phase", "未知 | Unknown")
            
            # 记录学习阶段历史 | Record learning phase history
            if "learning_phase_history" in self.engine_state:
                self.engine_state["learning_phase_history"].append({
                    "step": self.engine_state["total_states_processed"],
                    "phase": results["learning_phase"]
                })
            
            # 触发学习阶段变化回调 | Trigger learning phase change callback
            for callback in self.callbacks["on_learning_phase_change"]:
                callback(results["learning_phase"])
            
            return results
            
        except Exception as e:
            # 触发错误回调 | Trigger error callback
            for callback in self.callbacks["on_error"]:
                callback(e)
            
            # 重新抛出异常 | Re-raise exception
            raise
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """
        获取完整分析报告
        Get complete analysis report
        
        返回 | Returns:
        ------------
        Dict[str, Any]
            分析报告 | Analysis report
        """
        # 获取计算器统计信息 | Get calculator statistics
        stats = self.calculator.get_statistics()
        
        # 获取学习阶段分析 | Get learning phase analysis
        phase_info = self.calculator.analyze_learning_phases()
        
        # 获取相变摘要 | Get transition summary
        transition_summary = self.calculator.get_transition_summary()
        
        # 计算性能指标 | Calculate performance metrics
        state_changes = self.calculator.get_state_change_series()
        avg_change = sum(state_changes) / len(state_changes) if state_changes else 0.0
        
        # 构建报告 | Build report
        report = {
            "engine_info": {
                "version": __version__,
                "total_states_processed": self.engine_state["total_states_processed"],
                "total_processing_time": self.engine_state.get("total_processing_time", 0.0),
                "average_processing_time": self.engine_state.get("average_processing_time", 0.0)
            },
            "config_summary": self.config.to_dict(),
            "performance_metrics": {
                "average_state_change": avg_change,
                "state_change_std": np.std(state_changes) if state_changes else 0.0,
                "state_change_trend": "上升 | Increasing" if len(state_changes) > 1 and state_changes[-1] > state_changes[0] else "下降 | Decreasing",
                "transition_frequency": self.engine_state["total_transitions"] / max(self.engine_state["total_states_processed"], 1)
            },
            "learning_analysis": phase_info,
            "transition_analysis": transition_summary,
            "state_history_summary": {
                "total_states": stats["state_history_length"],
                "current_step": stats["current_step"],
                "transition_points": stats["transition_points"]
            },
            "recommendations": self._generate_recommendations(phase_info, transition_summary)
        }
        
        return report
    
    def _generate_recommendations(self, phase_info: Dict, transition_summary: Dict) -> List[str]:
        """
        生成优化建议
        Generate optimization recommendations
        
        参数 | Parameters:
        ------------
        phase_info : Dict
            学习阶段信息 | Learning phase information
        transition_summary : Dict
            相变摘要信息 | Transition summary information
        
        返回 | Returns:
        ------------
        List[str]
            建议列表 | Recommendation list
        """
        recommendations = []
        
        # 基于学习阶段的建议 | Recommendations based on learning phase
        phase = phase_info.get("phase", "")
        
        if "探索期" in phase or "Exploration" in phase:
            recommendations.append("当前处于探索期，建议增加探索率 | Currently in exploration phase, recommend increasing exploration rate")
            recommendations.append("可以尝试增加学习率以加速探索 | Try increasing learning rate to accelerate exploration")
        
        elif "学习期" in phase or "Learning" in phase:
            recommendations.append("当前处于高效学习期，保持当前配置 | Currently in efficient learning phase, maintain current configuration")
            recommendations.append("监控状态变化，准备调整学习策略 | Monitor state changes, prepare to adjust learning strategy")
        
        elif "稳定期" in phase or "Stable" in phase:
            recommendations.append("当前处于稳定期，考虑减少学习率 | Currently in stable phase, consider reducing learning rate")
            recommendations.append("可以增加正则化强度防止过拟合 | Increase regularization strength to prevent overfitting")
        
        elif "收敛期" in phase or "Convergence" in phase:
            recommendations.append("当前处于收敛期，学习即将完成 | Currently in convergence phase, learning nearly complete")
            recommendations.append("考虑提前停止训练或降低学习率 | Consider early stopping or reducing learning rate")
        
        # 基于相变的建议 | Recommendations based on phase transitions
        transitions = transition_summary.get("total_transitions", 0)
        
        if transitions > 5:
            recommendations.append(f"检测到多次相变({transitions}次)，学习过程不稳定 | Multiple phase transitions detected ({transitions}), learning process unstable")
            recommendations.append("建议降低学习率或增加批量大小 | Recommend reducing learning rate or increasing batch size")
        
        elif transitions == 0 and self.engine_state["total_states_processed"] > 10:
            recommendations.append("未检测到相变，学习过程可能过于平稳 | No phase transitions detected, learning process may be too smooth")
            recommendations.append("建议增加模型复杂度或调整学习策略 | Recommend increasing model complexity or adjusting learning strategy")
        
        return recommendations
    
    def save_state(self, filepath: str):
        """
        保存引擎状态
        Save engine state
        
        参数 | Parameters:
        ------------
        filepath : str
            保存路径 | Save path
        """
        import pickle
        
        state = {
            "calculator_state": {
                "state_history": self.calculator.state_history,
                "transition_points": self.calculator.transition_points,
                "phase_threshold": self.calculator.phase_threshold
            },
            "engine_state": self.engine_state,
            "config": self.config.to_dict(),
            "version": __version__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"✅ 引擎状态已保存到: {filepath} | Engine state saved to: {filepath}")
    
    def load_state(self, filepath: str):
        """
        加载引擎状态
        Load engine state
        
        参数 | Parameters:
        ------------
        filepath : str
            加载路径 | Load path
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # 验证版本 | Verify version
        if state.get("version") != __version__:
            warnings.warn(f"版本不匹配: 保存的版本={state.get('version')}, 当前版本={__version__} | Version mismatch: saved={state.get('version')}, current={__version__}")
        
        # 恢复计算器状态 | Restore calculator state
        calc_state = state["calculator_state"]
        self.calculator.state_history = calc_state["state_history"]
        self.calculator.transition_points = calc_state["transition_points"]
        self.calculator.phase_threshold = calc_state["phase_threshold"]
        
        # 恢复引擎状态 | Restore engine state
        self.engine_state = state["engine_state"]
        
        # 恢复配置 | Restore configuration
        self.config = RGAConfig.from_dict(state["config"])
        
        print(f"✅ 引擎状态已从 {filepath} 加载 | Engine state loaded from {filepath}")
    
    def reset(self):
        """
        重置引擎状态
        Reset engine state
        """
        self.calculator.reset()
        self.engine_state = {
            "total_states_processed": 0,
            "total_transitions": 0,
            "last_transition_step": -1,
            "learning_phase_history": [],
            "performance_metrics": {}
        }
        print("✅ 引擎状态已重置 | Engine state reset")
    
    def visualize_state_changes(self, save_path: Optional[str] = None):
        """
        可视化状态变化（需要matplotlib）
        Visualize state changes (requires matplotlib)
        
        参数 | Parameters:
        ------------
        save_path : Optional[str], optional
            保存路径，None则不保存 | Save path, None means don't save
        """
        try:
            import matplotlib.pyplot as plt
            
            changes = self.calculator.get_state_change_series()
            
            if not changes:
                print("⚠️ 无状态变化数据可可视化 | No state change data to visualize")
                return
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(changes) + 1), changes, marker='o', linestyle='-', color='b')
            plt.axhline(y=self.calculator.phase_threshold, color='r', linestyle='--', label=f'相变阈值={self.calculator.phase_threshold}')
            
            # 标记相变点 | Mark transition points
            for point in self.calculator.transition_points:
                timestamp = point.get('timestamp', 0)
                if 0 <= timestamp - 1 < len(changes):
                    plt.scatter(timestamp, changes[timestamp-1], color='r', s=100, zorder=5)
            
            plt.xlabel('步骤 | Step')
            plt.ylabel('状态变化量 | State Change')
            plt.title('RGA状态变化趋势 | RGA State Change Trend')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ 可视化图表已保存到: {save_path} | Visualization chart saved to: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("❌ 需要matplotlib进行可视化 | matplotlib required for visualization")
            print("   请安装: pip install matplotlib | Please install: pip install matplotlib")
    
    def __str__(self) -> str:
        """
        返回引擎的字符串表示
        Return string representation of engine
        
        返回 | Returns:
        ------------
        str
            引擎信息 | Engine information
        """
        stats = self.engine_state
        phase_info = self.calculator.analyze_learning_phases()
        
        lines = [
            "RGA Engine | RGA 引擎",
            "=" * 50,
            f"版本: {__version__} | Version: {__version__}",
            f"处理状态数: {stats['total_states_processed']} | States processed: {stats['total_states_processed']}",
            f"相变次数: {stats['total_transitions']} | Phase transitions: {stats['total_transitions']}",
            f"当前学习阶段: {phase_info.get('phase', '未知')} | Current learning phase: {phase_info.get('phase', 'Unknown')}",
            f"配置摘要:",
            f"  - 维度: {self.config.dim} | Dimension: {self.config.dim}",
            f"  - 词汇表大小: {self.config.vocab_size} | Vocabulary size: {self.config.vocab_size}",
            f"  - 相变阈值: {self.config.phase_threshold} | Phase threshold: {self.config.phase_threshold}"
        ]
        
        return "\n".join(lines)


# ==================== 工厂函数 | Factory Functions ====================
def create_rga_engine(
    config: Optional[Union[RGAConfig, Dict]] = None,
    enable_logging: bool = True,
    enable_monitoring: bool = True
) -> RGAEngine:
    """
    创建RGA引擎的工厂函数
    Factory function to create RGA engine
    
    参数 | Parameters:
    ------------
    config : Optional[Union[RGAConfig, Dict]], optional
        配置实例或字典 | Configuration instance or dictionary
    enable_logging : bool, optional
        是否启用日志记录 | Whether to enable logging
    enable_monitoring : bool, optional
        是否启用监控 | Whether to enable monitoring
    
    返回 | Returns:
    ------------
    RGAEngine
        RGA引擎实例 | RGA engine instance
    
    示例 | Example:
    ------------
        # 创建默认引擎 | Create default engine
        engine = create_rga_engine()
        
        # 创建自定义配置引擎 | Create custom configured engine
        engine = create_rga_engine({"vocab_size": 20000, "dim": 512})
    """
    engine = RGAEngine(config)
    
    # 添加默认回调函数 | Add default callback functions
    if enable_logging:
        def log_state_record(step, shape):
            if step % 10 == 0:
                print(f"📊 已记录 {step} 个状态，形状: {shape} | Recorded {step} states, shape: {shape}")
        
        engine.register_callback("on_state_record", log_state_record)
    
    if enable_monitoring:
        def log_phase_transition(step, delta):
            print(f"⚠️  检测到相变! 步骤: {step}, 变化量: {delta:.4f} | Phase transition detected! Step: {step}, Delta: {delta:.4f}")
        
        engine.register_callback("on_phase_transition", log_phase_transition)
    
    return engine


# ==================== 模块初始化 | Module Initialization ====================
def _initialize_module():
    """
    模块初始化函数
    Module initialization function
    """
    print(f"RGA Core Module v{__version__} 初始化成功 | Initialized successfully")
    print(f"可用组件: {', '.join(__all__)} | Available components: {', '.join(__all__)}")


# 自动初始化模块 | Automatically initialize module
_initialize_module()