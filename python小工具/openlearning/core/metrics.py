"""
核心度量计算模块 | Core Metrics Calculation Module
================================================

实现RGA架构的四个核心公式和状态监控系统。
Implements the four core formulas and state monitoring system of RGA architecture.

四个核心公式:
Four Core Formulas:
1. 单网络变化量公式（状态监控）| Single network change formula (state monitoring)
2. 相变检测公式（质变识别）| Phase transition detection formula (qualitative change recognition)
3. 三网络堆叠公式（多视角融合）| Three-network stacking formula (multi-perspective fusion)
4. 单向阀公式（信息控制）| One-way valve formula (information control)

设计原则:
Design Principles:
• 公式驱动，数学精确 | Formula-driven, mathematically precise
• 状态历史跟踪，支持回溯分析 | State history tracking, supporting retrospective analysis
• 实时相变检测，动态调整策略 | Real-time phase transition detection, dynamic strategy adjustment
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class CoreMetricsCalculator:
    """
    核心度量计算器 - 实现文中四个核心公式
    Core Metrics Calculator - Implements the Four Core Formulas
    ===========================================================
    
    功能: 监控网络状态变化，检测学习相变，管理信息流动。
    Purpose: Monitor network state changes, detect learning phase transitions, manage information flow.
    
    核心特性:
    Core Features:
    - 实时状态变化监控 | Real-time state change monitoring
    - 动态相变检测 | Dynamic phase transition detection
    - 多视角信息融合 | Multi-perspective information fusion
    - 智能信息流控制 | Intelligent information flow control
    
    物理意义:
    Physical Significance:
    - 状态变化量: 量化学习过程中的"能量流动" | State change: Quantifies "energy flow" in learning
    - 相变检测: 识别认知状态的"跃迁时刻" | Phase transition: Identifies "transition moments" of cognitive state
    - 网络堆叠: 实现"多视角平等融合" | Network stacking: Achieves "multi-perspective equal fusion"
    - 单向阀: 保护核心记忆，控制信息流动方向 | One-way valve: Protects core memory, controls information flow direction
    
    使用示例:
    Usage Example:
        # 创建计算器 | Create calculator
        calculator = CoreMetricsCalculator(phase_threshold=0.5)
        
        # 记录状态 | Record state
        calculator.record_state(Q, K, V)
        
        # 计算状态变化 | Calculate state change
        delta = calculator.compute_state_change(Q_t, K_t, V_t, Q_t_1, K_t_1, V_t_1)
        
        # 检测相变 | Detect phase transition
        delta_total, is_transition = calculator.detect_phase_transition(Q_t, K_t, V_t, Q_t_1, K_t_1, V_t_1)
    """
    
    def __init__(self, phase_threshold: float = 1.0):
        """
        初始化核心度量计算器
        Initialize Core Metrics Calculator
        
        参数 | Parameters:
        ------------
        phase_threshold : float, optional
            相变检测阈值，默认1.0 | Phase transition detection threshold, default 1.0
        
        属性 | Attributes:
        ------------
        state_history : List[Dict]
            状态历史记录 | State history records
        transition_points : List[Dict]
            相变点记录 | Phase transition point records
        phase_threshold : float
            相变检测阈值 | Phase transition detection threshold
        """
        self.state_history: List[Dict[str, Any]] = []  # 记录状态历史 | Record state history
        self.transition_points: List[Dict[str, Any]] = []  # 记录相变点 | Record phase transition points
        self.phase_threshold = phase_threshold  # 相变阈值 | Phase transition threshold
    
    # ==================== 公式1: 单网络变化量 ====================
    # ==================== Formula 1: Single Network Change ====================
    def compute_state_change(self, 
                           Q_t: torch.Tensor, 
                           K_t: torch.Tensor, 
                           V_t: torch.Tensor,
                           Q_t_1: torch.Tensor, 
                           K_t_1: torch.Tensor, 
                           V_t_1: torch.Tensor,
                           norm_type: str = 'l2') -> float:
        """
        单网络变化量公式: Δ = ||Q_t - Q_{t-1}|| + ||K_t - K_{t-1}|| + ||V_t - V_{t-1}||
        Single Network Change Formula: Δ = ||Q_t - Q_{t-1}|| + ||K_t - K_{t-1}|| + ||V_t - V_{t-1}||
        
        作用: 监控网络状态的连续变化量
        Purpose: Monitor continuous change in network state
        
        物理意义: 量化学习过程中的"能量流动"
        Physical Significance: Quantifies "energy flow" in the learning process
        
        参数 | Parameters:
        ------------
        Q_t, K_t, V_t : torch.Tensor
            当前时刻的Q、K、V矩阵 | Q, K, V matrices at current time
        Q_t_1, K_t_1, V_t_1 : torch.Tensor
            上一时刻的Q、K、V矩阵 | Q, K, V matrices at previous time
        norm_type : str, optional
            范数类型 ('l1' 或 'l2')，默认'l2' | Norm type ('l1' or 'l2'), default 'l2'
        
        返回 | Returns:
        ------------
        float
            总变化量 | Total change amount
        
        异常 | Raises:
        ------------
        ValueError
            当范数类型不支持时抛出 | Raised when norm type is not supported
        
        示例 | Example:
        ------------
            delta = calculator.compute_state_change(Q_t, K_t, V_t, Q_t_1, K_t_1, V_t_1, norm_type='l2')
            print(f"状态变化量: {delta:.4f}")
        """
        if norm_type == 'l2':
            delta_Q = torch.norm(Q_t - Q_t_1, p=2)
            delta_K = torch.norm(K_t - K_t_1, p=2)
            delta_V = torch.norm(V_t - V_t_1, p=2)
        elif norm_type == 'l1':
            delta_Q = torch.norm(Q_t - Q_t_1, p=1)
            delta_K = torch.norm(K_t - K_t_1, p=1)
            delta_V = torch.norm(V_t - V_t_1, p=1)
        else:
            raise ValueError(f"不支持的范数类型: {norm_type} | Unsupported norm type: {norm_type}")
        
        return (delta_Q + delta_K + delta_V).item()
    
    # ==================== 公式2: 相变检测 ====================
    # ==================== Formula 2: Phase Transition Detection ====================
    def detect_phase_transition(self,
                              Q_t: torch.Tensor, 
                              K_t: torch.Tensor, 
                              V_t: torch.Tensor,
                              Q_t_1: torch.Tensor, 
                              K_t_1: torch.Tensor, 
                              V_t_1: torch.Tensor) -> Tuple[float, bool]:
        """
        相变检测公式: Δ = ‖Q^(t) - Q^(t-1)‖_F + ‖K^(t) - K^(t-1)‖_F + ‖V^(t) - V^(t-1)‖_F
        Phase Transition Detection Formula: Δ = ‖Q^(t) - Q^(t-1)‖_F + ‖K^(t) - K^(t-1)‖_F + ‖V^(t) - V^(t-1)‖_F
        
        关键: 使用Frobenius范数（矩阵整体结构变化）
        Key: Use Frobenius norm (overall matrix structure change)
        
        作用: 检测质的变化，而不仅仅是量的变化
        Purpose: Detect qualitative changes, not just quantitative changes
        
        物理意义: 识别认知状态的"跃迁时刻"
        Physical Significance: Identify "transition moments" of cognitive state
        
        参数 | Parameters:
        ------------
        Q_t, K_t, V_t : torch.Tensor
            当前时刻的Q、K、V矩阵 | Q, K, V matrices at current time
        Q_t_1, K_t_1, V_t_1 : torch.Tensor
            上一时刻的Q、K、V矩阵 | Q, K, V matrices at previous time
        
        返回 | Returns:
        ------------
        Tuple[float, bool]
            delta_total: 总变化量 | Total change amount
            is_transition: 是否发生相变 | Whether phase transition occurred
        
        注意 | Note:
        ------------
        Frobenius范数捕捉矩阵的整体结构变化，适合检测质变。
        Frobenius norm captures the overall structure change of matrices, suitable for detecting qualitative changes.
        """
        # Frobenius范数：捕捉矩阵结构变化
        # Frobenius norm: Capture matrix structure change
        delta_Q = torch.norm(Q_t - Q_t_1, p='fro')
        delta_K = torch.norm(K_t - K_t_1, p='fro')
        delta_V = torch.norm(V_t - V_t_1, p='fro')
        
        delta_total = (delta_Q + delta_K + delta_V).item()
        
        # 检测是否为相变 | Detect if it's a phase transition
        is_transition = delta_total > self.phase_threshold
        
        if is_transition:
            self.transition_points.append({
                'delta': delta_total,
                'Q_shape': Q_t.shape,
                'timestamp': len(self.state_history)
            })
        
        return delta_total, is_transition
    
    # ==================== 公式3: 三网络堆叠 ====================
    # ==================== Formula 3: Three-Network Stacking ====================
    def stack_three_networks(self,
                           Q_list: List[torch.Tensor],
                           weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        三网络堆叠公式: Q_stack = ∑_{i=1}^3 α_i Q_i, where α = Softmax(w)
        Three-Network Stacking Formula: Q_stack = ∑_{i=1}^3 α_i Q_i, where α = Softmax(w)
        
        作用: 融合三个子网络的不同处理路径
        Purpose: Fuse different processing paths of three subnetworks
        
        物理意义: 实现"多视角平等融合"，避免层级偏见
        Physical Significance: Achieve "multi-perspective equal fusion", avoid hierarchical bias
        
        参数 | Parameters:
        ------------
        Q_list : List[torch.Tensor]
            三个Q矩阵的列表 [Q1, Q2, Q3] | List of three Q matrices [Q1, Q2, Q3]
        weights : Optional[torch.Tensor], optional
            权重张量，如果为None则使用均匀权重 | Weight tensor, if None use uniform weights
        
        返回 | Returns:
        ------------
        torch.Tensor
            堆叠后的Q矩阵 | Stacked Q matrix
        
        异常 | Raises:
        ------------
        ValueError
            当输入不是3个矩阵或形状不一致时抛出 | Raised when input is not 3 matrices or shapes are inconsistent
        
        设计原理:
        Design Principle:
        - 使用Softmax确保权重和为1 | Use Softmax to ensure weights sum to 1
        - 支持自定义权重或使用均匀权重 | Support custom weights or use uniform weights
        - 要求所有输入矩阵形状一致 | Require all input matrices to have consistent shape
        """
        if len(Q_list) != 3:
            raise ValueError(f"需要3个Q矩阵进行堆叠，得到{len(Q_list)}个 | Need 3 Q matrices for stacking, got {len(Q_list)}")
        
        # 验证形状一致性 | Verify shape consistency
        shapes = [Q.shape for Q in Q_list]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError("所有Q矩阵必须形状一致 | All Q matrices must have consistent shape")
        
        # 默认权重：均匀分布 | Default weights: Uniform distribution
        if weights is None:
            weights = torch.ones(3)
        
        # Softmax归一化权重（确保总和为1）
        # Softmax normalize weights (ensure sum to 1)
        alpha = F.softmax(weights, dim=0)
        
        # 加权堆叠 | Weighted stacking
        Q_stack = torch.zeros_like(Q_list[0])
        for i, Q in enumerate(Q_list):
            Q_stack += alpha[i] * Q
        
        return Q_stack
    
    # ==================== 公式4: 单向阀 ====================
    # ==================== Formula 4: One-Way Valve ====================
    def apply_one_way_valve(self,
                          h_in: torch.Tensor,
                          mode: str = 'detach',
                          gate_value: Optional[int] = None) -> torch.Tensor:
        """
        单向阀公式: h_out = detach(h_in) 或 h_out = g·h_in, g∈{0,1}
        One-Way Valve Formula: h_out = detach(h_in) or h_out = g·h_in, g∈{0,1}
        
        两种模式:
        Two Modes:
        1. detach模式: 切断梯度，创建不可逆记忆 | detach mode: Cut off gradient, create irreversible memory
        2. gate模式: 二进制门控，实现"全或无"信息流 | gate mode: Binary gating, achieve "all-or-nothing" information flow
        
        物理意义: 保护核心记忆，控制信息流动方向
        Physical Significance: Protect core memory, control information flow direction
        
        参数 | Parameters:
        ------------
        h_in : torch.Tensor
            输入张量 | Input tensor
        mode : str, optional
            操作模式 ('detach' 或 'gate')，默认'detach' | Operation mode ('detach' or 'gate'), default 'detach'
        gate_value : Optional[int], optional
            门控值 (0 或 1)，仅在gate模式下使用 | Gate value (0 or 1), only used in gate mode
        
        返回 | Returns:
        ------------
        torch.Tensor
            处理后的张量 | Processed tensor
        
        异常 | Raises:
        ------------
        ValueError
            当模式不支持或门控值无效时抛出 | Raised when mode is unsupported or gate value is invalid
        
        应用场景:
        Application Scenarios:
        - detach模式: 保存历史状态，防止梯度回传 | detach mode: Save historical state, prevent gradient backpropagation
        - gate模式: 选择性信息传递，实现注意力机制 | gate mode: Selective information transmission, implement attention mechanism
        """
        if mode == 'detach':
            # 模式1: 梯度阻断（不可逆记忆）
            # Mode 1: Gradient cutoff (irreversible memory)
            return h_in.detach()
        
        elif mode == 'gate':
            # 模式2: 二进制门控
            # Mode 2: Binary gating
            if gate_value not in [0, 1]:
                raise ValueError(f"门控值必须是0或1，得到{gate_value} | Gate value must be 0 or 1, got {gate_value}")
            
            if gate_value == 0:
                return torch.zeros_like(h_in)
            else:
                return h_in.clone()
        
        else:
            raise ValueError(f"不支持的模式: {mode} | Unsupported mode: {mode}")
    
    # ==================== 状态管理方法 ====================
    # ==================== State Management Methods ====================
    def record_state(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """
        记录当前状态到历史
        Record current state to history
        
        参数 | Parameters:
        ------------
        Q, K, V : torch.Tensor
            当前时刻的Q、K、V矩阵 | Q, K, V matrices at current time
        
        注意 | Note:
        ------------
        状态记录用于后续的状态变化分析和相变检测。
        State records are used for subsequent state change analysis and phase transition detection.
        """
        self.state_history.append({
            'Q': Q.detach().clone(),
            'K': K.detach().clone(),
            'V': V.detach().clone(),
            'step': len(self.state_history)
        })
    
    def get_state_change_series(self, norm_type: str = 'l2') -> List[float]:
        """
        获取状态变化序列
        Get state change series
        
        参数 | Parameters:
        ------------
        norm_type : str, optional
            范数类型 ('l1' 或 'l2')，默认'l2' | Norm type ('l1' or 'l2'), default 'l2'
        
        返回 | Returns:
        ------------
        List[float]
            状态变化量列表 | List of state change amounts
        
        注意 | Note:
        ------------
        返回从第二步开始的状态变化序列，用于趋势分析。
        Returns state change series starting from step 2, used for trend analysis.
        """
        if len(self.state_history) < 2:
            return []
        
        changes = []
        for i in range(1, len(self.state_history)):
            curr = self.state_history[i]
            prev = self.state_history[i-1]
            
            delta = self.compute_state_change(
                curr['Q'], curr['K'], curr['V'],
                prev['Q'], prev['K'], prev['V'],
                norm_type
            )
            changes.append(delta)
        
        return changes
    
    def analyze_learning_phases(self) -> Dict[str, Any]:
        """
        分析学习阶段
        Analyze learning phases
        
        返回 | Returns:
        ------------
        Dict[str, Any]
            学习阶段分析结果 | Learning phase analysis results
        
        分析逻辑:
        Analysis Logic:
        - 探索期: 平均变化量 > 25.0 | Exploration phase: Average change > 25.0
        - 学习期: 15.0 < 平均变化量 ≤ 25.0 | Learning phase: 15.0 < Average change ≤ 25.0
        - 稳定期: 5.0 < 平均变化量 ≤ 15.0 | Stable phase: 5.0 < Average change ≤ 15.0
        - 收敛期: 平均变化量 ≤ 5.0 | Convergence phase: Average change ≤ 5.0
        
        注意 | Note:
        ------------
        基于最近5步的平均变化量进行分析，如果数据不足则返回初始阶段。
        Analysis based on average change of last 5 steps, returns initial phase if insufficient data.
        """
        if len(self.state_history) < 2:
            return {'message': '数据不足 | Insufficient data', 'phase': '初始阶段 | Initial phase'}
        
        changes = self.get_state_change_series('l2')
        
        if not changes:
            return {'phase': '初始阶段 | Initial phase', 'stability': 0.0}
        
        # 分析最近5步的变化 | Analyze changes in last 5 steps
        recent_changes = changes[-min(5, len(changes)):]
        avg_change = sum(recent_changes) / len(recent_changes)
        
        # 基于平均变化量判断学习阶段
        # Determine learning phase based on average change
        if avg_change > 25.0:
            phase = "探索期 | Exploration phase"
        elif avg_change > 15.0:
            phase = "学习期 | Learning phase"
        elif avg_change > 5.0:
            phase = "稳定期 | Stable phase"
        else:
            phase = "收敛期 | Convergence phase"
        
        return {
            'phase': phase,
            'avg_change': avg_change,
            'total_steps': len(self.state_history),
            'transition_points': len(self.transition_points),
            'current_stability': 1.0 / (avg_change + 1e-6)  # 稳定性指标 | Stability metric
        }
    
    def reset(self):
        """
        重置计算器状态
        Reset calculator state
        
        注意 | Note:
        ------------
        清除所有历史记录和相变点，恢复初始状态。
        Clear all history records and transition points, restore to initial state.
        """
        self.state_history = []
        self.transition_points = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        Get statistics
        
        返回 | Returns:
        ------------
        Dict[str, Any]
            计算器统计信息 | Calculator statistics
        """
        return {
            'state_history_length': len(self.state_history),
            'transition_points': len(self.transition_points),
            'phase_threshold': self.phase_threshold,
            'current_step': len(self.state_history) if self.state_history else 0
        }
    
    def get_transition_summary(self) -> Dict[str, Any]:
        """
        获取相变摘要
        Get transition summary
        
        返回 | Returns:
        ------------
        Dict[str, Any]
            相变摘要信息 | Transition summary information
        """
        if not self.transition_points:
            return {'message': '无相变记录 | No transition records'}
        
        # 提取所有相变的变化量 | Extract all transition deltas
        deltas = [point['delta'] for point in self.transition_points]
        
        return {
            'total_transitions': len(self.transition_points),
            'max_delta': max(deltas) if deltas else 0.0,
            'min_delta': min(deltas) if deltas else 0.0,
            'avg_delta': sum(deltas) / len(deltas) if deltas else 0.0,
            'timestamps': [point['timestamp'] for point in self.transition_points]
        }
    
    def __str__(self) -> str:
        """
        返回计算器的字符串表示
        Return string representation of calculator
        
        返回 | Returns:
        ------------
        str
            格式化的计算器信息 | Formatted calculator information
        """
        stats = self.get_statistics()
        phase_info = self.analyze_learning_phases()
        
        lines = [
            "Core Metrics Calculator | 核心度量计算器",
            "=" * 50,
            f"状态历史长度: {stats['state_history_length']} | State history length: {stats['state_history_length']}",
            f"相变点数量: {stats['transition_points']} | Transition points: {stats['transition_points']}",
            f"相变阈值: {stats['phase_threshold']} | Phase threshold: {stats['phase_threshold']}",
            f"当前学习阶段: {phase_info.get('phase', '未知')} | Current learning phase: {phase_info.get('phase', 'Unknown')}",
        ]
        
        if 'avg_change' in phase_info:
            lines.append(f"平均变化量: {phase_info['avg_change']:.4f} | Average change: {phase_info['avg_change']:.4f}")
        
        return "\n".join(lines)
