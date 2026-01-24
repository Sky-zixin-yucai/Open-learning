"""
单向阀模块 | One-Way Valve Module
================================

实现信息流的单向控制，保护核心记忆，控制信息流动方向。
Implements one-way control of information flow, protecting core memory and controlling information flow direction.

设计原理:
Design Principles:
• detach模式：切断梯度，创建不可逆记忆 | Detach mode: cuts off gradients, creates irreversible memory
• gate模式：二进制门控，实现"全或无"信息流 | Gate mode: binary gating, implements "all-or-nothing" information flow

物理意义:
Physical Meaning:
• 保护核心记忆，防止梯度回传干扰 | Protects core memory from gradient backpropagation interference
• 控制信息流动方向，实现记忆层级的固化 | Controls information flow direction, solidifies memory hierarchy

数学公式:
Mathematical Formulas:
1. detach模式: h_out = detach(h_in)
2. gate模式: h_out = g·h_in, g∈{0,1}

应用场景:
Application Scenarios:
• 地质记忆检索时的梯度保护 | Gradient protection during geological memory retrieval
• 链式反应单元间的信息流控制 | Information flow control between chain reaction units
• V值主导性的梯度增强 | Gradient enhancement for V-value dominance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class OneWayValve(nn.Module):
    """
    单向阀组合 | One-Way Valve Combination
    
    核心功能:
    Core Functions:
    1. 三个独立的门控参数，每个对应一个值 | Three independent gating parameters, each for one value
    2. 三个独立的内部变换模块，每个处理一个值 | Three independent internal transformation modules, each processing one value
    3. 门控融合：将原始输入和变换后的输入按门控参数融合 | Gated fusion: fuses original input and transformed input via gating parameters
    
    数学表达:
    Mathematical Expression:
    Q_out = g_Q * Q + (1 - g_Q) * transform_Q(Q)
    K_out = g_K * K + (1 - g_K) * transform_K(K)
    V_out = g_V * V + (1 - g_V) * transform_V(V)
    其中g = sigmoid(gate_param) ∈ (0,1)
    where g = sigmoid(gate_param) ∈ (0,1)
    """
    
    def __init__(self, dim: int):
        """
        初始化单向阀组合 | Initialize One-Way Valve Combination
        
        参数:
        Parameters:
            dim: 特征维度 | Feature dimension
        """
        super().__init__()
        self.dim = dim
        
        # 三个独立的门控参数，每个对应一个值
        # Three independent gating parameters, each for one value
        self.gate_Q = nn.Parameter(torch.ones(1, 1, dim))
        self.gate_K = nn.Parameter(torch.ones(1, 1, dim))
        self.gate_V = nn.Parameter(torch.ones(1, 1, dim))
        
        # ==================== Q处理模块 ====================
        # ==================== Q Processing Module ====================
        # Q模块：两层线性，中间用ReLU激活
        # Q module: two linear layers with ReLU activation in between
        self.Q_linear1 = nn.Linear(dim, dim)
        self.Q_linear2 = nn.Linear(dim, dim)
        self.Q_act = nn.ReLU()
        
        # ==================== K处理模块 ====================
        # ==================== K Processing Module ====================
        # K模块：一层线性+Tanh激活
        # K module: one linear layer + Tanh activation
        self.K_linear = nn.Linear(dim, dim)
        self.K_act = nn.Tanh()
        
        # ==================== V处理模块 ====================
        # ==================== V Processing Module ====================
        # V模块：链式反应，两层线性，使用不同的激活函数
        # V module: chain reaction, two linear layers with different activation functions
        self.V_linear1 = nn.Linear(dim, dim)
        self.V_linear2 = nn.Linear(dim, dim)
        self.V_act1 = nn.ReLU()
        self.V_act2 = nn.Tanh()
    
    def _gated_fusion(self, x: torch.Tensor, transformed: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        门控融合函数 | Gated Fusion Function
        
        功能:
        Function:
            将原始输入和变换后的输入按门控参数融合
            Fuses original input and transformed input via gating parameters
        
        数学公式:
        Mathematical Formula:
            output = x * g + transformed * (1 - g)
            其中 g = sigmoid(gate) ∈ (0,1)
            where g = sigmoid(gate) ∈ (0,1)
        
        物理意义:
        Physical Meaning:
            • 当g接近1时，保留更多原始信息
            • 当g接近0时，使用更多变换信息
            • 门控参数可学习，自适应调整信息流
            • When g is close to 1, retains more original information
            • When g is close to 0, uses more transformed information
            • Gating parameters are learnable, adaptively adjusting information flow
        """
        g = torch.sigmoid(gate)  # 将门控参数映射到(0,1) | Map gating parameters to (0,1)
        return x * g + transformed * (1 - g)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播 | Forward Propagation
        
        参数:
        Parameters:
            Q: 查询张量，形状为[batch_size, seq_len, dim] | Query tensor, shape [batch_size, seq_len, dim]
            K: 键张量，形状为[batch_size, seq_len, dim] | Key tensor, shape [batch_size, seq_len, dim]
            V: 值张量，形状为[batch_size, seq_len, dim] | Value tensor, shape [batch_size, seq_len, dim]
        
        返回:
        Returns:
            (Q_out, K_out, V_out): 处理后的Q,K,V张量 | Processed Q, K, V tensors
        
        处理流程:
        Processing Flow:
            1. Q处理：两层线性变换 + ReLU激活 | Q processing: two linear transformations + ReLU activation
            2. K处理：一层线性变换 + Tanh激活 | K processing: one linear transformation + Tanh activation
            3. V处理：两层线性变换 + ReLU+Tanh激活 | V processing: two linear transformations + ReLU+Tanh activation
            4. 门控融合：原始值与变换值按门控参数融合 | Gated fusion: original and transformed values fused via gating parameters
        """
        # ==================== 处理Q值 ====================
        # ==================== Process Q Value ====================
        Q_transformed = self.Q_linear1(Q)
        Q_transformed = self.Q_act(Q_transformed)
        Q_transformed = self.Q_linear2(Q_transformed)
        Q_out = self._gated_fusion(Q, Q_transformed, self.gate_Q)
        
        # ==================== 处理K值 ====================
        # ==================== Process K Value ====================
        K_transformed = self.K_act(self.K_linear(K))
        K_out = self._gated_fusion(K, K_transformed, self.gate_K)
        
        # ==================== 处理V值 ====================
        # ==================== Process V Value ====================
        V_transformed = self.V_linear1(V)
        V_transformed = self.V_act1(V_transformed)
        V_transformed = self.V_linear2(V_transformed)
        V_transformed = self.V_act2(V_transformed)
        V_out = self._gated_fusion(V, V_transformed, self.gate_V)
        
        return Q_out, K_out, V_out
    
    def get_gate_values(self) -> dict:
        """
        获取门控值 | Get Gating Values
        
        返回:
        Returns:
            gate_values: 包含当前门控值的字典 | Dictionary containing current gating values
        
        监控用途:
        Monitoring Purpose:
            • 监控门控参数的学习状态
            • 分析信息流控制模式
            • Monitor learning status of gating parameters
            • Analyze information flow control patterns
        """
        with torch.no_grad():
            gate_Q_val = torch.sigmoid(self.gate_Q).mean().item()
            gate_K_val = torch.sigmoid(self.gate_K).mean().item()
            gate_V_val = torch.sigmoid(self.gate_V).mean().item()
        
        return {
            'gate_Q': gate_Q_val,
            'gate_K': gate_K_val,
            'gate_V': gate_V_val,
            'V_dominance': gate_V_val > max(gate_Q_val, gate_K_val),  # V值是否主导 | Whether V value dominates
        }
    
    def apply_detach_mode(self, h_in: torch.Tensor) -> torch.Tensor:
        """
        应用detach模式 | Apply Detach Mode
        
        功能:
        Function:
            切断梯度，创建不可逆记忆
            Cuts off gradients, creates irreversible memory
        
        数学公式:
        Mathematical Formula:
            h_out = detach(h_in)
        
        物理意义:
        Physical Meaning:
            • 创建不可逆的记忆层
            • 防止后续梯度回传干扰已有记忆
            • 适用于地质记忆的存储阶段
            • Creates irreversible memory layers
            • Prevents subsequent gradient backpropagation from interfering with existing memory
            • Suitable for storage phase of geological memory
        """
        return h_in.detach()
    
    def apply_gate_mode(self, h_in: torch.Tensor, gate_value: int) -> torch.Tensor:
        """
        应用gate模式 | Apply Gate Mode
        
        功能:
        Function:
            二进制门控，实现"全或无"信息流
            Binary gating, implements "all-or-nothing" information flow
        
        数学公式:
        Mathematical Formula:
            h_out = g·h_in, g∈{0,1}
        
        参数:
        Parameters:
            h_in: 输入张量 | Input tensor
            gate_value: 门控值，必须是0或1 | Gating value, must be 0 or 1
        
        返回:
        Returns:
            h_out: 输出张量 | Output tensor
        
        物理意义:
        Physical Meaning:
            • gate_value=1: 完全通过，信息流畅通
            • gate_value=0: 完全阻断，信息流中断
            • 实现信息的"全或无"传输
            • gate_value=1: Complete passage, information flow unobstructed
            • gate_value=0: Complete blocking, information flow interrupted
            • Implements "all-or-nothing" information transmission
        """
        if gate_value not in [0, 1]:
            raise ValueError(f"门控值必须是0或1，得到{gate_value} | Gating value must be 0 or 1, got {gate_value}")
        
        if gate_value == 0:
            return torch.zeros_like(h_in)
        else:
            return h_in.clone()
    
    def analyze_information_flow(self, Q_in: torch.Tensor, Q_out: torch.Tensor,
                                K_in: torch.Tensor, K_out: torch.Tensor,
                                V_in: torch.Tensor, V_out: torch.Tensor) -> dict:
        """
        分析信息流变化 | Analyze Information Flow Changes
        
        功能:
        Function:
            计算输入输出的变化量，分析信息流模式
            Calculates changes between input and output, analyzes information flow patterns
        
        参数:
        Parameters:
            Q_in, Q_out: 查询输入输出 | Query input and output
            K_in, K_out: 键输入输出 | Key input and output
            V_in, V_out: 值输入输出 | Value input and output
        
        返回:
        Returns:
            flow_analysis: 信息流分析结果 | Information flow analysis results
        """
        with torch.no_grad():
            # 计算变化量（使用L2范数）
            # Calculate changes (using L2 norm)
            delta_Q = torch.norm(Q_out - Q_in, p=2).item()
            delta_K = torch.norm(K_out - K_in, p=2).item()
            delta_V = torch.norm(V_out - V_in, p=2).item()
            
            # 计算保持率（原始信息保留比例）
            # Calculate retention rate (proportion of original information retained)
            retention_Q = 1.0 - delta_Q / (torch.norm(Q_in, p=2).item() + 1e-8)
            retention_K = 1.0 - delta_K / (torch.norm(K_in, p=2).item() + 1e-8)
            retention_V = 1.0 - delta_V / (torch.norm(V_in, p=2).item() + 1e-8)
            
            # 计算主导性指标（V相对于Q/K的变化）
            # Calculate dominance metric (V change relative to Q/K)
            V_dominance = delta_V / max(delta_Q, delta_K, 1e-8)
            
            # 门控值
            # Gating values
            gate_vals = self.get_gate_values()
        
        return {
            'changes': {
                'delta_Q': delta_Q,
                'delta_K': delta_K,
                'delta_V': delta_V,
            },
            'retention_rates': {
                'retention_Q': retention_Q,
                'retention_K': retention_K,
                'retention_V': retention_V,
            },
            'V_dominance': V_dominance,
            'flow_pattern': self._identify_flow_pattern(retention_Q, retention_K, retention_V, V_dominance),
            'gate_values': gate_vals,
        }
    
    def _identify_flow_pattern(self, r_Q: float, r_K: float, r_V: float, V_dom: float) -> str:
        """
        识别信息流模式 | Identify Information Flow Pattern
        
        参数:
        Parameters:
            r_Q, r_K, r_V: Q, K, V的保持率 | Retention rates for Q, K, V
            V_dom: V主导性指标 | V dominance metric
        
        返回:
        Returns:
            pattern: 信息流模式描述 | Information flow pattern description
        """
        patterns = []
        
        # 根据保持率判断
        # Determine based on retention rates
        if r_V < 0.3:
            patterns.append("V高变换")
        elif r_V > 0.7:
            patterns.append("V高保持")
        
        if r_Q < 0.3 and r_K < 0.3:
            patterns.append("QK高变换")
        elif r_Q > 0.7 and r_K > 0.7:
            patterns.append("QK高保持")
        
        # 根据主导性判断
        # Determine based on dominance
        if V_dom > 2.0:
            patterns.append("V强主导")
        elif V_dom < 0.5:
            patterns.append("V弱主导")
        
        # 根据门控值判断
        # Determine based on gating values
        gate_vals = self.get_gate_values()
        if gate_vals['V_dominance']:
            patterns.append("V门控主导")
        
        return " | ".join(patterns) if patterns else "平衡模式"
    
    def reset_parameters(self):
        """
        重置参数 | Reset Parameters
        
        功能:
        Function:
            重新初始化所有可学习参数
            Reinitialize all learnable parameters
        
        应用场景:
        Application Scenarios:
            • 模型重新训练
            • 参数初始化调优
            • Model retraining
            • Parameter initialization tuning
        """
        # 重置门控参数
        # Reset gating parameters
        nn.init.ones_(self.gate_Q)
        nn.init.ones_(self.gate_K)
        nn.init.ones_(self.gate_V)
        
        # 重置线性层参数
        # Reset linear layer parameters
        for layer in [self.Q_linear1, self.Q_linear2, self.K_linear, self.V_linear1, self.V_linear2]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        print("单向阀参数已重置 | One-way valve parameters reset")


# ==================== 简化的单向阀（纯函数版）====================
# ==================== Simplified One-Way Valve (Pure Function Version) ====================

class SimpleOneWayValve:
    """
    简化的单向阀（纯函数版）| Simplified One-Way Valve (Pure Function Version)
    
    特点:
    Features:
        • 无参数，纯函数实现
        • 支持detach和gate两种模式
        • 适用于轻量级应用
        • Parameter-free, pure function implementation
        • Supports both detach and gate modes
        • Suitable for lightweight applications
    """
    
    @staticmethod
    def apply_valve(h_in: torch.Tensor, mode: str = 'detach', gate_value: Optional[int] = None) -> torch.Tensor:
        """
        应用单向阀 | Apply One-Way Valve
        
        参数:
        Parameters:
            h_in: 输入张量 | Input tensor
            mode: 模式，'detach'或'gate' | Mode, 'detach' or 'gate'
            gate_value: 门控值（仅gate模式需要）| Gating value (only needed for gate mode)
        
        返回:
        Returns:
            h_out: 输出张量 | Output tensor
        
        核心公式:
        Core Formulas:
            1. detach模式: h_out = detach(h_in)
            2. gate模式: h_out = g·h_in, g∈{0,1}
        
        物理意义:
        Physical Meaning:
            • detach模式：创建不可逆记忆层
            • gate模式：实现二进制信息流控制
            • detach mode: creates irreversible memory layers
            • gate mode: implements binary information flow control
        """
        if mode == 'detach':
            # 模式1: 梯度阻断（不可逆记忆）
            # Mode 1: Gradient blocking (irreversible memory)
            return h_in.detach()
        
        elif mode == 'gate':
            # 模式2: 二进制门控
            # Mode 2: Binary gating
            if gate_value not in [0, 1]:
                raise ValueError(f"门控值必须是0或1，得到{gate_value} | Gating value must be 0 or 1, got {gate_value}")
            
            if gate_value == 0:
                return torch.zeros_like(h_in)
            else:
                return h_in.clone()
        
        else:
            raise ValueError(f"不支持的模式: {mode} | Unsupported mode: {mode}")


# ==================== 工厂函数 ====================
# ==================== Factory Functions ====================

def create_one_way_valve(dim: int, valve_type: str = 'learnable') -> nn.Module:
    """
    创建单向阀工厂函数 | One-Way Valve Factory Function
    
    参数:
    Parameters:
        dim: 特征维度 | Feature dimension
        valve_type: 阀类型，'learnable'或'simple' | Valve type, 'learnable' or 'simple'
    
    返回:
    Returns:
        valve: 单向阀实例 | One-way valve instance
    
    设计模式:
    Design Pattern:
        • 工厂模式：统一创建接口
        • 策略模式：支持不同类型的阀
        • Factory pattern: unified creation interface
        • Strategy pattern: supports different valve types
    """
    if valve_type == 'learnable':
        return OneWayValve(dim)
    elif valve_type == 'simple':
        return SimpleOneWayValve()
    else:
        raise ValueError(f"未知的阀类型: {valve_type} | Unknown valve type: {valve_type}")


# ==================== 测试函数 ====================
# ==================== Test Functions ====================

def test_one_way_valve():
    """
    测试单向阀功能 | Test One-Way Valve Functionality
    """
    print("🧪 测试单向阀... | Testing one-way valve...")
    
    # 测试可学习单向阀
    # Test learnable one-way valve
    dim = 64
    batch_size = 4
    seq_len = 10
    
    valve = OneWayValve(dim)
    
    # 创建测试数据
    # Create test data
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播
    # Forward propagation
    Q_out, K_out, V_out = valve(Q, K, V)
    
    # 验证输出形状
    # Verify output shapes
    assert Q_out.shape == (batch_size, seq_len, dim), f"Q输出形状错误: {Q_out.shape}"
    assert K_out.shape == (batch_size, seq_len, dim), f"K输出形状错误: {K_out.shape}"
    assert V_out.shape == (batch_size, seq_len, dim), f"V输出形状错误: {V_out.shape}"
    
    print(f"✅ 形状验证通过 | Shape verification passed")
    print(f"   输入形状: Q{K.shape}, K{K.shape}, V{V.shape}")
    print(f"   输出形状: Q{Q_out.shape}, K{K_out.shape}, V{V_out.shape}")
    
    # 测试门控值获取
    # Test gating value retrieval
    gate_vals = valve.get_gate_values()
    print(f"✅ 门控值获取: {gate_vals}")
    
    # 测试信息流分析
    # Test information flow analysis
    flow_analysis = valve.analyze_information_flow(Q, Q_out, K, K_out, V, V_out)
    print(f"✅ 信息流分析: {flow_analysis['flow_pattern']}")
    
    # 测试简化的单向阀
    # Test simplified one-way valve
    simple_valve = SimpleOneWayValve()
    
    # 测试detach模式
    # Test detach mode
    h = torch.randn(2, 3, requires_grad=True)
    h_detached = simple_valve.apply_valve(h, mode='detach')
    assert not h_detached.requires_grad, "detach模式应移除梯度 | Detach mode should remove gradient"
    
    # 测试gate模式
    # Test gate mode
    h_gated = simple_valve.apply_valve(h, mode='gate', gate_value=0)
    assert torch.allclose(h_gated, torch.zeros_like(h)), "gate模式(g=0)应输出零 | Gate mode (g=0) should output zeros"
    
    h_gated = simple_valve.apply_valve(h, mode='gate', gate_value=1)
    assert torch.allclose(h_gated, h), "gate模式(g=1)应保持输入 | Gate mode (g=1) should preserve input"
    
    print("✅ 所有测试通过 | All tests passed")
    return True

__all__ = [
    "OneWayValve",
    "SimpleOneWayValve",
    "create_one_way_valve",
    "test_one_way_valve",
]

# ==================== 模块主入口 ====================
# ==================== Module Main Entry ====================

if __name__ == "__main__":
    # 当直接运行此文件时，执行测试
    # Execute tests when this file is run directly
    test_one_way_valve()