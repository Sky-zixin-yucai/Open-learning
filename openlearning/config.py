"""
配置管理模块 | Configuration Management Module
===================================

"""
import time
import os
import json
import re
import random
import math
from collections import Counter, deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Function

@dataclass
class RGAConfig:
    """
    RGA配置类 | RGA Configuration Class
    ================================== 标准配置 | Standard Configuration ==============================
    特点 | Features
    - 反应单元数量 | Number of reaction units
    - 地质记忆深度 | Geological memory depth
    - 持续思考循环 | Continuous thinking loop
    """
    # ============================ 基础配置 | Basic Configuration ====================
    vocab_size: int = 10000        # 词汇表大小 | Vocabulary size
    dim: int = 32                  # 模型维度（必须为偶数）| Model dimension (must be even)

    # ============================ 特色配置 | Special Configuration ==================
    num_units: int = 3             # 反应单元数量 | Number of reaction units
    geo_depth: int = 3             # 地质记忆深度 | Geological memory depth
    max_loop: int = 3              # 持续思考循环 | Continuous thinking loop

    # ============================ 公式配置 | Formula Configuration ==================
    phase_threshold: float = 0.80      # 相变检测阈值 | Phase transition detection threshold
    v_scaling_factor: float = 1.0      # 能量缩放因子 | Energy scaling factor
    min_Q_concepts: float = 1.0        # 最小概念数量 | Minimum number of concepts
    history_length: int = 10           # 历史记忆长度 | History memory length
    density_method: str = "sum"        # 密度计算方法 | Density calculation method

#  =============================== 核心配置 | Auxiliary Configuration ================
@dataclass
class VRegularizationParams:
    max_V_mean: float = 2.0             # V值最大均值 | Maximum mean value of V 
    min_V_mean: float = 0.5             # V值最小均值 | Minimum mean value of V 
    target_V_mean: float = 1.0          # 目标V值均值 | Target mean value
    similarity_threshold: float = 0.3   # 相似度阈值 | Similarity threshold
    adjustment_strength: float = 0.1    # 调整强度 | Adjustment strength  
    cycle_decay_rate: float = 0.95      # 循环衰减率 | Cycle decay rate

V_regularization_params: VRegularizationParams = field(default_factory=VRegularizationParams)

class CoreMetricsCalculator:
    """
    监控计算器类，用于保证核心指标。| Core Metrics Calculator Class
    ==================================== 监控器 | Monitor ===============================================
    - 单网络变化量公式 (状态监控) | Single Network Change Formula (Status Monitoring)
    - 相变检测公式 (质变识别) | Phase Transition Detection Formula (Quality Identification)
    - 三网络堆叠公式 (多视角融合) | Three Network Stacking Formula (Multi-View Fusion)
    - 单向阀公式 (信息控制) | Unidirectional Valve Formula (Information Control)
    """

    def __init__(self, config):
        self.config = config
        self.state_history = [] # 状态历史记录 | State history record
        self.transition_points = [] # 记忆相变点 | Memory phase transition points

        # ================= 公式1: 单网络变化量 | Formula 1: Single Network Change =================================
    def compute_state_change(self, 
                           Q_t: torch.Tensor, K_t: torch.Tensor, V_t: torch.Tensor,
                           Q_t_1: torch.Tensor, K_t_1: torch.Tensor, V_t_1: torch.Tensor,
                           norm_type: str = 'l2') -> float:
        """
        单网络变化量公式: Δ = ||Q_t - Q_{t-1}|| + ||K_t - K_{t-1}|| + ||V_t - V_{t-1}||
        ------------------------------------------------------------------------------------------------------------
        Formula for the change in a single network: Δ = ||Q_t - Q_{t-1}|| + ||K_t - K_{t-1}|| + ||V_t - V_{t-1}||
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
            raise ValueError(f"不支持的范数类型 | Unsupported norm type |: {norm_type}")
        
        return (delta_Q + delta_K + delta_V).item()
    
    # ==================== 公式2: 相变检测 | Change Detection Formula 2 | ====================
    def detect_phase_transition(self,
                              Q_t: torch.Tensor, K_t: torch.Tensor, V_t: torch.Tensor,
                              Q_t_1: torch.Tensor, K_t_1: torch.Tensor, V_t_1: torch.Tensor) -> Tuple[float, bool]:
        """
        相变检测公式: Δ = ‖Q^(t) - Q^(t-1)‖_F + ‖K^(t) - K^(t-1)‖_F + ‖V^(t) - V^(t-1)‖_F
        --------------------------------------------------------------------------------------------------------------
        Phase transition detection formula: Δ = ‖Q^(t) - Q^(t-1)‖_F + ‖K^(t) - K^(t-1)‖_F + ‖V^(t) - V^(t-1)‖_F
        """
        # Frobenius范数：捕捉矩阵结构变化 | Frobenius norm: captures changes in matrix structure
        delta_Q = torch.norm(Q_t - Q_t_1, p='fro')
        delta_K = torch.norm(K_t - K_t_1, p='fro')
        delta_V = torch.norm(V_t - V_t_1, p='fro')
        
        delta_total = (delta_Q + delta_K + delta_V).item()
        
        # 检测是否为相变 | Detect phase transition
        threshold = self.config.phase_threshold
        is_transition = delta_total > threshold
        
        if is_transition:
            self.transition_points.append({
                'delta': delta_total,
                'Q_shape': Q_t.shape,
                'timestamp': len(self.state_history)
            })
        
        return delta_total, is_transition
    
    # ==================== 公式3: 三网络堆叠 | Formula 3: Stacked Three Networks ====================
    def stack_three_networks(self,
                           Q_list: List[torch.Tensor],
                           weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        三网络堆叠公式: Q_stack = ∑_{i=1}^3 α_i Q_i, where α = Softmax(w)
        -------------------------------------------------------------------------------------------------------------------
        Stack three networks formula: Q_stack = ∑_{i=1}^3 α_i Q_i, where α = Softmax(w)
        """
        if len(Q_list) != 3:
            raise ValueError(f"| 需要3个Q矩阵进行堆叠,得到 | 3 Q matrices need to be stacked to obtain {len(Q_list)}个 | piece |")
        
        # 验证形状一致性 | Verify shape consistency 
        shapes = [Q.shape for Q in Q_list]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError("| 所有Q矩阵必须形状一致。 | All Q matrices must have the same shape. | ")
        
        # 默认权重：均匀分布 | Default weight: uniform distribution
        if weights is None:
            weights = torch.ones(3)
        
        # Softmax归一化权重（确保总和为1）| Softmax normalized weights (ensuring the sum is 1)
        alpha = F.softmax(weights, dim=0)
        
        # 加权堆叠 | Weighted stacking
        Q_stack = torch.zeros_like(Q_list[0])
        for i, Q in enumerate(Q_list):
            Q_stack += alpha[i] * Q
        
        return Q_stack
    
    # =================== 公式4: 单向阀 | Formula 4: Check Valve ===================================
    def apply_one_way_valve(self,
                          h_in: torch.Tensor,
                          mode: str = 'detach',
                          gate_value: Optional[int] = None) -> torch.Tensor:
        """
        单向阀公式: h_out = detach(h_in) 或 h_out = g·h_in, g∈{0,1}
        ------------------------------------------------------------------------------------------------------------------------
        Check Valve Formula: h_out = detach(h_in) 或 h_out = g·h_in, g∈{0,1}
        """
        if mode == 'detach':
            # 模式1: 梯度阻断（不可逆记忆）| Gradient Blocking (Irreversible Memory)
            return h_in.detach()
        
        elif mode == 'gate':
            # 模式2: 二进制门控 | Binary gate control
            if gate_value not in [0, 1]:
                raise ValueError(f"门控值必须是0或1，得到{gate_value}")
            
            if gate_value == 0:
                return torch.zeros_like(h_in)
            else:
                return h_in.clone()
        
        else:
            raise ValueError(f"不支持的模式: {mode}")
    
    # ==================== 辅助方法 | Auxiliary Methods ====================
    def record_state(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """记录当前状态到历史 | Record the current state to history"""
        self.state_history.append({
            'Q': Q.detach().clone(),
            'K': K.detach().clone(),
            'V': V.detach().clone(),
            'step': len(self.state_history)
        })
    
    def get_state_change_series(self, norm_type: str = 'l2') -> List[float]:
        """获取状态变化序列 | Get the series of state changes"""
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
    
    def analyze_learning_phases(self) -> Dict:
        """分析学习阶段 | Analyze learning phases"""
        if len(self.state_history) < 2:
            return {'message': '数据不足 | Insufficient data'}
        
        changes = self.get_state_change_series('l2')
        
        # 检测学习阶段
        if not changes:
            return {'phase': '初始阶段 | Initial Phase', 'stability': 0.0}
        
        recent_changes = changes[-min(5, len(changes)):]  # 最近5步
        avg_change = sum(recent_changes) / len(recent_changes)
        
        if avg_change > 25.0:
            phase = "探索期 | Exploration Phase"
        elif avg_change > 15.0:
            phase = "学习期 | Learning Phase"
        elif avg_change > 5.0:
            phase = "稳定期 | Stabilization Phase"
        else:
            phase = "收敛期 | Convergence Phase"
        
        return {
            'phase': phase,
            'avg_change': avg_change,
            'total_steps': len(self.state_history),
            'transition_points': len(self.transition_points),
            'current_stability': 1.0 / (avg_change + 1e-6)
        }
    
    def reset(self):
        """重置计算器状态 | Reset the calculator state."""
        self.state_history = []
        self.transition_points = []

# ==================================== 三值平衡器 | Ternary Balancer ====================================
class TriValueBalancer(nn.Module):
    """三值平衡器 - 保持Q、K、V在统一维度上 | Ternary Balancer - Keep Q, K, V in the same dimension | """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 平衡权重（可学习） | Balance weights (learnable)
        self.qk_balance = nn.Parameter(torch.tensor(0.5))  # Q-K平衡权重 | Q-K balance weight
        self.v_balance = nn.Parameter(torch.tensor(0.3))   # V平衡权重 | V balance weight
        self.static_weight = nn.Parameter(torch.tensor(0.2))  # 静态平衡权重 | Static balance weight
        
        # 简化内存：使用线性变换 | Simplified memory: Use linear transformation
        self.q_transform = nn.Linear(dim, dim)
        self.k_transform = nn.Linear(dim, dim)
        self.v_transform = nn.Linear(dim, dim)
        
        # 记忆密度跟踪 | Memory density tracking
        self.register_buffer('density_history', torch.zeros(10))
        
    def _compute_tri_density(self, Q, K, V):
        """计算三值静态密度（内存优化版）| Calculate ternary static density (memory-optimized version)"""
        batch_size, seq_len, dim = Q.shape
        
        # 计算每个位置的Q、K、V范数 | Calculate the norms of Q, K, and V at each position
        q_norm = torch.norm(Q, dim=-1, keepdim=True)  # [batch, seq, 1]
        k_norm = torch.norm(K, dim=-1, keepdim=True)
        v_norm = torch.norm(V, dim=-1, keepdim=True)
        
        # 计算Q-K-V三角关系 | Calculate the triangular relationship of Q-K-V
        # 使用每个位置上Q、K、V的范数比例作为密度指标 | Use the ratio of norms at each position as a density indicator
        total_norm = q_norm + k_norm + v_norm + 1e-8
        q_ratio = q_norm / total_norm
        k_ratio = k_norm / total_norm
        v_ratio = v_norm / total_norm
        
        # 计算静态密度：三值的均衡程度 | Calculate static density: the balance level of three values
        # 当q_ratio, k_ratio, v_ratio接近0.33时密度最高 | The density is the highest when q_ratio, k_ratio, and v_ratio are close to 0.33.
        ideal_ratio = 1/3
        density = 1.0 - (torch.abs(q_ratio - ideal_ratio) + 
                        torch.abs(k_ratio - ideal_ratio) + 
                        torch.abs(v_ratio - ideal_ratio)).mean().item()
        
        # 计算连接数：Q-K相似度 | Calculate connections: Q-K similarity
        Q_flat = Q.view(-1, dim)[::seq_len]  # 采样，避免内存过大 | Sampling to avoid memory overflow
        K_flat = K.view(-1, dim)[::seq_len]
        
        if Q_flat.size(0) > 1:
            Q_norm = F.normalize(Q_flat, p=2, dim=1)
            K_norm = F.normalize(K_flat, p=2, dim=1)
            similarity = torch.mm(Q_norm, K_norm.T)
            connections = (similarity > 0.3).float().sum().item() / 2
        else:
            connections = 0
        
        return density, connections
        
    def forward(self, Q, K, V, return_density=False):
        """
        平衡Q、K、V三值
        核心公式：保持三者在统一密度关系下
        -------------------------------------
        Balancing the three values Q, K, and V
        Core formula: maintaining the three in a unified density relationship
        """
        batch_size, seq_len, dim = Q.shape
        
        # 1. 计算当前密度 | Compute current density
        density, connections = self._compute_tri_density(Q, K, V)
        
        # 更新密度历史 | Update density history
        self.density_history = torch.roll(self.density_history, shifts=1)
        self.density_history[0] = density
        
        # 2. 计算Q-K关系（简化版，避免大矩阵）| Calculating the Q-K relationship (simplified version to avoid large matrices)
        # 使用逐位置点积，而不是整个相似度矩阵 | Use position-wise dot products instead of the entire similarity matrix
        Q_norm = F.normalize(Q, p=2, dim=-1)  # [batch, seq, dim]
        K_norm = F.normalize(K, p=2, dim=-1)
        
        # 每个位置的Q-K点积 | Q-K dot product at each position
        qk_dot = torch.sum(Q_norm * K_norm, dim=-1, keepdim=True)  # [batch, seq, 1]
        
        # 3. 映射函数|mapping function：M(R(Q,K)) = tanh(R_QK)
        M_R = torch.tanh(qk_dot)  # [batch, seq, 1]
        
        # 4. 生成V：G(R, M) 的简化版本 | Simplified version of G(R, M)
        # V_generated = V * (1 + M_R * α)，其中α是V平衡权重 | where α is the V balance weight
        v_weight = torch.sigmoid(self.v_balance)
        V_generated = V * (1.0 + M_R * v_weight)
        
        # 5. 平衡公式：V_balanced = V_generated 经过变换 | Balanced formula: V_balanced = V_generated after transformation
        V_transformed = self.v_transform(V_generated)
        V_balanced = V * 0.7 + V_transformed * 0.3
        
        # 6. 同时平衡Q和K | Simultaneously balance Q and K
        qk_weight = torch.sigmoid(self.qk_balance)
        
        # Q应该根据V和K调整 | Q should be adjusted based on V and K
        Q_with_VK = (V_balanced + K) / 2
        Q_transformed = self.q_transform(Q_with_VK)
        Q_balanced = qk_weight * Q + (1 - qk_weight) * Q_transformed
        
        # K应该根据V和Q调整 | K should be adjusted based on V and Q
        K_with_VQ = (V_balanced + Q) / 2
        K_transformed = self.k_transform(K_with_VQ)
        K_balanced = qk_weight * K + (1 - qk_weight) * K_transformed
        
        # 7. 应用静态约束 | Apply static constraints
        static_factor = torch.sigmoid(self.static_weight)
        Q_final = static_factor * Q_balanced + (1 - static_factor) * Q
        K_final = static_factor * K_balanced + (1 - static_factor) * K
        V_final = static_factor * V_balanced + (1 - static_factor) * V
        
        # 8. 确保三值在合理范围内（防止爆炸）| Ensure that the three values are within a reasonable range (to prevent explosion)
        # 计算均值并适当缩放 | Calculate the mean and appropriately scale | 
        v_mean = V_final.mean().item()
        if v_mean > 2.0:
            scale = 1.5 / v_mean
            V_final = V_final * scale
            Q_final = Q_final * scale
            K_final = K_final * scale
        elif v_mean < 0.3:
            scale = 0.7 / v_mean
            V_final = V_final * scale
            Q_final = Q_final * scale
            K_final = K_final * scale
        
        if return_density:
            return Q_final, K_final, V_final, density, connections
        else:
            return Q_final, K_final, V_final

# ======================= 导出列表 | Export list ====================

__all__ = [
    'RGAConfig',
    'VRegularizationParams',
    'CoreMetricsCalculator',
    'TriValueBalancer',

]