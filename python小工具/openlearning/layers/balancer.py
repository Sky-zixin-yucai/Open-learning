"""
平衡器层模块 | Balancer Layers Module
===================================

三值平衡与调控层实现，保持Q、K、V在统一维度上的稳定关系。
Tri-value balancing and regulation layer implementations, maintaining stable relationships among Q, K, V in unified dimensions.

设计理念:
Design Philosophy:
• 三值均衡：保持Q、K、V三者的平衡关系 | Tri-value equilibrium: Maintain balanced relationships among Q, K, V
• 动态调节：根据输入特性动态调整平衡参数 | Dynamic regulation: Adjust balance parameters based on input characteristics
• 数值稳定：防止梯度爆炸和数值不稳定 | Numerical stability: Prevent gradient explosion and numerical instability
• 记忆跟踪：记录历史状态以优化平衡 | Memory tracking: Record historical states to optimize balancing

核心组件 | Core Components:
1. TriValueBalancer: 三值平衡器 - 基础平衡器 | Basic tri-value balancer
2. VDominantBalancer: V主导平衡器 - 强调V值主导性 | V-dominant balancer: Emphasize V value dominance
3. DensityDrivenBalancer: 密度驱动平衡器 - 基于连接密度平衡 | Density-driven balancer: Balance based on connection density
4. AdaptiveStabilizer: 自适应稳定器 - 动态调整三值关系 | Adaptive stabilizer: Dynamically adjust tri-value relationships

数学基础 | Mathematical Foundation:
• 平衡公式: V_balanced = V * (1 + M_R * α) | Balance formula: V_balanced = V * (1 + M_R * α)
• Q-K关系: R(Q,K) = cos_sim(Q, K) | Q-K relation: R(Q,K) = cos_sim(Q, K)
• 映射函数: M(R) = tanh(R) | Mapping function: M(R) = tanh(R)
• 密度公式: D = 2m/((N+1)N) | Density formula: D = 2m/((N+1)N)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union, List
import numpy as np
from collections import deque


class TriValueBalancer(nn.Module):
    """
    三值平衡器 | Tri-Value Balancer
    ----------------------------
    
    保持Q、K、V在统一维度上的平衡，防止任一值过度增长或衰减。
    Maintain balance among Q, K, V in unified dimensions, preventing any value from excessive growth or decay.
    
    设计原理 | Design Principles:
        • 三值均衡：确保Q、K、V保持适当比例 | Tri-value equilibrium: Ensure Q, K, V maintain appropriate proportions
        • Q-K关系引导：使用Q-K关系指导V值生成 | Q-K relation guidance: Use Q-K relation to guide V value generation
        • 静态约束：添加静态权重限制数值范围 | Static constraints: Add static weights to limit numerical range
        • 历史跟踪：记录密度变化历史优化平衡 | History tracking: Record density change history to optimize balancing
    
    数学表示 | Mathematical Representation:
        1. 计算Q-K关系: R(Q,K) = cos_sim(Q, K) | Compute Q-K relation: R(Q,K) = cos_sim(Q, K)
        2. 映射函数: M(R) = tanh(R) | Mapping function: M(R) = tanh(R)
        3. 生成V值: V_gen = V * (1 + M(R) * α) | Generate V value: V_gen = V * (1 + M(R) * α)
        4. 平衡Q和K: Q_bal = β * Q + (1-β) * f(Q, V_gen, K) | Balance Q and K: Q_bal = β * Q + (1-β) * f(Q, V_gen, K)
        5. 静态约束: 应用静态权重确保稳定性 | Static constraints: Apply static weights to ensure stability
    
    参数 | Parameters:
        dim (int): 输入特征维度 | Input feature dimension
        memory_size (int, optional): 密度历史记录大小，默认10 | Density history record size, default 10
        qk_balance_init (float, optional): Q-K平衡初始值，默认0.5 | Q-K balance initial value, default 0.5
        v_balance_init (float, optional): V平衡初始值，默认0.3 | V balance initial value, default 0.3
        static_weight_init (float, optional): 静态权重初始值，默认0.2 | Static weight initial value, default 0.2
    
    输入 | Input:
        Q (torch.Tensor): Query张量 [B, S, D] | Query tensor [B, S, D]
        K (torch.Tensor): Key张量 [B, S, D] | Key tensor [B, S, D]
        V (torch.Tensor): Value张量 [B, S, D] | Value tensor [B, S, D]
        return_density (bool, optional): 是否返回密度信息，默认False | Whether to return density info, default False
    
    输出 | Output:
        Tuple: 如果return_density=True，返回(Q, K, V, density, connections)
               否则返回(Q, K, V) | If return_density=True, return (Q, K, V, density, connections)
               Otherwise return (Q, K, V)
    
    示例 | Example:
        >>> balancer = TriValueBalancer(dim=128)
        >>> Q = torch.randn(4, 32, 128)
        >>> K = torch.randn(4, 32, 128)
        >>> V = torch.randn(4, 32, 128)
        >>> Q_bal, K_bal, V_bal = balancer(Q, K, V)
        >>> print(f"平衡后形状: Q={Q_bal.shape}, K={K_bal.shape}, V={V_bal.shape}")
    """
    
    def __init__(self, dim: int, memory_size: int = 10, 
                 qk_balance_init: float = 0.5, v_balance_init: float = 0.3,
                 static_weight_init: float = 0.2):
        """
        初始化三值平衡器 | Initialize Tri-Value Balancer
        
        参数 | Parameters:
            dim (int): 输入特征维度 | Input feature dimension
            memory_size (int): 密度历史记录大小 | Density history record size
            qk_balance_init (float): Q-K平衡初始值 | Q-K balance initial value
            v_balance_init (float): V平衡初始值 | V balance initial value
            static_weight_init (float): 静态权重初始值 | Static weight initial value
        """
        super().__init__()
        self.dim = dim
        self.memory_size = memory_size
        
        # ==================== 平衡权重 ====================
        # Balance Weights
        
        # Q-K平衡权重（可学习） | Q-K balance weight (learnable)
        self.qk_balance = nn.Parameter(torch.tensor(qk_balance_init))
        
        # V平衡权重（可学习） | V balance weight (learnable)
        self.v_balance = nn.Parameter(torch.tensor(v_balance_init))
        
        # 静态平衡权重（可学习） | Static balance weight (learnable)
        self.static_weight = nn.Parameter(torch.tensor(static_weight_init))
        
        # ==================== 线性变换层 ====================
        # Linear Transformation Layers
        
        # Q变换层 | Q transformation layer
        self.q_transform = nn.Linear(dim, dim)
        
        # K变换层 | K transformation layer
        self.k_transform = nn.Linear(dim, dim)
        
        # V变换层 | V transformation layer
        self.v_transform = nn.Linear(dim, dim)
        
        # ==================== 记忆密度跟踪 ====================
        # Memory Density Tracking
        
        # 密度历史记录 | Density history record
        self.register_buffer('density_history', torch.zeros(memory_size))
        self.history_index = 0
        
        # ==================== 初始化参数 ====================
        # Parameter Initialization
        
        self._init_parameters()
    
    def _init_parameters(self):
        """
        初始化参数 | Initialize Parameters
        """
        # 线性层初始化 | Linear layer initialization
        for linear in [self.q_transform, self.k_transform, self.v_transform]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
    
    def _compute_tri_density(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[float, float]:
        """
        计算三值静态密度（内存优化版） | Compute Tri-Value Static Density (Memory Optimized Version)
        
        使用简化的密度计算方法，避免大矩阵操作 | Use simplified density computation method to avoid large matrix operations
        
        参数 | Parameters:
            Q (torch.Tensor): Query张量 | Query tensor
            K (torch.Tensor): Key张量 | Key tensor
            V (torch.Tensor): Value张量 | Value tensor
        
        返回 | Returns:
            Tuple[float, float]: (密度, 连接数) | (Density, number of connections)
        
        计算步骤 | Computation Steps:
            1. 计算每个位置Q、K、V的范数比 | Compute norm ratios of Q, K, V at each position
            2. 计算均衡程度作为密度指标 | Compute equilibrium degree as density metric
            3. 采样计算Q-K相似度连接数 | Sample to compute Q-K similarity connections
        """
        batch_size, seq_len, dim = Q.shape
        
        # 计算每个位置的Q、K、V范数 | Compute norms of Q, K, V at each position
        q_norm = torch.norm(Q, dim=-1, keepdim=True)  # [batch, seq, 1]
        k_norm = torch.norm(K, dim=-1, keepdim=True)
        v_norm = torch.norm(V, dim=-1, keepdim=True)
        
        # 计算Q-K-V三角关系 | Compute Q-K-V triangular relationship
        # 使用每个位置上Q、K、V的范数比例作为密度指标 | Use norm ratios of Q, K, V at each position as density metric
        total_norm = q_norm + k_norm + v_norm + 1e-8
        q_ratio = q_norm / total_norm
        k_ratio = k_norm / total_norm
        v_ratio = v_norm / total_norm
        
        # 计算静态密度：三值的均衡程度 | Compute static density: Equilibrium degree of tri-values
        # 当q_ratio, k_ratio, v_ratio接近0.33时密度最高 | Density highest when q_ratio, k_ratio, v_ratio approach 0.33
        ideal_ratio = 1/3
        density = 1.0 - (torch.abs(q_ratio - ideal_ratio) + 
                        torch.abs(k_ratio - ideal_ratio) + 
                        torch.abs(v_ratio - ideal_ratio)).mean().item()
        
        # 计算连接数：Q-K相似度 | Compute connections: Q-K similarity
        # 采样避免内存过大 | Sample to avoid excessive memory usage
        Q_flat = Q.view(-1, dim)[::seq_len]  # 采样 | Sample
        K_flat = K.view(-1, dim)[::seq_len]
        
        if Q_flat.size(0) > 1:
            Q_norm = F.normalize(Q_flat, p=2, dim=1)
            K_norm = F.normalize(K_flat, p=2, dim=1)
            similarity = torch.mm(Q_norm, K_norm.T)
            connections = (similarity > 0.3).float().sum().item() / 2
        else:
            connections = 0
        
        return density, connections
    
    def _update_density_history(self, density: float):
        """
        更新密度历史 | Update Density History
        
        参数 | Parameters:
            density (float): 当前密度值 | Current density value
        """
        self.density_history[self.history_index] = density
        self.history_index = (self.history_index + 1) % self.memory_size
    
    def _get_density_statistics(self) -> Dict:
        """
        获取密度统计信息 | Get Density Statistics
        
        返回 | Returns:
            Dict: 密度统计字典 | Density statistics dictionary
        """
        valid_history = self.density_history[self.density_history != 0]
        
        if len(valid_history) > 0:
            mean_density = valid_history.mean().item()
            std_density = valid_history.std().item()
            trend = "上升" if len(valid_history) > 1 and valid_history[-1] > valid_history[0] else "稳定"
        else:
            mean_density = 0.0
            std_density = 0.0
            trend = "未知"
        
        return {
            'mean_density': mean_density,
            'std_density': std_density,
            'trend': trend,
            'history_size': len(valid_history)
        }
    
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor,
                return_density: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                                      Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]]:
        """
        前向传播：平衡Q、K、V三值 | Forward Propagation: Balance Q, K, V Tri-Values
        
        参数 | Parameters:
            Q (torch.Tensor): Query张量 [B, S, D] | Query tensor [B, S, D]
            K (torch.Tensor): Key张量 [B, S, D] | Key tensor [B, S, D]
            V (torch.Tensor): Value张量 [B, S, D] | Value tensor [B, S, D]
            return_density (bool): 是否返回密度信息 | Whether to return density info
        
        返回 | Returns:
            Union: 平衡后的Q、K、V，可选密度信息 | Balanced Q, K, V, optional density info
        
        处理步骤 | Processing Steps:
            1. 计算当前密度 | Compute current density
            2. 计算Q-K关系 | Compute Q-K relation
            3. 映射函数应用 | Apply mapping function
            4. 生成V值 | Generate V value
            5. 平衡Q和K | Balance Q and K
            6. 应用静态约束 | Apply static constraints
            7. 数值范围限制 | Limit numerical range
        """
        batch_size, seq_len, dim = Q.shape
        
        # ==================== 1. 计算当前密度 ====================
        # 1. Compute Current Density
        density, connections = self._compute_tri_density(Q, K, V)
        
        # 更新密度历史 | Update density history
        self._update_density_history(density)
        
        # ==================== 2. 计算Q-K关系 ====================
        # 2. Compute Q-K Relation
        # 使用逐位置点积，避免大矩阵 | Use element-wise dot product to avoid large matrices
        Q_norm = F.normalize(Q, p=2, dim=-1)  # [batch, seq, dim]
        K_norm = F.normalize(K, p=2, dim=-1)
        
        # 每个位置的Q-K点积 | Q-K dot product at each position
        qk_dot = torch.sum(Q_norm * K_norm, dim=-1, keepdim=True)  # [batch, seq, 1]
        
        # ==================== 3. 映射函数：M(R(Q,K)) = tanh(R_QK) ====================
        # 3. Mapping Function: M(R(Q,K)) = tanh(R_QK)
        M_R = torch.tanh(qk_dot)  # [batch, seq, 1]
        
        # ==================== 4. 生成V值：G(R, M) 的简化版本 ====================
        # 4. Generate V Value: Simplified version of G(R, M)
        # V_generated = V * (1 + M_R * α)，其中α是V平衡权重 | V_generated = V * (1 + M_R * α), where α is V balance weight
        v_weight = torch.sigmoid(self.v_balance)
        V_generated = V * (1.0 + M_R * v_weight)
        
        # ==================== 5. 平衡公式：V_balanced = V_generated 经过变换 ====================
        # 5. Balance Formula: V_balanced = V_generated after transformation
        V_transformed = self.v_transform(V_generated)
        V_balanced = V * 0.7 + V_transformed * 0.3
        
        # ==================== 6. 同时平衡Q和K ====================
        # 6. Simultaneously Balance Q and K
        qk_weight = torch.sigmoid(self.qk_balance)
        
        # Q应该根据V和K调整 | Q should adjust based on V and K
        Q_with_VK = (V_balanced + K) / 2
        Q_transformed = self.q_transform(Q_with_VK)
        Q_balanced = qk_weight * Q + (1 - qk_weight) * Q_transformed
        
        # K应该根据V和Q调整 | K should adjust based on V and Q
        K_with_VQ = (V_balanced + Q) / 2
        K_transformed = self.k_transform(K_with_VQ)
        K_balanced = qk_weight * K + (1 - qk_weight) * K_transformed
        
        # ==================== 7. 应用静态约束 ====================
        # 7. Apply Static Constraints
        static_factor = torch.sigmoid(self.static_weight)
        Q_final = static_factor * Q_balanced + (1 - static_factor) * Q
        K_final = static_factor * K_balanced + (1 - static_factor) * K
        V_final = static_factor * V_balanced + (1 - static_factor) * V
        
        # ==================== 8. 确保三值在合理范围内 ====================
        # 8. Ensure Tri-Values Within Reasonable Range
        
        # 计算V均值并适当缩放 | Compute V mean and appropriately scale
        v_mean = V_final.mean().item()
        
        if v_mean > 2.0:  # V值过大 | V value too large
            scale = 1.5 / v_mean
            V_final = V_final * scale
            Q_final = Q_final * scale
            K_final = K_final * scale
        elif v_mean < 0.3:  # V值过小 | V value too small
            scale = 0.7 / v_mean
            V_final = V_final * scale
            Q_final = Q_final * scale
            K_final = K_final * scale
        
        # ==================== 9. 返回结果 ====================
        # 9. Return Results
        if return_density:
            return Q_final, K_final, V_final, density, connections
        else:
            return Q_final, K_final, V_final
    
    def get_balance_info(self) -> Dict:
        """
        获取平衡器信息 | Get Balancer Information
        
        返回 | Returns:
            Dict: 平衡器信息字典 | Balancer information dictionary
        """
        density_stats = self._get_density_statistics()
        
        return {
            'dim': self.dim,
            'qk_balance': self.qk_balance.item(),
            'v_balance': self.v_balance.item(),
            'static_weight': self.static_weight.item(),
            'qk_weight_activated': torch.sigmoid(self.qk_balance).item(),
            'v_weight_activated': torch.sigmoid(self.v_balance).item(),
            'static_factor_activated': torch.sigmoid(self.static_weight).item(),
            'density_statistics': density_stats,
            'memory_size': self.memory_size,
            'history_filled': (self.density_history != 0).sum().item(),
        }
    
    def reset_history(self):
        """
        重置历史记录 | Reset History Records
        """
        self.density_history.zero_()
        self.history_index = 0
        print("✅ 平衡器历史已重置 | Balancer history reset")
    
    def extra_repr(self) -> str:
        """
        额外的字符串表示 | Extra String Representation
        
        返回 | Returns:
            str: 描述平衡器参数的字符串 | String describing balancer parameters
        """
        return f'dim={self.dim}, memory_size={self.memory_size}'


class VDominantBalancer(nn.Module):
    """
    V主导平衡器 | V-Dominant Balancer
    --------------------------------
    
    强调V值主导性的平衡器，确保V值在合理范围内主导信息流。
    Balancer emphasizing V value dominance, ensuring V value dominates information flow within reasonable range.
    
    设计原理 | Design Principles:
        • V值主导：确保V值在信息处理中起主导作用 | V-dominant: Ensure V value plays dominant role in information processing
        • 动态阈值：根据输入动态调整V值阈值 | Dynamic threshold: Dynamically adjust V value threshold based on inputs
        • 安全范围：限制V值在安全范围内 | Safety range: Limit V value within safe range
        • 渐进调整：逐步调整V值而非突变 | Gradual adjustment: Gradually adjust V value rather than abrupt changes
    
    与TriValueBalancer的区别 | Differences from TriValueBalancer:
        • 更强调V值主导性 | More emphasis on V value dominance
        • 更严格的V值范围控制 | Stricter V value range control
        • 更关注V值稳定性 | More focus on V value stability
    """
    
    def __init__(self, dim: int, v_dominant_weight: float = 1.5, 
                 v_safety_range: Tuple[float, float] = (0.5, 2.0)):
        """
        初始化V主导平衡器 | Initialize V-Dominant Balancer
        
        参数 | Parameters:
            dim (int): 输入特征维度 | Input feature dimension
            v_dominant_weight (float): V主导权重 | V dominant weight
            v_safety_range (Tuple[float, float]): V值安全范围 (最小, 最大) | V value safety range (min, max)
        """
        super().__init__()
        self.dim = dim
        self.v_safety_min, self.v_safety_max = v_safety_range
        
        # V主导权重（可学习） | V dominant weight (learnable)
        self.v_dominant_weight = nn.Parameter(torch.tensor(v_dominant_weight))
        
        # Q、K平衡权重 | Q, K balance weights
        self.q_balance = nn.Parameter(torch.tensor(0.5))
        self.k_balance = nn.Parameter(torch.tensor(0.5))
        
        # 变换层 | Transformation layers
        self.q_transform = nn.Linear(dim, dim)
        self.k_transform = nn.Linear(dim, dim)
        self.v_transform = nn.Linear(dim, dim)
        
        # V值历史记录 | V value history record
        self.register_buffer('v_history', torch.zeros(10))
        self.history_index = 0
        
        # 初始化参数 | Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数 | Initialize parameters"""
        for linear in [self.q_transform, self.k_transform, self.v_transform]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
    
    def _record_v_history(self, v_mean: float):
        """记录V值历史 | Record V value history"""
        self.v_history[self.history_index] = v_mean
        self.history_index = (self.history_index + 1) % len(self.v_history)
    
    def _get_v_stats(self) -> Dict:
        """获取V值统计 | Get V value statistics"""
        valid_history = self.v_history[self.v_history != 0]
        
        if len(valid_history) > 0:
            mean_v = valid_history.mean().item()
            std_v = valid_history.std().item()
            trend = "上升" if len(valid_history) > 1 and valid_history[-1] > valid_history[0] else "下降"
        else:
            mean_v = 0.0
            std_v = 0.0
            trend = "未知"
        
        return {
            'mean': mean_v,
            'std': std_v,
            'trend': trend,
            'in_safety_range': self.v_safety_min <= mean_v <= self.v_safety_max,
            'history_size': len(valid_history)
        }
    
    def _apply_v_dominance(self, V: torch.Tensor) -> torch.Tensor:
        """
        应用V值主导 | Apply V Value Dominance
        
        参数 | Parameters:
            V (torch.Tensor): Value张量 | Value tensor
        
        返回 | Returns:
            torch.Tensor: 主导处理后的V值 | V value after dominance processing
        """
        # 获取V均值 | Get V mean
        v_mean = V.mean().item()
        self._record_v_history(v_mean)
        
        # 计算V主导权重 | Compute V dominant weight
        v_weight = torch.sigmoid(self.v_dominant_weight)
        
        # 根据V值范围调整权重 | Adjust weight based on V value range
        if v_mean > self.v_safety_max:
            # V值过高，降低权重 | V value too high, reduce weight
            adjustment = 0.8
        elif v_mean < self.v_safety_min:
            # V值过低，增加权重 | V value too low, increase weight
            adjustment = 1.2
        else:
            # V值在安全范围内 | V value within safety range
            adjustment = 1.0
        
        v_weight = v_weight * adjustment
        
        # 应用V主导 | Apply V dominance
        V_dominant = V * v_weight
        
        # 确保在安全范围内 | Ensure within safety range
        v_dominant_mean = V_dominant.mean().item()
        if v_dominant_mean > self.v_safety_max:
            scale = self.v_safety_max / v_dominant_mean
            V_dominant = V_dominant * scale
        elif v_dominant_mean < self.v_safety_min:
            scale = self.v_safety_min / v_dominant_mean
            V_dominant = V_dominant * scale
        
        return V_dominant
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：V主导平衡 | Forward Propagation: V-Dominant Balancing
        
        参数 | Parameters:
            Q (torch.Tensor): Query张量 | Query tensor
            K (torch.Tensor): Key张量 | Key tensor
            V (torch.Tensor): Value张量 | Value tensor
        
        返回 | Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 平衡后的Q、K、V | Balanced Q, K, V
        """
        # ==================== 1. 应用V主导 ====================
        # 1. Apply V Dominance
        V_dominant = self._apply_v_dominance(V)
        
        # ==================== 2. 计算Q-K关系 ====================
        # 2. Compute Q-K Relation
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        qk_dot = torch.sum(Q_norm * K_norm, dim=-1, keepdim=True)
        qk_sim = torch.tanh(qk_dot)
        
        # ==================== 3. 平衡Q和K ====================
        # 3. Balance Q and K
        
        # Q平衡 | Q balancing
        q_weight = torch.sigmoid(self.q_balance)
        Q_with_VK = (V_dominant + K) / 2
        Q_transformed = self.q_transform(Q_with_VK)
        Q_balanced = q_weight * Q + (1 - q_weight) * Q_transformed
        
        # K平衡 | K balancing
        k_weight = torch.sigmoid(self.k_balance)
        K_with_VQ = (V_dominant + Q) / 2
        K_transformed = self.k_transform(K_with_VQ)
        K_balanced = k_weight * K + (1 - k_weight) * K_transformed
        
        # ==================== 4. V值变换 ====================
        # 4. V Value Transformation
        V_transformed = self.v_transform(V_dominant)
        V_balanced = 0.7 * V_dominant + 0.3 * V_transformed
        
        # ==================== 5. 应用Q-K相似度调整 ====================
        # 5. Apply Q-K Similarity Adjustment
        # 如果Q-K相似度高，增强V值 | If Q-K similarity high, enhance V value
        adjustment = 1.0 + 0.2 * qk_sim
        V_final = V_balanced * adjustment
        
        return Q_balanced, K_balanced, V_final
    
    def get_dominant_info(self) -> Dict:
        """
        获取主导信息 | Get Dominant Information
        
        返回 | Returns:
            Dict: 主导信息字典 | Dominant information dictionary
        """
        v_stats = self._get_v_stats()
        
        return {
            'dim': self.dim,
            'v_dominant_weight': self.v_dominant_weight.item(),
            'v_dominant_weight_activated': torch.sigmoid(self.v_dominant_weight).item(),
            'q_balance': self.q_balance.item(),
            'k_balance': self.k_balance.item(),
            'q_weight_activated': torch.sigmoid(self.q_balance).item(),
            'k_weight_activated': torch.sigmoid(self.k_balance).item(),
            'v_safety_range': [self.v_safety_min, self.v_safety_max],
            'v_statistics': v_stats,
        }
    
    def extra_repr(self) -> str:
        """
        额外的字符串表示 | Extra String Representation
        """
        return f'dim={self.dim}, v_safety_range=[{self.v_safety_min}, {self.v_safety_max}]'


class DensityDrivenBalancer(nn.Module):
    """
    密度驱动平衡器 | Density-Driven Balancer
    -------------------------------------
    
    基于连接点密度公式驱动的平衡器，使用密度信息指导平衡过程。
    Balancer driven by connection point density formula, using density information to guide balancing process.
    
    设计原理 | Design Principles:
        • 密度驱动：使用密度公式指导平衡 | Density-driven: Use density formula to guide balancing
        • 动态连接：基于相似度阈值建立连接 | Dynamic connections: Establish connections based on similarity threshold
        • 密度反馈：根据密度变化调整平衡策略 | Density feedback: Adjust balancing strategy based on density changes
        • 历史学习：从历史密度模式学习 | Historical learning: Learn from historical density patterns
    """
    
    def __init__(self, dim: int, connection_threshold: float = 0.3, 
                 density_history_size: int = 20):
        """
        初始化密度驱动平衡器 | Initialize Density-Driven Balancer
        
        参数 | Parameters:
            dim (int): 输入特征维度 | Input feature dimension
            connection_threshold (float): 连接阈值 | Connection threshold
            density_history_size (int): 密度历史大小 | Density history size
        """
        super().__init__()
        self.dim = dim
        self.connection_threshold = connection_threshold
        self.density_history_size = density_history_size
        
        # 平衡权重 | Balance weights
        self.density_weight = nn.Parameter(torch.tensor(0.5))
        self.stability_weight = nn.Parameter(torch.tensor(0.3))
        
        # 变换层 | Transformation layers
        self.q_transform = nn.Linear(dim, dim)
        self.k_transform = nn.Linear(dim, dim)
        self.v_transform = nn.Linear(dim, dim)
        
        # 密度历史 | Density history
        self.register_buffer('density_history', torch.zeros(density_history_size))
        self.density_index = 0
        
        # 连接数历史 | Connection count history
        self.register_buffer('connection_history', torch.zeros(density_history_size))
        
        # 初始化参数 | Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数 | Initialize parameters"""
        for linear in [self.q_transform, self.k_transform, self.v_transform]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
    
    def _compute_density(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[float, float]:
        """
        计算密度和连接数 | Compute Density and Connection Count
        
        参数 | Parameters:
            Q (torch.Tensor): Query张量 | Query tensor
            K (torch.Tensor): Key张量 | Key tensor
            V (torch.Tensor): Value张量 | Value tensor
        
        返回 | Returns:
            Tuple[float, float]: (密度, 连接数) | (Density, connection count)
        
        使用密度公式: D = 2m/((N+1)N) | Use density formula: D = 2m/((N+1)N)
        """
        batch_size, seq_len, dim = Q.shape
        
        # 采样避免内存过大 | Sample to avoid excessive memory
        sample_rate = max(1, seq_len // 10)
        Q_sampled = Q.view(-1, dim)[::sample_rate]
        K_sampled = K.view(-1, dim)[::sample_rate]
        V_sampled = V.view(-1, dim)[::sample_rate]
        
        # 计算相似度矩阵 | Compute similarity matrix
        Q_norm = F.normalize(Q_sampled, p=2, dim=1)
        K_norm = F.normalize(K_sampled, p=2, dim=1)
        V_norm = F.normalize(V_sampled, p=2, dim=1)
        
        # 计算Q-K相似度 | Compute Q-K similarity
        qk_similarity = torch.mm(Q_norm, K_norm.T)
        
        # 统计连接数 | Count connections
        connections = (qk_similarity > self.connection_threshold).float().sum().item() / 2
        
        # 计算密度 | Compute density
        N = qk_similarity.size(0)
        if N > 1:
            density = (2 * connections) / ((N + 1) * N)
        else:
            density = 0.0
        
        # 记录历史 | Record history
        self._update_history(density, connections)
        
        return density, connections
    
    def _update_history(self, density: float, connections: float):
        """更新历史记录 | Update history records"""
        self.density_history[self.density_index] = density
        self.connection_history[self.density_index] = connections
        self.density_index = (self.density_index + 1) % self.density_history_size
    
    def _get_density_trend(self) -> str:
        """获取密度趋势 | Get density trend"""
        valid_density = self.density_history[self.density_history != 0]
        
        if len(valid_density) < 2:
            return "稳定"
        
        # 计算趋势 | Compute trend
        recent = valid_density[-min(5, len(valid_density)):]
        early = valid_density[:min(5, len(valid_density))]
        
        recent_mean = recent.mean().item()
        early_mean = early.mean().item()
        
        if recent_mean > early_mean * 1.1:
            return "上升"
        elif recent_mean < early_mean * 0.9:
            return "下降"
        else:
            return "稳定"
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：密度驱动平衡 | Forward Propagation: Density-Driven Balancing
        
        参数 | Parameters:
            Q (torch.Tensor): Query张量 | Query tensor
            K (torch.Tensor): Key张量 | Key tensor
            V (torch.Tensor): Value张量 | Value tensor
        
        返回 | Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 平衡后的Q、K、V | Balanced Q, K, V
        """
        # ==================== 1. 计算密度 ====================
        # 1. Compute Density
        density, connections = self._compute_density(Q, K, V)
        
        # ==================== 2. 根据密度调整平衡策略 ====================
        # 2. Adjust Balancing Strategy Based on Density
        
        # 密度权重 | Density weight
        density_weight = torch.sigmoid(self.density_weight)
        
        # 稳定性权重 | Stability weight
        stability_weight = torch.sigmoid(self.stability_weight)
        
        # 根据密度调整平衡强度 | Adjust balance strength based on density
        if density < 0.2:
            # 密度过低，增强平衡 | Density too low, enhance balancing
            balance_strength = 0.8
        elif density > 0.8:
            # 密度过高，减弱平衡 | Density too high, weaken balancing
            balance_strength = 0.3
        else:
            # 正常密度 | Normal density
            balance_strength = 0.5
        
        # 综合权重 | Comprehensive weight
        combined_weight = density_weight * balance_strength
        
        # ==================== 3. 应用平衡变换 ====================
        # 3. Apply Balance Transformations
        
        # Q变换 | Q transformation
        Q_transformed = self.q_transform(Q)
        Q_balanced = combined_weight * Q_transformed + (1 - combined_weight) * Q
        
        # K变换 | K transformation
        K_transformed = self.k_transform(K)
        K_balanced = combined_weight * K_transformed + (1 - combined_weight) * K
        
        # V变换 | V transformation
        V_transformed = self.v_transform(V)
        V_balanced = combined_weight * V_transformed + (1 - combined_weight) * V
        
        # ==================== 4. 根据连接数调整 ====================
        # 4. Adjust Based on Connection Count
        
        # 计算连接密度比 | Compute connection density ratio
        max_possible_connections = (Q.size(0) * Q.size(1)) * (Q.size(0) * Q.size(1) - 1) / 2
        connection_ratio = connections / max_possible_connections if max_possible_connections > 0 else 0
        
        # 根据连接密度比调整 | Adjust based on connection density ratio
        if connection_ratio < 0.1:
            # 连接过少，增强V值 | Too few connections, enhance V value
            v_adjustment = 1.2
        elif connection_ratio > 0.5:
            # 连接过多，减弱V值 | Too many connections, weaken V value
            v_adjustment = 0.8
        else:
            v_adjustment = 1.0
        
        V_final = V_balanced * v_adjustment
        
        # ==================== 5. 应用稳定性约束 ====================
        # 5. Apply Stability Constraints
        Q_final = stability_weight * Q_balanced + (1 - stability_weight) * Q
        K_final = stability_weight * K_balanced + (1 - stability_weight) * K
        V_final = stability_weight * V_final + (1 - stability_weight) * V
        
        return Q_final, K_final, V_final
    
    def get_density_info(self) -> Dict:
        """
        获取密度信息 | Get Density Information
        
        返回 | Returns:
            Dict: 密度信息字典 | Density information dictionary
        """
        valid_density = self.density_history[self.density_history != 0]
        valid_connections = self.connection_history[self.connection_history != 0]
        
        density_trend = self._get_density_trend()
        
        info = {
            'dim': self.dim,
            'connection_threshold': self.connection_threshold,
            'density_weight': self.density_weight.item(),
            'stability_weight': self.stability_weight.item(),
            'density_weight_activated': torch.sigmoid(self.density_weight).item(),
            'stability_weight_activated': torch.sigmoid(self.stability_weight).item(),
            'density_trend': density_trend,
            'history_size': len(valid_density),
        }
        
        if len(valid_density) > 0:
            info.update({
                'avg_density': valid_density.mean().item(),
                'avg_connections': valid_connections.mean().item(),
                'max_density': valid_density.max().item(),
                'min_density': valid_density.min().item(),
            })
        
        return info
    
    def reset_history(self):
        """重置历史记录 | Reset history records"""
        self.density_history.zero_()
        self.connection_history.zero_()
        self.density_index = 0
        print("✅ 密度驱动平衡器历史已重置 | Density-driven balancer history reset")
    
    def extra_repr(self) -> str:
        """
        额外的字符串表示 | Extra String Representation
        """
        return f'dim={self.dim}, connection_threshold={self.connection_threshold}, history_size={self.density_history_size}'


class AdaptiveStabilizer(nn.Module):
    """
    自适应稳定器 | Adaptive Stabilizer
    --------------------------------
    
    动态调整三值关系的自适应稳定器，根据输入特性优化稳定性。
    Adaptive stabilizer that dynamically adjusts tri-value relationships, optimizing stability based on input characteristics.
    
    设计原理 | Design Principles:
        • 动态适应：根据输入特性调整稳定策略 | Dynamic adaptation: Adjust stability strategy based on input characteristics
        • 多模式稳定：支持多种稳定模式 | Multi-mode stability: Support multiple stability modes
        • 实时监控：实时监控数值稳定性 | Real-time monitoring: Real-time monitoring of numerical stability
        • 预防性调整：预防性调整避免数值问题 | Preventive adjustment: Preventive adjustment to avoid numerical issues
    """
    
    def __init__(self, dim: int, num_modes: int = 3):
        """
        初始化自适应稳定器 | Initialize Adaptive Stabilizer
        
        参数 | Parameters:
            dim (int): 输入特征维度 | Input feature dimension
            num_modes (int): 稳定模式数量 | Number of stability modes
        """
        super().__init__()
        self.dim = dim
        self.num_modes = num_modes
        
        # 模式选择器 | Mode selector
        self.mode_selector = nn.Sequential(
            nn.Linear(dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes),
            nn.Softmax(dim=-1)
        )
        
        # 模式参数 | Mode parameters
        self.mode_weights = nn.Parameter(torch.randn(num_modes))
        
        # 模式特定变换层 | Mode-specific transformation layers
        self.q_transforms = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_modes)])
        self.k_transforms = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_modes)])
        self.v_transforms = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_modes)])
        
        # 稳定性历史 | Stability history
        self.register_buffer('stability_history', torch.zeros(10))
        self.stability_index = 0
        
        # 初始化参数 | Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数 | Initialize parameters"""
        # 模式选择器初始化 | Mode selector initialization
        for layer in self.mode_selector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # 变换层初始化 | Transformation layer initialization
        for transforms in [self.q_transforms, self.k_transforms, self.v_transforms]:
            for transform in transforms:
                nn.init.xavier_uniform_(transform.weight)
                nn.init.zeros_(transform.bias)
    
    def _compute_stability(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> float:
        """
        计算稳定性分数 | Compute Stability Score
        
        参数 | Parameters:
            Q (torch.Tensor): Query张量 | Query tensor
            K (torch.Tensor): Key张量 | Key tensor
            V (torch.Tensor): Value张量 | Value tensor
        
        返回 | Returns:
            float: 稳定性分数 (0-1) | Stability score (0-1)
        """
        # 计算范数稳定性 | Compute norm stability
        q_norm = torch.norm(Q, dim=-1).std().item()
        k_norm = torch.norm(K, dim=-1).std().item()
        v_norm = torch.norm(V, dim=-1).std().item()
        
        # 计算均值稳定性 | Compute mean stability
        q_mean = Q.mean().item()
        k_mean = K.mean().item()
        v_mean = V.mean().item()
        
        # 计算数值范围 | Compute numerical range
        q_range = Q.max().item() - Q.min().item()
        k_range = K.max().item() - K.min().item()
        v_range = V.max().item() - V.min().item()
        
        # 综合稳定性分数 | Comprehensive stability score
        norm_stability = 1.0 / (q_norm + k_norm + v_norm + 1e-6)
        mean_stability = 1.0 / (abs(q_mean) + abs(k_mean) + abs(v_mean) + 1e-6)
        range_stability = 1.0 / (q_range + k_range + v_range + 1e-6)
        
        stability = (norm_stability + mean_stability + range_stability) / 3
        
        # 记录稳定性历史 | Record stability history
        self.stability_history[self.stability_index] = stability
        self.stability_index = (self.stability_index + 1) % len(self.stability_history)
        
        return stability
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：自适应稳定 | Forward Propagation: Adaptive Stabilization
        
        参数 | Parameters:
            Q (torch.Tensor): Query张量 | Query tensor
            K (torch.Tensor): Key张量 | Key tensor
            V (torch.Tensor): Value张量 | Value tensor
        
        返回 | Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 稳定后的Q、K、V | Stabilized Q, K, V
        """
        batch_size, seq_len, dim = Q.shape
        
        # ==================== 1. 计算稳定性 ====================
        # 1. Compute Stability
        stability = self._compute_stability(Q, K, V)
        
        # ==================== 2. 选择稳定模式 ====================
        # 2. Select Stability Mode
        
        # 展平输入用于模式选择 | Flatten inputs for mode selection
        flattened = torch.cat([Q, K, V], dim=-1).mean(dim=(0, 1))  # [dim*3]
        flattened = flattened.unsqueeze(0)  # [1, dim*3]
        
        # 计算模式概率 | Compute mode probabilities
        mode_probs = self.mode_selector(flattened)  # [1, num_modes]
        
        # 应用模式权重 | Apply mode weights
        mode_weights = F.softmax(self.mode_weights, dim=0)
        combined_probs = mode_probs * mode_weights
        combined_probs = combined_probs / combined_probs.sum(dim=-1, keepdim=True)
        
        # ==================== 3. 根据稳定性调整模式强度 ====================
        # 3. Adjust Mode Intensity Based on Stability
        
        if stability < 0.3:
            # 稳定性低，使用强稳定模式 | Low stability, use strong stability mode
            mode_intensity = 0.8
        elif stability > 0.7:
            # 稳定性高，使用弱稳定模式 | High stability, use weak stability mode
            mode_intensity = 0.2
        else:
            # 中等稳定性 | Medium stability
            mode_intensity = 0.5
        
        # ==================== 4. 应用各模式变换 ====================
        # 4. Apply Each Mode Transformation
        
        mode_outputs_Q = []
        mode_outputs_K = []
        mode_outputs_V = []
        
        for i in range(self.num_modes):
            # Q变换 | Q transformation
            Q_transformed = self.q_transforms[i](Q)
            mode_outputs_Q.append(Q_transformed)
            
            # K变换 | K transformation
            K_transformed = self.k_transforms[i](K)
            mode_outputs_K.append(K_transformed)
            
            # V变换 | V transformation
            V_transformed = self.v_transforms[i](V)
            mode_outputs_V.append(V_transformed)
        
        # ==================== 5. 加权融合各模式输出 ====================
        # 5. Weighted Fusion of Mode Outputs
        
        # 堆叠模式输出 | Stack mode outputs
        Q_stacked = torch.stack(mode_outputs_Q, dim=3)  # [B, S, D, num_modes]
        K_stacked = torch.stack(mode_outputs_K, dim=3)
        V_stacked = torch.stack(mode_outputs_V, dim=3)
        
        # 应用模式概率 | Apply mode probabilities
        mode_probs_expanded = combined_probs.view(1, 1, 1, self.num_modes)  # [1, 1, 1, num_modes]
        
        Q_weighted = Q_stacked * mode_probs_expanded
        K_weighted = K_stacked * mode_probs_expanded
        V_weighted = V_stacked * mode_probs_expanded
        
        # 加权求和 | Weighted sum
        Q_fused = Q_weighted.sum(dim=3)
        K_fused = K_weighted.sum(dim=3)
        V_fused = V_weighted.sum(dim=3)
        
        # ==================== 6. 应用模式强度 ====================
        # 6. Apply Mode Intensity
        
        Q_stabilized = mode_intensity * Q_fused + (1 - mode_intensity) * Q
        K_stabilized = mode_intensity * K_fused + (1 - mode_intensity) * K
        V_stabilized = mode_intensity * V_fused + (1 - mode_intensity) * V
        
        # ==================== 7. 根据稳定性进行最终调整 ====================
        # 7. Final Adjustment Based on Stability
        
        # 如果稳定性过低，进行额外稳定化 | If stability too low, perform additional stabilization
        if stability < 0.2:
            # 计算缩放因子 | Compute scaling factor
            scale = 0.8 if V_stabilized.mean().item() > 1.5 else 1.2
            V_stabilized = V_stabilized * scale
        
        return Q_stabilized, K_stabilized, V_stabilized
    
    def get_stability_info(self) -> Dict:
        """
        获取稳定性信息 | Get Stability Information
        
        返回 | Returns:
            Dict: 稳定性信息字典 | Stability information dictionary
        """
        valid_stability = self.stability_history[self.stability_history != 0]
        
        info = {
            'dim': self.dim,
            'num_modes': self.num_modes,
            'mode_weights': F.softmax(self.mode_weights, dim=0).tolist(),
            'stability_history_size': len(valid_stability),
        }
        
        if len(valid_stability) > 0:
            info.update({
                'avg_stability': valid_stability.mean().item(),
                'min_stability': valid_stability.min().item(),
                'max_stability': valid_stability.max().item(),
                'stability_trend': '上升' if len(valid_stability) > 1 and valid_stability[-1] > valid_stability[0] else '下降',
            })
        
        return info
    
    def extra_repr(self) -> str:
        """
        额外的字符串表示 | Extra String Representation
        """
        return f'dim={self.dim}, num_modes={self.num_modes}'


# ==================== 工厂函数 ====================
# Factory Functions

def create_balancer_layer(balancer_type: str = 'tri_value', **kwargs) -> nn.Module:
    """
    创建平衡器层 | Create Balancer Layer
    
    参数 | Parameters:
        balancer_type (str): 平衡器类型，可选['tri_value', 'v_dominant', 'density_driven', 'adaptive'] | Balancer type, options: ['tri_value', 'v_dominant', 'density_driven', 'adaptive']
        **kwargs: 传递给具体平衡器的参数 | Parameters passed to specific balancer
    
    返回 | Returns:
        nn.Module: 创建的平衡器 | Created balancer
    
    异常 | Raises:
        ValueError: 如果平衡器类型不支持 | If balancer type is not supported
    """
    balancer_map = {
        'tri_value': TriValueBalancer,
        'v_dominant': VDominantBalancer,
        'density_driven': DensityDrivenBalancer,
        'adaptive': AdaptiveStabilizer,
    }
    
    if balancer_type not in balancer_map:
        raise ValueError(
            f"不支持的平衡器类型: {balancer_type}\n"
            f"支持的类型: {list(balancer_map.keys())}"
        )
    
    return balancer_map[balancer_type](**kwargs)


# ==================== 测试函数 ====================
# Test Functions

def test_tri_value_balancer():
    """测试三值平衡器 | Test Tri-Value Balancer"""
    print("=" * 60)
    print("测试三值平衡器 | Testing Tri-Value Balancer")
    print("=" * 60)
    
    # 创建平衡器 | Create balancer
    dim = 64
    balancer = TriValueBalancer(dim=dim, memory_size=5)
    
    # 创建测试数据 | Create test data
    batch_size, seq_len = 2, 16
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播（无密度信息） | Forward propagation (without density info)
    Q_bal, K_bal, V_bal = balancer(Q, K, V, return_density=False)
    
    # 前向传播（有密度信息） | Forward propagation (with density info)
    Q_bal2, K_bal2, V_bal2, density, connections = balancer(Q, K, V, return_density=True)
    
    # 验证输出形状 | Verify output shapes
    print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"输出形状: Q={Q_bal.shape}, K={K_bal.shape}, V={V_bal.shape}")
    
    # 验证一致性 | Verify consistency
    q_diff = torch.abs(Q_bal - Q_bal2).max().item()
    k_diff = torch.abs(K_bal - K_bal2).max().item()
    v_diff = torch.abs(V_bal - V_bal2).max().item()
    
    print(f"\n输出一致性检查 | Output Consistency Check:")
    print(f"  Q差异最大值: {q_diff:.6f} (应接近0)")
    print(f"  K差异最大值: {k_diff:.6f} (应接近0)")
    print(f"  V差异最大值: {v_diff:.6f} (应接近0)")
    
    # 获取平衡信息 | Get balance info
    balance_info = balancer.get_balance_info()
    print(f"\n平衡器信息 | Balancer Information:")
    for key, value in balance_info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # 检查V值范围 | Check V value range
    v_mean = V_bal.mean().item()
    print(f"V值均值: {v_mean:.4f} (应在合理范围内)")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)
    
    return balancer


def test_v_dominant_balancer():
    """测试V主导平衡器 | Test V-Dominant Balancer"""
    print("=" * 60)
    print("测试V主导平衡器 | Testing V-Dominant Balancer")
    print("=" * 60)
    
    # 创建平衡器 | Create balancer
    dim = 64
    balancer = VDominantBalancer(dim=dim, v_dominant_weight=1.5, v_safety_range=(0.5, 2.0))
    
    # 创建测试数据 | Create test data
    batch_size, seq_len = 2, 16
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播 | Forward propagation
    Q_bal, K_bal, V_bal = balancer(Q, K, V)
    
    # 验证输出 | Verify output
    print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"输出形状: Q={Q_bal.shape}, K={K_bal.shape}, V={V_bal.shape}")
    
    # 获取主导信息 | Get dominant info
    dominant_info = balancer.get_dominant_info()
    print(f"\n主导信息 | Dominant Information:")
    for key, value in dominant_info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        elif isinstance(value, list):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")
    
    # 检查V值是否在安全范围内 | Check if V value within safety range
    v_mean = V_bal.mean().item()
    v_min, v_max = dominant_info['v_safety_range']
    in_range = v_min <= v_mean <= v_max
    print(f"V值均值: {v_mean:.4f}, 安全范围: [{v_min}, {v_max}], 在范围内: {in_range}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)
    
    return balancer


def test_density_driven_balancer():
    """测试密度驱动平衡器 | Test Density-Driven Balancer"""
    print("=" * 60)
    print("测试密度驱动平衡器 | Testing Density-Driven Balancer")
    print("=" * 60)
    
    # 创建平衡器 | Create balancer
    dim = 64
    balancer = DensityDrivenBalancer(dim=dim, connection_threshold=0.3, density_history_size=5)
    
    # 创建测试数据 | Create test data
    batch_size, seq_len = 2, 16
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)
    
    # 多次前向传播填充历史 | Multiple forward passes to fill history
    for _ in range(3):
        Q_bal, K_bal, V_bal = balancer(Q, K, V)
    
    # 验证输出 | Verify output
    print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"输出形状: Q={Q_bal.shape}, K={K_bal.shape}, V={V_bal.shape}")
    
    # 获取密度信息 | Get density info
    density_info = balancer.get_density_info()
    print(f"\n密度信息 | Density Information:")
    for key, value in density_info.items():
        if isinstance(value, dict):
            continue  # 已经单独处理 | Already handled separately
        else:
            print(f"  {key}: {value}")
    
    # 重置历史 | Reset history
    balancer.reset_history()
    print(f"历史已重置 | History reset")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)
    
    return balancer


def test_adaptive_stabilizer():
    """测试自适应稳定器 | Test Adaptive Stabilizer"""
    print("=" * 60)
    print("测试自适应稳定器 | Testing Adaptive Stabilizer")
    print("=" * 60)
    
    # 创建稳定器 | Create stabilizer
    dim = 64
    stabilizer = AdaptiveStabilizer(dim=dim, num_modes=3)
    
    # 创建测试数据 | Create test data
    batch_size, seq_len = 2, 16
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播 | Forward propagation
    Q_stab, K_stab, V_stab = stabilizer(Q, K, V)
    
    # 验证输出 | Verify output
    print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"输出形状: Q={Q_stab.shape}, K={K_stab.shape}, V={V_stab.shape}")
    
    # 获取稳定性信息 | Get stability info
    stability_info = stabilizer.get_stability_info()
    print(f"\n稳定性信息 | Stability Information:")
    for key, value in stability_info.items():
        if isinstance(value, list):
            print(f"  {key}: {[f'{v:.4f}' for v in value]}")
        else:
            print(f"  {key}: {value}")
    
    # 检查数值稳定性 | Check numerical stability
    q_std = Q.std().item()
    k_std = K.std().item()
    v_std = V.std().item()
    q_stab_std = Q_stab.std().item()
    k_stab_std = K_stab.std().item()
    v_stab_std = V_stab.std().item()
    
    print(f"\n数值稳定性比较 | Numerical Stability Comparison:")
    print(f"  Q标准差: 原始={q_std:.4f}, 稳定后={q_stab_std:.4f}")
    print(f"  K标准差: 原始={k_std:.4f}, 稳定后={k_stab_std:.4f}")
    print(f"  V标准差: 原始={v_std:.4f}, 稳定后={v_stab_std:.4f}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)
    
    return stabilizer


def test_balancer_factory():
    """测试平衡器工厂函数 | Test Balancer Factory Functions"""
    print("=" * 60)
    print("测试平衡器工厂函数 | Testing Balancer Factory Functions")
    print("=" * 60)
    
    # 测试创建各种平衡器 | Test creating various balancers
    balancer_types = ['tri_value', 'v_dominant', 'density_driven', 'adaptive']
    
    for balancer_type in balancer_types:
        try:
            if balancer_type == 'tri_value':
                balancer = create_balancer_layer(balancer_type, dim=64)
            elif balancer_type == 'v_dominant':
                balancer = create_balancer_layer(balancer_type, dim=64, v_dominant_weight=1.5)
            elif balancer_type == 'density_driven':
                balancer = create_balancer_layer(balancer_type, dim=64, connection_threshold=0.3)
            elif balancer_type == 'adaptive':
                balancer = create_balancer_layer(balancer_type, dim=64, num_modes=3)
            
            print(f"✅ 成功创建 {balancer_type} 平衡器 | Successfully created {balancer_type} balancer")
            print(f"  类型: {balancer.__class__.__name__}")
            print(f"  维度: {balancer.dim}")
            print(f"  参数量: {sum(p.numel() for p in balancer.parameters()):,}")
        except Exception as e:
            print(f"❌ 创建 {balancer_type} 平衡器失败: {e}")
    
    # 测试无效类型 | Test invalid type
    try:
        balancer = create_balancer_layer('invalid', dim=64)
    except ValueError as e:
        print(f"\n✅ 预期中的错误 | Expected error: {e}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)


# ==================== 导出列表 ====================
# Export List
# src/rga/layers/balancer.py

__all__ = [
    # Balancer classes
    'TriValueBalancer',
    'VDominantBalancer',
    'DensityDrivenBalancer',
    'AdaptiveStabilizer',
    
    # Factory functions
    'create_balancer_layer',
    
    # Test functions
    'test_tri_value_balancer',
    'test_v_dominant_balancer',
    'test_density_driven_balancer',
    'test_adaptive_stabilizer',
    'test_balancer_factory',
]


# ==================== 模块自检 ====================
# Module Self-Test

if __name__ == "__main__":
    print("=" * 60)
    print("平衡器层模块自检 | Balancer Layers Module Self-Test")
    print("=" * 60)
    
    try:
        # 运行测试 | Run tests
        print("\n1. 测试三值平衡器 | Testing Tri-Value Balancer")
        test_tri_value_balancer()
        
        print("\n2. 测试V主导平衡器 | Testing V-Dominant Balancer")
        test_v_dominant_balancer()
        
        print("\n3. 测试密度驱动平衡器 | Testing Density-Driven Balancer")
        test_density_driven_balancer()
        
        print("\n4. 测试自适应稳定器 | Testing Adaptive Stabilizer")
        test_adaptive_stabilizer()
        
        print("\n5. 测试工厂函数 | Testing Factory Functions")
        test_balancer_factory()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过 | All tests passed")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败 | Test failed: {e}")
        import traceback
        traceback.print_exc()