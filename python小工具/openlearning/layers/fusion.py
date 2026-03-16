"""
三明治融合模块 | Sandwich Fusion Module
======================================

实现三明治融合层，将深层、当前和原始状态按固定权重融合。
Implements sandwich fusion layer, fusing deep, current, and original states with fixed weights.

设计原理:
Design Principles:
• 三层融合：深层状态（稳定）、当前状态（活跃）、原始状态（基准）
• 固定权重：权重不可学习，按照文档设计固定
• 维度扩展：将深层均值向量扩展为完整序列形状
• Three-layer fusion: deep state (stable), current state (active), original state (baseline)
• Fixed weights: weights are not learnable, fixed according to document design
• Dimension expansion: expands deep mean vector to full sequence shape

数学公式:
Mathematical Formulas:
1. Q融合: Q_final = 0.5×Q_deep + 0.3×Q_current + 0.2×Q_original
2. K融合: K_final = 0.5×K_deep + 0.3×K_current + 0.2×K_original
3. V融合: V_final = 0.6×V_deep + 0.3×V_current + 0.1×V_original

物理意义:
Physical Meaning:
• 深层状态：地质记忆中的稳定模式，权重最大
• 当前状态：链式反应单元的最新输出，权重中等
• 原始状态：初始输入状态，权重最小，保持基准
• Deep state: stable patterns from geological memory, largest weight
• Current state: latest output from chain reaction units, medium weight
• Original state: initial input state, smallest weight, maintains baseline

权重设计:
Weight Design:
• Q/K权重：[深层0.5, 当前0.3, 原始0.2]
• V权重：[深层0.6, 当前0.3, 原始0.1] - V值主导性更强
• Q/K weights: [deep 0.5, current 0.3, original 0.2]
• V weights: [deep 0.6, current 0.3, original 0.1] - V value has stronger dominance

维度处理:
Dimension Handling:
• 深层状态通常是从地质层检索的均值向量[dim]或[1,dim]
• 需要将其扩展为[batch_size, seq_len, dim]以匹配当前状态
• Deep state is usually mean vector from geological layer [dim] or [1,dim]
• Needs to be expanded to [batch_size, seq_len, dim] to match current state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class SandwichFusion(nn.Module):
    """
    三明治融合层 | Sandwich Fusion Layer
    
    核心功能:
    Core Functions:
    1. 将地质层的深层状态（均值向量）扩展为完整序列形状
    2. 按固定权重融合深层、当前和原始状态
    3. 确保V值的主导性（V权重不同）
    
    Expands deep state (mean vector) from geological layer to full sequence shape
    Fuses deep, current, and original states with fixed weights
    Ensures V value dominance (different V weights)
    
    权重设计:
    Weight Design:
        Q权重：深层0.5，当前0.3，原始0.2
        K权重：深层0.5，当前0.3，原始0.2
        V权重：深层0.6，当前0.3，原始0.1 - V值主导性更强
        
        Q weights: deep 0.5, current 0.3, original 0.2
        K weights: deep 0.5, current 0.3, original 0.2
        V weights: deep 0.6, current 0.3, original 0.1 - V value has stronger dominance
    """
    
    def __init__(self):
        """
        初始化三明治融合层 | Initialize Sandwich Fusion Layer
        """
        super().__init__()
        
        # ==================== 固定权重 ====================
        # ==================== Fixed Weights ====================
        # 按照文档中的权重，不可学习
        # Fixed weights according to document, not learnable
        
        # Q权重：深层0.5，当前0.3，原始0.2
        # Q weights: deep 0.5, current 0.3, original 0.2
        self.register_buffer('q_weights', torch.tensor([0.5, 0.3, 0.2]))
        
        # K权重：深层0.5，当前0.3，原始0.2
        # K weights: deep 0.5, current 0.3, original 0.2
        self.register_buffer('k_weights', torch.tensor([0.5, 0.3, 0.2]))
        
        # V权重：深层0.6，当前0.3，原始0.1
        # V weights: deep 0.6, current 0.3, original 0.1
        self.register_buffer('v_weights', torch.tensor([0.6, 0.3, 0.1]))
        
        # 验证权重和为1
        # Verify weight sums equal 1
        self._validate_weights()
        
        print(f"✅ 三明治融合层初始化 | Sandwich fusion layer initialized")
        print(f"   Q权重: {self.q_weights.tolist()}")
        print(f"   K权重: {self.k_weights.tolist()}")
        print(f"   V权重: {self.v_weights.tolist()}")
    
    def _validate_weights(self):
        """
        验证权重和是否为1 | Verify weight sums equal 1
        
        确保:
        Ensures:
            1. 所有权重和接近1.0（允许微小误差）
            2. 所有权重非负
            3. V权重不同，体现V值主导性
            
            1. All weight sums close to 1.0 (allow small error)
            2. All weights non-negative
            3. V weights different, reflecting V value dominance
        """
        tolerance = 1e-6
        
        # 检查权重和
        # Check weight sums
        q_sum = self.q_weights.sum().item()
        k_sum = self.k_weights.sum().item()
        v_sum = self.v_weights.sum().item()
        
        assert abs(q_sum - 1.0) < tolerance, f"Q权重和应为1.0，得到{q_sum}"
        assert abs(k_sum - 1.0) < tolerance, f"K权重和应为1.0，得到{k_sum}"
        assert abs(v_sum - 1.0) < tolerance, f"V权重和应为1.0，得到{v_sum}"
        
        # 检查权重非负
        # Check weights non-negative
        assert torch.all(self.q_weights >= 0).item(), "Q权重应为非负"
        assert torch.all(self.k_weights >= 0).item(), "K权重应为非负"
        assert torch.all(self.v_weights >= 0).item(), "V权重应为非负"
        
        # 检查V权重独特性
        # Check V weight uniqueness
        assert not torch.allclose(self.v_weights, self.q_weights), "V权重应与Q权重不同"
        assert not torch.allclose(self.v_weights, self.k_weights), "V权重应与K权重不同"
        
        print(f"✅ 权重验证通过 | Weight validation passed")
    
    def _expand_deep_state(self, 
                          deep_state: torch.Tensor, 
                          batch_size: int, 
                          seq_len: int) -> torch.Tensor:
        """
        将地质层的深层状态（均值向量）扩展为完整序列形状
        Expand deep state (mean vector) from geological layer to full sequence shape
        
        输入处理:
        Input Processing:
            支持多种输入形状：
            1. [dim] → 扩展为 [batch_size, seq_len, dim]
            2. [1, dim] → 扩展为 [batch_size, seq_len, dim]
            3. [1, 1, dim] → 扩展为 [batch_size, seq_len, dim]
            4. [1, seq, dim] → 取均值后扩展
        
        Supports multiple input shapes:
            1. [dim] → expanded to [batch_size, seq_len, dim]
            2. [1, dim] → expanded to [batch_size, seq_len, dim]
            3. [1, 1, dim] → expanded to [batch_size, seq_len, dim]
            4. [1, seq, dim] → averaged then expanded
        
        参数:
        Parameters:
            deep_state: 深层状态张量，可能为各种形状
            batch_size: 目标批次大小
            seq_len: 目标序列长度
            
            deep_state: Deep state tensor, could be various shapes
            batch_size: Target batch size
            seq_len: Target sequence length
        
        返回:
        Returns:
            expanded_state: 扩展后的张量，形状为[batch_size, seq_len, dim]
            expanded_state: Expanded tensor with shape [batch_size, seq_len, dim]
        
        算法步骤:
        Algorithm Steps:
            1. 清理维度，确保至少3维
            2. 如果是[1, seq, dim]，取均值变成[1, 1, dim]
            3. 扩展维度到目标形状
        
        1. Clean dimensions, ensure at least 3D
        2. If [1, seq, dim], average to [1, 1, dim]
        3. Expand dimensions to target shape
        """
        # 记录原始形状用于调试
        # Record original shape for debugging
        original_shape = deep_state.shape
        
        # 清理维度
        # Clean dimensions
        if deep_state.dim() == 1:  # [dim]
            deep_state = deep_state.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
        elif deep_state.dim() == 2:  # [1, dim] 或 [seq, dim]
            if deep_state.size(0) == 1:  # [1, dim]
                deep_state = deep_state.unsqueeze(1)  # [1, 1, dim]
            else:  # [seq, dim] - 不常见
                # 取均值变成[1, 1, dim]
                # Average to [1, 1, dim]
                deep_state = deep_state.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, dim]
        
        # 现在应该是3维： [1, 1, dim] 或 [1, seq, dim]
        # Now should be 3D: [1, 1, dim] or [1, seq, dim]
        
        # 确保我们得到 [1, 1, dim]
        # Ensure we get [1, 1, dim]
        if deep_state.size(1) != 1:
            # 如果是[1, seq, dim]，取均值变成[1, 1, dim]
            # If [1, seq, dim], average to [1, 1, dim]
            deep_state = deep_state.mean(dim=1, keepdim=True)
        
        # 扩展维度到 [batch_size, seq_len, dim]
        # Expand dimensions to [batch_size, seq_len, dim]
        expanded = deep_state.expand(batch_size, seq_len, -1)
        
        # 调试信息
        # Debug information
        if hasattr(self, '_debug') and self._debug:
            print(f"  深层状态扩展: {original_shape} → {deep_state.shape} → {expanded.shape}")
        
        return expanded
    
    def forward(self,
               # 深层状态（从地质层检索）
               # Deep state (retrieved from geological layer)
               Q_deep: torch.Tensor,      # 深层Q状态，可能是[dim]、[1,dim]或[1,1,dim]
               K_deep: torch.Tensor,      # 深层K状态
               V_deep: torch.Tensor,      # 深层V状态
               
               # 当前状态（最新链式反应输出）
               # Current state (latest chain reaction output)
               Q_current: torch.Tensor,   # [B, S, dim] 当前Q3⁽ⁿ⁾
               K_current: torch.Tensor,   # [B, S, dim] 当前K3⁽ⁿ⁾
               V_current: torch.Tensor,   # [B, S, dim] 当前V3⁽ⁿ⁾
               
               # 原始状态（初始输入）- 可选
               # Original state (initial input) - optional
               Q_original: Optional[torch.Tensor] = None,
               K_original: Optional[torch.Tensor] = None,
               V_original: Optional[torch.Tensor] = None
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        三明治融合前向传播 | Sandwich Fusion Forward Propagation
        
        核心功能:
        Core Functions:
            将深层状态扩展到完整维度，然后进行加权融合
            Expand deep state to full dimensions, then perform weighted fusion
        
        参数:
        Parameters:
            Q_deep, K_deep, V_deep: 深层状态，可能是各种形状
            Q_current, K_current, V_current: 当前状态，形状为[batch_size, seq_len, dim]
            Q_original, K_original, V_original: 原始状态，形状为[batch_size, seq_len, dim]或None
            
            Q_deep, K_deep, V_deep: Deep states, could be various shapes
            Q_current, K_current, V_current: Current states, shape [batch_size, seq_len, dim]
            Q_original, K_original, V_original: Original states, shape [batch_size, seq_len, dim] or None
        
        返回:
        Returns:
            (Q_final, K_final, V_final): 融合后的Q,K,V张量，形状为[batch_size, seq_len, dim]
            (Q_final, K_final, V_final): Fused Q, K, V tensors, shape [batch_size, seq_len, dim]
        
        融合公式:
        Fusion Formulas:
            Q_final = 0.5×Q_deep_expanded + 0.3×Q_current + 0.2×Q_original
            K_final = 0.5×K_deep_expanded + 0.3×K_current + 0.2×K_original
            V_final = 0.6×V_deep_expanded + 0.3×V_current + 0.1×V_original
        """
        # 获取批次和序列维度
        # Get batch and sequence dimensions
        batch_size, seq_len, dim = Q_current.shape
        
        # ==================== 维度扩展 ====================
        # ==================== Dimension Expansion ====================
        # 将深层状态扩展到完整序列形状
        # Expand deep states to full sequence shape
        Q_deep_expanded = self._expand_deep_state(Q_deep, batch_size, seq_len)
        K_deep_expanded = self._expand_deep_state(K_deep, batch_size, seq_len)
        V_deep_expanded = self._expand_deep_state(V_deep, batch_size, seq_len)
        
        # ==================== 处理原始状态 ====================
        # ==================== Process Original States ====================
        # 如果未提供原始状态，使用零张量
        # If original states not provided, use zero tensors
        if Q_original is None:
            Q_original = torch.zeros_like(Q_current)
        if K_original is None:
            K_original = torch.zeros_like(K_current)
        if V_original is None:
            V_original = torch.zeros_like(V_current)
        
        # 验证所有输入形状一致
        # Verify all input shapes are consistent
        self._validate_shapes(
            Q_deep_expanded, K_deep_expanded, V_deep_expanded,
            Q_current, K_current, V_current,
            Q_original, K_original, V_original,
            batch_size, seq_len, dim
        )
        
        # ==================== 加权融合 ====================
        # ==================== Weighted Fusion ====================
        # Q融合: 深层0.5，当前0.3，原始0.2
        # Q fusion: deep 0.5, current 0.3, original 0.2
        Q_final = (self.q_weights[0] * Q_deep_expanded +
                  self.q_weights[1] * Q_current +
                  self.q_weights[2] * Q_original)
        
        # K融合: 深层0.5，当前0.3，原始0.2
        # K fusion: deep 0.5, current 0.3, original 0.2
        K_final = (self.k_weights[0] * K_deep_expanded +
                  self.k_weights[1] * K_current +
                  self.k_weights[2] * K_original)
        
        # V融合: 深层0.6，当前0.3，原始0.1 (V值主导性更强)
        # V fusion: deep 0.6, current 0.3, original 0.1 (V value has stronger dominance)
        V_final = (self.v_weights[0] * V_deep_expanded +
                  self.v_weights[1] * V_current +
                  self.v_weights[2] * V_original)
        
        return Q_final, K_final, V_final
    
    def _validate_shapes(self,
                        Q_deep: torch.Tensor, K_deep: torch.Tensor, V_deep: torch.Tensor,
                        Q_current: torch.Tensor, K_current: torch.Tensor, V_current: torch.Tensor,
                        Q_original: torch.Tensor, K_original: torch.Tensor, V_original: torch.Tensor,
                        batch_size: int, seq_len: int, dim: int):
        """
        验证输入形状 | Validate Input Shapes
        
        确保:
        Ensures:
            1. 所有张量形状为 [batch_size, seq_len, dim]
            2. 深层状态已正确扩展
            3. 原始状态存在或为零张量
            
            1. All tensors have shape [batch_size, seq_len, dim]
            2. Deep states properly expanded
            3. Original states exist or are zero tensors
        """
        expected_shape = (batch_size, seq_len, dim)
        
        # 检查深层状态
        # Check deep states
        assert Q_deep.shape == expected_shape, f"Q_deep形状应为{expected_shape}，得到{Q_deep.shape}"
        assert K_deep.shape == expected_shape, f"K_deep形状应为{expected_shape}，得到{K_deep.shape}"
        assert V_deep.shape == expected_shape, f"V_deep形状应为{expected_shape}，得到{V_deep.shape}"
        
        # 检查当前状态
        # Check current states
        assert Q_current.shape == expected_shape, f"Q_current形状应为{expected_shape}，得到{Q_current.shape}"
        assert K_current.shape == expected_shape, f"K_current形状应为{expected_shape}，得到{K_current.shape}"
        assert V_current.shape == expected_shape, f"V_current形状应为{expected_shape}，得到{V_current.shape}"
        
        # 检查原始状态
        # Check original states
        assert Q_original.shape == expected_shape, f"Q_original形状应为{expected_shape}，得到{Q_original.shape}"
        assert K_original.shape == expected_shape, f"K_original形状应为{expected_shape}，得到{K_original.shape}"
        assert V_original.shape == expected_shape, f"V_original形状应为{expected_shape}，得到{V_original.shape}"
    
    def apply_single_fusion(self,
                           deep_state: torch.Tensor,
                           current_state: torch.Tensor,
                           original_state: torch.Tensor,
                           value_type: str = 'Q') -> torch.Tensor:
        """
        对单个值进行融合，自动处理维度扩展
        Fuse single value, automatically handles dimension expansion
        
        参数:
        Parameters:
            deep_state: 深层状态，可能是各种形状
            current_state: 当前状态，形状为[batch_size, seq_len, dim]
            original_state: 原始状态，形状为[batch_size, seq_len, dim]或None
            value_type: 值类型，'Q'、'K'或'V'
            
            deep_state: Deep state, could be various shapes
            current_state: Current state, shape [batch_size, seq_len, dim]
            original_state: Original state, shape [batch_size, seq_len, dim] or None
            value_type: Value type, 'Q', 'K', or 'V'
        
        返回:
        Returns:
            fused_state: 融合后的张量
            fused_state: Fused tensor
        """
        batch_size, seq_len, dim = current_state.shape
        
        # 扩展深层状态
        # Expand deep state
        deep_expanded = self._expand_deep_state(deep_state, batch_size, seq_len)
        
        # 选择权重
        # Select weights
        if value_type == 'Q':
            weights = self.q_weights
        elif value_type == 'K':
            weights = self.k_weights
        elif value_type == 'V':
            weights = self.v_weights
        else:
            raise ValueError(f"未知的值类型: {value_type} | Unknown value type: {value_type}")
        
        # 确保原始状态有正确形状
        # Ensure original state has correct shape
        if original_state is None:
            original_state = torch.zeros_like(current_state)
        elif original_state.shape != (batch_size, seq_len, dim):
            original_state = self._expand_deep_state(original_state, batch_size, seq_len)
        
        # 加权融合
        # Weighted fusion
        return (weights[0] * deep_expanded +
                weights[1] * current_state +
                weights[2] * original_state)
    
    def analyze_fusion_contribution(self,
                                  Q_deep: torch.Tensor,
                                  Q_current: torch.Tensor,
                                  Q_original: torch.Tensor = None) -> dict:
        """
        分析融合贡献度 | Analyze Fusion Contribution
        
        功能:
        Function:
            计算各组成部分在最终输出中的贡献比例
            Calculate contribution ratio of each component in final output
        
        参数:
        Parameters:
            Q_deep, Q_current, Q_original: Q值的深层、当前、原始状态
            (可类似分析K和V)
        
        返回:
        Returns:
            contribution_analysis: 贡献度分析结果
            contribution_analysis: Contribution analysis results
        """
        # 获取融合结果
        # Get fusion result
        Q_fused = self.apply_single_fusion(Q_deep, Q_current, Q_original, 'Q')
        
        with torch.no_grad():
            # 计算各组成部分的贡献（能量）
            # Calculate contribution (energy) of each component
            batch_size, seq_len, dim = Q_current.shape
            
            # 扩展深层状态
            # Expand deep state
            Q_deep_expanded = self._expand_deep_state(Q_deep, batch_size, seq_len)
            
            # 处理原始状态
            # Process original state
            if Q_original is None:
                Q_original = torch.zeros_like(Q_current)
            
            # 加权后的各组成部分
            # Weighted components
            deep_contribution = self.q_weights[0] * Q_deep_expanded
            current_contribution = self.q_weights[1] * Q_current
            original_contribution = self.q_weights[2] * Q_original
            
            # 计算能量（L2范数）
            # Calculate energy (L2 norm)
            total_energy = torch.norm(Q_fused, p=2).item()
            deep_energy = torch.norm(deep_contribution, p=2).item()
            current_energy = torch.norm(current_contribution, p=2).item()
            original_energy = torch.norm(original_contribution, p=2).item()
            
            # 计算贡献比例
            # Calculate contribution ratios
            if total_energy > 0:
                deep_ratio = deep_energy / total_energy
                current_ratio = current_energy / total_energy
                original_ratio = original_energy / total_energy
            else:
                deep_ratio = current_ratio = original_ratio = 0.0
        
        return {
            'total_energy': total_energy,
            'contributions': {
                'deep': {
                    'energy': deep_energy,
                    'ratio': deep_ratio,
                    'weight': self.q_weights[0].item(),
                },
                'current': {
                    'energy': current_energy,
                    'ratio': current_ratio,
                    'weight': self.q_weights[1].item(),
                },
                'original': {
                    'energy': original_energy,
                    'ratio': original_ratio,
                    'weight': self.q_weights[2].item(),
                },
            },
            'weight_design': {
                'designed_ratios': self.q_weights.tolist(),
                'actual_ratios': [deep_ratio, current_ratio, original_ratio],
                'alignment': self._calculate_alignment_score(
                    [deep_ratio, current_ratio, original_ratio],
                    self.q_weights.tolist()
                ),
            },
        }
    
    def _calculate_alignment_score(self, actual_ratios: list, designed_ratios: list) -> float:
        """
        计算实际比例与设计比例的对齐分数 | Calculate alignment score between actual and designed ratios
        
        参数:
        Parameters:
            actual_ratios: 实际贡献比例
            designed_ratios: 设计权重比例
            
            actual_ratios: Actual contribution ratios
            designed_ratios: Designed weight ratios
        
        返回:
        Returns:
            alignment_score: 对齐分数，范围[0,1]，1表示完全对齐
            alignment_score: Alignment score, range [0,1], 1 means perfect alignment
        """
        if len(actual_ratios) != len(designed_ratios):
            return 0.0
        
        # 计算余弦相似度作为对齐分数
        # Calculate cosine similarity as alignment score
        actual_tensor = torch.tensor(actual_ratios)
        designed_tensor = torch.tensor(designed_ratios)
        
        # 归一化
        # Normalize
        actual_norm = actual_tensor / (torch.norm(actual_tensor, p=2) + 1e-8)
        designed_norm = designed_tensor / (torch.norm(designed_tensor, p=2) + 1e-8)
        
        # 计算余弦相似度
        # Calculate cosine similarity
        alignment = torch.dot(actual_norm, designed_norm).item()
        
        # 映射到[0,1]范围
        # Map to [0,1] range
        alignment = max(0.0, min(1.0, alignment))
        
        return alignment
    
    def get_fusion_stats(self,
                        Q_deep: torch.Tensor,
                        Q_current: torch.Tensor,
                        Q_original: torch.Tensor = None) -> dict:
        """
        获取融合统计信息 | Get Fusion Statistics
        
        返回:
        Returns:
            fusion_stats: 融合过程统计信息
            fusion_stats: Fusion process statistics
        """
        # 分析贡献度
        # Analyze contribution
        contribution = self.analyze_fusion_contribution(Q_deep, Q_current, Q_original)
        
        # 计算其他统计
        # Calculate other statistics
        batch_size, seq_len, dim = Q_current.shape
        
        # 扩展前的深层状态形状
        # Deep state shape before expansion
        deep_shape_before = Q_deep.shape
        
        # 扩展后的深层状态
        # Expanded deep state
        Q_deep_expanded = self._expand_deep_state(Q_deep, batch_size, seq_len)
        
        # 计算变化量
        # Calculate changes
        with torch.no_grad():
            # 深层与当前的差异
            # Difference between deep and current
            deep_current_diff = torch.norm(Q_deep_expanded - Q_current, p=2).item()
            
            # 融合后的变化（相对于当前）
            # Change after fusion (relative to current)
            Q_fused = self.apply_single_fusion(Q_deep, Q_current, Q_original, 'Q')
            fusion_change = torch.norm(Q_fused - Q_current, p=2).item()
            
            # 计算相对变化率
            # Calculate relative change rate
            current_norm = torch.norm(Q_current, p=2).item()
            if current_norm > 0:
                fusion_change_ratio = fusion_change / current_norm
            else:
                fusion_change_ratio = 0.0
        
        return {
            'dimension_transformation': {
                'deep_state_shape_before': list(deep_shape_before),
                'deep_state_shape_after': list(Q_deep_expanded.shape),
                'current_state_shape': list(Q_current.shape),
            },
            'weight_configuration': {
                'Q': self.q_weights.cpu().numpy().tolist(),
                'K': self.k_weights.cpu().numpy().tolist(),
                'V': self.v_weights.cpu().numpy().tolist(),
            },
            'change_metrics': {
                'deep_current_difference': deep_current_diff,
                'fusion_change': fusion_change,
                'fusion_change_ratio': fusion_change_ratio,
            },
            'contribution_analysis': contribution,
            'fusion_effectiveness': self._assess_fusion_effectiveness(contribution, fusion_change_ratio),
        }
    
    def _assess_fusion_effectiveness(self, contribution: dict, change_ratio: float) -> str:
        """
        评估融合效果 | Assess Fusion Effectiveness
        
        参数:
        Parameters:
            contribution: 贡献度分析结果
            change_ratio: 融合变化率
            
            contribution: Contribution analysis results
            change_ratio: Fusion change ratio
        
        返回:
        Returns:
            effectiveness: 效果评估描述
            effectiveness: Effectiveness assessment description
        """
        alignment = contribution['weight_design']['alignment']
        
        if alignment > 0.9 and change_ratio > 0.1:
            return "优秀融合：权重对齐好，变化显著 | Excellent fusion: good weight alignment, significant change"
        elif alignment > 0.7 and change_ratio > 0.05:
            return "良好融合：权重对齐较好，有适度变化 | Good fusion: decent weight alignment, moderate change"
        elif alignment > 0.5:
            return "一般融合：权重对齐一般，变化较小 | Fair fusion: average weight alignment, small change"
        else:
            return "需要改进：权重对齐差或变化不足 | Needs improvement: poor weight alignment or insufficient change"
    
    def enable_debug_mode(self, enabled: bool = True):
        """
        启用调试模式 | Enable Debug Mode
        
        参数:
        Parameters:
            enabled: 是否启用调试模式
            enabled: Whether to enable debug mode
        """
        self._debug = enabled
        print(f"调试模式{'启用' if enabled else '禁用'} | Debug mode {'enabled' if enabled else 'disabled'}")


# ==================== 工厂函数 ====================
# ==================== Factory Functions ====================

def create_sandwich_fusion(**kwargs) -> SandwichFusion:
    """
    创建三明治融合工厂函数 | Sandwich Fusion Factory Function
    
    返回:
    Returns:
        fusion: 三明治融合实例 | Sandwich fusion instance
    """
    return SandwichFusion(**kwargs)


# ==================== 测试函数 ====================
# ==================== Test Functions ====================

def test_sandwich_fusion():
    """
    测试三明治融合功能 | Test Sandwich Fusion Functionality
    """
    print("🧪 测试三明治融合... | Testing sandwich fusion...")
    
    # 创建融合层实例
    # Create fusion layer instance
    fusion = SandwichFusion()
    
    # 启用调试模式
    # Enable debug mode
    fusion.enable_debug_mode(True)
    
    # 创建测试数据
    # Create test data
    batch_size = 2
    seq_len = 10
    dim = 64
    
    # 当前状态（标准形状）
    # Current states (standard shape)
    Q_current = torch.randn(batch_size, seq_len, dim)
    K_current = torch.randn(batch_size, seq_len, dim)
    V_current = torch.randn(batch_size, seq_len, dim)
    
    # 原始状态（可选）
    # Original states (optional)
    Q_original = torch.randn(batch_size, seq_len, dim) * 0.5
    K_original = torch.randn(batch_size, seq_len, dim) * 0.5
    V_original = torch.randn(batch_size, seq_len, dim) * 0.5
    
    # 测试不同形状的深层状态
    # Test deep states with different shapes
    test_cases = [
        ("[dim]", torch.randn(dim)),  # 形状1: [dim]
        ("[1, dim]", torch.randn(1, dim)),  # 形状2: [1, dim]
        ("[1, 1, dim]", torch.randn(1, 1, dim)),  # 形状3: [1, 1, dim]
        ("[1, seq, dim]", torch.randn(1, seq_len, dim)),  # 形状4: [1, seq, dim]
    ]
    
    for shape_desc, Q_deep in test_cases:
        print(f"\n测试深层状态形状: {shape_desc} | Testing deep state shape: {shape_desc}")
        
        # 创建对应的K和V深层状态
        # Create corresponding K and V deep states
        if shape_desc == "[dim]":
            K_deep = torch.randn(dim)
            V_deep = torch.randn(dim)
        elif shape_desc == "[1, dim]":
            K_deep = torch.randn(1, dim)
            V_deep = torch.randn(1, dim)
        elif shape_desc == "[1, 1, dim]":
            K_deep = torch.randn(1, 1, dim)
            V_deep = torch.randn(1, 1, dim)
        else:  # [1, seq, dim]
            K_deep = torch.randn(1, seq_len, dim)
            V_deep = torch.randn(1, seq_len, dim)
        
        # 执行融合
        # Execute fusion
        Q_final, K_final, V_final = fusion(
            Q_deep, K_deep, V_deep,
            Q_current, K_current, V_current,
            Q_original, K_original, V_original
        )
        
        # 验证输出形状
        # Verify output shapes
        expected_shape = (batch_size, seq_len, dim)
        assert Q_final.shape == expected_shape, f"Q_final形状错误: {Q_final.shape}"
        assert K_final.shape == expected_shape, f"K_final形状错误: {K_final.shape}"
        assert V_final.shape == expected_shape, f"V_final形状错误: {V_final.shape}"
        
        print(f"✅ 形状验证通过: {Q_final.shape}")
        
        # 测试单个值融合
        # Test single value fusion
        print("测试单个值融合... | Testing single value fusion...")
        Q_single_fused = fusion.apply_single_fusion(Q_deep, Q_current, Q_original, 'Q')
        assert Q_single_fused.shape == expected_shape, f"单个融合形状错误: {Q_single_fused.shape}"
        
        # 验证与完整融合的一致性
        # Verify consistency with full fusion
        assert torch.allclose(Q_single_fused, Q_final, rtol=1e-5), "单个融合与完整融合不一致"
        print("✅ 单个融合验证通过")
        
        # 分析融合统计
        # Analyze fusion statistics
        print("分析融合统计... | Analyzing fusion statistics...")
        fusion_stats = fusion.get_fusion_stats(Q_deep, Q_current, Q_original)
        
        print(f"   深层状态扩展: {fusion_stats['dimension_transformation']['deep_state_shape_before']} → "
              f"{fusion_stats['dimension_transformation']['deep_state_shape_after']}")
        print(f"   权重对齐分数: {fusion_stats['contribution_analysis']['weight_design']['alignment']:.3f}")
        print(f"   融合效果: {fusion_stats['fusion_effectiveness']}")
    
    # 测试无原始状态的情况
    # Test case without original states
    print("\n测试无原始状态... | Testing without original states...")
    Q_final_no_orig, K_final_no_orig, V_final_no_orig = fusion(
        torch.randn(dim), torch.randn(dim), torch.randn(dim),
        Q_current, K_current, V_current
    )
    
    assert Q_final_no_orig.shape == expected_shape, "无原始状态融合形状错误"
    print("✅ 无原始状态测试通过")
    
    # 禁用调试模式
    # Disable debug mode
    fusion.enable_debug_mode(False)
    
    print("\n" + "="*60)
    print("✅ 所有三明治融合测试通过 | All sandwich fusion tests passed")
    print("="*60)
    
    return True

__all__ = [
    "SandwichFusion",
    "create_sandwich_fusion",
    "test_sandwich_fusion",
]

# ==================== 模块主入口 ====================
# ==================== Module Main Entry ====================

if __name__ == "__main__":
    # 当直接运行此文件时，执行测试
    # Execute tests when this file is run directly
    test_sandwich_fusion()