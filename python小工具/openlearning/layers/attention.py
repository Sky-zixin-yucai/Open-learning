"""
注意力层模块 | Attention Layers Module
====================================

规则治理架构的注意力层实现，包含多种处理Q、K、V值的子网络。
Attention layer implementations for Rule-Governed Architecture, including multiple subnetworks for processing Q, K, V values.

设计理念:
Design Philosophy:
• 多路径处理：三种不同的处理顺序 (V→K→Q, Q→V→K, K→Q→V) | Multi-path processing: Three different processing orders (V→K→Q, Q→V→K, K→Q→V)
• 固定归一化：使用FixedRMSNorm确保数值稳定性 | Fixed normalization: Use FixedRMSNorm to ensure numerical stability
• 残差连接：防止梯度消失，促进深层训练 | Residual connections: Prevent gradient vanishing, facilitate deep training
• 链式反应：多单元协同工作，形成认知链 | Chain reaction: Multiple units work together to form cognitive chains

核心组件 | Core Components:
1. VKQ_SubNet_WithFixedNorm: V→K→Q顺序处理的子网络 | Subnetwork with V→K→Q sequential processing
2. QVK_SubNet_WithFixedNorm: Q→V→K顺序处理的子网络 | Subnetwork with Q→V→K sequential processing
3. KQV_SubNet_WithFixedNorm: K→Q→V顺序处理的子网络 | Subnetwork with K→Q→V sequential processing
4. ChainReactionUnit_Final: 完整链式反应单元 | Complete chain reaction unit

架构特点 | Architecture Features:
• 顺序依赖：后一阶段的输入依赖于前一阶段的输出 | Sequential dependency: Input of later stage depends on output of previous stage
• 信息流动：模拟认知过程中的信息传播 | Information flow: Simulates information propagation in cognitive processes
• 状态演化：逐步演化Q、K、V状态 | State evolution: Gradually evolves Q, K, V states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union
# 尝试导入FixedRMSNorm模块 | Attempt to import FixedRMSNorm module
try:
    from normalization import FixedRMSNorm
except ImportError:
    # 如果直接导入失败，尝试相对导入
    # If direct import fails, try relative import
    try:
        from .normalization import FixedRMSNorm
    except ImportError as e:
        raise ImportError(f"Failed to import core modules: {e}")


class VKQ_SubNet_WithFixedNorm(nn.Module):
    """
    V→K→Q顺序处理子网络 | V→K→Q Sequential Processing Subnetwork
    ------------------------------------------------------------
    
    严格按照V→K→Q顺序处理输入的Q、K、V值。
    Strictly processes input Q, K, V values in V→K→Q order.
    
    处理流程 | Processing Flow:
        1. 第一阶段：处理V值，使用原始Q、K、V | Stage 1: Process V value using original Q, K, V
        2. 第二阶段：处理K值，使用原始Q、K和更新后的V | Stage 2: Process K value using original Q, K and updated V
        3. 第三阶段：处理Q值，使用原始Q、更新后的K和V | Stage 3: Process Q value using original Q, updated K and V
    
    设计原理 | Design Principles:
        • V值主导：首先处理V值，体现V值主导原则 | V-dominant: Process V value first, embodying V-dominant principle
        • 信息传递：将处理后的值传递到下一阶段 | Information transfer: Pass processed values to next stage
        • 残差学习：每个阶段都使用残差连接 | Residual learning: Each stage uses residual connections
    
    数学表示 | Mathematical Representation:
        V' = V + α_V * f_V(Q, K, V)
        K' = K + α_K * f_K(Q, K, V')
        Q' = Q + α_Q * f_Q(Q, K', V')
    
    其中f为处理函数，α为残差权重 | Where f is processing function, α is residual weight.
    
    参数 | Parameters:
        dim (int): 输入特征维度 | Input feature dimension
    
    输入 | Input:
        Q (torch.Tensor): Query张量，形状为[B, S, D] | Query tensor with shape [B, S, D]
        K (torch.Tensor): Key张量，形状为[B, S, D] | Key tensor with shape [B, S, D]
        V (torch.Tensor): Value张量，形状为[B, S, D] | Value tensor with shape [B, S, D]
    
    输出 | Output:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 处理后的Q、K、V | Processed Q, K, V
    
    示例 | Example:
        >>> subnet = VKQ_SubNet_WithFixedNorm(dim=128)
        >>> Q = torch.randn(4, 32, 128)
        >>> K = torch.randn(4, 32, 128)
        >>> V = torch.randn(4, 32, 128)
        >>> Q_out, K_out, V_out = subnet(Q, K, V)
        >>> print(f"输出形状: Q={Q_out.shape}, K={K_out.shape}, V={V_out.shape}")
    """
    
    def __init__(self, dim: int):
        """
        初始化V→K→Q子网络 | Initialize V→K→Q Subnetwork
        
        参数 | Parameters:
            dim (int): 输入特征维度 | Input feature dimension
        """
        super().__init__()
        self.dim = dim
        
        # ==================== V处理模块 ====================
        # V Processing Module
        self.V_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),  # 融合Q、K、V信息 | Fuse Q, K, V information
            FixedRMSNorm(dim * 2),        # ✅ 固定归一化 | Fixed normalization
            nn.GELU(),                    # 非线性激活 | Non-linear activation
            nn.Linear(dim * 2, dim),      # 投影回原始维度 | Project back to original dimension
            FixedRMSNorm(dim),            # ✅ 固定归一化 | Fixed normalization
            nn.Dropout(0.1)               # 防止过拟合 | Prevent overfitting
        )
        
        # ==================== K处理模块 ====================
        # K Processing Module
        self.K_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),  # 接收V'的输出 | Receives output of V'
            FixedRMSNorm(dim * 2),        # ✅ 固定归一化 | Fixed normalization
            nn.GELU(),                    # 非线性激活 | Non-linear activation
            nn.Linear(dim * 2, dim),      # 投影回原始维度 | Project back to original dimension
            FixedRMSNorm(dim),            # ✅ 固定归一化 | Fixed normalization
            nn.Dropout(0.1)               # 防止过拟合 | Prevent overfitting
        )
        
        # ==================== Q处理模块 ====================
        # Q Processing Module
        self.Q_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),  # 接收V'和K'的输出 | Receives outputs of V' and K'
            FixedRMSNorm(dim * 2),        # ✅ 固定归一化 | Fixed normalization
            nn.GELU(),                    # 非线性激活 | Non-linear activation
            nn.Linear(dim * 2, dim),      # 投影回原始维度 | Project back to original dimension
            FixedRMSNorm(dim),            # ✅ 固定归一化 | Fixed normalization
            nn.Dropout(0.1)               # 防止过拟合 | Prevent overfitting
        )
        
        # ==================== 残差权重 ====================
        # Residual Weights
        self.res_weight_V = nn.Parameter(torch.tensor(0.1))  # V残差权重 | V residual weight
        self.res_weight_K = nn.Parameter(torch.tensor(0.1))  # K残差权重 | K residual weight
        self.res_weight_Q = nn.Parameter(torch.tensor(0.1))  # Q残差权重 | Q residual weight
        
        # ==================== 层标识 ====================
        # Layer Identification
        self.register_buffer('layer_id', torch.tensor(1))  # 标识此为VKQ处理顺序 | Identifies this as VKQ processing order
    
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：严格按照V→K→Q顺序处理 | Forward Propagation: Strictly process in V→K→Q order
        
        参数 | Parameters:
            Q (torch.Tensor): Query张量，形状为[B, S, D] | Query tensor with shape [B, S, D]
            K (torch.Tensor): Key张量，形状为[B, S, D] | Key tensor with shape [B, S, D]
            V (torch.Tensor): Value张量，形状为[B, S, D] | Value tensor with shape [B, S, D]
        
        返回 | Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 处理后的Q、K、V | Processed Q, K, V
        
        处理步骤 | Processing Steps:
            1. 展平张量以便处理 | Flatten tensors for processing
            2. 阶段1：处理V值 | Stage 1: Process V value
            3. 阶段2：处理K值 | Stage 2: Process K value
            4. 阶段3：处理Q值 | Stage 3: Process Q value
            5. 恢复原始形状 | Restore original shape
        """
        batch_size, seq_len, dim = Q.shape
        
        # ==================== 展平处理 ====================
        # Flatten for Processing
        Q_flat = Q.view(batch_size * seq_len, dim)
        K_flat = K.view(batch_size * seq_len, dim)
        V_flat = V.view(batch_size * seq_len, dim)
        
        # ==================== 阶段1: V处理 ====================
        # Stage 1: V Processing
        combined_V = torch.cat([Q_flat, K_flat, V_flat], dim=-1)  # 连接Q、K、V | Concatenate Q, K, V
        delta_V = self.V_processing(combined_V)                    # 处理变化量 | Process delta
        V_new = V_flat + self.res_weight_V * delta_V              # 残差连接 | Residual connection
        
        # ==================== 阶段2: K处理（使用新的V） ====================
        # Stage 2: K Processing (using new V)
        combined_K = torch.cat([Q_flat, K_flat, V_new], dim=-1)   # 使用V_new | Use V_new
        delta_K = self.K_processing(combined_K)                   # 处理变化量 | Process delta
        K_new = K_flat + self.res_weight_K * delta_K              # 残差连接 | Residual connection
        
        # ==================== 阶段3: Q处理（使用新的V和K） ====================
        # Stage 3: Q Processing (using new V and K)
        combined_Q = torch.cat([Q_flat, K_new, V_new], dim=-1)    # 使用K_new和V_new | Use K_new and V_new
        delta_Q = self.Q_processing(combined_Q)                   # 处理变化量 | Process delta
        Q_new = Q_flat + self.res_weight_Q * delta_Q              # 残差连接 | Residual connection
        
        # ==================== 恢复原始形状 ====================
        # Restore Original Shape
        Q_out = Q_new.view(batch_size, seq_len, dim)
        K_out = K_new.view(batch_size, seq_len, dim)
        V_out = V_new.view(batch_size, seq_len, dim)
        
        return Q_out, K_out, V_out
    
    def get_config(self) -> Dict:
        """
        获取子网络配置 | Get Subnetwork Configuration
        
        返回 | Returns:
            Dict: 配置字典 | Configuration dictionary
        """
        return {
            'type': 'VKQ_SubNet_WithFixedNorm',
            'dim': self.dim,
            'layer_id': self.layer_id.item(),
            'residual_weights': {
                'V': self.res_weight_V.item(),
                'K': self.res_weight_K.item(),
                'Q': self.res_weight_Q.item(),
            }
        }
    
    def extra_repr(self) -> str:
        """
        额外的字符串表示 | Extra String Representation
        
        返回 | Returns:
            str: 描述子网络参数的字符串 | String describing subnetwork parameters
        """
        return f'V→K→Q顺序, dim={self.dim}, layer_id={self.layer_id.item()}'


class QVK_SubNet_WithFixedNorm(nn.Module):
    """
    Q→V→K顺序处理子网络 | Q→V→K Sequential Processing Subnetwork
    ------------------------------------------------------------
    
    严格按照Q→V→K顺序处理输入的Q、K、V值。
    Strictly processes input Q, K, V values in Q→V→K order.
    
    处理流程 | Processing Flow:
        1. 第一阶段：处理Q值，使用原始Q、K、V | Stage 1: Process Q value using original Q, K, V
        2. 第二阶段：处理V值，使用更新后的Q、原始K、V | Stage 2: Process V value using updated Q, original K, V
        3. 第三阶段：处理K值，使用更新后的Q、V和原始K | Stage 3: Process K value using updated Q, V and original K
    
    设计原理 | Design Principles:
        • Q值优先：首先处理Q值，体现查询驱动原则 | Q-priority: Process Q value first, embodying query-driven principle
        • 逐步传播：Q值变化影响V值，再影响K值 | Gradual propagation: Q changes affect V, then K
        • 对称结构：与VKQ子网络形成对称 | Symmetric structure: Forms symmetry with VKQ subnetwork
    
    数学表示 | Mathematical Representation:
        Q' = Q + α_Q * f_Q(Q, K, V)
        V' = V + α_V * f_V(Q', K, V)
        K' = K + α_K * f_K(Q', K, V')
    
    参数 | Parameters:
        dim (int): 输入特征维度 | Input feature dimension
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # ==================== Q处理模块 ====================
        # Q Processing Module
        self.Q_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            FixedRMSNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            FixedRMSNorm(dim),
            nn.Dropout(0.1)
        )
        
        # ==================== V处理模块 ====================
        # V Processing Module
        self.V_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            FixedRMSNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            FixedRMSNorm(dim),
            nn.Dropout(0.1)
        )
        
        # ==================== K处理模块 ====================
        # K Processing Module
        self.K_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            FixedRMSNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            FixedRMSNorm(dim),
            nn.Dropout(0.1)
        )
        
        # ==================== 残差权重 ====================
        # Residual Weights
        self.res_weight_Q = nn.Parameter(torch.tensor(0.1))
        self.res_weight_V = nn.Parameter(torch.tensor(0.1))
        self.res_weight_K = nn.Parameter(torch.tensor(0.1))
        
        # ==================== 层标识 ====================
        # Layer Identification
        self.register_buffer('layer_id', torch.tensor(2))
    
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：严格按照Q→V→K顺序处理 | Forward Propagation: Strictly process in Q→V→K order
        """
        batch_size, seq_len, dim = Q.shape
        
        # 展平处理 | Flatten for processing
        Q_flat = Q.view(batch_size * seq_len, dim)
        K_flat = K.view(batch_size * seq_len, dim)
        V_flat = V.view(batch_size * seq_len, dim)
        
        # === 阶段1: Q处理 === | Stage 1: Q Processing
        combined_Q = torch.cat([Q_flat, K_flat, V_flat], dim=-1)
        delta_Q = self.Q_processing(combined_Q)
        Q_new = Q_flat + self.res_weight_Q * delta_Q
        
        # === 阶段2: V处理（使用新的Q） === | Stage 2: V Processing (using new Q)
        combined_V = torch.cat([Q_new, K_flat, V_flat], dim=-1)
        delta_V = self.V_processing(combined_V)
        V_new = V_flat + self.res_weight_V * delta_V
        
        # === 阶段3: K处理（使用新的Q和V） === | Stage 3: K Processing (using new Q and V)
        combined_K = torch.cat([Q_new, K_flat, V_new], dim=-1)
        delta_K = self.K_processing(combined_K)
        K_new = K_flat + self.res_weight_K * delta_K
        
        # 恢复原始形状 | Restore original shape
        Q_out = Q_new.view(batch_size, seq_len, dim)
        K_out = K_new.view(batch_size, seq_len, dim)
        V_out = V_new.view(batch_size, seq_len, dim)
        
        return Q_out, K_out, V_out
    
    def get_config(self) -> Dict:
        """
        获取子网络配置 | Get Subnetwork Configuration
        """
        return {
            'type': 'QVK_SubNet_WithFixedNorm',
            'dim': self.dim,
            'layer_id': self.layer_id.item(),
            'residual_weights': {
                'Q': self.res_weight_Q.item(),
                'V': self.res_weight_V.item(),
                'K': self.res_weight_K.item(),
            }
        }
    
    def extra_repr(self) -> str:
        """
        额外的字符串表示 | Extra String Representation
        """
        return f'Q→V→K顺序, dim={self.dim}, layer_id={self.layer_id.item()}'


class KQV_SubNet_WithFixedNorm(nn.Module):
    """
    K→Q→V顺序处理子网络 | K→Q→V Sequential Processing Subnetwork
    ------------------------------------------------------------
    
    严格按照K→Q→V顺序处理输入的Q、K、V值。
    Strictly processes input Q, K, V values in K→Q→V order.
    
    处理流程 | Processing Flow:
        1. 第一阶段：处理K值，使用原始Q、K、V | Stage 1: Process K value using original Q, K, V
        2. 第二阶段：处理Q值，使用原始Q、更新后的K、V | Stage 2: Process Q value using original Q, updated K, V
        3. 第三阶段：处理V值，使用更新后的Q、K和原始V | Stage 3: Process V value using updated Q, K and original V
    
    设计原理 | Design Principles:
        • K值优先：首先处理K值，体现关键信息优先原则 | K-priority: Process K value first, embodying key information priority
        • 信息中介：K值作为Q和V之间的桥梁 | Information mediation: K acts as bridge between Q and V
        • 完整覆盖：三种顺序覆盖所有可能性 | Complete coverage: Three orders cover all possibilities
    
    数学表示 | Mathematical Representation:
        K' = K + α_K * f_K(Q, K, V)
        Q' = Q + α_Q * f_Q(Q, K', V)
        V' = V + α_V * f_V(Q', K', V)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # ==================== K处理模块 ====================
        # K Processing Module
        self.K_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            FixedRMSNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            FixedRMSNorm(dim),
            nn.Dropout(0.1)
        )
        
        # ==================== Q处理模块 ====================
        # Q Processing Module
        self.Q_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            FixedRMSNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            FixedRMSNorm(dim),
            nn.Dropout(0.1)
        )
        
        # ==================== V处理模块 ====================
        # V Processing Module
        self.V_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            FixedRMSNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            FixedRMSNorm(dim),
            nn.Dropout(0.1)
        )
        
        # ==================== 残差权重 ====================
        # Residual Weights
        self.res_weight_K = nn.Parameter(torch.tensor(0.1))
        self.res_weight_Q = nn.Parameter(torch.tensor(0.1))
        self.res_weight_V = nn.Parameter(torch.tensor(0.1))
        
        # ==================== 层标识 ====================
        # Layer Identification
        self.register_buffer('layer_id', torch.tensor(3))
    
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：严格按照K→Q→V顺序处理 | Forward Propagation: Strictly process in K→Q→V order
        """
        batch_size, seq_len, dim = Q.shape
        
        # 展平处理 | Flatten for processing
        Q_flat = Q.view(batch_size * seq_len, dim)
        K_flat = K.view(batch_size * seq_len, dim)
        V_flat = V.view(batch_size * seq_len, dim)
        
        # === 阶段1: K处理 === | Stage 1: K Processing
        combined_K = torch.cat([Q_flat, K_flat, V_flat], dim=-1)
        delta_K = self.K_processing(combined_K)
        K_new = K_flat + self.res_weight_K * delta_K
        
        # === 阶段2: Q处理（使用新的K） === | Stage 2: Q Processing (using new K)
        combined_Q = torch.cat([Q_flat, K_new, V_flat], dim=-1)
        delta_Q = self.Q_processing(combined_Q)
        Q_new = Q_flat + self.res_weight_Q * delta_Q
        
        # === 阶段3: V处理（使用新的K和Q） === | Stage 3: V Processing (using new K and Q)
        combined_V = torch.cat([Q_new, K_new, V_flat], dim=-1)
        delta_V = self.V_processing(combined_V)
        V_new = V_flat + self.res_weight_V * delta_V
        
        # 恢复原始形状 | Restore original shape
        Q_out = Q_new.view(batch_size, seq_len, dim)
        K_out = K_new.view(batch_size, seq_len, dim)
        V_out = V_new.view(batch_size, seq_len, dim)
        
        return Q_out, K_out, V_out
    
    def get_config(self) -> Dict:
        """
        获取子网络配置 | Get Subnetwork Configuration
        """
        return {
            'type': 'KQV_SubNet_WithFixedNorm',
            'dim': self.dim,
            'layer_id': self.layer_id.item(),
            'residual_weights': {
                'K': self.res_weight_K.item(),
                'Q': self.res_weight_Q.item(),
                'V': self.res_weight_V.item(),
            }
        }
    
    def extra_repr(self) -> str:
        """
        额外的字符串表示 | Extra String Representation
        """
        return f'K→Q→V顺序, dim={self.dim}, layer_id={self.layer_id.item()}'


class ChainReactionUnit_Final(nn.Module):
    """
    完整链式反应单元 | Complete Chain Reaction Unit
    --------------------------------------------
    
    集成三个子网络(VKQ, QVK, KQV)的链式反应单元，实现多层次认知处理。
    Chain reaction unit integrating three subnetworks (VKQ, QVK, KQV), implementing multi-level cognitive processing.
    
    架构设计 | Architecture Design:
        • 串联结构：三个子网络依次处理 | Series structure: Three subnetworks process sequentially
        • 状态演化：Q、K、V状态逐步演化 | State evolution: Q, K, V states evolve gradually
        • 权重融合：不同路径加权融合 | Weighted fusion: Different paths are weighted and fused
        • 残差保留：保留输入信息 | Residual preservation: Preserve input information
    
    处理流程 | Processing Flow:
        输入: Q⁽⁰⁾, K⁽⁰⁾, V⁽⁰⁾
        ↓
        子网络1: Q⁽¹⁾, K⁽¹⁾, V⁽¹⁾ = VKQ_SubNet(Q⁽⁰⁾, K⁽⁰⁾, V⁽⁰⁾)
        ↓
        子网络2: Q⁽²⁾, K⁽²⁾, V⁽²⁾ = QVK_SubNet(Q⁽¹⁾, K⁽¹⁾, V⁽¹⁾)
        ↓
        子网络3: Q⁽³⁾, K⁽³⁾, V⁽³⁾ = KQV_SubNet(Q⁽²⁾, K⁽²⁾, V⁽²⁾)
        ↓
        权重融合: Q_out = w_Q * Q⁽³⁾, K_out = w_K * K⁽³⁾, V_out = w_V * V⁽³⁾
        ↓
        残差连接: 输出 = 融合结果 + α * 输入
    
    特点 | Features:
        • 多层次处理：三个子网络实现多层次认知 | Multi-level processing: Three subnetworks achieve multi-level cognition
        • V值主导：V权重最大，体现V值主导原则 | V-dominant: V has largest weight, embodying V-dominant principle
        • 状态监控：记录V值演化历史 | State monitoring: Record V value evolution history
        • 单元标识：每个单元有唯一标识 | Unit identification: Each unit has unique identifier
    """
    
    def __init__(self, dim: int, unit_id: int):
        """
        初始化链式反应单元 | Initialize Chain Reaction Unit
        
        参数 | Parameters:
            dim (int): 输入特征维度 | Input feature dimension
            unit_id (int): 单元标识符 | Unit identifier
        """
        super().__init__()
        self.dim = dim
        self.unit_id = unit_id
        
        # ==================== 三个子网络 ====================
        # Three Subnetworks
        self.subnet1 = VKQ_SubNet_WithFixedNorm(dim)  # V→K→Q
        self.subnet2 = QVK_SubNet_WithFixedNorm(dim)  # Q→V→K
        self.subnet3 = KQV_SubNet_WithFixedNorm(dim)  # K→Q→V
        
        # ==================== 权重参数 ====================
        # Weight Parameters
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))      # 残差连接权重 | Residual connection weight
        self.V_dominant_weight = nn.Parameter(torch.tensor(1.5))   # V主导权重 | V dominant weight
        self.K_weight = nn.Parameter(torch.tensor(1.0))            # K权重 | K weight
        self.Q_weight = nn.Parameter(torch.tensor(1.0))            # Q权重 | Q weight
        
        # ==================== V值历史记录 ====================
        # V Value History Recording
        self.register_buffer('V_history', torch.zeros(3))  # 记录三个阶段的V均值 | Record V means of three stages
        
        # ==================== 单元标识 ====================
        # Unit Identification
        self.register_buffer('unit_id_tensor', torch.tensor(unit_id))
    
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor,
                return_evolution: bool = False) -> Tuple:
        """
        前向传播：依次通过三个子网络 | Forward Propagation: Pass through three subnetworks sequentially
        
        参数 | Parameters:
            Q (torch.Tensor): Query张量 | Query tensor
            K (torch.Tensor): Key张量 | Key tensor
            V (torch.Tensor): Value张量 | Value tensor
            return_evolution (bool): 是否返回演化信息 | Whether to return evolution information
        
        返回 | Returns:
            Tuple: 如果return_evolution=True，返回(Q, K, V, V_evolution)
                   否则返回(Q, K, V) | If return_evolution=True, return (Q, K, V, V_evolution)
                   Otherwise return (Q, K, V)
        """
        # 保存输入状态（用于残差连接） | Save input states (for residual connections)
        Q_in, K_in, V_in = Q.clone(), K.clone(), V.clone()
        
        # ==================== 阶段1: 通过第一个子网 (V→K→Q) ====================
        # Stage 1: Through first subnetwork (V→K→Q)
        Q11, K11, V11 = self.subnet1(Q, K, V)
        
        # ==================== 阶段2: 通过第二个子网 (Q→V→K) ====================
        # Stage 2: Through second subnetwork (Q→V→K)
        Q12, K12, V12 = self.subnet2(Q11, K11, V11)
        
        # ==================== 阶段3: 通过第三个子网 (K→Q→V) ====================
        # Stage 3: Through third subnetwork (K→Q→V)
        Q13, K13, V13 = self.subnet3(Q12, K12, V12)
        
        # ==================== 权重应用 ====================
        # Weight Application
        Q_out = Q13 * self.Q_weight
        K_out = K13 * self.K_weight
        V_out = V13 * self.V_dominant_weight  # V值主导 | V value dominant
        
        # ==================== 单元级残差连接 ====================
        # Unit-level Residual Connection
        Q_out = Q_out + self.residual_alpha * Q_in
        K_out = K_out + self.residual_alpha * K_in
        V_out = V_out + self.residual_alpha * V_in
        
        # ==================== 记录V值演化历史 ====================
        # Record V Value Evolution History
        if V11.numel() > 0 and V12.numel() > 0 and V13.numel() > 0:
            with torch.no_grad():
                self.V_history[0] = V11.mean().detach()
                self.V_history[1] = V12.mean().detach()
                self.V_history[2] = V13.mean().detach()
        
        # ==================== 返回结果 ====================
        # Return Results
        if return_evolution:
            # 返回详细演化信息 | Return detailed evolution information
            V_evolution = {
                'unit_id': self.unit_id,
                'layer_id': self.unit_id_tensor.item(),
                'V11_mean': V11.mean().item() if V11.numel() > 0 else 0.0,
                'V12_mean': V12.mean().item() if V12.numel() > 0 else 0.0,
                'V13_mean': V13.mean().item() if V13.numel() > 0 else 0.0,
                'V_history_0': self.V_history[0].item(),
                'V_history_1': self.V_history[1].item(),
                'V_history_2': self.V_history[2].item(),
                'V_dominant_weight': self.V_dominant_weight.item(),
                'Q_weight': self.Q_weight.item(),
                'K_weight': self.K_weight.item(),
                'residual_alpha': self.residual_alpha.item()
            }
            return Q_out, K_out, V_out, V_evolution
        else:
            return Q_out, K_out, V_out
    
    def get_unit_info(self) -> Dict:
        """
        获取单元信息 | Get Unit Information
        
        返回 | Returns:
            Dict: 单元信息字典 | Unit information dictionary
        """
        return {
            'unit_id': self.unit_id,
            'dim': self.dim,
            'V_dominant_weight': self.V_dominant_weight.item(),
            'Q_weight': self.Q_weight.item(),
            'K_weight': self.K_weight.item(),
            'residual_alpha': self.residual_alpha.item(),
            'V_history': self.V_history.tolist(),
            'subnet_configs': {
                'subnet1': self.subnet1.get_config(),
                'subnet2': self.subnet2.get_config(),
                'subnet3': self.subnet3.get_config(),
            }
        }
    
    def extra_repr(self) -> str:
        """
        额外的字符串表示 | Extra String Representation
        """
        return f'链式反应单元{self.unit_id}, dim={self.dim}'


# ==================== 工厂函数 ====================
# Factory Functions

def create_attention_subnet(subnet_type: str, dim: int) -> nn.Module:
    """
    创建注意力子网络 | Create Attention Subnetwork
    
    参数 | Parameters:
        subnet_type (str): 子网络类型，可选['vkq', 'qvk', 'kqv'] | Subnetwork type, options: ['vkq', 'qvk', 'kqv']
        dim (int): 输入特征维度 | Input feature dimension
    
    返回 | Returns:
        nn.Module: 创建的注意力子网络 | Created attention subnetwork
    
    异常 | Raises:
        ValueError: 如果子网络类型不支持 | If subnet type is not supported
    """
    subnet_map = {
        'vkq': VKQ_SubNet_WithFixedNorm,
        'qvk': QVK_SubNet_WithFixedNorm,
        'kqv': KQV_SubNet_WithFixedNorm,
    }
    
    if subnet_type not in subnet_map:
        raise ValueError(
            f"不支持的子网络类型: {subnet_type}\n"
            f"支持的类型: {list(subnet_map.keys())}"
        )
    
    return subnet_map[subnet_type](dim)


def create_chain_reaction_unit(dim: int, unit_id: int = 0) -> ChainReactionUnit_Final:
    """
    创建链式反应单元 | Create Chain Reaction Unit
    
    参数 | Parameters:
        dim (int): 输入特征维度 | Input feature dimension
        unit_id (int): 单元标识符，默认0 | Unit identifier, default 0
    
    返回 | Returns:
        ChainReactionUnit_Final: 链式反应单元实例 | Chain reaction unit instance
    """
    return ChainReactionUnit_Final(dim, unit_id)


# ==================== 测试函数 ====================
# Test Functions

def test_vkq_subnet():
    """测试VKQ子网络 | Test VKQ Subnetwork"""
    print("=" * 60)
    print("测试VKQ子网络 | Testing VKQ Subnetwork")
    print("=" * 60)
    
    # 创建测试数据 | Create test data
    batch_size, seq_len, dim = 2, 16, 64
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)
    
    # 创建子网络 | Create subnetwork
    subnet = VKQ_SubNet_WithFixedNorm(dim=dim)
    
    # 前向传播 | Forward propagation
    Q_out, K_out, V_out = subnet(Q, K, V)
    
    # 验证输出形状 | Verify output shapes
    print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"输出形状: Q={Q_out.shape}, K={K_out.shape}, V={V_out.shape}")
    
    # 验证处理顺序 | Verify processing order
    print(f"子网络类型: {subnet.__class__.__name__}")
    print(f"处理顺序: V→K→Q")
    
    # 获取配置 | Get configuration
    config = subnet.get_config()
    print(f"配置: {config}")
    
    # 检查参数 | Check parameters
    total_params = sum(p.numel() for p in subnet.parameters())
    print(f"总参数量: {total_params:,}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)
    
    return subnet


def test_qvk_subnet():
    """测试QVK子网络 | Test QVK Subnetwork"""
    print("=" * 60)
    print("测试QVK子网络 | Testing QVK Subnetwork")
    print("=" * 60)
    
    # 创建测试数据 | Create test data
    batch_size, seq_len, dim = 2, 16, 64
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)
    
    # 创建子网络 | Create subnetwork
    subnet = QVK_SubNet_WithFixedNorm(dim=dim)
    
    # 前向传播 | Forward propagation
    Q_out, K_out, V_out = subnet(Q, K, V)
    
    # 验证输出形状 | Verify output shapes
    print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"输出形状: Q={Q_out.shape}, K={K_out.shape}, V={V_out.shape}")
    
    # 验证处理顺序 | Verify processing order
    print(f"子网络类型: {subnet.__class__.__name__}")
    print(f"处理顺序: Q→V→K")
    
    # 获取配置 | Get configuration
    config = subnet.get_config()
    print(f"配置: {config}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)
    
    return subnet


def test_kqv_subnet():
    """测试KQV子网络 | Test KQV Subnetwork"""
    print("=" * 60)
    print("测试KQV子网络 | Testing KQV Subnetwork")
    print("=" * 60)
    
    # 创建测试数据 | Create test data
    batch_size, seq_len, dim = 2, 16, 64
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)
    
    # 创建子网络 | Create subnetwork
    subnet = KQV_SubNet_WithFixedNorm(dim=dim)
    
    # 前向传播 | Forward propagation
    Q_out, K_out, V_out = subnet(Q, K, V)
    
    # 验证输出形状 | Verify output shapes
    print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"输出形状: Q={Q_out.shape}, K={K_out.shape}, V={V_out.shape}")
    
    # 验证处理顺序 | Verify processing order
    print(f"子网络类型: {subnet.__class__.__name__}")
    print(f"处理顺序: K→Q→V")
    
    # 获取配置 | Get configuration
    config = subnet.get_config()
    print(f"配置: {config}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)
    
    return subnet


def test_chain_reaction_unit():
    """测试链式反应单元 | Test Chain Reaction Unit"""
    print("=" * 60)
    print("测试链式反应单元 | Testing Chain Reaction Unit")
    print("=" * 60)
    
    # 创建测试数据 | Create test data
    batch_size, seq_len, dim = 2, 16, 64
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)
    
    # 创建链式反应单元 | Create chain reaction unit
    unit = ChainReactionUnit_Final(dim=dim, unit_id=1)
    
    # 前向传播（不返回演化信息） | Forward propagation (without evolution info)
    Q_out1, K_out1, V_out1 = unit(Q, K, V, return_evolution=False)
    
    # 前向传播（返回演化信息） | Forward propagation (with evolution info)
    Q_out2, K_out2, V_out2, evolution_info = unit(Q, K, V, return_evolution=True)
    
    # 验证输出形状 | Verify output shapes
    print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"输出形状（无演化）: Q={Q_out1.shape}, K={K_out1.shape}, V={V_out1.shape}")
    print(f"输出形状（有演化）: Q={Q_out2.shape}, K={K_out2.shape}, V={V_out2.shape}")
    
    # 验证一致性 | Verify consistency
    q_diff = torch.abs(Q_out1 - Q_out2).max().item()
    k_diff = torch.abs(K_out1 - K_out2).max().item()
    v_diff = torch.abs(V_out1 - V_out2).max().item()
    
    print(f"输出一致性检查:")
    print(f"  Q差异最大值: {q_diff:.6f} (应接近0)")
    print(f"  K差异最大值: {k_diff:.6f} (应接近0)")
    print(f"  V差异最大值: {v_diff:.6f} (应接近0)")
    
    # 显示演化信息 | Show evolution information
    print(f"\n演化信息 | Evolution Information:")
    for key, value in evolution_info.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
    
    # 获取单元信息 | Get unit information
    unit_info = unit.get_unit_info()
    print(f"\n单元信息 | Unit Information:")
    print(f"  单元ID: {unit_info['unit_id']}")
    print(f"  维度: {unit_info['dim']}")
    print(f"  V主导权重: {unit_info['V_dominant_weight']:.3f}")
    print(f"  V历史: {unit_info['V_history']}")
    
    # 检查参数 | Check parameters
    total_params = sum(p.numel() for p in unit.parameters())
    print(f"总参数量: {total_params:,}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)
    
    return unit


def test_attention_factory():
    """测试注意力工厂函数 | Test Attention Factory Functions"""
    print("=" * 60)
    print("测试注意力工厂函数 | Testing Attention Factory Functions")
    print("=" * 60)
    
    dim = 64
    
    # 测试创建子网络 | Test creating subnetworks
    for subnet_type in ['vkq', 'qvk', 'kqv']:
        try:
            subnet = create_attention_subnet(subnet_type, dim)
            print(f"成功创建 {subnet_type.upper()} 子网络 | Successfully created {subnet_type.upper()} subnetwork")
            print(f"  类型: {subnet.__class__.__name__}")
            print(f"  维度: {dim}")
            print(f"  参数量: {sum(p.numel() for p in subnet.parameters()):,}")
        except Exception as e:
            print(f"创建 {subnet_type.upper()} 子网络失败: {e}")
    
    # 测试创建链式反应单元 | Test creating chain reaction unit
    try:
        unit = create_chain_reaction_unit(dim, unit_id=5)
        print(f"\n成功创建链式反应单元 | Successfully created chain reaction unit")
        print(f"  单元ID: {unit.unit_id}")
        print(f"  维度: {unit.dim}")
        print(f"  参数量: {sum(p.numel() for p in unit.parameters()):,}")
    except Exception as e:
        print(f"创建链式反应单元失败: {e}")
    
    # 测试无效类型 | Test invalid type
    try:
        subnet = create_attention_subnet('invalid', dim)
    except ValueError as e:
        print(f"\n预期中的错误 | Expected error: {e}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)


# ==================== 导出列表 ====================
# Export List
# src/rga/layers/attention.py

__all__ = [
    # Subnetwork classes
    'VKQ_SubNet_WithFixedNorm',
    'QVK_SubNet_WithFixedNorm',
    'KQV_SubNet_WithFixedNorm',
    
    # Chain reaction unit
    'ChainReactionUnit_Final',
    
    # Factory functions
    'create_attention_subnet',
    'create_chain_reaction_unit',
    
    # Test functions
    'test_vkq_subnet',
    'test_qvk_subnet',
    'test_kqv_subnet',
    'test_chain_reaction_unit',
    'test_attention_factory',
]


# ==================== 模块自检 ====================
# Module Self-Test

if __name__ == "__main__":
    print("=" * 60)
    print("注意力层模块自检 | Attention Layers Module Self-Test")
    print("=" * 60)
    
    try:
        # 运行测试 | Run tests
        print("\n1. 测试VKQ子网络 | Testing VKQ Subnetwork")
        test_vkq_subnet()
        
        print("\n2. 测试QVK子网络 | Testing QVK Subnetwork")
        test_qvk_subnet()
        
        print("\n3. 测试KQV子网络 | Testing KQV Subnetwork")
        test_kqv_subnet()
        
        print("\n4. 测试链式反应单元 | Testing Chain Reaction Unit")
        test_chain_reaction_unit()
        
        print("\n5. 测试工厂函数 | Testing Factory Functions")
        test_attention_factory()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过 | All tests passed")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败 | Test failed: {e}")
        import traceback
        traceback.print_exc()