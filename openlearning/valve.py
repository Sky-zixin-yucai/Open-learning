"""
====================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class FixedRMSNorm(nn.Module):
    """固定RMSNorm - 无可学习参数"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # 🚫 没有可学习参数
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算RMS
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps
        return x / rms  # 固定缩放，无学习参数

class VKQ_SubNet_WithFixedNorm(nn.Module):
    """带固定归一化的子网络1: V→K→Q 顺序处理"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # V处理模块
        self.V_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), 
            FixedRMSNorm(dim * 2),      # ✅ 固定归一化
            nn.GELU(),
            nn.Linear(dim * 2, dim), 
            FixedRMSNorm(dim),          # ✅ 固定归一化
            nn.Dropout(0.1)
        )
        
        # K处理模块（接收V的输出）
        self.K_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), 
            FixedRMSNorm(dim * 2),      # ✅ 固定归一化
            nn.GELU(),
            nn.Linear(dim * 2, dim), 
            FixedRMSNorm(dim),          # ✅ 固定归一化
            nn.Dropout(0.1)
        )
        
        # Q处理模块（接收K的输出）
        self.Q_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), 
            FixedRMSNorm(dim * 2),      # ✅ 固定归一化
            nn.GELU(),
            nn.Linear(dim * 2, dim), 
            FixedRMSNorm(dim),          # ✅ 固定归一化
            nn.Dropout(0.1)
        )
        
        # 残差权重（可学习）
        self.res_weight_V = nn.Parameter(torch.tensor(0.1))
        self.res_weight_K = nn.Parameter(torch.tensor(0.1))
        self.res_weight_Q = nn.Parameter(torch.tensor(0.1))
        
        # 可选：保留层ID标记（用于连接层区分）
        self.register_buffer('layer_id', torch.tensor(1))
    
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        严格遵循 V→K→Q 顺序处理
        返回：处理后的 Q, K, V
        """
        batch_size, seq_len, dim = Q.shape
        
        # 展平处理
        Q_flat = Q.view(batch_size * seq_len, dim)
        K_flat = K.view(batch_size * seq_len, dim)
        V_flat = V.view(batch_size * seq_len, dim)
        
        # === 阶段1: V处理 ===
        combined_V = torch.cat([Q_flat, K_flat, V_flat], dim=-1)
        delta_V = self.V_processing(combined_V)
        V_new = V_flat + self.res_weight_V * delta_V
        
        # === 阶段2: K处理（使用新的V）===
        combined_K = torch.cat([Q_flat, K_flat, V_new], dim=-1)
        delta_K = self.K_processing(combined_K)
        K_new = K_flat + self.res_weight_K * delta_K
        
        # === 阶段3: Q处理（使用新的V和K）===
        combined_Q = torch.cat([Q_flat, K_new, V_new], dim=-1)
        delta_Q = self.Q_processing(combined_Q)
        Q_new = Q_flat + self.res_weight_Q * delta_Q
        
        # 恢复原始形状
        Q_out = Q_new.view(batch_size, seq_len, dim)
        K_out = K_new.view(batch_size, seq_len, dim)
        V_out = V_new.view(batch_size, seq_len, dim)
        
        return Q_out, K_out, V_out
    
class QVK_SubNet_WithFixedNorm(nn.Module):
    """带固定归一化的子网络2: Q→V→K 顺序处理"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Q处理模块
        self.Q_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), 
            FixedRMSNorm(dim * 2),      # ✅ 固定归一化
            nn.GELU(),
            nn.Linear(dim * 2, dim), 
            FixedRMSNorm(dim),          # ✅ 固定归一化
            nn.Dropout(0.1)
        )
        
        # V处理模块（接收Q的输出）
        self.V_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), 
            FixedRMSNorm(dim * 2),      # ✅ 固定归一化
            nn.GELU(),
            nn.Linear(dim * 2, dim), 
            FixedRMSNorm(dim),          # ✅ 固定归一化
            nn.Dropout(0.1)
        )
        
        # K处理模块（接收V的输出）
        self.K_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), 
            FixedRMSNorm(dim * 2),      # ✅ 固定归一化
            nn.GELU(),
            nn.Linear(dim * 2, dim), 
            FixedRMSNorm(dim),          # ✅ 固定归一化
            nn.Dropout(0.1)
        )
        
        # 残差权重（可学习）
        self.res_weight_Q = nn.Parameter(torch.tensor(0.1))
        self.res_weight_V = nn.Parameter(torch.tensor(0.1))
        self.res_weight_K = nn.Parameter(torch.tensor(0.1))
        
        # 可选：保留层ID标记（用于连接层区分）
        self.register_buffer('layer_id', torch.tensor(2))
    
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        严格遵循 Q→V→K 顺序处理
        返回：处理后的 Q, K, V
        """
        batch_size, seq_len, dim = Q.shape
        
        # 展平处理
        Q_flat = Q.view(batch_size * seq_len, dim)
        K_flat = K.view(batch_size * seq_len, dim)
        V_flat = V.view(batch_size * seq_len, dim)
        
        # === 阶段1: Q处理 ===
        combined_Q = torch.cat([Q_flat, K_flat, V_flat], dim=-1)
        delta_Q = self.Q_processing(combined_Q)
        Q_new = Q_flat + self.res_weight_Q * delta_Q
        
        # === 阶段2: V处理（使用新的Q）===
        combined_V = torch.cat([Q_new, K_flat, V_flat], dim=-1)
        delta_V = self.V_processing(combined_V)
        V_new = V_flat + self.res_weight_V * delta_V
        
        # === 阶段3: K处理（使用新的Q和V）===
        combined_K = torch.cat([Q_new, K_flat, V_new], dim=-1)
        delta_K = self.K_processing(combined_K)
        K_new = K_flat + self.res_weight_K * delta_K
        
        # 恢复原始形状
        Q_out = Q_new.view(batch_size, seq_len, dim)
        K_out = K_new.view(batch_size, seq_len, dim)
        V_out = V_new.view(batch_size, seq_len, dim)
        
        return Q_out, K_out, V_out
    
class KQV_SubNet_WithFixedNorm(nn.Module):
    """带固定归一化的子网络3: K→Q→V 顺序处理"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # K处理模块
        self.K_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), 
            FixedRMSNorm(dim * 2),      # ✅ 固定归一化
            nn.GELU(),
            nn.Linear(dim * 2, dim), 
            FixedRMSNorm(dim),          # ✅ 固定归一化
            nn.Dropout(0.1)
        )
        
        # Q处理模块（接收K的输出）
        self.Q_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), 
            FixedRMSNorm(dim * 2),      # ✅ 固定归一化
            nn.GELU(),
            nn.Linear(dim * 2, dim), 
            FixedRMSNorm(dim),          # ✅ 固定归一化
            nn.Dropout(0.1)
        )
        
        # V处理模块（接收Q的输出）
        self.V_processing = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), 
            FixedRMSNorm(dim * 2),      # ✅ 固定归一化
            nn.GELU(),
            nn.Linear(dim * 2, dim), 
            FixedRMSNorm(dim),          # ✅ 固定归一化
            nn.Dropout(0.1)
        )
        
        # 残差权重（可学习）
        self.res_weight_K = nn.Parameter(torch.tensor(0.1))
        self.res_weight_Q = nn.Parameter(torch.tensor(0.1))
        self.res_weight_V = nn.Parameter(torch.tensor(0.1))
        
        # 可选：保留层ID标记（用于连接层区分）
        self.register_buffer('layer_id', torch.tensor(3))
    
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        严格遵循 K→Q→V 顺序处理
        返回：处理后的 Q, K, V
        """
        batch_size, seq_len, dim = Q.shape
        
        # 展平处理
        Q_flat = Q.view(batch_size * seq_len, dim)
        K_flat = K.view(batch_size * seq_len, dim)
        V_flat = V.view(batch_size * seq_len, dim)
        
        # === 阶段1: K处理 ===
        combined_K = torch.cat([Q_flat, K_flat, V_flat], dim=-1)
        delta_K = self.K_processing(combined_K)
        K_new = K_flat + self.res_weight_K * delta_K
        
        # === 阶段2: Q处理（使用新的K）===
        combined_Q = torch.cat([Q_flat, K_new, V_flat], dim=-1)
        delta_Q = self.Q_processing(combined_Q)
        Q_new = Q_flat + self.res_weight_Q * delta_Q
        
        # === 阶段3: V处理（使用新的K和Q）===
        combined_V = torch.cat([Q_new, K_new, V_flat], dim=-1)
        delta_V = self.V_processing(combined_V)
        V_new = V_flat + self.res_weight_V * delta_V
        
        # 恢复原始形状
        Q_out = Q_new.view(batch_size, seq_len, dim)
        K_out = K_new.view(batch_size, seq_len, dim)
        V_out = V_new.view(batch_size, seq_len, dim)
        
        return Q_out, K_out, V_out
    
class ChainReactionUnit_Final(nn.Module):
    """完整的链式反应单元 - 集成三个子网络"""
    
    def __init__(self, dim: int, unit_id: int):
        super().__init__()
        self.dim = dim
        self.unit_id = unit_id
        
        # 三个子网络（使用FixedRMSNorm的版本）
        self.subnet1 = VKQ_SubNet_WithFixedNorm(dim)  # V→K→Q
        self.subnet2 = QVK_SubNet_WithFixedNorm(dim)  # Q→V→K
        self.subnet3 = KQV_SubNet_WithFixedNorm(dim)  # K→Q→V
        
        # 权重参数（可学习）
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))
        self.V_dominant_weight = nn.Parameter(torch.tensor(1.5))
        self.K_weight = nn.Parameter(torch.tensor(1.0))
        self.Q_weight = nn.Parameter(torch.tensor(1.0))
        
        # 🚫 移除传统的归一化层
        # 子网络内部已有FixedRMSNorm，此处不需要额外归一化
        
        # V值历史记录（用于监控）
        self.register_buffer('V_history', torch.zeros(3))
        
        # 单元标识
        self.register_buffer('unit_id_tensor', torch.tensor(unit_id))
    
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor,
                return_evolution: bool = False) -> Tuple:
        """
        前向传播，依次通过三个子网络
        可选返回详细的演化过程信息
        """
        
        # 保存输入状态（用于残差连接）
        Q_in, K_in, V_in = Q.clone(), K.clone(), V.clone()
        
        # === 阶段1: 通过第一个子网 (V→K→Q) ===
        Q11, K11, V11 = self.subnet1(Q, K, V)
        # 🚫 不使用传统归一化，子网络内部已有FixedRMSNorm
        
        # === 阶段2: 通过第二个子网 (Q→V→K) ===
        Q12, K12, V12 = self.subnet2(Q11, K11, V11)
        # 🚫 不使用传统归一化，子网络内部已有FixedRMSNorm
        
        # === 阶段3: 通过第三个子网 (K→Q→V) ===
        Q13, K13, V13 = self.subnet3(Q12, K12, V12)
        # 🚫 不使用传统归一化，子网络内部已有FixedRMSNorm
        
        # === 权重应用 ===
        # 根据你的设计，V值起主导作用
        Q_out = Q13 * self.Q_weight
        K_out = K13 * self.K_weight
        V_out = V13 * self.V_dominant_weight
        
        # === 单元级残差连接 ===
        Q_out = Q_out + self.residual_alpha * Q_in
        K_out = K_out + self.residual_alpha * K_in
        V_out = V_out + self.residual_alpha * V_in
        
        # === 记录V值演化历史 ===
        # 用于相变检测和状态监控
        if V11.numel() > 0 and V12.numel() > 0 and V13.numel() > 0:
            self.V_history[0] = V11.mean().detach()
            self.V_history[1] = V12.mean().detach()
            self.V_history[2] = V13.mean().detach()
        
        # === 返回结果 ===
        if return_evolution:
            # 返回详细演化信息（用于分析和调试）
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
        """获取单元信息"""
        return {
            'unit_id': self.unit_id,
            'dim': self.dim,
            'V_dominant_weight': self.V_dominant_weight.item(),
            'Q_weight': self.Q_weight.item(),
            'K_weight': self.K_weight.item(),
            'residual_alpha': self.residual_alpha.item(),
            'V_history': self.V_history.tolist()
        }
    
__all__ = [
    'ChainReactionUnit_Final',
]