"""
地质记忆模块 | Geological Memory Module
======================================
三明治融合模块 | Sandwich Fusion Module
======================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np

class GeologicalMemory(nn.Module):
    """
    地质记忆层 - 严格按照文档设计
    三层深度 × 三个时间层 × 三个V子值
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # ==================== 三层存储结构 ====================
        # 最新层 (n)
        self.register_buffer('depth0_Q', torch.zeros(3, dim))  # [时间层, dim]
        self.register_buffer('depth0_K', torch.zeros(3, dim))
        self.register_buffer('depth0_V', torch.zeros(3, 3, dim))  # [时间层, 3个子值, dim]
        
        # 中期层 (n-1)
        self.register_buffer('depth1_Q', torch.zeros(3, dim))
        self.register_buffer('depth1_K', torch.zeros(3, dim))
        self.register_buffer('depth1_V', torch.zeros(3, 3, dim))
        
        # 深层 (n-2)
        self.register_buffer('depth2_Q', torch.zeros(3, dim))
        self.register_buffer('depth2_K', torch.zeros(3, dim))
        self.register_buffer('depth2_V', torch.zeros(3, 3, dim))
        
        # ==================== 能量系统 ====================
        self.register_buffer('energy_depth0', torch.ones(3))  # 最新层能量
        self.register_buffer('energy_depth1', torch.ones(3))  # 中期层能量
        self.register_buffer('energy_depth2', torch.ones(3))  # 深层能量
        
        # 线性衰退率
        self.decay_factor = 0.7
        
        # ==================== 状态跟踪 ====================
        self.step = 0
        self.register_buffer('timestamps_depth0', torch.zeros(3, dtype=torch.long))
        self.register_buffer('timestamps_depth1', torch.zeros(3, dtype=torch.long))
        self.register_buffer('timestamps_depth2', torch.zeros(3, dtype=torch.long))
    
    def reset(self):
        """重置记忆"""
        for buffer in self.buffers():
            if buffer.dim() > 0:  # 跳过标量
                buffer.zero_()
        
        self.energy_depth0.fill_(1.0)
        self.energy_depth1.fill_(1.0)
        self.energy_depth2.fill_(1.0)
        self.step = 0
    
    def _compute_energy_by_density(self, values: torch.Tensor) -> float:
        """
        使用密度公式计算能量
        静态密度公式: D = 2m/((N+1)N)
        其中N=节点数，m=连接数
        """
        if values.numel() == 0 or values.size(0) < 2:
            return 1.0
        
        N = values.size(0)
        
        # 计算余弦相似度矩阵
        values_norm = F.normalize(values, p=2, dim=1)
        similarity = torch.mm(values_norm, values_norm.T)
        
        # 统计连接数 (相似度 > 0.3)
        connections = (similarity > 0.3).float()
        m = connections.sum().item() / 2  # 无向图，每条边计一次
        
        # 计算密度
        if N > 1:
            density = (2 * m) / ((N + 1) * N)
        else:
            density = 0.0
        
        # 能量 = 密度归一化到[0,1]
        energy = min(max(density, 0.0), 1.0)
        return energy
    
    def _linear_decay(self):
        """线性衰退：能量 *= 衰退因子"""
        with torch.no_grad():
            self.energy_depth0.mul_(self.decay_factor)
            self.energy_depth1.mul_(self.decay_factor)
            self.energy_depth2.mul_(self.decay_factor)
    
    def _shift_and_update(self, Q_list: List[torch.Tensor], K_list: List[torch.Tensor], 
                         V_list: List[List[torch.Tensor]]):
        """
        执行三层移位和更新
        最新(n) → 中期(n-1) → 深层(n-2) → 衰退
        """
        self.step += 1
        
        # ==================== 1. 检查深层衰退 ====================
        # 找到能量最低的时间层（最可能衰退）
        min_energy_idx = torch.argmin(self.energy_depth2).item()
        min_energy = self.energy_depth2[min_energy_idx].item()
        
        # 如果能量低于阈值，用中期层数据覆盖
        if min_energy < 0.1:
            if self.energy_depth1[min_energy_idx] > 0.1:  # 中期层有有效数据
                self.depth2_Q[min_energy_idx] = self.depth1_Q[min_energy_idx].clone()
                self.depth2_K[min_energy_idx] = self.depth1_K[min_energy_idx].clone()
                self.depth2_V[min_energy_idx] = self.depth1_V[min_energy_idx].clone()
                self.energy_depth2[min_energy_idx] = self.energy_depth1[min_energy_idx].clone() * 0.8
                self.timestamps_depth2[min_energy_idx] = self.step
        
        # ==================== 2. 中期层 ← 最新层 ====================
        for i in range(3):
            if self.energy_depth0[i] > 0.1:  # 最新层有有效数据
                self.depth1_Q[i] = self.depth0_Q[i].clone()
                self.depth1_K[i] = self.depth0_K[i].clone()
                self.depth1_V[i] = self.depth0_V[i].clone()
                self.energy_depth1[i] = self.energy_depth0[i].clone() * 0.9
                self.timestamps_depth1[i] = self.step
        
        # ==================== 3. 存储新数据到最新层 ====================
        for time_layer in range(min(3, len(Q_list))):
            if Q_list[time_layer] is not None:
                # 计算批次和序列的均值
                Q_mean = Q_list[time_layer].mean(dim=(0, 1))  # [dim]
                K_mean = K_list[time_layer].mean(dim=(0, 1))
                
                # 存储三个V子值
                V_means = []
                for v_idx in range(3):
                    if v_idx < len(V_list[time_layer]):
                        V_mean = V_list[time_layer][v_idx].mean(dim=(0, 1))  # [dim]
                        V_means.append(V_mean)
                    else:
                        V_means.append(torch.zeros_like(Q_mean))
                
                # 存储到最新层
                self.depth0_Q[time_layer] = Q_mean
                self.depth0_K[time_layer] = K_mean
                self.depth0_V[time_layer] = torch.stack(V_means, dim=0)  # [3, dim]
                
                # 更新时间戳
                self.timestamps_depth0[time_layer] = self.step
                
                # 计算新能量（使用Q, K, V的平均状态）
                combined = torch.stack([Q_mean, K_mean] + V_means, dim=0)  # [5, dim]
                new_energy = self._compute_energy_by_density(combined)
                self.energy_depth0[time_layer] = new_energy
    
    def store(self, 
              Q_list: List[torch.Tensor],      # [Q1, Q2, Q3]
              K_list: List[torch.Tensor],      # [K1, K2, K3]
              V_list: List[List[torch.Tensor]]):  # [[1V1,2V1,3V1], [1V2,2V2,3V2], [1V3,2V3,3V3]]
        """
        存储三个链式反应单元的输出
        """
        # 应用线性衰退
        self._linear_decay()
        
        # 执行移位和更新
        self._shift_and_update(Q_list, K_list, V_list)
    
    def retrieve(self, 
                depth: int = 2,        # 0:最新, 1:中期, 2:深层
                time_layer: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        检索地质记忆
        默认：深层(depth=2)的第3个时间层(time_layer=2)
        """
        if depth == 0:
            Q_mem = self.depth0_Q
            K_mem = self.depth0_K
            V_mem = self.depth0_V
        elif depth == 1:
            Q_mem = self.depth1_Q
            K_mem = self.depth1_K
            V_mem = self.depth1_V
        else:
            Q_mem = self.depth2_Q
            K_mem = self.depth2_K
            V_mem = self.depth2_V
        
        time_layer_idx = min(time_layer, 2)
        
        # 检索并detach防止梯度回流
        Q_det = Q_mem[time_layer_idx].detach().clone()
        K_ret = K_mem[time_layer_idx].detach().clone()
        V_ret = V_mem[time_layer_idx, 2].detach().clone()  # 第3个V子值
        
        return Q_det, K_ret, V_ret
    
    def get_energy_stats(self) -> Dict:
        """获取能量统计"""
        return {
            'step': self.step,
            'energy_depth0': self.energy_depth0.cpu().numpy().tolist(),
            'energy_depth1': self.energy_depth1.cpu().numpy().tolist(),
            'energy_depth2': self.energy_depth2.cpu().numpy().tolist(),
        }
    
    def get_v_stats(self, depth: int = 0, time_layer: int = 0) -> Dict:
        """获取V值统计"""
        if depth == 0:
            V_mem = self.depth0_V
        elif depth == 1:
            V_mem = self.depth1_V
        else:
            V_mem = self.depth2_V
        
        V_tensor = V_mem[time_layer]  # [3, dim]
        
        return {
            'V1_mean': V_tensor[0].mean().item(),
            'V2_mean': V_tensor[1].mean().item(),
            'V3_mean': V_tensor[2].mean().item(),
            'V_all_different': torch.all(V_tensor[0] != V_tensor[1]).item() and 
                              torch.all(V_tensor[1] != V_tensor[2]).item()
        }
    
    def forward(self, 
               Q_list: List[torch.Tensor],
               K_list: List[torch.Tensor],
               V_list: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：存储并检索
        """
        self.store(Q_list, K_list, V_list)
        return self.retrieve(depth=2, time_layer=2)

class SandwichFusion(nn.Module):
    """
    三明治融合层 - 修正版
    处理维度不匹配：将地质层的均值向量扩展为完整序列形状
    """
    
    def __init__(self):
        super().__init__()
        
        # ==================== 固定权重 ====================
        # 按照文档中的权重，不可学习
        # Q权重：深层0.5，当前0.3，原始0.2
        self.register_buffer('q_weights', torch.tensor([0.5, 0.3, 0.2]))
        
        # K权重：深层0.5，当前0.3，原始0.2
        self.register_buffer('k_weights', torch.tensor([0.5, 0.3, 0.2]))
        
        # V权重：深层0.6，当前0.3，原始0.1
        self.register_buffer('v_weights', torch.tensor([0.6, 0.3, 0.1]))
        
        # 验证权重和为1
        assert torch.allclose(self.q_weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(self.k_weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(self.v_weights.sum(), torch.tensor(1.0), atol=1e-6)
    
    def _expand_deep_state(self, 
                          deep_state: torch.Tensor, 
                          batch_size: int, 
                          seq_len: int) -> torch.Tensor:
        """
        将地质层的深层状态（均值向量）扩展为完整序列形状
        输入：deep_state [dim] 或 [1, dim] 或 [1, 1, dim]
        输出：[batch_size, seq_len, dim]
        """
        # 清理维度
        if deep_state.dim() == 1:  # [dim]
            deep_state = deep_state.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
        elif deep_state.dim() == 2:  # [1, dim] 或 [seq, dim]
            if deep_state.size(0) == 1:  # [1, dim]
                deep_state = deep_state.unsqueeze(1)  # [1, 1, dim]
            else:  # [seq, dim] - 不常见
                deep_state = deep_state.unsqueeze(0)  # [1, seq, dim]
        
        # 现在应该是3维： [1, 1, dim] 或 [1, seq, dim]
        # 确保我们得到 [1, 1, dim]
        if deep_state.size(1) != 1:
            # 如果是[1, seq, dim]，取均值变成[1, 1, dim]
            deep_state = deep_state.mean(dim=1, keepdim=True)
        
        # 扩展维度到 [batch_size, seq_len, dim]
        expanded = deep_state.expand(batch_size, seq_len, -1)
        
        return expanded
    
    def forward(self,
               # 深层状态（从地质层检索）- 可能是各种形状
               Q_deep: torch.Tensor,      # 深层稳定状态 [dim] 或 [1, dim] 或 [1, 1, dim]
               K_deep: torch.Tensor,      # 同上
               V_deep: torch.Tensor,      # 同上
               
               # 当前状态（最新链式反应输出）
               Q_current: torch.Tensor,   # [B, S, dim] 当前Q3⁽ⁿ⁾
               K_current: torch.Tensor,   # [B, S, dim] 当前K3⁽ⁿ⁾
               V_current: torch.Tensor,   # [B, S, dim] 当前V3⁽ⁿ⁾
               
               # 原始状态（初始输入）- 可选，如果不提供则设为0
               Q_original: torch.Tensor = None,  # [B, S, dim] 原始Q
               K_original: torch.Tensor = None,  # [B, S, dim] 原始K
               V_original: torch.Tensor = None   # [B, S, dim] 原始V
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        三明治融合前向传播
        将深层状态扩展到完整维度，然后进行加权融合
        """
        
        # 获取批次和序列维度
        batch_size, seq_len, dim = Q_current.shape
        
        # ==================== 维度扩展 ====================
        # 将深层状态扩展到完整序列形状
        Q_deep_expanded = self._expand_deep_state(Q_deep, batch_size, seq_len)
        K_deep_expanded = self._expand_deep_state(K_deep, batch_size, seq_len)
        V_deep_expanded = self._expand_deep_state(V_deep, batch_size, seq_len)
        
        # ==================== 处理原始状态 ====================
        # 如果未提供原始状态，使用零张量
        if Q_original is None:
            Q_original = torch.zeros_like(Q_current)
        if K_original is None:
            K_original = torch.zeros_like(K_current)
        if V_original is None:
            V_original = torch.zeros_like(V_current)
        
        # 验证所有输入形状一致
        assert Q_deep_expanded.shape == (batch_size, seq_len, dim)
        assert K_deep_expanded.shape == (batch_size, seq_len, dim)
        assert V_deep_expanded.shape == (batch_size, seq_len, dim)
        assert Q_original.shape == (batch_size, seq_len, dim)
        assert K_original.shape == (batch_size, seq_len, dim)
        assert V_original.shape == (batch_size, seq_len, dim)
        
        # ==================== 加权融合 ====================
        # Q融合
        Q_final = (self.q_weights[0] * Q_deep_expanded +
                  self.q_weights[1] * Q_current +
                  self.q_weights[2] * Q_original)
        
        # K融合
        K_final = (self.k_weights[0] * K_deep_expanded +
                  self.k_weights[1] * K_current +
                  self.k_weights[2] * K_original)
        
        # V融合（注意V使用不同的权重）
        V_final = (self.v_weights[0] * V_deep_expanded +
                  self.v_weights[1] * V_current +
                  self.v_weights[2] * V_original)
        
        return Q_final, K_final, V_final
    
    def apply_single_fusion(self,
                           deep_state: torch.Tensor,
                           current_state: torch.Tensor,
                           original_state: torch.Tensor,
                           value_type: str = 'Q') -> torch.Tensor:
        """
        对单个值进行融合，自动处理维度扩展
        """
        batch_size, seq_len, dim = current_state.shape
        
        # 扩展深层状态
        deep_expanded = self._expand_deep_state(deep_state, batch_size, seq_len)
        
        # 选择权重
        if value_type == 'Q':
            weights = self.q_weights
        elif value_type == 'K':
            weights = self.k_weights
        elif value_type == 'V':
            weights = self.v_weights
        else:
            raise ValueError(f"未知的值类型: {value_type}")
        
        # 确保原始状态有正确形状
        if original_state is None:
            original_state = torch.zeros_like(current_state)
        
        return (weights[0] * deep_expanded +
                weights[1] * current_state +
                weights[2] * original_state)
    
    def get_debug_info(self,
                      Q_deep: torch.Tensor,
                      Q_current: torch.Tensor,
                      Q_original: torch.Tensor = None) -> dict:
        """
        调试信息：显示维度转换过程
        """
        batch_size, seq_len, dim = Q_current.shape
        
        # 扩展前的形状
        deep_shape_before = Q_deep.shape
        
        # 扩展
        Q_deep_expanded = self._expand_deep_state(Q_deep, batch_size, seq_len)
        
        info = {
            'deep_state_shape_before': list(deep_shape_before),
            'deep_state_shape_after': list(Q_deep_expanded.shape),
            'current_state_shape': list(Q_current.shape),
            'weights': {
                'Q': self.q_weights.cpu().numpy().tolist(),
                'K': self.k_weights.cpu().numpy().tolist(),
                'V': self.v_weights.cpu().numpy().tolist(),
            }
        }
        
        if Q_original is not None:
            info['original_state_shape'] = list(Q_original.shape)
        
        return info
    
__all__ = [
    'GeologicalMemory',
    'SandwichFusion',
]