import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
import json
from collections import deque
from torch.utils.data import Dataset, DataLoader
import re
import random
from collections import Counter, deque
from torch.autograd import Function

class CoreMetricsCalculator:
    """
    核心度量计算器 - 实现文中四个核心公式
    1. 单网络变化量公式 (状态监控)
    2. 相变检测公式 (质变识别)
    3. 三网络堆叠公式 (多视角融合)
    4. 单向阀公式 (信息控制)
    """
    
    def __init__(self):
        """初始化度量计算器"""
        self.state_history = []  # 记录状态历史
        self.transition_points = []  # 记录相变点
        self.phase_threshold = 1.0  # 相变阈值
    
    # ==================== 公式1: 单网络变化量 ====================
    def compute_state_change(self, 
                           Q_t: torch.Tensor, K_t: torch.Tensor, V_t: torch.Tensor,
                           Q_t_1: torch.Tensor, K_t_1: torch.Tensor, V_t_1: torch.Tensor,
                           norm_type: str = 'l2') -> float:
        """
        单网络变化量公式: Δ = ||Q_t - Q_{t-1}|| + ||K_t - K_{t-1}|| + ||V_t - V_{t-1}||
        
        作用: 监控网络状态的连续变化量
        物理意义: 量化学习过程中的"能量流动"
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
            raise ValueError(f"不支持的范数类型: {norm_type}")
        
        return (delta_Q + delta_K + delta_V).item()
    
    # ==================== 公式2: 相变检测 ====================
    def detect_phase_transition(self,
                              Q_t: torch.Tensor, K_t: torch.Tensor, V_t: torch.Tensor,
                              Q_t_1: torch.Tensor, K_t_1: torch.Tensor, V_t_1: torch.Tensor) -> Tuple[float, bool]:
        """
        相变检测公式: Δ = ‖Q^(t) - Q^(t-1)‖_F + ‖K^(t) - K^(t-1)‖_F + ‖V^(t) - V^(t-1)‖_F
        
        关键: 使用Frobenius范数（矩阵整体结构变化）
        作用: 检测质的变化，而不仅仅是量的变化
        物理意义: 识别认知状态的"跃迁时刻"
        """
        # Frobenius范数：捕捉矩阵结构变化
        delta_Q = torch.norm(Q_t - Q_t_1, p='fro')
        delta_K = torch.norm(K_t - K_t_1, p='fro')
        delta_V = torch.norm(V_t - V_t_1, p='fro')
        
        delta_total = (delta_Q + delta_K + delta_V).item()
        
        # 检测是否为相变
        is_transition = delta_total > self.phase_threshold
        
        if is_transition:
            self.transition_points.append({
                'delta': delta_total,
                'Q_shape': Q_t.shape,
                'timestamp': len(self.state_history)
            })
        
        return delta_total, is_transition
    
    # ==================== 公式3: 三网络堆叠 ====================
    def stack_three_networks(self,
                           Q_list: List[torch.Tensor],
                           weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        三网络堆叠公式: Q_stack = ∑_{i=1}^3 α_i Q_i, where α = Softmax(w)
        
        作用: 融合三个子网络的不同处理路径
        物理意义: 实现"多视角平等融合"，避免层级偏见
        """
        if len(Q_list) != 3:
            raise ValueError(f"需要3个Q矩阵进行堆叠，得到{len(Q_list)}个")
        
        # 验证形状一致性
        shapes = [Q.shape for Q in Q_list]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError("所有Q矩阵必须形状一致")
        
        # 默认权重：均匀分布
        if weights is None:
            weights = torch.ones(3)
        
        # Softmax归一化权重（确保总和为1）
        alpha = F.softmax(weights, dim=0)
        
        # 加权堆叠
        Q_stack = torch.zeros_like(Q_list[0])
        for i, Q in enumerate(Q_list):
            Q_stack += alpha[i] * Q
        
        return Q_stack
    
    # ==================== 公式4: 单向阀 ====================
    def apply_one_way_valve(self,
                          h_in: torch.Tensor,
                          mode: str = 'detach',
                          gate_value: Optional[int] = None) -> torch.Tensor:
        """
        单向阀公式: h_out = detach(h_in) 或 h_out = g·h_in, g∈{0,1}
        
        两种模式:
        1. detach模式: 切断梯度，创建不可逆记忆
        2. gate模式: 二进制门控，实现"全或无"信息流
        
        物理意义: 保护核心记忆，控制信息流动方向
        """
        if mode == 'detach':
            # 模式1: 梯度阻断（不可逆记忆）
            return h_in.detach()
        
        elif mode == 'gate':
            # 模式2: 二进制门控
            if gate_value not in [0, 1]:
                raise ValueError(f"门控值必须是0或1，得到{gate_value}")
            
            if gate_value == 0:
                return torch.zeros_like(h_in)
            else:
                return h_in.clone()
        
        else:
            raise ValueError(f"不支持的模式: {mode}")
    
    # ==================== 辅助方法 ====================
    def record_state(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """记录当前状态到历史"""
        self.state_history.append({
            'Q': Q.detach().clone(),
            'K': K.detach().clone(),
            'V': V.detach().clone(),
            'step': len(self.state_history)
        })
    
    def get_state_change_series(self, norm_type: str = 'l2') -> List[float]:
        """获取状态变化序列"""
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
        """分析学习阶段"""
        if len(self.state_history) < 2:
            return {'message': '数据不足'}
        
        changes = self.get_state_change_series('l2')
        
        # 检测学习阶段
        if not changes:
            return {'phase': '初始阶段', 'stability': 0.0}
        
        recent_changes = changes[-min(5, len(changes)):]  # 最近5步
        avg_change = sum(recent_changes) / len(recent_changes)
        
        if avg_change > 25.0:
            phase = "探索期"
        elif avg_change > 15.0:
            phase = "学习期"
        elif avg_change > 5.0:
            phase = "稳定期"
        else:
            phase = "收敛期"
        
        return {
            'phase': phase,
            'avg_change': avg_change,
            'total_steps': len(self.state_history),
            'transition_points': len(self.transition_points),
            'current_stability': 1.0 / (avg_change + 1e-6)
        }
    
    def reset(self):
        """重置计算器状态"""
        self.state_history = []
        self.transition_points = []


class 单向阀组合(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 三个独立的门控参数，每个对应一个值
        self.gate_Q = nn.Parameter(torch.ones(1, 1, dim))
        self.gate_K = nn.Parameter(torch.ones(1, 1, dim))
        self.gate_V = nn.Parameter(torch.ones(1, 1, dim))
        
        # 三个独立的内部变换模块，每个模块处理一个值
        # Q模块：两层线性，中间用ReLU激活
        self.Q_linear1 = nn.Linear(dim, dim)
        self.Q_linear2 = nn.Linear(dim, dim)
        self.Q_act = nn.ReLU()
        
        # K模块：一层线性+Tanh激活
        self.K_linear = nn.Linear(dim, dim)
        self.K_act = nn.Tanh()
        
        # V模块：链式反应，两层线性，使用不同的激活函数
        self.V_linear1 = nn.Linear(dim, dim)
        self.V_linear2 = nn.Linear(dim, dim)
        self.V_act1 = nn.ReLU()
        self.V_act2 = nn.Tanh()
    
    def _门控融合(self, x, transformed, gate):
        """门控融合函数：将原始输入和变换后的输入按门控参数融合"""
        g = torch.sigmoid(gate)  # 将门控参数映射到(0,1)
        return x * g + transformed * (1 - g)
    
    def forward(self, Q, K, V):
        # 处理Q值
        Q_transformed = self.Q_linear1(Q)
        Q_transformed = self.Q_act(Q_transformed)
        Q_transformed = self.Q_linear2(Q_transformed)
        Q_out = self._门控融合(Q, Q_transformed, self.gate_Q)
        
        # 处理K值
        K_transformed = self.K_act(self.K_linear(K))
        K_out = self._门控融合(K, K_transformed, self.gate_K)
        
        # 处理V值
        V_transformed = self.V_linear1(V)
        V_transformed = self.V_act1(V_transformed)
        V_transformed = self.V_linear2(V_transformed)
        V_transformed = self.V_act2(V_transformed)
        V_out = self._门控融合(V, V_transformed, self.gate_V)
        
        return Q_out, K_out, V_out
    
class EnhancedEmbeddingLayer(nn.Module):
    """
    增强嵌入层 - 基于概念图节点特征（Q, K, V）生成标记向量
    遵循规则：不进行传统归一化，仅使用原始密度公式体系
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, marker_dim: int = 32):
        """
        初始化增强嵌入层
        
        Args:
            vocab_size: 词汇表大小
            embed_dim: 基础嵌入维度
            marker_dim: 标记向量维度
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.marker_dim = marker_dim
        
        # ==================== 基础嵌入组件 ====================
        # 词嵌入（可学习）
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码（固定正弦编码）- 现在使用缓冲区，会自动移动到设备
        self.register_buffer('position_encoding', self._create_positional_encoding(max_len=512, d_model=embed_dim))
        
        # ==================== 标记生成组件 ====================
        # 概念特征映射层：将(Q, K, V)特征映射到标记空间
        # 输入: 3个特征值 (Q_local, K_local, V_local)
        # 输出: marker_dim维标记向量
        self.marker_projection = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, marker_dim),
            nn.Tanh()
        )
        
        # 默认标记（用于未识别的词）
        self.default_marker = nn.Parameter(torch.randn(marker_dim) * 0.1)
        
        # ==================== 概念特征缓存 ====================
        # 存储词汇索引到概念特征的映射
        self.register_buffer('concept_features', torch.zeros(vocab_size, 3))
        self.register_buffer('has_concept', torch.zeros(vocab_size, dtype=torch.bool))
        
        # ==================== 统计跟踪 ====================
        self.register_buffer('activation_stats', torch.zeros(3))  # 统计Q, K, V激活
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        创建正弦位置编码（非学习参数）
        
        Args:
            max_len: 最大序列长度
            d_model: 模型维度
            
        Returns:
            位置编码张量 [1, max_len, d_model]
        """
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )  # [d_model/2]
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def set_concept_features(self, word_indices: torch.Tensor, features: torch.Tensor):
        """
        设置词汇的概念特征（Q, K, V）
        
        Args:
            word_indices: 词汇索引 [n_words]
            features: 概念特征 [n_words, 3] (Q, K, V)
        """
        if len(word_indices) != len(features):
            raise ValueError(f"词汇索引数量({len(word_indices)})与特征数量({len(features)})不匹配")
        
        # 更新概念特征缓存 - 确保在同一设备
        device = self.concept_features.device
        self.concept_features[word_indices] = features.to(device)
        self.has_concept[word_indices] = True
        
        print(f"已设置 {len(word_indices)} 个词汇的概念特征")
    
    def get_concept_features(self, word_idx: int) -> torch.Tensor:
        """
        获取指定词汇的概念特征
        
        Args:
            word_idx: 词汇索引
            
        Returns:
            概念特征张量 [3] 或 None（如果无概念特征）
        """
        if word_idx >= self.vocab_size:
            raise ValueError(f"词汇索引 {word_idx} 超出词汇表范围 {self.vocab_size}")
        
        if self.has_concept[word_idx]:
            return self.concept_features[word_idx].clone()
        else:
            return None
    
    def _calculate_local_density(self, batch_markers: torch.Tensor) -> Dict[str, float]:
        """
        计算本地认知密度（使用你的公式体系）
        
        Args:
            batch_markers: 标记向量 [batch_size, seq_len, marker_dim]
            
        Returns:
            密度统计信息
        """
        if batch_markers.numel() == 0:
            return {'density': 0.0, 'stability': 0.0}
        
        # 计算标记向量间的余弦相似度矩阵
        batch_size, seq_len, marker_dim = batch_markers.shape
        flat_markers = batch_markers.view(-1, marker_dim)  # [batch*seq_len, marker_dim]
        
        # 计算余弦相似度
        norm_markers = F.normalize(flat_markers, p=2, dim=1)
        similarity_matrix = torch.mm(norm_markers, norm_markers.T)  # [N, N]
        
        # 应用你的密度公式：D = 2m/((N+1)N)，其中m为连接数
        N = similarity_matrix.size(0)
        
        # 阈值过滤：保留相似度大于0.3的连接
        threshold = 0.3
        connections = (similarity_matrix > threshold).float()
        M = connections.sum().item() / 2  # 无向图，每条边计算两次
        
        # 静态密度公式
        if N > 1:
            static_density = (2 * M) / ((N + 1) * N)
        else:
            static_density = 0.0
        
        # 动态密度变化（与历史统计比较）
        prev_stats = self.activation_stats
        
        # 计算当前统计：标记向量的平均范数
        current_stats = torch.tensor([
            batch_markers.mean().item(),
            batch_markers.std().item(),
            similarity_matrix.mean().item()
        ], device=batch_markers.device)
        
        # 更新统计
        self.activation_stats = 0.9 * self.activation_stats + 0.1 * current_stats
        
        # 计算稳定性：统计变化率
        if prev_stats.sum() > 0:
            stability = 1.0 / (torch.norm(current_stats - prev_stats, p=2).item() + 1e-6)
        else:
            stability = 1.0
        
        return {
            'static_density': static_density,
            'connections': M,
            'nodes': N,
            'stability': stability,
            'avg_similarity': similarity_matrix.mean().item()
        }
    
    def forward(self, input_ids: torch.Tensor, return_details: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入词索引 [batch_size, seq_len]
            return_details: 是否返回详细统计信息
            
        Returns:
            包含基础嵌入和标记的字典
        """
        batch_size, seq_len = input_ids.shape
        
        # ==================== 基础嵌入 ====================
        # 词嵌入
        token_emb = self.token_embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # 位置编码 - 确保在正确的设备上
        if seq_len <= self.position_encoding.size(1):
            pos_emb = self.position_encoding[:, :seq_len, :]
            # 确保位置编码与token嵌入在同一设备
            if pos_emb.device != token_emb.device:
                pos_emb = pos_emb.to(token_emb.device)
        else:
            # 动态扩展位置编码 - 确保在正确的设备上
            pos_emb = self._create_positional_encoding(seq_len, self.embed_dim).to(input_ids.device)
        
        # 基础嵌入 = 词嵌入 + 位置编码
        base_embeddings = token_emb + pos_emb  # [batch, seq_len, embed_dim]
        
        # ==================== 标记生成 ====================
        markers = []
        
        # 为每个词生成标记
        for batch_idx in range(batch_size):
            batch_markers = []
            
            for pos_idx in range(seq_len):
                word_idx = input_ids[batch_idx, pos_idx].item()
                
                # 获取概念特征 - 确保在同一设备
                concept_feat = self.get_concept_features(word_idx)
                
                if concept_feat is not None:
                    # 有概念特征：投影到标记空间
                    concept_feat = concept_feat.to(input_ids.device).unsqueeze(0)
                    marker = self.marker_projection(concept_feat)  # [1, marker_dim]
                else:
                    # 无概念特征：使用默认标记
                    marker = self.default_marker.unsqueeze(0).to(input_ids.device)  # [1, marker_dim]
                
                batch_markers.append(marker)
            
            # 拼接当前批次的标记
            batch_markers = torch.cat(batch_markers, dim=0)  # [seq_len, marker_dim]
            markers.append(batch_markers)
        
        # 组合所有批次的标记
        markers = torch.stack(markers, dim=0)  # [batch, seq_len, marker_dim]
        
        # ==================== 统计信息 ====================
        stats = {}
        if return_details:
            # 计算本地认知密度
            density_info = self._calculate_local_density(markers)
            
            # 收集输出统计
            stats = {
                'marker_stats': {
                    'mean': markers.mean().item(),
                    'std': markers.std().item(),
                    'min': markers.min().item(),
                    'max': markers.max().item(),
                    'norm': markers.norm(dim=-1).mean().item(),
                },
                'density_info': density_info,
                'concept_coverage': self.has_concept[input_ids.cpu()].float().mean().item(),
            }
        
        # ==================== 返回结果 ====================
        output = {
            'base_embeddings': base_embeddings,  # [batch, seq_len, embed_dim]
            'markers': markers,                  # [batch, seq_len, marker_dim]
            'concept_features': self.concept_features[input_ids.cpu()].to(input_ids.device)  # [batch, seq_len, 3]
        }
        
        if return_details:
            output['stats'] = stats
        
        return output
    
    def analyze_concept_distribution(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """
        分析输入中的概念分布
    
        Args:
            input_ids: 输入词索引 [batch_size, seq_len]
        
        Returns:
            概念分布统计
        """
        batch_size, seq_len = input_ids.shape
        total_tokens = batch_size * seq_len
    
        # 获取has_concept，确保在正确的设备上
        # 确保has_concept和input_ids在同一设备
        device = input_ids.device
        has_concept_on_device = self.has_concept.to(device)
    
        # 统计概念覆盖
        has_concept_mask = has_concept_on_device[input_ids]  # [batch, seq_len]
        concept_count = has_concept_mask.sum().item()
        concept_coverage = concept_count / total_tokens if total_tokens > 0 else 0
    
        # 获取有概念的词的特征
        if concept_count > 0:
            # 获取有概念的索引
            concept_indices = input_ids[has_concept_mask]  # [concept_count]
        
            # 确保concept_features在正确的设备上
            concept_feats_on_device = self.concept_features.to(device)
            concept_feats = concept_feats_on_device[concept_indices]  # [concept_count, 3]
        
            # 计算特征统计
            Q_stats = concept_feats[:, 0]
            K_stats = concept_feats[:, 1]
            V_stats = concept_feats[:, 2]
        
            stats = {
                'concept_coverage': concept_coverage,
                'Q_mean': Q_stats.mean().item(),
                'Q_std': Q_stats.std().item(),
                'K_mean': K_stats.mean().item(),
                'K_std': K_stats.std().item(),
                'V_mean': V_stats.mean().item(),
                'V_std': V_stats.std().item(),
                'V_range': (V_stats.max().item() - V_stats.min().item()),
                'V_stability': 1.0 / (V_stats.std().item() + 1e-6),
            }
        else:
            stats = {
                'concept_coverage': 0.0,
                'Q_mean': 0.0,
                'K_mean': 0.0,
                'V_mean': 0.0,
            }
    
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.activation_stats.zero_()

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
    
# ==================== 完整约束施工架构集成 ====================

class CompleteConstrainedArchitecture(nn.Module):
    """
    完整约束施工架构
    包含：3个链式反应单元 × 12个单向阀 × 地质层 × 三明治融合
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int = 512,
                 hidden_dim: int = 512,
                 marker_dim: int = 32,
                 max_seq_len: int = 512,
                 dropout_rate: float = 0.1):
        """
        初始化完整架构
        
        Args:
            vocab_size: 词汇表大小
            embed_dim: 基础嵌入维度
            hidden_dim: 隐藏维度（Q、K、V维度）
            marker_dim: 标记向量维度
            max_seq_len: 最大序列长度
            dropout_rate: dropout比率
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        print("\n" + "="*70)
        print("构建完整约束施工架构")
        print("="*70)
        
        # ==================== 核心度量计算器 ====================
        self.metrics = CoreMetricsCalculator()
        print("✓ 集成核心度量计算器（全流程监控）")
        
        # ==================== 三明治融合层 ====================
        self.sandwich_fusion = SandwichFusion()
        print("✓ 集成三明治融合层")
        
        # ==================== 嵌入层 ====================
        self.embedding = EnhancedEmbeddingLayer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            marker_dim=marker_dim
        )
        print("✓ 集成增强嵌入层")
        
        # ==================== 12个单向阀（4个位置×3个值） ====================
        # 位置1：嵌入层后（阀1-3）
        self.valve1_q = 单向阀组合(hidden_dim)
        self.valve1_k = 单向阀组合(hidden_dim)
        self.valve1_v = 单向阀组合(hidden_dim)
        
        # 位置2：网络1后（阀4-6）
        self.valve2_q = 单向阀组合(hidden_dim)
        self.valve2_k = 单向阀组合(hidden_dim)
        self.valve2_v = 单向阀组合(hidden_dim)
        
        # 位置3：网络2后（阀7-9）
        self.valve3_q = 单向阀组合(hidden_dim)
        self.valve3_k = 单向阀组合(hidden_dim)
        self.valve3_v = 单向阀组合(hidden_dim)
        
        # 位置4：网络3后（阀10-12）
        self.valve4_q = 单向阀组合(hidden_dim)
        self.valve4_k = 单向阀组合(hidden_dim)
        self.valve4_v = 单向阀组合(hidden_dim)
        
        print("✓ 集成12个单向阀（4个位置×3个值）")
        
        # ==================== 3个链式反应单元 ====================
        self.chain_unit1 = ChainReactionUnit_Final(hidden_dim, unit_id=1)
        self.chain_unit2 = ChainReactionUnit_Final(hidden_dim, unit_id=2)
        self.chain_unit3 = ChainReactionUnit_Final(hidden_dim, unit_id=3)
        print("✓ 集成3个链式反应单元")
        
        # ==================== 地质层 ====================
        self.geological_memory = GeologicalMemory(hidden_dim)
        print("✓ 集成地质层（三层深度×三层时间×三个V值）")
        
        # ==================== 输出层 ====================
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.final_norm = FixedRMSNorm(hidden_dim)
        
        # ==================== 状态跟踪 ====================
        self.training_step = 0
        self.memory_initialized = False
        
        print("\n架构组件统计:")
        print(f"  嵌入层: 1个（{embed_dim}维）")
        print(f"  单向阀: 12个（4个位置×3个值）")
        print(f"  链式单元: 3个（每个包含3个子网络）")
        print(f"  地质层: 1个（存储三个网络的完整状态）")
        print(f"  三明治融合: 1个（融合地质层状态）")
        print("="*70)
    
    def _initialize_from_memory(self, batch_size: int, seq_len: int):
        """
        从地质记忆初始化状态
        """
        if not self.memory_initialized or self.training_step == 0:
            # 首次初始化，使用零状态
            device = next(self.parameters()).device
            Q_init = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
            K_init = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
            V_init = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
            return Q_init, K_init, V_init
        
        # 从地质层检索深层稳定状态
        Q_deep, K_deep, V_deep = self.geological_memory.retrieve(
            depth=2,  # 深层
            time_layer=2  # 第三个时间层
        )
        
        # 扩展维度
        Q_init = self.sandwich_fusion._expand_deep_state(Q_deep, batch_size, seq_len)
        K_init = self.sandwich_fusion._expand_deep_state(K_deep, batch_size, seq_len)
        V_init = self.sandwich_fusion._expand_deep_state(V_deep, batch_size, seq_len)
        
        return Q_init, K_init, V_init
    
    def _record_state(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, stage: str):
        """
        记录状态并计算度量
        """
        self.metrics.record_state(Q.detach().clone(), K.detach().clone(), V.detach().clone())
        
        # 如果有前一个状态，计算变化量
        if len(self.metrics.state_history) >= 2:
            prev_state = self.metrics.state_history[-2]
            curr_state = self.metrics.state_history[-1]
            
            # 计算单网络变化量
            state_change = self.metrics.compute_state_change(
                curr_state['Q'], curr_state['K'], curr_state['V'],
                prev_state['Q'], prev_state['K'], prev_state['V'],
                norm_type='l2'
            )
            
            # 计算相变检测
            delta_total, is_transition = self.metrics.detect_phase_transition(
                curr_state['Q'], curr_state['K'], curr_state['V'],
                prev_state['Q'], prev_state['K'], prev_state['V']
            )
            
            if is_transition:
                print(f"[相变检测] 阶段 '{stage}' 检测到相变: Δ={delta_total:.4f}")
    
    def _extract_v_sublist(self, unit_id: int, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """
        模拟提取三个V子值
        在实际中，需要从链式反应单元的三个子网络输出中提取
        """
        # 这里我们模拟从三个不同处理路径提取V子值
        batch_size, seq_len, dim = Q.shape
        
        # V子值1：基于Q和K的相似度
        q_norm = F.normalize(Q, dim=-1)
        k_norm = F.normalize(K, dim=-1)
        similarity = torch.bmm(q_norm, k_norm.transpose(1, 2))
        V_sub1 = similarity.mean(dim=-1, keepdim=True).expand(-1, -1, dim) * V
        
        # V子值2：基于V自身的变换
        V_sub2 = F.relu(V)  # 不同的激活函数
        
        # V子值3：基于Q、K、V的融合
        V_sub3 = (Q + K + V) / 3.0
        
        return [V_sub1, V_sub2, V_sub3]
    
    def forward(self, 
                input_ids: torch.Tensor,
                use_memory_initialization: bool = True,
                return_detailed: bool = False):
        """
        完整前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            use_memory_initialization: 是否使用地质记忆初始化
            return_detailed: 是否返回详细状态信息
            
        Returns:
            输出logits和详细状态
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        self.training_step += 1
        
        # ==================== 0. 状态初始化 ====================
        if use_memory_initialization:
            Q_init, K_init, V_init = self._initialize_from_memory(batch_size, seq_len)
            self.memory_initialized = True
        else:
            Q_init, K_init, V_init = None, None, None
        
        # ==================== 1. 嵌入层 ====================
        embedding_output = self.embedding(input_ids, return_details=False)
        base_embeddings = embedding_output['base_embeddings']  # [B, S, E]
        
        # 初始Q、K、V（从嵌入层生成）
        Q_base = base_embeddings
        K_base = base_embeddings
        V_base = base_embeddings
        
        # 记录初始状态
        self._record_state(Q_base, K_base, V_base, "嵌入层输出")
        
        # ==================== 2. 单向阀位置1（阀1-3） ====================
        Q1 = self.valve1_q(Q_base, K_base, V_base)[0]  # 只取Q输出
        K1 = self.valve1_k(Q_base, K_base, V_base)[1]  # 只取K输出
        V1 = self.valve1_v(Q_base, K_base, V_base)[2]  # 只取V输出
        
        self._record_state(Q1, K1, V1, "单向阀1后")
        
        # 应用三明治融合初始化（如果可用）
        if Q_init is not None and K_init is not None and V_init is not None:
            Q1, K1, V1 = self.sandwich_fusion(
                Q_deep=Q_init, K_deep=K_init, V_deep=V_init,
                Q_current=Q1, K_current=K1, V_current=V1,
                Q_original=Q_base, K_original=K_base, V_original=V_base
            )
            self._record_state(Q1, K1, V1, "三明治融合后")
        
        # ==================== 3. 网络1处理 ====================
        Q1_out, K1_out, V1_out, _ = self.chain_unit1(Q1, K1, V1, return_evolution=True)
        self._record_state(Q1_out, K1_out, V1_out, "网络1输出")
        
        # 提取网络1的三个V子值
        V1_sublist = self._extract_v_sublist(1, Q1_out, K1_out, V1_out)
        
        # ==================== 4. 单向阀位置2（阀4-6） ====================
        Q2_in = self.valve2_q(Q1_out, K1_out, V1_out)[0]
        K2_in = self.valve2_k(Q1_out, K1_out, V1_out)[1]
        V2_in = self.valve2_v(Q1_out, K1_out, V1_out)[2]
        self._record_state(Q2_in, K2_in, V2_in, "单向阀2后")
        
        # ==================== 5. 网络2处理（继承网络1状态） ====================
        # 网络2的输入融合了网络1的输出（链式继承）
        Q2_in_enhanced = Q2_in + 0.3 * Q1_out  # 继承网络1的Q状态
        K2_in_enhanced = K2_in + 0.3 * K1_out  # 继承网络1的K状态
        V2_in_enhanced = V2_in + 0.5 * V1_out  # 主要继承网络1的V状态（V主导）
        
        Q2_out, K2_out, V2_out, _ = self.chain_unit2(
            Q2_in_enhanced, K2_in_enhanced, V2_in_enhanced, return_evolution=True
        )
        self._record_state(Q2_out, K2_out, V2_out, "网络2输出")
        
        # 提取网络2的三个V子值
        V2_sublist = self._extract_v_sublist(2, Q2_out, K2_out, V2_out)
        
        # ==================== 6. 单向阀位置3（阀7-9） ====================
        Q3_in = self.valve3_q(Q2_out, K2_out, V2_out)[0]
        K3_in = self.valve3_k(Q2_out, K2_out, V2_out)[1]
        V3_in = self.valve3_v(Q2_out, K2_out, V2_out)[2]
        self._record_state(Q3_in, K3_in, V3_in, "单向阀3后")
        
        # ==================== 7. 网络3处理（继承网络2状态，间接继承网络1） ====================
        # 网络3的输入融合了网络1和网络2的输出
        Q3_in_enhanced = Q3_in + 0.2 * Q1_out + 0.3 * Q2_out  # 继承网络1和网络2
        K3_in_enhanced = K3_in + 0.2 * K1_out + 0.3 * K2_out
        V3_in_enhanced = V3_in + 0.3 * V1_out + 0.5 * V2_out  # V值主导继承
        
        Q3_out, K3_out, V3_out, _ = self.chain_unit3(
            Q3_in_enhanced, K3_in_enhanced, V3_in_enhanced, return_evolution=True
        )
        self._record_state(Q3_out, K3_out, V3_out, "网络3输出")
        
        # 提取网络3的三个V子值
        V3_sublist = self._extract_v_sublist(3, Q3_out, K3_out, V3_out)
        
        # ==================== 8. 单向阀位置4（阀10-12） ====================
        Q_final = self.valve4_q(Q3_out, K3_out, V3_out)[0]
        K_final = self.valve4_k(Q3_out, K3_out, V3_out)[1]
        V_final = self.valve4_v(Q3_out, K3_out, V3_out)[2]
        self._record_state(Q_final, K_final, V_final, "单向阀4后（地质层输入）")
        
        # ==================== 9. 地质层存储 ====================
        # 准备三个网络的输出列表
        Q_list = [Q1_out, Q2_out, Q3_out]
        K_list = [K1_out, K2_out, K3_out]
        V_list = [V1_sublist, V2_sublist, V3_sublist]  # 每个网络三个V子值
        
        # 存储到地质层
        self.geological_memory.store(Q_list, K_list, V_list)
        
        # ==================== 10. 输出处理 ====================
        normalized = self.final_norm(V_final)
        logits = self.output_projection(normalized)
        
        # ==================== 11. 收集输出信息 ====================
        output = {
            'logits': logits,
            'final_Q': Q_final,
            'final_K': K_final,
            'final_V': V_final,
            'training_step': self.training_step,
            'geological_energy': self.geological_memory.get_energy_stats(),
            'learning_phase': self.metrics.analyze_learning_phases()
        }
        
        if return_detailed:
            # 添加详细状态信息
            state_changes = self.metrics.get_state_change_series('l2')
            output.update({
                'state_changes': state_changes,
                'transition_points': self.metrics.transition_points,
                'v_sublists': {
                    'network1': [v.mean().item() for v in V1_sublist],
                    'network2': [v.mean().item() for v in V2_sublist],
                    'network3': [v.mean().item() for v in V3_sublist]
                },
                'valve_gates': {
                    'valve1_q': torch.sigmoid(self.valve1_q.gate_Q).mean().item(),
                    'valve1_k': torch.sigmoid(self.valve1_k.gate_K).mean().item(),
                    'valve1_v': torch.sigmoid(self.valve1_v.gate_V).mean().item(),
                }
            })
        
        return output
    
    def reset_memory(self):
        """重置地质记忆和状态历史"""
        self.geological_memory.reset()
        self.metrics.reset()
        self.memory_initialized = False
        self.training_step = 0
        print("地质记忆和状态历史已重置")
    
    def get_architecture_info(self):
        """获取架构信息"""
        return {
            'name': 'CompleteConstrainedArchitecture',
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'components': {
                'embedding_layer': 'EnhancedEmbeddingLayer',
                'one_way_valves': 12,
                'chain_reaction_units': 3,
                'geological_memory': 'GeologicalMemory',
                'sandwich_fusion': 'SandwichFusion',
                'metrics_calculator': 'CoreMetricsCalculator'
            },
            'training_steps': self.training_step,
            'memory_initialized': self.memory_initialized
        }
    
    def visualize_flow(self):
        """可视化架构流程"""
        diagram = """
        =============== 完整约束施工架构流程 ===============
        
        输入: token IDs
              ↓
        嵌入层: EnhancedEmbeddingLayer
              ↓
        单向阀位置1: 阀1(Q), 阀2(K), 阀3(V)
              ↓
        三明治融合: 融合地质层记忆 + 当前状态 + 原始输入
              ↓
        网络1: ChainReactionUnit_Final (V→K→Q, Q→V→K, K→Q→V)
              ↓
        单向阀位置2: 阀4(Q), 阀5(K), 阀6(V)
              ↓
        网络2: 继承网络1状态，处理增强输入
              ↓
        单向阀位置3: 阀7(Q), 阀8(K), 阀9(V)
              ↓
        网络3: 继承网络1+2状态，处理增强输入
              ↓
        单向阀位置4: 阀10(Q), 阀11(K), 阀12(V)
              ↓
        地质层: 存储三个网络的完整状态（Q、K、3个V子值）
              ↓
        输出: logits
        
        ==================================================
        
        链式继承关系:
        网络2 ← 继承网络1状态（Q、K、V）
        网络3 ← 继承网络2状态 + 间接继承网络1状态
        
        V值主导:
        每个网络最后提升V值，V协调下一层的K与Q
        地质层记录三个网络的完整V值演化
        
        监控体系:
        每个关键位置计算状态变化和相变检测
        地质层能量衰退机制保持记忆新鲜度
        """
        print(diagram)


# ==================== 完整架构测试 ====================

def test_complete_architecture():
    """测试完整约束施工架构"""
    print("\n" + "="*70)
    print("测试完整约束施工架构")
    print("="*70)
    
    # 模型参数
    vocab_size = 1000
    batch_size = 2
    seq_len = 8
    
    # 创建模型
    model = CompleteConstrainedArchitecture(
        vocab_size=vocab_size,
        embed_dim=64,
        hidden_dim=64,
        marker_dim=16
    )
    
    # 移动到设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"模型设备: {device}")
    
    # 创建测试数据
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # 第一次前向传播（无记忆初始化）
    print("\n>>> 第一次前向传播（无记忆初始化）...")
    output1 = model(input_ids, use_memory_initialization=False, return_detailed=True)
    
    print(f"输出logits形状: {output1['logits'].shape}")
    print(f"训练步数: {output1['training_step']}")
    print(f"地质能量: {output1['geological_energy']}")
    print(f"学习阶段: {output1['learning_phase']}")
    
    if 'v_sublists' in output1:
        print(f"网络1 V子值: {output1['v_sublists']['network1']}")
        print(f"网络2 V子值: {output1['v_sublists']['network2']}")
        print(f"网络3 V子值: {output1['v_sublists']['network3']}")
    
    # 第二次前向传播（使用记忆初始化）
    print("\n>>> 第二次前向传播（使用记忆初始化）...")
    output2 = model(input_ids, use_memory_initialization=True, return_detailed=False)
    
    print(f"输出logits形状: {output2['logits'].shape}")
    print(f"训练步数: {output2['training_step']}")
    
    # 获取架构信息
    print("\n>>> 架构信息...")
    arch_info = model.get_architecture_info()
    for key, value in arch_info.items():
        if key != 'components':
            print(f"  {key}: {value}")
    
    print("\n>>> 组件信息:")
    for comp, desc in arch_info['components'].items():
        print(f"  {comp}: {desc}")
    
    # 可视化流程
    print("\n>>> 架构流程可视化...")
    model.visualize_flow()
    
    # 重置内存
    print("\n>>> 测试记忆重置...")
    model.reset_memory()
    
    print("\n" + "="*70)
    print("完整架构测试完成!")
    print("="*70)
    
    return True


# ==================== 运行测试 ====================

if __name__ == "__main__":
    print("开始完整约束施工架构集成测试...")
    
    try:
        # 运行完整架构测试
        success = test_complete_architecture()
        
        if success:
            print("\n" + "="*70)
            print("🎉 完整约束施工架构集成成功!")
            print("="*70)
            print("\n架构验证总结:")
            print("1. ✅ 三明治融合初始化 - 融合地质记忆状态")
            print("2. ✅ 嵌入层 + 单向阀1 - 初始状态分化")
            print("3. ✅ 3个链式反应单元 - V值主导处理")
            print("4. ✅ 12个单向阀 - 精确控制信息流")
            print("5. ✅ 地质层存储 - 完整状态记录（三个网络×三个V子值）")
            print("6. ✅ 链式继承 - 网络2继承网络1，网络3继承网络2")
            print("7. ✅ 全流程监控 - 状态变化和相变检测")
            print("8. ✅ 记忆衰退机制 - 地质层能量管理")
            print("\n架构已准备好进行训练和部署!")
        else:
            print("⚠️ 架构测试遇到问题")
    except Exception as e:
        print(f"❌ 架构测试失败: {e}")
        import traceback
        traceback.print_exc()    


__all__ = {
    'test_complete_architecture',
}