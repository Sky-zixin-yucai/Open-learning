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

# ==================== 测试代码替换部分 ====================
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import time
    import json
    import os
    from pathlib import Path
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    # ==================== 配置参数 ====================
    CONFIG = {
        "model_path": "D:/桌面/HAI/models/Qwen2.5-1.5B",  # 本地模型路径
        "vocab_size": 151936,  # Qwen2.5的词汇表大小
        "embed_dim": 1024,  # Qwen2.5的嵌入维度
        "marker_dim": 64,  # 标记向量维度
        "batch_size": 2,  # 批处理大小
        "seq_len": 256,  # 序列长度
        "test_samples": 20,  # 测试样本数
        "save_dir": "./analysis_results",  # 结果保存目录
        "device": "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测设备
    }
    
    # 创建保存目录
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    print("=" * 80)
    print("增强嵌入层状态分析测试")
    print(f"设备: {CONFIG['device']}")
    print("=" * 80)
    
    # 移动设备
    device = torch.device(CONFIG["device"])
    
    # ==================== 1. 初始化核心组件 ====================
    print("\n1. 初始化核心组件...")
    
    # 初始化度量计算器
    metrics_calculator = CoreMetricsCalculator()
    
    # 初始化增强嵌入层
    embedding_layer = EnhancedEmbeddingLayer(
        vocab_size=CONFIG["vocab_size"],
        embed_dim=CONFIG["embed_dim"],
        marker_dim=CONFIG["marker_dim"]
    ).to(device)
    
    # 初始化单向阀组合（用于测试）
    one_way_valve_module = 单向阀组合(dim=CONFIG["embed_dim"]).to(device)
    
    print(f"   嵌入层参数: vocab_size={CONFIG['vocab_size']}, embed_dim={CONFIG['embed_dim']}")
    print(f"   度量计算器已初始化")
    print(f"   单向阀组合已初始化")
    print(f"   单向阀组合参数量: {sum(p.numel() for p in one_way_valve_module.parameters()):,}")
    
    # ==================== 2. 加载开源模型 ====================
    print("\n2. 加载开源模型 Qwen2.5-1.5B...")
    
    try:
        # 尝试加载本地模型
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["model_path"],
            trust_remote_code=True,
            padding_side='right'
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_path"],
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if CONFIG["device"] == "cuda" else None
        ).to(device)
        
        print(f"   模型加载成功: {CONFIG['model_path']}")
        print(f"   词汇表大小: {len(tokenizer)}")
        print(f"   模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # 获取模型的嵌入层作为参考
        if hasattr(model, 'get_input_embeddings'):
            original_embedding = model.get_input_embeddings()
            print(f"   原始嵌入层维度: {original_embedding.embedding_dim}")
            
            # 提取模型的部分嵌入权重作为概念特征
            with torch.no_grad():
                # 随机选择一些词汇作为概念词
                n_concepts = min(5000, CONFIG["vocab_size"])
                concept_indices = torch.randint(0, CONFIG["vocab_size"], (n_concepts,)).to(device)
                
                # 从原始嵌入层获取权重作为特征
                if hasattr(original_embedding, 'weight'):
                    embed_weights = original_embedding.weight
                    
                    # 计算每个词向量的统计特征作为概念特征
                    concept_features = []
                    for idx in concept_indices:
                        vec = embed_weights[idx].to(device).float()
                        
                        # 计算Q, K, V特征
                        # Q: 向量均值（全局特征）
                        Q_val = vec.mean().item()
                        # K: 向量标准差（复杂度）
                        K_val = vec.std().item()
                        # V: 向量最大最小值差（动态范围）
                        V_val = (vec.max() - vec.min()).item()
                        
                        concept_features.append([Q_val, K_val, V_val])
                    
                    concept_features = torch.tensor(concept_features).to(device)
                    
                    # 设置到增强嵌入层
                    embedding_layer.set_concept_features(concept_indices.cpu(), concept_features.cpu())
                    print(f"   已设置 {n_concepts} 个概念特征")
        
    except Exception as e:
        print(f"   模型加载失败: {e}")
        print("   使用随机初始化...")
        tokenizer = None
        model = None
        
        # 使用随机概念特征
        n_concepts = 5000
        concept_indices = torch.randint(0, CONFIG["vocab_size"], (n_concepts,))
        
        # 随机生成概念特征
        Q_vals = torch.randn(n_concepts, 1) * 2.0 + 5.0
        K_vals = torch.randn(n_concepts, 1) * 5.0 + 10.0
        V_vals = torch.abs(K_vals / (Q_vals + 1e-6))
        
        concept_features = torch.cat([Q_vals, K_vals, V_vals], dim=1)
        embedding_layer.set_concept_features(concept_indices, concept_features)
        print(f"   已设置 {n_concepts} 个随机概念特征")
    
    # ==================== 3. 模拟训练过程 ====================
    print("\n3. 模拟训练过程并记录状态...")
    
    # 存储所有时刻的状态
    all_states = []
    all_metrics = []
    all_density_info = []
    all_valve_results = []  # 存储单向阀测试结果
    
    # 记录开始时间
    start_time = time.time()
    
    # 模拟多个训练步骤
    for step in tqdm(range(CONFIG["test_samples"]), desc="训练步骤"):
        step_start_time = time.time()
        
        # 生成随机输入
        if tokenizer is not None:
            # 如果有tokenizer，生成有意义的文本
            texts = [
                "The quick brown fox jumps over the lazy dog while the artificial intelligence analyzes the pattern.",
                "Machine learning algorithms are transforming the way we process natural language and understand semantics.",
                "Quantum computing represents a paradigm shift in computational power and algorithmic complexity.",
                "The future of technology depends on innovative approaches to data representation and knowledge extraction.",
                "Cognitive architectures aim to replicate human-like reasoning and decision-making processes.",
                "Deep neural networks have revolutionized computer vision, natural language processing, and speech recognition.",
                "Reinforcement learning enables agents to learn optimal behaviors through interaction with environments.",
                "Transfer learning allows models to leverage knowledge from one domain to improve performance in another.",
                "Explainable AI is crucial for building trust and understanding in automated decision systems.",
                "Multi-modal learning integrates information from different sensory modalities for richer understanding."
            ]
            
            # 随机选择一个文本
            text = texts[step % len(texts)]
            
            # 分词
            inputs = tokenizer(
                text, 
                return_tensors="pt",
                max_length=CONFIG["seq_len"],
                truncation=True,
                padding="max_length"
            )
            input_ids = inputs["input_ids"].to(device)
        else:
            # 随机生成输入
            input_ids = torch.randint(
                0, CONFIG["vocab_size"], 
                (CONFIG["batch_size"], CONFIG["seq_len"])
            ).to(device)
        
        # ==================== 增强嵌入层前向传播 ====================
        with torch.no_grad():
            embedding_output = embedding_layer(input_ids, return_details=True)
        
        # 提取Q, K, V特征（使用概念特征的均值）
        concept_feats = embedding_output['concept_features']  # [batch, seq_len, 3]
        Q_features = concept_feats[:, :, 0].mean(dim=1, keepdim=True)  # [batch, 1]
        K_features = concept_feats[:, :, 1].mean(dim=1, keepdim=True)  # [batch, 1]
        V_features = concept_feats[:, :, 2].mean(dim=1, keepdim=True)  # [batch, 1]
        
        # 记录当前状态到计算器（使用Q, K, V特征）
        metrics_calculator.record_state(Q_features, K_features, V_features)
        
        # ==================== 计算状态变化和相变检测 ====================
        state_change = 0.0
        phase_delta = 0.0
        is_transition = False
        
        if len(metrics_calculator.state_history) >= 2:
            # 获取历史状态（从计算器中获取）
            prev_state = metrics_calculator.state_history[-2]
            curr_state = metrics_calculator.state_history[-1]
            
            # 计算单网络变化量
            state_change = metrics_calculator.compute_state_change(
                curr_state['Q'], curr_state['K'], curr_state['V'],
                prev_state['Q'], prev_state['K'], prev_state['V'],
                norm_type='l2'
            )
            
            # 检测相变
            phase_delta, is_transition = metrics_calculator.detect_phase_transition(
                curr_state['Q'], curr_state['K'], curr_state['V'],
                prev_state['Q'], prev_state['K'], prev_state['V']
            )
            
        # 记录度量
        step_metrics = {
            'step': step,
            'state_change': state_change,
            'phase_delta': phase_delta,
            'is_transition': is_transition,
            'timestamp': time.time()
        }
        all_metrics.append(step_metrics)
        
        if is_transition:
            print(f"\n   步骤 {step}: 检测到相变! Δ={phase_delta:.4f}")
        
        # ==================== 测试CoreMetricsCalculator的单向阀 ====================
        # 对Q, K, V分别应用不同的单向阀模式
        Q_valve_detach = metrics_calculator.apply_one_way_valve(
            Q_features, mode='detach'
        )
        K_valve_gate_1 = metrics_calculator.apply_one_way_valve(
            K_features, mode='gate', gate_value=1
        )
        K_valve_gate_0 = metrics_calculator.apply_one_way_valve(
            K_features, mode='gate', gate_value=0
        )
        V_valve_gate_1 = metrics_calculator.apply_one_way_valve(
            V_features, mode='gate', gate_value=1
        )
        
        # ==================== 测试单向阀组合模块 ====================
        # 准备测试数据（扩展维度以匹配模块输入）
        batch_size = Q_features.size(0)
        
        # 创建测试张量 [batch, seq_len=1, dim]
        Q_test = torch.randn(batch_size, 1, CONFIG["embed_dim"]).to(device) * 0.1 + Q_features.unsqueeze(-1).expand(-1, -1, CONFIG["embed_dim"])
        K_test = torch.randn(batch_size, 1, CONFIG["embed_dim"]).to(device) * 0.1 + K_features.unsqueeze(-1).expand(-1, -1, CONFIG["embed_dim"])
        V_test = torch.randn(batch_size, 1, CONFIG["embed_dim"]).to(device) * 0.1 + V_features.unsqueeze(-1).expand(-1, -1, CONFIG["embed_dim"])
        
        # 通过单向阀组合模块
        Q_valve_out, K_valve_out, V_valve_out = one_way_valve_module(Q_test, K_test, V_test)
        
        # ==================== 三网络堆叠 ====================
        # 创建三个不同的Q矩阵（模拟不同视角）
        Q_list = [
            Q_test,
            Q_test * 0.8 + torch.randn_like(Q_test) * 0.2,
            Q_test * 1.2 - torch.randn_like(Q_test) * 0.2
        ]
        
        # 应用三网络堆叠
        Q_stack = metrics_calculator.stack_three_networks(Q_list)
        
        # ==================== 记录状态信息 ====================
        step_state = {
            'step': step,
            'input_shape': input_ids.shape,
            'embedding_output_shape': embedding_output['base_embeddings'].shape,
            'marker_shape': embedding_output['markers'].shape,
            'Q_mean': Q_features.mean().item(),
            'K_mean': K_features.mean().item(),
            'V_mean': V_features.mean().item(),
            'state_change': state_change,
            'phase_delta': phase_delta,
            'is_transition': is_transition,
            'density_info': embedding_output.get('stats', {}).get('density_info', {}),
            'concept_coverage': embedding_output.get('stats', {}).get('concept_coverage', 0.0),
            'Q_stack_mean': Q_stack.mean().item(),
            'valve_results': {
                'Q_valve_detach_mean': Q_valve_detach.mean().item(),
                'K_valve_gate_1_mean': K_valve_gate_1.mean().item(),
                'K_valve_gate_0_mean': K_valve_gate_0.mean().item(),
                'V_valve_gate_1_mean': V_valve_gate_1.mean().item(),
                'one_way_valve_module': {
                    'Q_out_mean': Q_valve_out.mean().item(),
                    'K_out_mean': K_valve_out.mean().item(),
                    'V_out_mean': V_valve_out.mean().item(),
                    'gate_Q': one_way_valve_module.gate_Q.mean().item(),
                    'gate_K': one_way_valve_module.gate_K.mean().item(),
                    'gate_V': one_way_valve_module.gate_V.mean().item()
                }
            },
            'step_time': time.time() - step_start_time
        }
        
        all_states.append(step_state)
        all_valve_results.append(step_state['valve_results'])
        
        # 记录密度信息
        if 'stats' in embedding_output and 'density_info' in embedding_output['stats']:
            density_info = embedding_output['stats']['density_info']
            density_info['step'] = step
            all_density_info.append(density_info)
        
        # ==================== 分析学习阶段 ====================
        if step % 5 == 0 and step > 0:
            phase_analysis = metrics_calculator.analyze_learning_phases()
            print(f"\n   步骤 {step} 学习阶段分析: {phase_analysis.get('phase', '未知')}")
            print(f"   平均变化量: {phase_analysis.get('avg_change', 0):.4f}")
            print(f"   相变点数量: {phase_analysis.get('transition_points', 0)}")
    
    total_time = time.time() - start_time
    print(f"\n   总模拟时间: {total_time:.2f}秒")
    print(f"   平均每步时间: {total_time/CONFIG['test_samples']:.4f}秒")
    
    # ==================== 4. 分析结果 ====================
    print("\n4. 分析结果...")
    
    # 统计信息
    total_steps = len(all_states)
    total_transitions = sum(1 for s in all_states if s.get('is_transition', False))
    
    # 计算平均变化
    state_changes = [s['state_change'] for s in all_states if s['state_change'] > 0]
    avg_state_change = sum(state_changes) / len(state_changes) if state_changes else 0
    
    # 计算概念覆盖率
    concept_coverages = [s['concept_coverage'] for s in all_states]
    avg_concept_coverage = sum(concept_coverages) / len(concept_coverages) if concept_coverages else 0
    
    # 密度分析
    if all_density_info:
        avg_density = sum(d.get('static_density', 0) for d in all_density_info) / len(all_density_info)
        avg_stability = sum(d.get('stability', 0) for d in all_density_info) / len(all_density_info)
        avg_connections = sum(d.get('connections', 0) for d in all_density_info) / len(all_density_info)
    else:
        avg_density = 0
        avg_stability = 0
        avg_connections = 0
    
    # 单向阀分析
    if all_valve_results:
        avg_Q_detach = sum(v['Q_valve_detach_mean'] for v in all_valve_results) / len(all_valve_results)
        avg_module_Q_out = sum(v['one_way_valve_module']['Q_out_mean'] for v in all_valve_results) / len(all_valve_results)
        avg_gate_Q = sum(v['one_way_valve_module']['gate_Q'] for v in all_valve_results) / len(all_valve_results)
    
    # 打印总体分析
    print("\n" + "=" * 80)
    print("总体分析结果")
    print("=" * 80)
    print(f"总训练步骤: {total_steps}")
    print(f"总相变点: {total_transitions}")
    print(f"平均状态变化量: {avg_state_change:.4f}")
    print(f"平均概念覆盖率: {avg_concept_coverage:.2%}")
    print(f"平均静态密度: {avg_density:.4f}")
    print(f"平均稳定性: {avg_stability:.4f}")
    print(f"平均连接数: {avg_connections:.1f}")
    
    if all_valve_results:
        print(f"平均Q单向阀(detach)输出: {avg_Q_detach:.4f}")
        print(f"平均单向阀组合Q输出: {avg_module_Q_out:.4f}")
        print(f"平均门控参数Q: {avg_gate_Q:.4f}")
    
    # 打印相变点详情
    if metrics_calculator.transition_points:
        print("\n相变点详情:")
        for i, tp in enumerate(metrics_calculator.transition_points):
            print(f"  相变点 {i+1}: 步骤{tp['timestamp']}, Δ={tp['delta']:.4f}")
    
    # 学习阶段最终分析
    final_phase_analysis = metrics_calculator.analyze_learning_phases()
    print(f"\n最终学习阶段: {final_phase_analysis.get('phase', '未知')}")
    print(f"平均变化量: {final_phase_analysis.get('avg_change', 0):.4f}")
    print(f"当前稳定性: {final_phase_analysis.get('current_stability', 0):.4f}")
    
    # ==================== 5. 保存结果 ====================
    print(f"\n5. 保存分析结果到 {CONFIG['save_dir']}...")
    
    # 保存所有状态
    states_file = os.path.join(CONFIG["save_dir"], "embedding_states.json")
    with open(states_file, 'w', encoding='utf-8') as f:
        json.dump(all_states, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存所有度量
    metrics_file = os.path.join(CONFIG["save_dir"], "embedding_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存密度信息
    density_file = os.path.join(CONFIG["save_dir"], "density_info.json")
    with open(density_file, 'w', encoding='utf-8') as f:
        json.dump(all_density_info, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存单向阀结果
    valve_file = os.path.join(CONFIG["save_dir"], "valve_results.json")
    with open(valve_file, 'w', encoding='utf-8') as f:
        json.dump(all_valve_results, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存配置
    config_file = os.path.join(CONFIG["save_dir"], "test_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)
    
    # 保存最终分析报告
    final_report = {
        "total_steps": total_steps,
        "total_transitions": total_transitions,
        "avg_state_change": avg_state_change,
        "avg_concept_coverage": avg_concept_coverage,
        "avg_density": avg_density,
        "avg_stability": avg_stability,
        "avg_connections": avg_connections,
        "final_phase": final_phase_analysis.get('phase', '未知'),
        "final_stability": final_phase_analysis.get('current_stability', 0),
        "transition_points": metrics_calculator.transition_points,
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": total_time,
        "avg_step_time": total_time / CONFIG['test_samples'] if CONFIG['test_samples'] > 0 else 0
    }
    
    report_file = os.path.join(CONFIG["save_dir"], "final_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"   状态信息已保存: {states_file}")
    print(f"   度量信息已保存: {metrics_file}")
    print(f"   密度信息已保存: {density_file}")
    print(f"   单向阀结果已保存: {valve_file}")
    print(f"   最终报告已保存: {report_file}")
    
    # ==================== 6. 生成可视化摘要 ====================
    print("\n6. 生成分析摘要...")
    
    print("\n" + "=" * 80)
    print("嵌入层性能摘要")
    print("=" * 80)
    
    # 嵌入层维度信息
    print(f"嵌入层维度:")
    print(f"  - 基础嵌入: {CONFIG['embed_dim']} 维")
    print(f"  - 标记向量: {CONFIG['marker_dim']} 维")
    print(f"  - 总输出维度: {CONFIG['embed_dim'] + CONFIG['marker_dim']} 维")
    
    # 内存占用估计
    embed_params = sum(p.numel() for p in embedding_layer.parameters())
    valve_params = sum(p.numel() for p in one_way_valve_module.parameters())
    print(f"\n参数统计:")
    print(f"  - 嵌入层可训练参数: {embed_params:,}")
    print(f"  - 单向阀组合可训练参数: {valve_params:,}")
    print(f"  - 总可训练参数: {embed_params + valve_params:,}")
    print(f"  - 嵌入层内存占用: {embed_params * 4 / (1024**2):.2f} MB (FP32)")
    print(f"  - 单向阀组合内存占用: {valve_params * 4 / (1024**2):.2f} MB (FP32)")
    
    # 计算效率
    if total_steps > 0:
        avg_inference_time = total_time / total_steps
        print(f"\n效率统计:")
        print(f"  - 平均推理时间: {avg_inference_time*1000:.2f} ms/步")
        print(f"  - 处理速度: {CONFIG['batch_size'] * CONFIG['seq_len'] / avg_inference_time:.0f} tokens/s")
    
    # 概念特征统计
    if tokenizer is not None:
        concept_stats = embedding_layer.analyze_concept_distribution(input_ids)
        print(f"\n概念特征统计:")
        for key, value in concept_stats.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
    
    # 密度分析摘要
    if all_density_info:
        final_density = all_density_info[-1]
        print(f"\n最终密度状态:")
        for key, value in final_density.items():
            if isinstance(value, (int, float)):
                print(f"  - {key}: {value:.4f}")
    
    # 单向阀组合分析
    print(f"\n单向阀组合门控参数状态:")
    print(f"  - gate_Q: {one_way_valve_module.gate_Q.mean().item():.4f} ± {one_way_valve_module.gate_Q.std().item():.4f}")
    print(f"  - gate_K: {one_way_valve_module.gate_K.mean().item():.4f} ± {one_way_valve_module.gate_K.std().item():.4f}")
    print(f"  - gate_V: {one_way_valve_module.gate_V.mean().item():.4f} ± {one_way_valve_module.gate_V.std().item():.4f}")
    
    # 四大公式测试结果摘要
    print(f"\n四大公式测试结果:")
    print(f"  1. 单网络变化量公式: 成功测试，平均变化量={avg_state_change:.4f}")
    print(f"  2. 相变检测公式: 检测到{total_transitions}个相变点")
    print(f"  3. 三网络堆叠公式: 成功测试，平均堆叠输出={all_states[-1]['Q_stack_mean']:.4f}")
    print(f"  4. 单向阀公式: 成功测试detach和gate两种模式")
    
    print("\n" + "=" * 80)
    print("测试完成! 所有结果已保存到分析目录")
    print("=" * 80)
    
    # ==================== 7. 生成简单可视化（可选） ====================
    try:
        # 创建可视化目录
        viz_dir = os.path.join(CONFIG["save_dir"], "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 生成状态变化图
        if len(all_states) > 1:
            steps = [s['step'] for s in all_states]
            state_changes = [s['state_change'] for s in all_states]
            phase_deltas = [s['phase_delta'] for s in all_states]
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(steps, state_changes, 'b-', label='状态变化量')
            plt.xlabel('训练步骤')
            plt.ylabel('状态变化量')
            plt.title('单网络变化量公式监控')
            plt.grid(True)
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(steps, phase_deltas, 'r-', label='相变检测')
            # 标记相变点
            transition_steps = [s['step'] for s in all_states if s.get('is_transition', False)]
            if transition_steps:
                transition_values = [all_states[s]['phase_delta'] for s in transition_steps]
                plt.scatter(transition_steps, transition_values, color='red', s=100, zorder=5, label='相变点')
            plt.xlabel('训练步骤')
            plt.ylabel('相变检测值')
            plt.title('相变检测公式监控')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'state_changes.png'), dpi=150)
            plt.close()
            
            print(f"\n7. 可视化图表已保存到: {viz_dir}")
            
    except Exception as e:
        print(f"\n7. 可视化生成失败: {e}")