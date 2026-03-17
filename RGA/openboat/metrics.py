"""
单向阀模块 | One-Way Valve Module
================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class OneWayValve(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 三个独立的门控参数，每个对应一个值 | Three independent gating parameters, each corresponding to a value
        self.gate_Q = nn.Parameter(torch.ones(1, 1, dim))
        self.gate_K = nn.Parameter(torch.ones(1, 1, dim))
        self.gate_V = nn.Parameter(torch.ones(1, 1, dim))
        
        # 三个独立的内部变换模块，每个模块处理一个值 | Three independent internal transformation modules, each module processing a value
        # Q模块：两层线性，中间用ReLU激活 | Q module: two linear layers, with ReLU activation in between
        self.Q_linear1 = nn.Linear(dim, dim)
        self.Q_linear2 = nn.Linear(dim, dim)
        self.Q_act = nn.ReLU()
        
        # K模块：一层线性+Tanh激活 | K module: one linear layer + Tanh activation
        self.K_linear = nn.Linear(dim, dim)
        self.K_act = nn.Tanh()
        
        # V模块：链式反应，两层线性，使用不同的激活函数 | V module: chain reaction, two linear layers
        self.V_linear1 = nn.Linear(dim, dim)
        self.V_linear2 = nn.Linear(dim, dim)
        self.V_act1 = nn.ReLU()
        self.V_act2 = nn.Tanh()

    def _gated_fusion(self, x, transformed, gate):
        """
        门控融合函数：将原始输入和变换后的输入按门控参数融合
        -----------------------------------------------
        Gating fusion function: fuses the original input and the transformed input according to the gating parameters        
        """
        g = torch.sigmoid(gate)  # 将门控参数映射到(0,1) | Map the gating parameters to (0, 1)
        return x * g + transformed * (1 - g)
    
    def forward(self, Q, K, V):
        # 处理Q值 | Processing Q-values
        Q_transformed = self.Q_linear1(Q)
        Q_transformed = self.Q_act(Q_transformed)
        Q_transformed = self.Q_linear2(Q_transformed)
        Q_out = self._gated_fusion(Q, Q_transformed, self.gate_Q)
        
        # 处理K值 | Processing K-values
        K_transformed = self.K_act(self.K_linear(K))
        K_out = self._gated_fusion(K, K_transformed, self.gate_K)
        
        # 处理V值 | Processing V-values
        V_transformed = self.V_linear1(V)
        V_transformed = self.V_act1(V_transformed)
        V_transformed = self.V_linear2(V_transformed)
        V_transformed = self.V_act2(V_transformed)
        V_out = self._gated_fusion(V, V_transformed, self.gate_V)
        
        return Q_out, K_out, V_out
    
class EnhancedEmbeddingLayer(nn.Module):
    """
    增强嵌入层 - 基于概念图节点特征（Q, K, V）生成标记向量
    遵循规则：不进行传统归一化，仅使用原始密度公式体系
    --------------------------------------------------
    Enhanced Embedding Layer - Generate token vectors based on concept graph node features (Q, K, V)
    Follow the rule: Do not perform traditional normalization, only use the original density formula system
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, marker_dim: int = 32):
        """
        初始化增强嵌入层
        -----------------
        Initialize the enhanced embedding layer
        Args:
            vocab_size: 词汇表大小 | Vocabulary size
            embed_dim: 基础嵌入维度 | Basic embedding dimension
            marker_dim: 标记向量维度 | Mark vector dimension
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.marker_dim = marker_dim
        
        # ==================== 基础嵌入组件 | Basic Embedding Components ====================
        # 词嵌入（可学习） | Token embedding (learnable)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码（固定正弦编码）- 现在使用缓冲区，会自动移动到设备 | Positional encoding (fixed sine encoding) - now using buffer, will automatically move to device
        self.register_buffer('position_encoding', self._create_positional_encoding(max_len=512, d_model=embed_dim))
        
        # ==================== 标记生成组件 ====================
        # 概念特征映射层：将(Q, K, V)特征映射到标记空间
        # 输入: 3个特征值 (Q_local, K_local, V_local)
        # 输出: marker_dim维标记向量
        # ==================== Marker Generation Component ====================
        # Concept Feature Mapping Layer: Maps (Q, K, V) features to the marker space
        # Input: 3 feature values (Q_local, K_local, V_local)
        # Output: marker_dim-dimensional marker vector
        self.marker_projection = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, marker_dim),
            nn.Tanh()
        )
        
        # 默认标记（用于未识别的词）| Default tag (for unrecognized words)
        self.default_marker = nn.Parameter(torch.randn(marker_dim) * 0.1)
        
        # ==================== 概念特征缓存 | Concept Feature Cache ====================
        # 存储词汇索引到概念特征的映射 | Store the mapping from vocabulary indices to conceptual features
        self.register_buffer('concept_features', torch.zeros(vocab_size, 3))
        self.register_buffer('has_concept', torch.zeros(vocab_size, dtype=torch.bool))
        
        # ==================== 统计跟踪 | Statistics Tracking ====================
        self.register_buffer('activation_stats', torch.zeros(3))  # 统计Q, K, V激活 | Count Q, K, V activations
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        创建正弦位置编码（非学习参数）
        ---------------------------
        Create sinusoidal positional encoding (non-learnable parameters)
        
        Args:
            max_len: 最大序列长度 | maximum sequence length
            d_model: 模型维度 | model dimension
            
        Returns:
            位置编码张量 [1, max_len, d_model]
            positional encoding tensor [1, max_len, d_model]
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
        ----------------------------
        Set the conceptual features (Q, K, V) of words
        
        Args:
            word_indices: 词汇索引 [n_words] | Vocabulary Index [n_words]
            features: 概念特征 [n_words, 3] (Q, K, V) | Concept features [n_words, 3] (Q, K, V)
        """
        if len(word_indices) != len(features):
            raise ValueError(f"词汇索引数量| Number of vocabulary index entries({len(word_indices)})与特征数量 | and the number of features({len(features)})不匹配 | Mismatch")
        
        # 更新概念特征缓存 - 确保在同一设备
        device = self.concept_features.device
        self.concept_features[word_indices] = features.to(device)
        self.has_concept[word_indices] = True
        
        print(f"已设置 | Set up {len(word_indices)} 个词汇的概念特征 | The conceptual features of a word")
    
    def get_concept_features(self, word_idx: int) -> torch.Tensor:
        """
        获取指定词汇的概念特征
        --------------------
        Obtain the conceptual features of a specified vocabulary
        
        Args:
            word_idx: 词汇索引 | Vocabulary Index
            
        Returns:
            概念特征张量 [3] 或 None（如果无概念特征）
            ----------------------------------------
            Concept feature tensor [3] or None (if there are no concept features)
        """
        if word_idx >= self.vocab_size:
            raise ValueError(f"词汇索引 | Vocabulary Index{word_idx} 超出词汇表范围 | Beyond the scope of the vocabulary list{self.vocab_size}")
        
        if self.has_concept[word_idx]:
            return self.concept_features[word_idx].clone()
        else:
            return None
    
    def _calculate_local_density(self, batch_markers: torch.Tensor) -> Dict[str, float]:
        """
        计算本地认知密度（使用你的公式体系）
        -------------------------------
        Calculate the local cognitive density (using your formula system)
        
        Args:
            batch_markers: 标记向量 [batch_size, seq_len, marker_dim] | Marker vector [batch_size, seq_len, marker_dim]
            
        Returns:
            密度统计信息 | Density statistical information
        """
        if batch_markers.numel() == 0:
            return {'density': 0.0, 'stability': 0.0}
        
        # 计算标记向量间的余弦相似度矩阵 | Compute cosine similarity matrix between marker vectors
        batch_size, seq_len, marker_dim = batch_markers.shape
        flat_markers = batch_markers.view(-1, marker_dim)  # [batch*seq_len, marker_dim]
        
        # 计算余弦相似度 | Compute cosine similarity
        norm_markers = F.normalize(flat_markers, p=2, dim=1)
        similarity_matrix = torch.mm(norm_markers, norm_markers.T)  # [N, N]
        
        # 应用你的密度公式：D = 2m/((N+1)N)，其中m为连接数 | Apply your density formula: D = 2m
        N = similarity_matrix.size(0)
        
        # 阈值过滤：保留相似度大于0.3的连接 | Threshold filtering: retain connections with similarity greater than 0.3
        threshold = 0.3
        connections = (similarity_matrix > threshold).float()
        M = connections.sum().item() / 2  # 无向图，每条边计算两次 | Undirected graph, each edge is counted twice
        
        # 静态密度公式 | Dynamic density change (compared with historical statistics)
        if N > 1:
            static_density = (2 * M) / ((N + 1) * N)
        else:
            static_density = 0.0
        
        # 动态密度变化（与历史统计比较）| Dynamic density change (compared with historical statistics)
        prev_stats = self.activation_stats
        
        # 计算当前统计：标记向量的平均范数 | Calculate the current statistics: the average norm of the marker vectors
        current_stats = torch.tensor([
            batch_markers.mean().item(),
            batch_markers.std().item(),
            similarity_matrix.mean().item()
        ], device=batch_markers.device)
        
        # 更新统计 | Update statistics
        self.activation_stats = 0.9 * self.activation_stats + 0.1 * current_stats
        
        # 计算稳定性：统计变化率 | Calculation stability: statistical rate of change
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
        -------
        Forward propagation
        
        Args:
            input_ids: 输入词索引 [batch_size, seq_len] | Input word index [batch_size, seq_len]
            return_details: 是否返回详细统计信息 | Whether to return detailed statistical information
            
        Returns:
            包含基础嵌入和标记的字典 | A dictionary containing basic embeddings and tokens
        """
        batch_size, seq_len = input_ids.shape
        
        # ==================== 基础嵌入 | Basic embedding ====================
        # 词嵌入 | Word embedding
        token_emb = self.token_embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # 位置编码 - 确保在正确的设备上 | Position encoding - Ensure it is on the correct device
        if seq_len <= self.position_encoding.size(1):
            pos_emb = self.position_encoding[:, :seq_len, :]
            # 确保位置编码与token嵌入在同一设备 | Ensure position encoding is on the same device as token embedding
            if pos_emb.device != token_emb.device:
                pos_emb = pos_emb.to(token_emb.device)
        else:
            # 动态扩展位置编码 - 确保在正确的设备上 | Dynamic expansion position encoding - Ensure it is on the correct device
            pos_emb = self._create_positional_encoding(seq_len, self.embed_dim).to(input_ids.device)
        
        # 基础嵌入 = 词嵌入 + 位置编码 | Basic embedding = word embedding + positional encoding
        base_embeddings = token_emb + pos_emb  # [batch, seq_len, embed_dim]
        
        # ==================== 标记生成 | Token Generation ====================
        markers = []
        
        # 为每个词生成标记 | Generate a token for each word
        for batch_idx in range(batch_size):
            batch_markers = []
            
            for pos_idx in range(seq_len):
                word_idx = input_ids[batch_idx, pos_idx].item()
                
                # 获取概念特征 - 确保在同一设备 | Obtain concept features - ensure it is on the same device
                concept_feat = self.get_concept_features(word_idx)
                
                if concept_feat is not None:
                    # 有概念特征：投影到标记空间 | With concept features: project to the token space
                    concept_feat = concept_feat.to(input_ids.device).unsqueeze(0)
                    marker = self.marker_projection(concept_feat)  # [1, marker_dim]
                else:
                    # 无概念特征：使用默认标记 | Without concept features: use default token
                    marker = self.default_marker.unsqueeze(0).to(input_ids.device)  # [1, marker_dim]
                
                batch_markers.append(marker)
            
            # 拼接当前批次的标记 | Concatenate the current batch's markers
            batch_markers = torch.cat(batch_markers, dim=0)  # [seq_len, marker_dim]
            markers.append(batch_markers)
        
        # 组合所有批次的标记 | Combine all batch's markers
        markers = torch.stack(markers, dim=0)  # [batch, seq_len, marker_dim]
        
        # ==================== 统计信息 ====================
        stats = {}
        if return_details:
            # 计算本地认知密度 | Calculate local cognitive density
            density_info = self._calculate_local_density(markers)
            
            # 收集输出统计 | Collect output statistics
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
        ------------------
        Analyze the distribution of concepts in the input
    
        Args:
            input_ids: 输入词索引 [batch_size, seq_len] | Input word indices [batch_size, seq_len]
        
        Returns:
            概念分布统计 | Concept distribution statistics
        """
        batch_size, seq_len = input_ids.shape
        total_tokens = batch_size * seq_len
    
        # 获取has_concept，确保在正确的设备上 | Get has_concept and ensure it is on the correct device
        # 确保has_concept和input_ids在同一设备 | Ensure has_concept and input_ids are on
        device = input_ids.device
        has_concept_on_device = self.has_concept.to(device)
    
        # 统计概念覆盖 | Count concepts covered
        has_concept_mask = has_concept_on_device[input_ids]  # [batch, seq_len]
        concept_count = has_concept_mask.sum().item()
        concept_coverage = concept_count / total_tokens if total_tokens > 0 else 0
    
        # 获取有概念的词的特征 | Get features of words with concepts
        if concept_count > 0:
            # 获取有概念的索引 | Get indices of words with concepts
            concept_indices = input_ids[has_concept_mask]  # [concept_count]
        
            # 确保concept_features在正确的设备上 | Ensure concept_features is on the correct device
            concept_feats_on_device = self.concept_features.to(device)
            concept_feats = concept_feats_on_device[concept_indices]  # [concept_count, 3]
        
            # 计算特征统计 | Calculate feature statistics
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
        """重置统计信息 | Reset statistics information."""
        self.activation_stats.zero_()

__all__ = [
    'OneWayValve',
    'EnhancedEmbeddingLayer',
]
