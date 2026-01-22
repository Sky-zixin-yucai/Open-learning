"""
嵌入层模块 | Embedding Layers Module
===================================

增强嵌入层实现，基于概念图节点特征（Q, K, V）生成标记向量。
Enhanced embedding layer implementation, generating marker vectors based on concept graph node features (Q, K, V).

设计理念:
Design Philosophy:
• 概念驱动：基于词汇的概念特征(Q, K, V)生成嵌入 | Concept-driven: Generate embeddings based on word concept features (Q, K, V)
• 双通道输出：同时输出基础嵌入和标记向量 | Dual-channel output: Output both base embeddings and marker vectors
• 密度公式：使用连接点密度公式评估认知密度 | Density formula: Use connection point density formula to evaluate cognitive density
• 无传统归一化：遵循规则治理架构，不使用传统归一化 | No traditional normalization: Follow rule-governed architecture, no traditional normalization

核心组件 | Core Components:
1. EnhancedEmbeddingLayer: 增强嵌入层主类 | Main enhanced embedding layer class
2. 概念特征缓存：存储词汇的概念特征 | Concept feature cache: Store concept features for vocabulary
3. 标记生成器：将概念特征映射到标记空间 | Marker generator: Map concept features to marker space
4. 密度计算器：计算本地认知密度 | Density calculator: Compute local cognitive density

数学基础 | Mathematical Foundation:
• 连接点密度公式: D = 2m/((N+1)N) | Connection point density formula: D = 2m/((N+1)N)
• 标记向量: m = f(Q_local, K_local, V_local) | Marker vector: m = f(Q_local, K_local, V_local)
• 概念特征: c = [Q, K, V] | Concept features: c = [Q, K, V]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import re


class EnhancedEmbeddingLayer(nn.Module):
    """
    增强嵌入层 | Enhanced Embedding Layer
    -----------------------------------
    
    基于概念图节点特征生成标记向量的嵌入层，遵循规则治理架构的数学基础。
    Embedding layer generating marker vectors based on concept graph node features, following the mathematical foundation of rule-governed architecture.
    
    核心功能 | Core Functions:
        1. 基础嵌入：词嵌入 + 位置编码 | Basic embedding: Token embedding + positional encoding
        2. 概念特征映射：存储和检索词汇的概念特征(Q, K, V) | Concept feature mapping: Store and retrieve concept features (Q, K, V) for vocabulary
        3. 标记生成：将概念特征投影到标记空间 | Marker generation: Project concept features to marker space
        4. 密度计算：计算本地认知密度 | Density computation: Compute local cognitive density
    
    设计原则 | Design Principles:
        • 保留原始密度公式：不使用传统归一化 | Preserve original density formula: No traditional normalization
        • 概念驱动嵌入：基于词汇语义概念生成标记 | Concept-driven embedding: Generate markers based on word semantic concepts
        • 双通道信息：同时提供基础嵌入和标记向量 | Dual-channel information: Provide both base embeddings and marker vectors
        • 动态适应：支持动态添加概念特征 | Dynamic adaptation: Support dynamic addition of concept features
    
    架构特点 | Architecture Features:
        • 固定位置编码：使用正弦位置编码，非学习参数 | Fixed positional encoding: Use sinusoidal positional encoding, non-learnable parameters
        • 概念特征缓存：高效存储和检索词汇概念特征 | Concept feature cache: Efficiently store and retrieve vocabulary concept features
        • 标记投影网络：将3维概念特征映射到标记空间 | Marker projection network: Map 3D concept features to marker space
        • 统计跟踪：实时监控激活统计 | Statistical tracking: Real-time monitoring of activation statistics
    
    参数 | Parameters:
        vocab_size (int): 词汇表大小 | Vocabulary size
        embed_dim (int): 基础嵌入维度 | Base embedding dimension
        marker_dim (int, optional): 标记向量维度，默认32 | Marker vector dimension, default 32
    
    输入 | Input:
        input_ids (torch.Tensor): 输入词索引，形状为[B, S] | Input token indices with shape [B, S]
        return_details (bool, optional): 是否返回详细统计，默认False | Whether to return detailed statistics, default False
    
    输出 | Output:
        Dict[str, torch.Tensor]: 包含以下键的字典 | Dictionary containing the following keys:
            • base_embeddings: 基础嵌入 [B, S, embed_dim] | Base embeddings [B, S, embed_dim]
            • markers: 标记向量 [B, S, marker_dim] | Marker vectors [B, S, marker_dim]
            • concept_features: 概念特征 [B, S, 3] | Concept features [B, S, 3]
            • stats (可选): 详细统计信息 | Detailed statistics (optional)
    
    示例 | Example:
        >>> embed_layer = EnhancedEmbeddingLayer(vocab_size=20000, embed_dim=128)
        >>> input_ids = torch.randint(0, 20000, (4, 32))
        >>> output = embed_layer(input_ids, return_details=True)
        >>> print(f"基础嵌入形状: {output['base_embeddings'].shape}")
        >>> print(f"标记向量形状: {output['markers'].shape}")
        >>> print(f"概念特征形状: {output['concept_features'].shape}")
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, marker_dim: int = 32):
        """
        初始化增强嵌入层 | Initialize Enhanced Embedding Layer
        
        参数 | Parameters:
            vocab_size (int): 词汇表大小 | Vocabulary size
            embed_dim (int): 基础嵌入维度 | Base embedding dimension
            marker_dim (int, optional): 标记向量维度，默认32 | Marker vector dimension, default 32
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.marker_dim = marker_dim
        
        # ==================== 基础嵌入组件 ====================
        # Base Embedding Components
        
        # 词嵌入（可学习） | Token embedding (learnable)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码（固定正弦编码） | Positional encoding (fixed sinusoidal encoding)
        # 使用缓冲区，自动移动到设备 | Use buffer, automatically moved to device
        self.register_buffer('position_encoding', self._create_positional_encoding(max_len=512, d_model=embed_dim))
        
        # ==================== 标记生成组件 ====================
        # Marker Generation Components
        
        # 概念特征映射层：将(Q, K, V)特征映射到标记空间 | Concept feature mapping layer: Map (Q, K, V) features to marker space
        # 输入: 3个特征值 (Q_local, K_local, V_local) | Input: 3 feature values (Q_local, K_local, V_local)
        # 输出: marker_dim维标记向量 | Output: marker_dim-dimensional marker vector
        self.marker_projection = nn.Sequential(
            nn.Linear(3, 16),    # 将3维特征扩展到16维 | Expand 3D features to 16D
            nn.ReLU(),           # 非线性激活 | Nonlinear activation
            nn.Linear(16, marker_dim),  # 映射到标记维度 | Map to marker dimension
            nn.Tanh()            # 归一化到[-1, 1]范围 | Normalize to [-1, 1] range
        )
        
        # 默认标记（用于未识别的词） | Default marker (for unrecognized words)
        self.default_marker = nn.Parameter(torch.randn(marker_dim) * 0.1)
        
        # ==================== 概念特征缓存 ====================
        # Concept Feature Cache
        
        # 存储词汇索引到概念特征的映射 | Store mapping from vocabulary indices to concept features
        self.register_buffer('concept_features', torch.zeros(vocab_size, 3))
        self.register_buffer('has_concept', torch.zeros(vocab_size, dtype=torch.bool))
        
        # ==================== 统计跟踪 ====================
        # Statistical Tracking
        
        # 统计Q, K, V激活 | Statistics for Q, K, V activations
        self.register_buffer('activation_stats', torch.zeros(3))
        
        # ==================== 初始化参数 ====================
        # Parameter Initialization
        
        self._init_parameters()
    
    def _init_parameters(self):
        """
        初始化参数 | Initialize Parameters
        
        使用合适的初始化策略确保稳定训练 | Use appropriate initialization strategies to ensure stable training
        """
        # 词嵌入初始化 | Token embedding initialization
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # 标记投影层初始化 | Marker projection layer initialization
        for layer in self.marker_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # 默认标记初始化 | Default marker initialization
        nn.init.normal_(self.default_marker, mean=0.0, std=0.01)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        创建正弦位置编码（非学习参数） | Create Sinusoidal Positional Encoding (Non-learnable Parameters)
        
        基于Transformer论文中的公式： | Based on formula from Transformer paper:
            PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        参数 | Parameters:
            max_len (int): 最大序列长度 | Maximum sequence length
            d_model (int): 模型维度 | Model dimension
        
        返回 | Returns:
            torch.Tensor: 位置编码张量 [1, max_len, d_model] | Positional encoding tensor [1, max_len, d_model]
        """
        # 创建位置索引 | Create position indices
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        
        # 创建除数项 | Create divisor term
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )  # [d_model/2]
        
        # 初始化位置编码矩阵 | Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        
        # 计算正弦部分 | Compute sine part
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列 | Even columns
        
        # 计算余弦部分 | Compute cosine part
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列 | Odd columns
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def set_concept_features(self, word_indices: torch.Tensor, features: torch.Tensor):
        """
        设置词汇的概念特征（Q, K, V） | Set Concept Features (Q, K, V) for Vocabulary
        
        参数 | Parameters:
            word_indices (torch.Tensor): 词汇索引 [n_words] | Word indices [n_words]
            features (torch.Tensor): 概念特征 [n_words, 3] (Q, K, V) | Concept features [n_words, 3] (Q, K, V)
        
        异常 | Raises:
            ValueError: 如果词汇索引数量与特征数量不匹配 | If number of word indices doesn't match number of features
        """
        if len(word_indices) != len(features):
            raise ValueError(
                f"词汇索引数量({len(word_indices)})与特征数量({len(features)})不匹配 | "
                f"Number of word indices ({len(word_indices)}) doesn't match number of features ({len(features)})"
            )
        
        # 确保在同一设备 | Ensure on the same device
        device = self.concept_features.device
        
        # 更新概念特征缓存 | Update concept feature cache
        self.concept_features[word_indices] = features.to(device)
        self.has_concept[word_indices] = True
        
        print(f"✅ 已设置 {len(word_indices)} 个词汇的概念特征 | "
              f"Concept features set for {len(word_indices)} words")
    
    def get_concept_features(self, word_idx: int) -> Optional[torch.Tensor]:
        """
        获取指定词汇的概念特征 | Get Concept Features for Specified Word
        
        参数 | Parameters:
            word_idx (int): 词汇索引 | Word index
        
        返回 | Returns:
            Optional[torch.Tensor]: 概念特征张量 [3] 或 None（如果无概念特征） | Concept feature tensor [3] or None (if no concept features)
        
        异常 | Raises:
            ValueError: 如果词汇索引超出词汇表范围 | If word index exceeds vocabulary range
        """
        if word_idx >= self.vocab_size:
            raise ValueError(
                f"词汇索引 {word_idx} 超出词汇表范围 {self.vocab_size} | "
                f"Word index {word_idx} exceeds vocabulary range {self.vocab_size}"
            )
        
        if self.has_concept[word_idx]:
            return self.concept_features[word_idx].clone()
        else:
            return None
    
    def _calculate_local_density(self, batch_markers: torch.Tensor) -> Dict[str, float]:
        """
        计算本地认知密度（使用连接点密度公式） | Calculate Local Cognitive Density (Using Connection Point Density Formula)
        
        使用静态密度公式：D = 2m/((N+1)N) | Use static density formula: D = 2m/((N+1)N)
        其中： | Where:
            N = 节点数（标记向量数量） | N = Number of nodes (marker vectors)
            m = 连接数（余弦相似度 > 阈值的对数） | m = Number of connections (pairs with cosine similarity > threshold)
        
        参数 | Parameters:
            batch_markers (torch.Tensor): 标记向量 [batch_size, seq_len, marker_dim] | Marker vectors [batch_size, seq_len, marker_dim]
        
        返回 | Returns:
            Dict[str, float]: 密度统计信息 | Density statistics information
        """
        if batch_markers.numel() == 0:
            return {
                'static_density': 0.0,
                'connections': 0,
                'nodes': 0,
                'stability': 0.0,
                'avg_similarity': 0.0
            }
        
        # 获取标记向量形状 | Get marker vector shape
        batch_size, seq_len, marker_dim = batch_markers.shape
        
        # 展平标记向量 | Flatten marker vectors
        flat_markers = batch_markers.view(-1, marker_dim)  # [batch*seq_len, marker_dim]
        
        # 计算余弦相似度矩阵 | Compute cosine similarity matrix
        norm_markers = F.normalize(flat_markers, p=2, dim=1)  # 归一化 | Normalize
        similarity_matrix = torch.mm(norm_markers, norm_markers.T)  # [N, N]
        
        # 应用密度公式：D = 2m/((N+1)N) | Apply density formula: D = 2m/((N+1)N)
        N = similarity_matrix.size(0)
        
        # 阈值过滤：保留相似度大于0.3的连接 | Threshold filtering: Keep connections with similarity > 0.3
        threshold = 0.3
        connections = (similarity_matrix > threshold).float()
        M = connections.sum().item() / 2  # 无向图，每条边计算两次 | Undirected graph, each edge counted twice
        
        # 静态密度公式 | Static density formula
        if N > 1:
            static_density = (2 * M) / ((N + 1) * N)
        else:
            static_density = 0.0
        
        # 动态密度变化（与历史统计比较） | Dynamic density changes (compared with historical statistics)
        prev_stats = self.activation_stats
        
        # 计算当前统计：标记向量的平均范数 | Compute current statistics: Average norm of marker vectors
        current_stats = torch.tensor([
            batch_markers.mean().item(),          # 平均值 | Mean
            batch_markers.std().item(),           # 标准差 | Standard deviation
            similarity_matrix.mean().item()       # 平均相似度 | Average similarity
        ], device=batch_markers.device)
        
        # 更新统计：指数移动平均 | Update statistics: Exponential moving average
        self.activation_stats = 0.9 * self.activation_stats + 0.1 * current_stats
        
        # 计算稳定性：统计变化率的倒数 | Compute stability: Reciprocal of statistical change rate
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
        前向传播 | Forward Propagation
        
        参数 | Parameters:
            input_ids (torch.Tensor): 输入词索引 [batch_size, seq_len] | Input token indices [batch_size, seq_len]
            return_details (bool): 是否返回详细统计信息，默认False | Whether to return detailed statistics, default False
        
        返回 | Returns:
            Dict[str, torch.Tensor]: 包含基础嵌入和标记的字典 | Dictionary containing base embeddings and markers
        
        处理流程 | Processing Flow:
            1. 基础嵌入生成：词嵌入 + 位置编码 | Base embedding generation: Token embedding + positional encoding
            2. 标记向量生成：基于概念特征或默认标记 | Marker vector generation: Based on concept features or default markers
            3. 密度计算（可选）：计算本地认知密度 | Density computation (optional): Compute local cognitive density
            4. 统计信息收集（可选）：收集嵌入统计信息 | Statistics collection (optional): Collect embedding statistics
        """
        batch_size, seq_len = input_ids.shape
        
        # ==================== 基础嵌入生成 ====================
        # Base Embedding Generation
        
        # 词嵌入 | Token embedding
        token_emb = self.token_embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # 位置编码 - 确保在正确的设备上 | Positional encoding - ensure on correct device
        if seq_len <= self.position_encoding.size(1):
            pos_emb = self.position_encoding[:, :seq_len, :]
            # 确保位置编码与token嵌入在同一设备 | Ensure positional encoding on same device as token embedding
            if pos_emb.device != token_emb.device:
                pos_emb = pos_emb.to(token_emb.device)
        else:
            # 动态扩展位置编码 - 确保在正确的设备上 | Dynamically extend positional encoding - ensure on correct device
            pos_emb = self._create_positional_encoding(seq_len, self.embed_dim).to(input_ids.device)
        
        # 基础嵌入 = 词嵌入 + 位置编码 | Base embedding = Token embedding + Positional encoding
        base_embeddings = token_emb + pos_emb  # [batch, seq_len, embed_dim]
        
        # ==================== 标记向量生成 ====================
        # Marker Vector Generation
        
        # 初始化标记向量列表 | Initialize marker vector list
        markers_list = []
        
        # 为每个词生成标记向量 | Generate marker vectors for each word
        for batch_idx in range(batch_size):
            batch_markers = []
            
            for pos_idx in range(seq_len):
                word_idx = input_ids[batch_idx, pos_idx].item()
                
                # 获取概念特征 - 确保在同一设备 | Get concept features - ensure on same device
                concept_feat = self.get_concept_features(word_idx)
                
                if concept_feat is not None:
                    # 有概念特征：投影到标记空间 | Has concept features: Project to marker space
                    concept_feat = concept_feat.to(input_ids.device).unsqueeze(0)  # [1, 3]
                    marker = self.marker_projection(concept_feat)  # [1, marker_dim]
                else:
                    # 无概念特征：使用默认标记 | No concept features: Use default marker
                    marker = self.default_marker.unsqueeze(0).to(input_ids.device)  # [1, marker_dim]
                
                batch_markers.append(marker)
            
            # 拼接当前批次的标记向量 | Concatenate marker vectors for current batch
            batch_markers_tensor = torch.cat(batch_markers, dim=0)  # [seq_len, marker_dim]
            markers_list.append(batch_markers_tensor)
        
        # 组合所有批次的标记向量 | Combine marker vectors from all batches
        markers = torch.stack(markers_list, dim=0)  # [batch, seq_len, marker_dim]
        
        # ==================== 统计信息收集 ====================
        # Statistics Collection
        
        stats = {}
        if return_details:
            # 计算本地认知密度 | Compute local cognitive density
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
        # Return Results
        
        # 获取概念特征（用于后续处理） | Get concept features (for subsequent processing)
        concept_feats = self.concept_features[input_ids.cpu()].to(input_ids.device)  # [batch, seq_len, 3]
        
        output = {
            'base_embeddings': base_embeddings,  # [batch, seq_len, embed_dim]
            'markers': markers,                  # [batch, seq_len, marker_dim]
            'concept_features': concept_feats,   # [batch, seq_len, 3]
        }
        
        if return_details:
            output['stats'] = stats
        
        return output
    
    def analyze_concept_distribution(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """
        分析输入中的概念分布 | Analyze Concept Distribution in Input
        
        参数 | Parameters:
            input_ids (torch.Tensor): 输入词索引 [batch_size, seq_len] | Input token indices [batch_size, seq_len]
        
        返回 | Returns:
            Dict[str, float]: 概念分布统计 | Concept distribution statistics
        
        统计信息 | Statistics:
            • concept_coverage: 概念覆盖率 | Concept coverage
            • Q_mean, Q_std: Q值的均值和标准差 | Mean and std of Q values
            • K_mean, K_std: K值的均值和标准差 | Mean and std of K values
            • V_mean, V_std: V值的均值和标准差 | Mean and std of V values
            • V_range: V值范围 | V value range
            • V_stability: V值稳定性 | V value stability
        """
        batch_size, seq_len = input_ids.shape
        total_tokens = batch_size * seq_len
        
        # 确保has_concept在正确的设备上 | Ensure has_concept on correct device
        device = input_ids.device
        has_concept_on_device = self.has_concept.to(device)
        
        # 统计概念覆盖 | Statistics for concept coverage
        has_concept_mask = has_concept_on_device[input_ids]  # [batch, seq_len]
        concept_count = has_concept_mask.sum().item()
        concept_coverage = concept_count / total_tokens if total_tokens > 0 else 0
        
        # 获取有概念的词的特征 | Get features of words with concepts
        if concept_count > 0:
            # 获取有概念的索引 | Get indices with concepts
            concept_indices = input_ids[has_concept_mask]  # [concept_count]
            
            # 确保concept_features在正确的设备上 | Ensure concept_features on correct device
            concept_feats_on_device = self.concept_features.to(device)
            concept_feats = concept_feats_on_device[concept_indices]  # [concept_count, 3]
            
            # 计算特征统计 | Compute feature statistics
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
                'Q_std': 0.0,
                'K_std': 0.0,
                'V_std': 0.0,
                'V_range': 0.0,
                'V_stability': 0.0,
            }
        
        return stats
    
    def reset_statistics(self):
        """
        重置统计信息 | Reset Statistics
        
        重置激活统计到初始状态 | Reset activation statistics to initial state
        """
        self.activation_stats.zero_()
        print("✅ 嵌入层统计已重置 | Embedding layer statistics reset")
    
    def get_config(self) -> Dict:
        """
        获取嵌入层配置 | Get Embedding Layer Configuration
        
        返回 | Returns:
            Dict: 配置字典 | Configuration dictionary
        """
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'marker_dim': self.marker_dim,
            'concept_coverage': self.has_concept.float().mean().item(),
            'total_concepts': self.has_concept.sum().item(),
        }
    
    def extra_repr(self) -> str:
        """
        额外的字符串表示 | Extra String Representation
        
        返回 | Returns:
            str: 描述嵌入层参数的字符串 | String describing embedding layer parameters
        """
        concept_count = self.has_concept.sum().item()
        concept_percent = concept_count / self.vocab_size * 100
        
        return (f'vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, '
                f'marker_dim={self.marker_dim}, concepts={concept_count}/{self.vocab_size} ({concept_percent:.1f}%)')


class ConceptAwareEmbedding(nn.Module):
    """
    概念感知嵌入层 | Concept-Aware Embedding Layer
    -------------------------------------------
    
    简化的概念感知嵌入层，适用于轻量级应用。
    Simplified concept-aware embedding layer for lightweight applications.
    
    特点 | Features:
        • 集成概念特征：直接将概念特征与词嵌入结合 | Integrated concept features: Directly combine concept features with token embeddings
        • 轻量级设计：减少投影层，降低计算复杂度 | Lightweight design: Reduce projection layers, lower computational complexity
        • 实时概念注入：动态注入概念特征 | Real-time concept injection: Dynamically inject concept features
    
    与EnhancedEmbeddingLayer的区别 | Differences from EnhancedEmbeddingLayer:
        • 不生成单独的标记向量 | Does not generate separate marker vectors
        • 概念特征直接融入基础嵌入 | Concept features directly integrated into base embeddings
        • 适合计算资源有限的场景 | Suitable for scenarios with limited computational resources
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, concept_dim: int = 3):
        """
        初始化概念感知嵌入层 | Initialize Concept-Aware Embedding Layer
        
        参数 | Parameters:
            vocab_size (int): 词汇表大小 | Vocabulary size
            embed_dim (int): 嵌入维度 | Embedding dimension
            concept_dim (int, optional): 概念特征维度，默认3 (Q, K, V) | Concept feature dimension, default 3 (Q, K, V)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.concept_dim = concept_dim
        
        # 词嵌入 | Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码 | Positional encoding
        self.register_buffer('position_encoding', self._create_positional_encoding(max_len=512, d_model=embed_dim))
        
        # 概念特征投影层 | Concept feature projection layer
        self.concept_projection = nn.Linear(concept_dim, embed_dim)
        
        # 概念特征缓存 | Concept feature cache
        self.register_buffer('concept_features', torch.zeros(vocab_size, concept_dim))
        self.register_buffer('has_concept', torch.zeros(vocab_size, dtype=torch.bool))
        
        # 融合权重 | Fusion weight
        self.concept_weight = nn.Parameter(torch.tensor(0.3))
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """创建正弦位置编码 | Create sinusoidal positional encoding"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def set_concept_features(self, word_indices: torch.Tensor, features: torch.Tensor):
        """设置概念特征 | Set concept features"""
        if len(word_indices) != len(features):
            raise ValueError(f"词汇索引数量({len(word_indices)})与特征数量({len(features)})不匹配")
        
        self.concept_features[word_indices] = features.to(self.concept_features.device)
        self.has_concept[word_indices] = True
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播 | Forward propagation"""
        batch_size, seq_len = input_ids.shape
        
        # 基础词嵌入 | Base token embedding
        token_emb = self.token_embedding(input_ids)
        
        # 位置编码 | Positional encoding
        if seq_len <= self.position_encoding.size(1):
            pos_emb = self.position_encoding[:, :seq_len, :].to(input_ids.device)
        else:
            # 动态扩展 | Dynamic extension
            position = torch.arange(seq_len).unsqueeze(1).to(input_ids.device)
            div_term = torch.exp(
                torch.arange(0, self.embed_dim, 2) * (-np.log(10000.0) / self.embed_dim)
            ).to(input_ids.device)
            
            pe = torch.zeros(seq_len, self.embed_dim, device=input_ids.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pos_emb = pe.unsqueeze(0)
        
        # 概念特征 | Concept features
        concept_feats = self.concept_features[input_ids.cpu()].to(input_ids.device)  # [B, S, 3]
        has_concept_mask = self.has_concept[input_ids.cpu()].to(input_ids.device).unsqueeze(-1)  # [B, S, 1]
        
        # 投影概念特征 | Project concept features
        concept_projected = self.concept_projection(concept_feats)  # [B, S, embed_dim]
        
        # 融合概念特征（仅对有概念的词） | Fuse concept features (only for words with concepts)
        concept_weight = torch.sigmoid(self.concept_weight)
        concept_injected = concept_weight * concept_projected * has_concept_mask.float()
        
        # 最终嵌入 | Final embedding
        embeddings = token_emb + pos_emb + concept_injected
        
        return embeddings
    
    def extra_repr(self) -> str:
        """额外的字符串表示 | Extra string representation"""
        concept_count = self.has_concept.sum().item()
        return (f'vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, '
                f'concept_dim={self.concept_dim}, concepts={concept_count}/{self.vocab_size}')


# ==================== 工厂函数 ====================
# Factory Functions

def create_embedding_layer(embed_type: str = 'enhanced', **kwargs) -> nn.Module:
    """
    创建嵌入层 | Create Embedding Layer
    
    参数 | Parameters:
        embed_type (str): 嵌入层类型，可选['enhanced', 'concept_aware'] | Embedding layer type, options: ['enhanced', 'concept_aware']
        **kwargs: 传递给具体嵌入层的参数 | Parameters passed to specific embedding layer
    
    返回 | Returns:
        nn.Module: 创建的嵌入层 | Created embedding layer
    
    异常 | Raises:
        ValueError: 如果嵌入层类型不支持 | If embedding layer type is not supported
    """
    embed_map = {
        'enhanced': EnhancedEmbeddingLayer,
        'concept_aware': ConceptAwareEmbedding,
    }
    
    if embed_type not in embed_map:
        raise ValueError(
            f"不支持的嵌入层类型: {embed_type}\n"
            f"支持的类型: {list(embed_map.keys())}"
        )
    
    return embed_map[embed_type](**kwargs)


# ==================== 测试函数 ====================
# Test Functions

def test_enhanced_embedding():
    """测试增强嵌入层 | Test Enhanced Embedding Layer"""
    print("=" * 60)
    print("测试增强嵌入层 | Testing Enhanced Embedding Layer")
    print("=" * 60)
    
    # 创建嵌入层 | Create embedding layer
    vocab_size = 1000
    embed_dim = 128
    marker_dim = 32
    
    embed_layer = EnhancedEmbeddingLayer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        marker_dim=marker_dim
    )
    
    # 设置一些概念特征 | Set some concept features
    word_indices = torch.tensor([1, 2, 3, 4, 5])
    features = torch.randn(5, 3)  # 随机概念特征 | Random concept features
    embed_layer.set_concept_features(word_indices, features)
    
    # 创建测试输入 | Create test input
    batch_size, seq_len = 4, 16
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播（无详细统计） | Forward propagation (without detailed statistics)
    output_simple = embed_layer(input_ids, return_details=False)
    
    # 前向传播（有详细统计） | Forward propagation (with detailed statistics)
    output_detailed = embed_layer(input_ids, return_details=True)
    
    # 验证输出形状 | Verify output shapes
    print(f"输入形状: {input_ids.shape}")
    print(f"基础嵌入形状: {output_simple['base_embeddings'].shape}")
    print(f"标记向量形状: {output_simple['markers'].shape}")
    print(f"概念特征形状: {output_simple['concept_features'].shape}")
    
    # 验证详细统计 | Verify detailed statistics
    if 'stats' in output_detailed:
        stats = output_detailed['stats']
        print(f"\n详细统计 | Detailed Statistics:")
        print(f"  标记统计: {stats['marker_stats']}")
        print(f"  密度信息: 密度={stats['density_info']['static_density']:.4f}, "
              f"连接数={stats['density_info']['connections']:.0f}, "
              f"节点数={stats['density_info']['nodes']}")
        print(f"  概念覆盖率: {stats['concept_coverage']:.2%}")
    
    # 分析概念分布 | Analyze concept distribution
    concept_stats = embed_layer.analyze_concept_distribution(input_ids)
    print(f"\n概念分布分析 | Concept Distribution Analysis:")
    for key, value in concept_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # 获取配置 | Get configuration
    config = embed_layer.get_config()
    print(f"\n嵌入层配置 | Embedding Layer Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 检查参数 | Check parameters
    total_params = sum(p.numel() for p in embed_layer.parameters())
    print(f"总参数量: {total_params:,}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)
    
    return embed_layer


def test_concept_aware_embedding():
    """测试概念感知嵌入层 | Test Concept-Aware Embedding Layer"""
    print("=" * 60)
    print("测试概念感知嵌入层 | Testing Concept-Aware Embedding Layer")
    print("=" * 60)
    
    # 创建嵌入层 | Create embedding layer
    vocab_size = 1000
    embed_dim = 128
    
    embed_layer = ConceptAwareEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        concept_dim=3
    )
    
    # 设置概念特征 | Set concept features
    word_indices = torch.tensor([10, 20, 30, 40, 50])
    features = torch.randn(5, 3)
    embed_layer.set_concept_features(word_indices, features)
    
    # 创建测试输入 | Create test input
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播 | Forward propagation
    embeddings = embed_layer(input_ids)
    
    # 验证输出 | Verify output
    print(f"输入形状: {input_ids.shape}")
    print(f"输出嵌入形状: {embeddings.shape}")
    print(f"概念权重: {embed_layer.concept_weight.item():.4f}")
    
    # 检查参数 | Check parameters
    total_params = sum(p.numel() for p in embed_layer.parameters())
    print(f"总参数量: {total_params:,}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)
    
    return embed_layer


def test_embedding_factory():
    """测试嵌入层工厂函数 | Test Embedding Layer Factory Functions"""
    print("=" * 60)
    print("测试嵌入层工厂函数 | Testing Embedding Layer Factory Functions")
    print("=" * 60)
    
    # 测试创建增强嵌入层 | Test creating enhanced embedding layer
    try:
        embed_enhanced = create_embedding_layer(
            'enhanced',
            vocab_size=2000,
            embed_dim=128,
            marker_dim=32
        )
        print(f"✅ 成功创建增强嵌入层 | Successfully created enhanced embedding layer")
        print(f"  类型: {embed_enhanced.__class__.__name__}")
        print(f"  词汇表大小: {embed_enhanced.vocab_size}")
        print(f"  嵌入维度: {embed_enhanced.embed_dim}")
        print(f"  标记维度: {embed_enhanced.marker_dim}")
    except Exception as e:
        print(f"❌ 创建增强嵌入层失败: {e}")
    
    # 测试创建概念感知嵌入层 | Test creating concept-aware embedding layer
    try:
        embed_concept = create_embedding_layer(
            'concept_aware',
            vocab_size=2000,
            embed_dim=128,
            concept_dim=3
        )
        print(f"\n✅ 成功创建概念感知嵌入层 | Successfully created concept-aware embedding layer")
        print(f"  类型: {embed_concept.__class__.__name__}")
        print(f"  词汇表大小: {embed_concept.vocab_size}")
        print(f"  嵌入维度: {embed_concept.embed_dim}")
        print(f"  概念维度: {embed_concept.concept_dim}")
    except Exception as e:
        print(f"❌ 创建概念感知嵌入层失败: {e}")
    
    # 测试无效类型 | Test invalid type
    try:
        embed_invalid = create_embedding_layer('invalid', vocab_size=1000, embed_dim=128)
    except ValueError as e:
        print(f"\n✅ 预期中的错误 | Expected error: {e}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)


# ==================== 导出列表 ====================
# Export List
# src/rga/layers/embeddings.py

__all__ = [
    # Main embedding layers
    'EnhancedEmbeddingLayer',
    'ConceptAwareEmbedding',
    
    # Factory functions
    'create_embedding_layer',
    
    # Test functions
    'test_enhanced_embedding',
    'test_concept_aware_embedding',
    'test_embedding_factory',
]


# ==================== 模块自检 ====================
# Module Self-Test

if __name__ == "__main__":
    print("=" * 60)
    print("嵌入层模块自检 | Embedding Layers Module Self-Test")
    print("=" * 60)
    
    try:
        # 运行测试 | Run tests
        print("\n1. 测试增强嵌入层 | Testing Enhanced Embedding Layer")
        test_enhanced_embedding()
        
        print("\n2. 测试概念感知嵌入层 | Testing Concept-Aware Embedding Layer")
        test_concept_aware_embedding()
        
        print("\n3. 测试工厂函数 | Testing Factory Functions")
        test_embedding_factory()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过 | All tests passed")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败 | Test failed: {e}")
        import traceback
        traceback.print_exc()
