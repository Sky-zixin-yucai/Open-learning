# Rule-Governed Architecture/rga/integration/__init__.py
"""
规则治理架构集成模块 | Rule-Governed Architecture Integration Module
==============================================================

集成核心引擎、层模块和伪装功能，提供完整的前向传播流程。
Integrates core engine, layers modules and disguise functions, provides complete forward propagation pipeline.

核心功能 | Core Features:
1. 完整的前向传播流程 | Complete forward propagation pipeline
2. 伪装保存/加载系统 | Disguise save/load system
3. 统一配置管理 | Unified configuration management
4. 中英双语接口 | Bilingual Chinese-English interface

模块结构 | Module Structure:
├── RGAIntegrator - 主要集成类 | Main integration class
├── create_integrator - 工厂函数 | Factory function
├── save_disguised_model - 伪装保存函数 | Disguise save function
└── load_disguised_model - 伪装加载函数 | Disguise load function

设计原则 | Design Principles:
• 开箱即用 | Out-of-the-box usability
• 模块化设计 | Modular design
• 向后兼容 | Backward compatibility
• 性能优化 | Performance optimization

使用示例 | Usage Examples:
    >>> # 创建集成器 | Create integrator
    >>> from rga.integration import create_integrator
    >>> integrator = create_integrator(vocab_size=20000, dim=512)
    >>>
    >>> # 前向传播 | Forward propagation
    >>> output = integrator.forward(input_ids, num_cycles=3)
    >>>
    >>> # 伪装保存 | Disguise save
    >>> integrator.save_pretrained("./saved_model")
    >>>
    >>> # 伪装加载 | Disguise load
    >>> from rga.integration import load_disguised_model
    >>> integrator = load_disguised_model("./saved_model")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import os
import json
import pickle
import warnings
from dataclasses import dataclass, field
from collections import deque
import os

# ==================== 导入核心模块 ====================
# ==================== Import Core Modules ====================

# 导入RGA核心模块
from ..core import (
    RGAConfig, 
    RGAEngine, 
    CoreMetricsCalculator,
    create_rga_engine,
    get_default_config,
    validate_config
)

# 导入层模块
from ..layers import (
    # 层类 | Layer classes
    VKQ_SubNet_WithFixedNorm,
    QVK_SubNet_WithFixedNorm,
    KQV_SubNet_WithFixedNorm,
    ChainReactionUnit_Final,
    TriValueBalancer,
    VDominantBalancer,
    DensityDrivenBalancer,
    AdaptiveStabilizer,
    EnhancedEmbeddingLayer,
    ConceptAwareEmbedding,
    SandwichFusion,
    GeologicalMemory,
    OneWayValve,
    SimpleOneWayValve,
    FixedRMSNorm,
    FixedGroupRMSNorm,
    ScaledFixedRMSNorm,
    
    # 工厂函数 | Factory functions
    create_attention_subnet,
    create_balancer_layer,
    create_embedding_layer,
    create_one_way_valve,
    
    # 管理类 | Management classes
    LayerFactory,
    LayerRegistry,
    LayerConfig,
    LayerConfigManager,
    
    # 便捷函数 | Convenience functions
    get_layer_factory,
    create_layer,
    list_available_layers
)

# ==================== 集成配置 ====================
# ==================== Integration Configuration ====================

@dataclass
class IntegrationConfig:
    """
    集成配置数据类 | Integration Configuration Dataclass
    
    统一管理所有组件的配置参数。
    Unified management of all component configuration parameters.
    """
    
    # 基础配置 | Basic configuration
    vocab_size: int = 10000                    # 词汇表大小 | Vocabulary size
    dim: int = 512                            # 模型维度 | Model dimension
    num_units: int = 3                        # 链式反应单元数量 | Number of chain reaction units
    max_cycles: int = 3                       # 最大持续思考循环次数 | Maximum continuous thinking cycles
    phase_threshold: float = 0.43             # 相变检测阈值 | Phase transition detection threshold
    
    # 内存配置 | Memory configuration
    geo_depth: int = 3                        # 地质记忆深度 | Geological memory depth
    memory_size: int = 1000                   # 记忆大小 | Memory size
    history_length: int = 10                  # 历史长度 | History length
    
    # 性能配置 | Performance configuration
    enable_mixed_precision: bool = True       # 启用混合精度 | Enable mixed precision
    enable_gradient_checkpointing: bool = False  # 启用梯度检查点 | Enable gradient checkpointing
    gradient_accumulation_steps: int = 1      # 梯度累积步数 | Gradient accumulation steps
    
    # 层配置 | Layer configuration
    embedding_type: str = "enhanced"          # 嵌入层类型 | Embedding layer type
    attention_subnet_type: str = "vkq"        # 注意力子网络类型 | Attention subnet type
    balancer_type: str = "tri_value"          # 平衡器类型 | Balancer type
    valve_type: str = "learnable"             # 单向阀类型 | One-way valve type
    
    # V值调控配置 | V-value regulation configuration
    v_scaling_factor: float = 1.0             # V值缩放因子 | V-value scaling factor
    min_v_mean: float = 0.3                   # 最小V均值 | Minimum V mean
    max_v_mean: float = 2.0                   # 最大V均值 | Maximum V mean
    target_v_mean: float = 1.0                # 目标V均值 | Target V mean
    
    # 其他参数 | Additional parameters
    kwargs: Dict[str, Any] = field(default_factory=dict)  # 其他参数 | Additional parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 | Convert to dictionary"""
        config_dict = {
            'vocab_size': self.vocab_size,
            'dim': self.dim,
            'num_units': self.num_units,
            'max_cycles': self.max_cycles,
            'phase_threshold': self.phase_threshold,
            'geo_depth': self.geo_depth,
            'memory_size': self.memory_size,
            'history_length': self.history_length,
            'enable_mixed_precision': self.enable_mixed_precision,
            'enable_gradient_checkpointing': self.enable_gradient_checkpointing,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'embedding_type': self.embedding_type,
            'attention_subnet_type': self.attention_subnet_type,
            'balancer_type': self.balancer_type,
            'valve_type': self.valve_type,
            'v_scaling_factor': self.v_scaling_factor,
            'min_v_mean': self.min_v_mean,
            'max_v_mean': self.max_v_mean,
            'target_v_mean': self.target_v_mean,
        }
        
        # 合并其他参数 | Merge additional parameters
        config_dict.update(self.kwargs)
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IntegrationConfig':
        """从字典创建配置 | Create configuration from dictionary"""
        # 提取已知字段 | Extract known fields
        known_fields = {
            'vocab_size', 'dim', 'num_units', 'max_cycles', 'phase_threshold',
            'geo_depth', 'memory_size', 'history_length', 'enable_mixed_precision',
            'enable_gradient_checkpointing', 'gradient_accumulation_steps',
            'embedding_type', 'attention_subnet_type', 'balancer_type', 'valve_type',
            'v_scaling_factor', 'min_v_mean', 'max_v_mean', 'target_v_mean'
        }
        
        base_fields = {}
        kwargs = {}
        
        for key, value in config_dict.items():
            if key in known_fields:
                base_fields[key] = value
            else:
                kwargs[key] = value
        
        return cls(**base_fields, kwargs=kwargs)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置 | Validate configuration"""
        errors = []
        
        # 验证正整数参数 | Validate positive integer parameters
        for param_name in ['vocab_size', 'dim', 'num_units', 'max_cycles', 'geo_depth', 'memory_size', 'history_length']:
            value = getattr(self, param_name)
            if not isinstance(value, int) or value <= 0:
                errors.append(f"{param_name} 必须是正整数 | {param_name} must be positive integer")
        
        # 验证范围参数 | Validate range parameters
        if not (0 <= self.phase_threshold <= 1):
            errors.append("phase_threshold 必须在0-1之间 | phase_threshold must be between 0 and 1")
        
        if not (0 < self.min_v_mean < self.max_v_mean):
            errors.append("min_v_mean 必须小于 max_v_mean | min_v_mean must be less than max_v_mean")
        
        # 验证字符串参数 | Validate string parameters
        valid_embedding_types = ['enhanced', 'concept_aware']
        if self.embedding_type not in valid_embedding_types:
            errors.append(f"embedding_type 必须是 {valid_embedding_types} 之一 | embedding_type must be one of {valid_embedding_types}")
        
        valid_attention_types = ['vkq', 'qvk', 'kqv']
        if self.attention_subnet_type not in valid_attention_types:
            errors.append(f"attention_subnet_type 必须是 {valid_attention_types} 之一 | attention_subnet_type must be one of {valid_attention_types}")
        
        valid_balancer_types = ['tri_value', 'v_dominant', 'density_driven', 'adaptive']
        if self.balancer_type not in valid_balancer_types:
            errors.append(f"balancer_type 必须是 {valid_balancer_types} 之一 | balancer_type must be one of {valid_balancer_types}")
        
        valid_valve_types = ['learnable', 'simple', 'detach', 'gate']
        if self.valve_type not in valid_valve_types:
            errors.append(f"valve_type 必须是 {valid_valve_types} 之一 | valve_type must be one of {valid_valve_types}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def __str__(self) -> str:
        """字符串表示 | String representation"""
        config_dict = self.to_dict()
        items = []
        
        for key, value in config_dict.items():
            if key != 'kwargs':
                items.append(f"  {key}: {value}")
        
        return "IntegrationConfig:\n" + "\n".join(items)


# ==================== 核心集成类 ====================
# ==================== Core Integration Class ====================

class RGAIntegrator:
    """
    规则治理架构集成器 | Rule-Governed Architecture Integrator
    
    集成核心功能：前向传播、伪装保存/加载、状态监控等。
    Integrates core functionality: forward propagation, disguise save/load, state monitoring, etc.
    
    功能特性 | Features:
    • 完整的前向传播流程 | Complete forward propagation pipeline
    • 伪装保存/加载系统 | Disguise save/load system
    • 实时状态监控 | Real-time state monitoring
    • 多循环持续思考 | Multi-cycle continuous thinking
    • 自动内存优化 | Automatic memory optimization
    
    设计原则 | Design Principles:
    1. 模块化：每个组件独立可替换 | Modularity: Each component is independently replaceable
    2. 可扩展：易于添加新功能 | Extensibility: Easy to add new features
    3. 高性能：优化内存和计算 | High performance: Optimized memory and computation
    4. 易用性：简洁的API接口 | Usability: Clean API interface
    
    使用示例 | Usage Example:
        >>> # 创建集成器 | Create integrator
        >>> integrator = RGAIntegrator(vocab_size=20000, dim=512)
        >>>
        >>> # 前向传播 | Forward propagation
        >>> input_ids = torch.randint(0, 20000, (2, 16))
        >>> output = integrator.forward(input_ids, num_cycles=3)
        >>>
        >>> # 获取分析报告 | Get analysis report
        >>> report = integrator.get_analysis_report()
        >>>
        >>> # 伪装保存 | Disguise save
        >>> integrator.save_pretrained("./saved_model")
        >>>
        >>> # 可视化状态变化 | Visualize state changes
        >>> integrator.visualize_state_changes()
    """
    
    def __init__(self, config: Optional[Union[IntegrationConfig, Dict]] = None):
        """
        初始化RGA集成器 | Initialize RGA Integrator
        
        参数 | Parameters:
        ------------
        config : Optional[Union[IntegrationConfig, Dict]], optional
            配置实例或字典，None则使用默认配置 | Configuration instance or dict, None uses default config
        """
        # ==================== 1. 配置初始化 ====================
        # ==================== 1. Configuration Initialization ====================
        if config is None:
            self.config = IntegrationConfig()
        elif isinstance(config, dict):
            self.config = IntegrationConfig.from_dict(config)
        else:
            self.config = config
        
        # 验证配置 | Validate configuration
        is_valid, errors = self.config.validate()
        if not is_valid:
            raise ValueError(f"配置验证失败 | Configuration validation failed: {errors}")
        
        print(f"✅ RGA集成器配置初始化完成 | RGA Integrator configuration initialized")
        
        # ==================== 2. 设备初始化 ====================
        # ==================== 2. Device Initialization ====================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   设备: {self.device} | Device: {self.device}")
        
        # ==================== 3. 核心引擎初始化 ====================
        # ==================== 3. Core Engine Initialization ====================
        self.engine = create_rga_engine(
            config={
                'vocab_size': self.config.vocab_size,
                'dim': self.config.dim,
                'phase_threshold': self.config.phase_threshold
            },
            enable_logging=True,
            enable_monitoring=True
        )
        
        # ==================== 4. 层工厂初始化 ====================
        # ==================== 4. Layer Factory Initialization ====================
        self.layer_factory = get_layer_factory()
        
        # ==================== 5. 组件初始化 ====================
        # ==================== 5. Component Initialization ====================
        self._initialize_components()
        
        # ==================== 6. 性能优化设置 ====================
        # ==================== 6. Performance Optimization Setup ====================
        self._setup_performance_optimization()
        
        # ==================== 7. 状态跟踪 ====================
        # ==================== 7. State Tracking ====================
        self.forward_history = []
        self.v_history = []
        self.phase_history = []
        
        print(f"✅ RGA集成器初始化完成 | RGA Integrator initialization completed")
        print(f"   版本: 1.0.0 | Version: 1.0.0")
        print(f"   总参数量: {self.count_parameters():,}")
    
    def _initialize_components(self):
        """初始化所有组件 | Initialize all components"""
        print("🔧 初始化组件 | Initializing components...")
        
        # 1. 嵌入层 | Embedding layer
        self.embedding = create_embedding_layer(
            embed_type=self.config.embedding_type,
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.dim,
            marker_dim=self.config.dim
        )
        
        # 2. 单向阀 | One-way valve
        self.one_way_valve = create_one_way_valve(
            dim=self.config.dim,
            valve_type=self.config.valve_type
        )
        
        # 3. 链式反应单元 | Chain reaction units
        self.chain_units = nn.ModuleList()
        for unit_id in range(self.config.num_units):
            unit = ChainReactionUnit_Final(
                dim=self.config.dim,
                unit_id=unit_id
            )
            self.chain_units.append(unit)
        
        # 4. 三值平衡器 | Tri-value balancers
        self.tri_balancers = nn.ModuleList()
        for unit_id in range(self.config.num_units):
            balancer = create_balancer_layer(
                balancer_type=self.config.balancer_type,
                dim=self.config.dim
            )
            self.tri_balancers.append(balancer)
        
        # 5. 地质记忆 | Geological memory
        self.geological_memory = GeologicalMemory(dim=self.config.dim)
        
        # 6. 三明治融合 | Sandwich fusion
        self.sandwich_fusion = SandwichFusion()
        
        # 7. 输出层 | Output layer
        self.layer_norm = nn.LayerNorm(self.config.dim)
        self.output_projection = nn.Linear(self.config.dim, self.config.vocab_size)
        
        # 8. 循环投影层（可选） | Cycle projection layer (optional)
        self.cycle_projection = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim * 2),
            nn.ReLU(),
            nn.Linear(self.config.dim * 2, self.config.dim),
            nn.Tanh()
        )
        
        # 9. QKV初始化层 | QKV initialization layers
        self.init_Q = nn.Linear(self.config.dim, self.config.dim)
        self.init_K = nn.Linear(self.config.dim, self.config.dim)
        self.init_V = nn.Linear(self.config.dim, self.config.dim)
        
        # 移动到设备 | Move to device
        self.to(self.device)
        
        print(f"✅ 组件初始化完成 | Component initialization completed")
    
    def _setup_performance_optimization(self):
        """设置性能优化 | Setup performance optimization"""
        print("⚡ 设置性能优化 | Setting up performance optimization...")
        
        # 1. 混合精度 | Mixed precision
        self.use_mixed_precision = self.config.enable_mixed_precision and torch.cuda.is_available()
        if self.use_mixed_precision:
            try:
                version = torch.__version__.split('.')
                major_version = int(version[0])
                
                if major_version >= 2:
                    self.scaler = torch.amp.GradScaler('cuda')
                else:
                    self.scaler = torch.cuda.amp.GradScaler()
                
                print(f"  ✅ 混合精度已启用 | Mixed precision enabled")
            except Exception as e:
                print(f"  ⚠️  混合精度初始化失败: {e} | Mixed precision initialization failed: {e}")
                self.use_mixed_precision = False
                self.scaler = None
        else:
            self.scaler = None
        
        # 2. 梯度检查点 | Gradient checkpointing
        self.enable_gradient_checkpointing = self.config.enable_gradient_checkpointing
        
        # 3. 梯度累积 | Gradient accumulation
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        
        # 4. CUDA优化 | CUDA optimization
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"  ✅ CUDA优化已启用 | CUDA optimization enabled")
        
        print(f"✅ 性能优化设置完成 | Performance optimization setup completed")
    
    def count_parameters(self) -> int:
        """计算总参数量 | Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _adjust_v_by_qk_relation(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                cycle: int, unit_num: int) -> torch.Tensor:
        """
        基于Q-K关系调整V值 | Adjust V value based on Q-K relationship
        
        参数 | Parameters:
        ------------
        Q, K, V : torch.Tensor
            Q, K, V矩阵 | Q, K, V matrices
        cycle : int
            当前循环序号 | Current cycle number
        unit_num : int
            当前单元序号 | Current unit number
            
        返回 | Returns:
        ------------
        torch.Tensor
            调整后的V值 | Adjusted V value
        """
        batch_size, seq_len, dim = Q.shape
        
        # 计算Q和K的相似度 | Calculate similarity between Q and K
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        R_QK = torch.sum(Q_norm * K_norm, dim=-1, keepdim=True)
        
        # 映射函数 | Mapping function
        M_R = torch.tanh(R_QK)
        
        # 根据循环和单元调整权重 | Adjust weights based on cycle and unit
        cycle_factor = max(0.5, 1.0 - cycle * 0.1)
        
        if unit_num == 1:
            # 单元1：允许较大变化 | Unit 1: Allow larger changes
            G_output = V * (0.7 + 0.3 * M_R)
        elif unit_num == 2:
            # 单元2：保守调整 | Unit 2: Conservative adjustment
            G_output = V * (0.5 + 0.5 * M_R)
        else:
            # 单元3：精细调整 | Unit 3: Fine adjustment
            similarity_threshold = 0.5
            adjustment = torch.where(
                R_QK > similarity_threshold,
                1.2 * M_R,  # 增加 | Increase
                0.8 * M_R   # 减少 | Decrease
            )
            G_output = V * (0.6 + 0.4 * adjustment)
        
        # 应用循环因子 | Apply cycle factor
        G_output = G_output * cycle_factor
        
        # 限制V值范围 | Limit V value range
        V_mean = G_output.mean().item()
        if V_mean > self.config.max_v_mean:
            scaling_factor = self.config.target_v_mean / V_mean
            G_output = G_output * scaling_factor
        elif V_mean < self.config.min_v_mean:
            scaling_factor = self.config.target_v_mean / V_mean
            G_output = G_output * scaling_factor
        
        # 残差连接 | Residual connection
        V_adjusted = 0.7 * G_output + 0.3 * V
        
        return V_adjusted
    
    def _post_process_v(self, V: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, 
                       cycle: int) -> torch.Tensor:
        """
        V值后处理 | V value post-processing
        
        参数 | Parameters:
        ------------
        V, Q, K : torch.Tensor
            V, Q, K矩阵 | V, Q, K matrices
        cycle : int
            当前循环序号 | Current cycle number
            
        返回 | Returns:
        ------------
        torch.Tensor
            处理后的V值 | Processed V value
        """
        batch_size, seq_len, dim = V.shape
        
        # 计算Q-K相似度 | Calculate Q-K similarity
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        dot_product = torch.sum(Q_norm * K_norm, dim=-1)
        avg_similarity = dot_product.mean().item()
        
        # 基于相似度调整 | Adjust based on similarity
        adjustment = 1.0
        if avg_similarity > 0.7:
            adjustment = 1.05
        elif avg_similarity < 0.3:
            adjustment = 0.95
        
        V = V * adjustment
        
        # 循环衰减 | Cycle decay
        cycle_decay = max(0.85, 1.0 - cycle * 0.03)
        V = V * cycle_decay
        
        # 范围限制 | Range limitation
        v_mean = V.mean().item()
        v_std = V.std().item()
        
        if v_mean > self.config.max_v_mean or v_std > 1.5:
            scale_factor = self.config.target_v_mean / (v_mean + 1e-8)
            scale_factor = max(0.5, min(2.0, scale_factor))
            V = V * scale_factor
        elif v_mean < self.config.min_v_mean:
            scale_factor = self.config.target_v_mean / v_mean
            V = V * scale_factor
        
        return V
    
    def _check_v_health(self, V_list: List[torch.Tensor]) -> float:
        """
        检查V值健康度 | Check V value health
        
        参数 | Parameters:
        ------------
        V_list : List[torch.Tensor]
            V值列表 | List of V values
            
        返回 | Returns:
        ------------
        float
            健康度分数（0-1） | Health score (0-1)
        """
        if len(V_list) < 2:
            return 1.0
        
        health_score = 0.0
        
        # 计算V值均值 | Calculate V value means
        v_means = [V.mean().item() for V in V_list]
        
        # 检查变化率 | Check change rate
        changes = []
        for i in range(1, len(v_means)):
            if v_means[i-1] != 0:
                change = abs(v_means[i] - v_means[i-1]) / abs(v_means[i-1])
                changes.append(change)
        
        avg_change = sum(changes) / len(changes) if changes else 0
        
        if avg_change < 0.3:
            health_score += 0.3
        
        # 检查范围 | Check range
        v_fused_mean = v_means[-1]
        if self.config.min_v_mean < v_fused_mean < self.config.max_v_mean:
            health_score += 0.3
        
        # 检查方差 | Check variance
        v_fused_std = V_list[-1].std().item()
        if v_fused_std < 1.0:
            health_score += 0.2
        
        # 检查单调性 | Check monotonicity
        is_monotonic = all(v_means[i] <= v_means[i+1] for i in range(len(v_means)-1))
        if not is_monotonic:
            health_score += 0.2
        
        return health_score
    
    def forward(self, input_ids: torch.Tensor, num_cycles: int = None, 
               return_details: bool = False) -> Dict[str, Any]:
        """
        前向传播 | Forward propagation
        
        参数 | Parameters:
        ------------
        input_ids : torch.Tensor
            输入词索引 [batch_size, seq_len] | Input token indices [batch_size, seq_len]
        num_cycles : int, optional
            持续思考循环次数，None则使用配置值 | Continuous thinking cycles, None uses config value
        return_details : bool, optional
            是否返回详细信息 | Whether to return detailed information
            
        返回 | Returns:
        ------------
        Dict[str, Any]
            输出结果 | Output results
        """
        # 确定循环次数 | Determine cycle count
        if num_cycles is None:
            num_cycles = min(self.config.max_cycles, 3)
        
        # 确保输入在正确设备上 | Ensure input is on correct device
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        
        # 确保输入类型正确 | Ensure input type is correct
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        
        # 使用混合精度上下文 | Use mixed precision context
        if self.use_mixed_precision:
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                return self._forward_impl(input_ids, num_cycles, return_details)
        else:
            return self._forward_impl(input_ids, num_cycles, return_details)
    
    def _forward_impl(self, input_ids: torch.Tensor, num_cycles: int, 
                     return_details: bool) -> Dict[str, Any]:
        """
        前向传播实现 | Forward propagation implementation
        """
        # ==================== 第1步：嵌入层 ====================
        # ==================== Step 1: Embedding Layer ====================
        emb_result = self.embedding(input_ids, return_details=False)
        base_emb = emb_result['base_embeddings']
        
        if base_emb.dtype != torch.float32:
            base_emb = base_emb.float()
        
        # ==================== 第2步：持续思考循环 ====================
        # ==================== Step 2: Continuous Thinking Cycles ====================
        all_cycle_results = []
        
        for cycle in range(num_cycles):
            print(f"🔄 持续思考循环 {cycle+1}/{num_cycles} | Continuous thinking cycle {cycle+1}/{num_cycles}")
            
            # ==================== 第3步：初始化嵌入 ====================
            # ==================== Step 3: Initialize Embeddings ====================
            if cycle == 0:
                current_emb = base_emb.clone()
            else:
                prev_result = all_cycle_results[-1]
                current_emb = prev_result['Q_final']
                current_emb = self.cycle_projection(current_emb)
            
            # ==================== 第4步：单向阀处理 ====================
            # ==================== Step 4: One-way Valve Processing ====================
            Q = current_emb.clone()
            K = current_emb.clone()
            V = current_emb.clone()
            
            Q, K, V = self.one_way_valve(Q, K, V)
            
            # 记录状态 | Record state
            self.engine.process_state(Q, K, V)
            
            # ==================== 第5步：链式反应单元处理 ====================
            # ==================== Step 5: Chain Reaction Unit Processing ====================
            V_list = []
            
            for unit_idx in range(self.config.num_units):
                # 通过链式反应单元 | Through chain reaction unit
                unit = self.chain_units[unit_idx]
                Q, K, V = unit(Q, K, V)
                
                # 通过平衡器 | Through balancer
                balancer = self.tri_balancers[unit_idx]
                Q, K, V, density, connections = balancer(Q, K, V, return_density=True)
                
                # 调整V值 | Adjust V value
                V = self._adjust_v_by_qk_relation(Q, K, V, cycle, unit_idx + 1)
                
                # 存储V值 | Store V value
                V_list.append(V)
            
            # ==================== 第6步：地质记忆存储 ====================
            # ==================== Step 6: Geological Memory Storage ====================
            Q_list = [Q] * 3  # 简化处理 | Simplified processing
            K_list = [K] * 3
            V_sublist_list = [V_list] * 3  # 简化处理 | Simplified processing
            
            self.geological_memory.store(Q_list, K_list, V_sublist_list)
            
            # ==================== 第7步：地质记忆检索 ====================
            # ==================== Step 7: Geological Memory Retrieval ====================
            depth = min(cycle, 2)
            time_layer = min(cycle, 2)
            
            Q_deep, K_deep, V_deep = self.geological_memory.retrieve(
                depth=depth,
                time_layer=time_layer
            )
            
            # ==================== 第8步：三明治融合 ====================
            # ==================== Step 8: Sandwich Fusion ====================
            # 动态调整权重 | Dynamic weight adjustment
            if cycle == 0:
                q_weights = [0.5, 0.3, 0.2]
                k_weights = [0.5, 0.3, 0.2]
                v_weights = [0.6, 0.3, 0.1]
            else:
                # 基于历史调整权重 | Adjust weights based on history
                historical_v_means = [r['V_stats']['V_fused_mean'] for r in all_cycle_results]
                current_v_mean = V.mean().item()
                historical_v_mean = np.mean(historical_v_means) if historical_v_means else 0
                
                if current_v_mean > historical_v_mean * 1.2:
                    v_weights = [0.7, 0.2, 0.1]
                elif current_v_mean < historical_v_mean * 0.8:
                    v_weights = [0.5, 0.4, 0.1]
                else:
                    v_weights = [0.6, 0.3, 0.1]
                
                q_weights = [0.5, 0.3, 0.2]
                k_weights = [0.5, 0.3, 0.2]
            
            # 融合计算 | Fusion calculation
            batch_size, seq_len, dim = Q.shape
            
            # 扩展深层状态 | Expand deep state
            Q_deep_expanded = Q_deep.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            K_deep_expanded = K_deep.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            V_deep_expanded = V_deep.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            
            Q_fused = (q_weights[0] * Q_deep_expanded + 
                      q_weights[1] * Q + 
                      q_weights[2] * current_emb)
            
            K_fused = (k_weights[0] * K_deep_expanded + 
                      k_weights[1] * K + 
                      k_weights[2] * current_emb)
            
            V_fused = (v_weights[0] * V_deep_expanded + 
                      v_weights[1] * V + 
                      v_weights[2] * current_emb)
            
            # V值后处理 | V value post-processing
            V_fused = self._post_process_v(V_fused, Q_fused, K_fused, cycle)
            
            # ==================== 第9步：输出层处理 ====================
            # ==================== Step 9: Output Layer Processing ====================
            Q_normalized = self.layer_norm(Q_fused)
            logits = self.output_projection(Q_normalized)
            
            # ==================== 第10步：保存结果 ====================
            # ==================== Step 10: Save Results ====================
            cycle_result = {
                'cycle_num': cycle + 1,
                'logits': logits,
                'Q_final': Q_fused,
                'K_final': K_fused,
                'V_final': V_fused,
                'current_emb': current_emb,
                'V_stats': {
                    'V_list_means': [v.mean().item() for v in V_list],
                    'V_fused_mean': V_fused.mean().item(),
                    'QK_similarity': F.cosine_similarity(Q.flatten(), K.flatten(), dim=0).item(),
                },
                'fusion_weights': {
                    'Q': q_weights,
                    'K': k_weights,
                    'V': v_weights
                }
            }
            
            all_cycle_results.append(cycle_result)
            
            # 健康检查 | Health check
            if self._check_v_health(V_list + [V_fused]) < 0.25:
                print(f"⚠️ V值健康度低，提前结束思考循环 | Low V value health, ending thinking cycle early")
                break
        
        # ==================== 第11步：准备最终输出 ====================
        # ==================== Step 11: Prepare Final Output ====================
        final_result = all_cycle_results[-1]
        
        # 添加V值演化分析 | Add V value evolution analysis
        if len(all_cycle_results) > 1:
            v_values = [r['V_stats']['V_fused_mean'] for r in all_cycle_results]
            final_result['V_evolution'] = self._analyze_v_evolution(v_values)
        
        # 添加详细信息 | Add detailed information
        if return_details:
            final_result['all_cycles'] = all_cycle_results
            final_result['engine_report'] = self.engine.get_analysis_report()
            final_result['config'] = self.config.to_dict()
        
        # 更新历史记录 | Update history
        self.forward_history.append({
            'timestamp': len(self.forward_history),
            'num_cycles': num_cycles,
            'input_shape': input_ids.shape,
            'output_keys': list(final_result.keys())
        })
        
        self.v_history.append(final_result['V_stats']['V_fused_mean'])
        
        return final_result
    
    def _analyze_v_evolution(self, v_values: List[float]) -> Dict[str, Any]:
        """
        分析V值演化 | Analyze V value evolution
        
        参数 | Parameters:
        ------------
        v_values : List[float]
            V值历史列表 | V value history list
            
        返回 | Returns:
        ------------
        Dict[str, Any]
            演化分析结果 | Evolution analysis results
        """
        if len(v_values) < 2:
            return {'analysis': '数据不足 | Insufficient data'}
        
        # 计算趋势 | Calculate trend
        trend = '稳定 | Stable'
        if len(v_values) >= 3:
            slope = (v_values[-1] - v_values[0]) / len(v_values)
            if slope > 0.1:
                trend = '上升 | Increasing'
            elif slope < -0.1:
                trend = '下降 | Decreasing'
        
        # 计算波动性 | Calculate volatility
        volatility = np.std(v_values) / (np.mean(v_values) + 1e-8)
        
        # 检测相变点 | Detect phase transition points
        phase_transitions = 0
        for i in range(1, len(v_values)):
            if abs(v_values[i] - v_values[i-1]) / (abs(v_values[i-1]) + 1e-8) > 0.3:
                phase_transitions += 1
        
        # 生成建议 | Generate recommendations
        recommendation = self._get_v_management_recommendation(v_values)
        
        return {
            'trend': trend,
            'volatility': volatility,
            'phase_transitions': phase_transitions,
            'final_V': v_values[-1],
            'V_range': [min(v_values), max(v_values)],
            'recommendation': recommendation
        }
    
    def _get_v_management_recommendation(self, v_values: List[float]) -> str:
        """
        获取V值管理建议 | Get V value management recommendation
        
        参数 | Parameters:
        ------------
        v_values : List[float]
            V值历史列表 | V value history list
            
        返回 | Returns:
        ------------
        str
            管理建议 | Management recommendation
        """
        if len(v_values) < 2:
            return "继续收集数据 | Continue collecting data"
        
        # 检查是否过高 | Check if too high
        if v_values[-1] > self.config.max_v_mean:
            return "V值过高，建议降低V权重或增加Q-K相似度 | V value too high, suggest reducing V weight or increasing Q-K similarity"
        
        # 检查是否持续下降 | Check if continuously decreasing
        if all(v_values[i] > v_values[i+1] for i in range(len(v_values)-1)):
            return "V值持续下降，建议增加V权重 | V value continuously decreasing, suggest increasing V weight"
        
        # 检查波动性 | Check volatility
        volatility = np.std(v_values) / (np.mean(v_values) + 1e-8)
        if volatility > 0.5:
            return "V值波动过大，建议稳定Q-K关系 | V value volatility too high, suggest stabilizing Q-K relationship"
        
        return "V值健康，继续保持 | V value healthy, continue maintaining"
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """
        获取完整分析报告 | Get complete analysis report
        
        返回 | Returns:
        ------------
        Dict[str, Any]
            分析报告 | Analysis report
        """
        # 获取引擎报告 | Get engine report
        engine_report = self.engine.get_analysis_report()
        
        # 组件统计 | Component statistics
        component_stats = {
            'embedding': str(self.embedding.__class__.__name__),
            'one_way_valve': str(self.one_way_valve.__class__.__name__),
            'chain_units': len(self.chain_units),
            'tri_balancers': len(self.tri_balancers),
            'geological_memory': str(self.geological_memory.__class__.__name__),
            'sandwich_fusion': str(self.sandwich_fusion.__class__.__name__),
        }
        
        # 性能统计 | Performance statistics
        performance_stats = {
            'total_parameters': self.count_parameters(),
            'mixed_precision_enabled': self.use_mixed_precision,
            'gradient_checkpointing_enabled': self.enable_gradient_checkpointing,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'device': str(self.device),
        }
        
        # V值统计 | V value statistics
        v_stats = {}
        if self.v_history:
            v_stats = {
                'current_V': self.v_history[-1],
                'V_history_length': len(self.v_history),
                'V_mean': np.mean(self.v_history),
                'V_std': np.std(self.v_history),
                'V_trend': '上升 | Increasing' if len(self.v_history) > 1 and self.v_history[-1] > self.v_history[0] else '稳定 | Stable',
            }
        
        # 构建报告 | Build report
        report = {
            'integrator_info': {
                'version': '1.0.0',
                'forward_calls': len(self.forward_history),
                'last_forward': self.forward_history[-1] if self.forward_history else None,
            },
            'config_summary': self.config.to_dict(),
            'engine_report': engine_report,
            'component_stats': component_stats,
            'performance_stats': performance_stats,
            'V_stats': v_stats,
            'recommendations': self._generate_recommendations(),
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """
        生成优化建议 | Generate optimization recommendations
        
        返回 | Returns:
        ------------
        List[str]
            建议列表 | Recommendation list
        """
        recommendations = []
        
        # 基于V值历史的建议 | Recommendations based on V value history
        if self.v_history:
            current_v = self.v_history[-1]
            
            if current_v > self.config.max_v_mean:
                recommendations.append("当前V值过高，建议降低V权重 | Current V value too high, suggest reducing V weight")
            
            if current_v < self.config.min_v_mean:
                recommendations.append("当前V值过低，建议增加V权重 | Current V value too low, suggest increasing V weight")
            
            # 检查V值趋势 | Check V value trend
            if len(self.v_history) >= 3:
                recent_trend = self.v_history[-3:]
                if all(recent_trend[i] < recent_trend[i+1] for i in range(2)):
                    recommendations.append("V值持续上升，考虑调整融合权重 | V value continuously increasing, consider adjusting fusion weights")
                elif all(recent_trend[i] > recent_trend[i+1] for i in range(2)):
                    recommendations.append("V值持续下降，考虑调整融合权重 | V value continuously decreasing, consider adjusting fusion weights")
        
        # 基于引擎状态的建议 | Recommendations based on engine state
        engine_report = self.engine.get_analysis_report()
        
        if engine_report.get('transition_analysis', {}).get('total_transitions', 0) > 5:
            recommendations.append("检测到多次相变，学习过程不稳定 | Multiple phase transitions detected, learning process unstable")
            recommendations.append("建议降低学习率或增加批量大小 | Suggest reducing learning rate or increasing batch size")
        
        # 基于配置的建议 | Recommendations based on configuration
        if self.config.num_cycles > 5:
            recommendations.append("持续思考循环次数较多，可能增加计算成本 | Many continuous thinking cycles, may increase computational cost")
            recommendations.append("考虑减少循环次数或启用梯度检查点 | Consider reducing cycles or enabling gradient checkpointing")
        
        return recommendations
    
    def save_pretrained(self, save_directory: str, include_config: bool = True):
        """
        伪装保存：将模型保存为Transformer格式 | Disguise save: Save model as Transformer format
        
        参数 | Parameters:
        ------------
        save_directory : str
            保存目录 | Save directory
        include_config : bool, optional
            是否包含配置文件 | Whether to include configuration file
        """
        import os
        
        # 创建目录 | Create directory
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. 保存完整状态字典 | Save complete state dictionary
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")
        
        # 2. 创建Transformer标准配置文件 | Create Transformer standard configuration file
        config = {
            # Transformer标准字段 | Transformer standard fields
            "model_type": "bert",
            "architectures": ["BertForMaskedLM"],
            "hidden_size": self.config.dim,
            "num_hidden_layers": self.config.num_units,
            "vocab_size": self.config.vocab_size,
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "max_position_embeddings": 512,
            
            # RGA识别字段（隐藏） | RGA identification fields (hidden)
            "_is_rga_disguised": True,
            "_rga_version": "2.0",
            "_rga_integrator": True,
            "_rga_config": self.config.to_dict()
        }
        
        with open(f"{save_directory}/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 3. 保存集成器配置文件（可选） | Save integrator configuration file (optional)
        if include_config:
            with open(f"{save_directory}/integrator_config.json", "w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 4. 创建词汇表文件 | Create vocabulary file
        with open(f"{save_directory}/vocab.txt", "w", encoding="utf-8") as f:
            f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
            for i in range(min(100, self.config.vocab_size)):
                f.write(f"[WORD{i}]\n")
        
        # 5. 创建分词器配置文件 | Create tokenizer configuration file
        tokenizer_config = {
            "do_lower_case": True,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "tokenizer_class": "BertTokenizer"
        }
        
        with open(f"{save_directory}/tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        
        # 6. 保存引擎状态（可选） | Save engine state (optional)
        engine_state_path = f"{save_directory}/engine_state.pkl"
        self.engine.save_state(engine_state_path)
        
        # 统计信息 | Statistics
        total_params = self.count_parameters()
        file_list = os.listdir(save_directory)
        
        print(f"✅ 伪装保存完成 | Disguise save completed: {save_directory}")
        print(f"   参数数量 | Parameter count: {total_params:,}")
        print(f"   文件结构 | File structure: {file_list}")
        
        return save_directory
    
    def load_pretrained(self, load_directory: str, strict: bool = True):
        """
        伪装加载：从Transformer格式加载模型 | Disguise load: Load model from Transformer format
        
        参数 | Parameters:
        ------------
        load_directory : str
            加载目录 | Load directory
        strict : bool, optional
            是否严格加载 | Whether to load strictly
        """
        import json
        
        # 1. 加载配置文件 | Load configuration file
        config_path = f"{load_directory}/config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在 | Configuration file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        # 检查是否是伪装模型 | Check if it's a disguised model
        if not config_data.get("_is_rga_disguised", False):
            warnings.warn("⚠️  警告：这可能不是RGA伪装模型，继续尝试加载... | Warning: This may not be an RGA disguised model, continuing to try loading...")
        
        # 2. 创建配置 | Create configuration
        if config_data.get("_rga_integrator", False):
            # 使用保存的RGA配置 | Use saved RGA configuration
            rga_config = config_data.get("_rga_config", {})
            self.config = IntegrationConfig.from_dict(rga_config)
        else:
            # 从Transformer配置创建 | Create from Transformer configuration
            self.config = IntegrationConfig(
                vocab_size=config_data.get("vocab_size", 10000),
                dim=config_data.get("hidden_size", 512),
                num_units=config_data.get("num_hidden_layers", 3)
            )
        
        # 3. 重新初始化组件 | Reinitialize components
        self._initialize_components()
        
        # 4. 加载权重 | Load weights
        model_path = f"{load_directory}/pytorch_model.bin"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在 | Model file not found: {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        
        # 5. 加载引擎状态（可选） | Load engine state (optional)
        engine_state_path = f"{load_directory}/engine_state.pkl"
        if os.path.exists(engine_state_path):
            try:
                self.engine.load_state(engine_state_path)
            except Exception as e:
                print(f"⚠️  加载引擎状态失败: {e} | Failed to load engine state: {e}")
        
        # 统计信息 | Statistics
        total_params = self.count_parameters()
        loaded_params = sum(p.numel() for p in state_dict.values())
        
        print(f"✅ 伪装加载完成 | Disguise load completed: {load_directory}")
        print(f"   加载参数 | Loaded parameters: {loaded_params:,}")
        print(f"   模型参数 | Model parameters: {total_params:,}")
        
        if missing_keys:
            print(f"   ⚠️  缺失参数 | Missing parameters: {len(missing_keys)} 个 | {len(missing_keys)} items")
            for key in missing_keys[:5]:  # 只显示前5个 | Only show first 5
                print(f"     - {key}")
            if len(missing_keys) > 5:
                print(f"     ... 还有 {len(missing_keys) - 5} 个 | ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"   ⚠️  意外参数 | Unexpected parameters: {len(unexpected_keys)} 个 | {len(unexpected_keys)} items")
            for key in unexpected_keys[:5]:
                print(f"     - {key}")
            if len(unexpected_keys) > 5:
                print(f"     ... 还有 {len(unexpected_keys) - 5} 个 | ... and {len(unexpected_keys) - 5} more")
        
        return self
    
    def visualize_state_changes(self, save_path: Optional[str] = None):
        """
        可视化状态变化 | Visualize state changes
        
        参数 | Parameters:
        ------------
        save_path : Optional[str], optional
            保存路径，None则不保存 | Save path, None means don't save
        """
        self.engine.visualize_state_changes(save_path)
    
    def reset(self):
        """重置集成器状态 | Reset integrator state"""
        self.engine.reset()
        self.forward_history = []
        self.v_history = []
        self.phase_history = []
        print("✅ 集成器状态已重置 | Integrator state reset")
    
    def __str__(self) -> str:
        """字符串表示 | String representation"""
        stats = self.engine.engine_state
        component_count = len(self.chain_units) + len(self.tri_balancers)
        
        lines = [
            "RGA Integrator | RGA 集成器",
            "=" * 60,
            f"版本 | Version: 1.0.0",
            f"设备 | Device: {self.device}",
            f"总参数量 | Total parameters: {self.count_parameters():,}",
            f"组件数量 | Component count: {component_count}",
            f"前向传播次数 | Forward calls: {len(self.forward_history)}",
            f"V值历史长度 | V value history length: {len(self.v_history)}",
            f"当前V值 | Current V value: {self.v_history[-1] if self.v_history else 'N/A'}",
            f"配置摘要 | Configuration summary:",
            f"  - 词汇表大小 | Vocabulary size: {self.config.vocab_size}",
            f"  - 模型维度 | Model dimension: {self.config.dim}",
            f"  - 链式反应单元 | Chain reaction units: {self.config.num_units}",
            f"  - 最大循环数 | Max cycles: {self.config.max_cycles}",
        ]
        
        return "\n".join(lines)


# ==================== 工厂函数 ====================
# ==================== Factory Functions ====================

def create_integrator(config: Optional[Union[IntegrationConfig, Dict]] = None, 
                     device: Optional[str] = None) -> RGAIntegrator:
    """
    创建RGA集成器 | Create RGA Integrator
    
    参数 | Parameters:
    ------------
    config : Optional[Union[IntegrationConfig, Dict]], optional
        配置实例或字典 | Configuration instance or dictionary
    device : Optional[str], optional
        设备 ('cpu' 或 'cuda')，None则自动选择 | Device ('cpu' or 'cuda'), None auto-selects
        
    返回 | Returns:
    ------------
    RGAIntegrator
        RGA集成器实例 | RGA integrator instance
        
    示例 | Example:
    ------------
        # 创建默认集成器 | Create default integrator
        integrator = create_integrator()
        
        # 创建自定义配置集成器 | Create custom configured integrator
        config = {"vocab_size": 20000, "dim": 512, "num_units": 4}
        integrator = create_integrator(config)
        
        # 指定设备 | Specify device
        integrator = create_integrator(device="cuda")
    """
    # 创建集成器 | Create integrator
    integrator = RGAIntegrator(config)
    
    # 如果指定了设备，则移动到该设备 | If device specified, move to that device
    if device is not None:
        target_device = torch.device(device)
        if target_device != integrator.device:
            integrator.to(target_device)
            integrator.device = target_device
            print(f"✅ 集成器已移动到设备 | Integrator moved to device: {target_device}")
    
    return integrator


def save_disguised_model(integrator: RGAIntegrator, save_directory: str, 
                        include_config: bool = True) -> str:
    """
    伪装保存模型 | Disguise save model
    
    参数 | Parameters:
    ------------
    integrator : RGAIntegrator
        RGA集成器实例 | RGA integrator instance
    save_directory : str
        保存目录 | Save directory
    include_config : bool, optional
        是否包含配置文件 | Whether to include configuration file
        
    返回 | Returns:
    ------------
    str
        保存路径 | Save path
        
    示例 | Example:
    ------------
        # 创建并保存集成器 | Create and save integrator
        integrator = create_integrator()
        save_disguised_model(integrator, "./saved_model")
    """
    return integrator.save_pretrained(save_directory, include_config)


def load_disguised_model(load_directory: str, device: Optional[str] = None, 
                        config: Optional[Union[IntegrationConfig, Dict]] = None) -> RGAIntegrator:
    """
    伪装加载模型 | Disguise load model
    
    参数 | Parameters:
    ------------
    load_directory : str
        加载目录 | Load directory
    device : Optional[str], optional
        设备 ('cpu' 或 'cuda')，None则自动选择 | Device ('cpu' or 'cuda'), None auto-selects
    config : Optional[Union[IntegrationConfig, Dict]], optional
        配置实例或字典（如果保存的配置不完整） | Configuration instance or dictionary (if saved config incomplete)
        
    返回 | Returns:
    ------------
    RGAIntegrator
        加载的RGA集成器实例 | Loaded RGA integrator instance
        
    示例 | Example:
    ------------
        # 加载集成器 | Load integrator
        integrator = load_disguised_model("./saved_model")
        
        # 指定设备加载 | Load with specified device
        integrator = load_disguised_model("./saved_model", device="cuda")
    """
    import json
    
    # 1. 检查是否是集成器保存的 | Check if it's integrator saved
    config_path = f"{load_directory}/config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在 | Configuration file not found: {load_directory}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    
    # 2. 创建配置 | Create configuration
    if config is None:
        if config_data.get("_rga_integrator", False):
            # 使用保存的配置 | Use saved configuration
            rga_config = config_data.get("_rga_config", {})
            config = IntegrationConfig.from_dict(rga_config)
        else:
            # 从Transformer配置创建 | Create from Transformer configuration
            config = IntegrationConfig(
                vocab_size=config_data.get("vocab_size", 10000),
                dim=config_data.get("hidden_size", 512),
                num_units=config_data.get("num_hidden_layers", 3)
            )
    
    # 3. 创建集成器 | Create integrator
    integrator = create_integrator(config, device)
    
    # 4. 加载权重 | Load weights
    integrator.load_pretrained(load_directory)
    
    return integrator


# ==================== 便捷函数 ====================
# ==================== Convenience Functions ====================

def get_default_integration_config(**kwargs) -> IntegrationConfig:
    """
    获取默认集成配置 | Get default integration configuration
    
    参数 | Parameters:
    ------------
    **kwargs : dict
        要覆盖的配置参数 | Configuration parameters to override
        
    返回 | Returns:
    ------------
    IntegrationConfig
        集成配置实例 | Integration configuration instance
        
    示例 | Example:
    ------------
        # 获取默认配置 | Get default configuration
        config = get_default_integration_config()
        
        # 获取自定义配置 | Get customized configuration
        config = get_default_integration_config(vocab_size=20000, dim=512, num_units=4)
    """
    return IntegrationConfig(**kwargs)


def validate_integration_config(config: Union[IntegrationConfig, Dict]) -> Tuple[bool, List[str]]:
    """
    验证集成配置 | Validate integration configuration
    
    参数 | Parameters:
    ------------
    config : Union[IntegrationConfig, Dict]
        集成配置实例或字典 | Integration configuration instance or dictionary
        
    返回 | Returns:
    ------------
    Tuple[bool, List[str]]
        (是否有效, 错误消息列表) | (Whether valid, error message list)
    """
    try:
        if isinstance(config, dict):
            config = IntegrationConfig.from_dict(config)
        return config.validate()
    except Exception as e:
        return False, [f"配置验证异常: {e} | Configuration validation exception: {e}"]


def test_integrator():
    """
    测试集成器 | Test integrator
    
    返回 | Returns:
    ------------
    bool
        测试是否通过 | Whether test passed
    """
    print("🧪 测试RGA集成器 | Testing RGA Integrator")
    print("=" * 60)
    
    try:
        # 1. 创建集成器 | Create integrator
        integrator = create_integrator({
            "vocab_size": 1000,
            "dim": 32,
            "num_units": 2,
            "max_cycles": 2
        })
        
        print("✅ 集成器创建成功 | Integrator created successfully")
        
        # 2. 创建测试输入 | Create test input
        input_ids = torch.randint(0, 1000, (2, 16))
        
        # 3. 前向传播 | Forward propagation
        output = integrator.forward(input_ids, num_cycles=2, return_details=False)
        
        print("✅ 前向传播成功 | Forward propagation successful")
        print(f"   输出键: {list(output.keys())} | Output keys: {list(output.keys())}")
        print(f"   Logits形状: {output['logits'].shape} | Logits shape: {output['logits'].shape}")
        
        # 4. 获取分析报告 | Get analysis report
        report = integrator.get_analysis_report()
        
        print("✅ 分析报告获取成功 | Analysis report retrieved successfully")
        print(f"   报告键: {list(report.keys())} | Report keys: {list(report.keys())}")
        
        # 5. 伪装保存 | Disguise save
        save_dir = "./test_saved_model"
        integrator.save_pretrained(save_dir, include_config=True)
        
        print("✅ 伪装保存成功 | Disguise save successful")
        
        # 6. 伪装加载 | Disguise load
        loaded_integrator = load_disguised_model(save_dir)
        
        print("✅ 伪装加载成功 | Disguise load successful")
        
        # 7. 清理测试文件 | Clean up test files
        import shutil
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        
        print("✅ 测试文件清理完成 | Test files cleaned up")
        
        print("=" * 60)
        print("✅ 所有测试通过 | All tests passed")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败 | Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 模块导出 ====================
# ==================== Module Exports ====================

__all__ = [
    # 配置类 | Configuration classes
    'IntegrationConfig',
    
    # 核心类 | Core classes
    'RGAIntegrator',
    
    # 工厂函数 | Factory functions
    'create_integrator',
    'save_disguised_model',
    'load_disguised_model',
    
    # 便捷函数 | Convenience functions
    'get_default_integration_config',
    'validate_integration_config',
    
    # 测试函数 | Test functions
    'test_integrator',
]

__version__ = "1.0.0"
__author__ = "RGA Team"
__description__ = "规则治理架构集成模块 | Rule-Governed Architecture Integration Module"


# ==================== 模块初始化 ====================
# ==================== Module Initialization ====================

def _init_module():
    """初始化模块 | Initialize module"""
    print(f"✅ RGA集成模块已加载 | RGA Integration Module loaded")
    print(f"   版本 | Version: {__version__}")
    print(f"   导出项 | Exports: {len(__all__)} 个 | {len(__all__)} items")
    print(f"   描述 | Description: {__description__}")


# 自动初始化 | Auto-initialize
if __name__ != "__main__":
    _init_module()


# ==================== 主程序入口 ====================
# ==================== Main Program Entry ====================

if __name__ == "__main__":
    # 运行测试 | Run test
    print("🚀 启动RGA集成模块测试 | Starting RGA Integration Module test...")
    success = test_integrator()
    
    if success:
        print("\n🎉 RGA集成模块测试成功！ | RGA Integration Module test successful!")
        print("   可以使用以下方式导入: | You can import using:")
        print("   from rga.integration import create_integrator, RGAIntegrator")
    else:
        print("\n❌ RGA集成模块测试失败！ | RGA Integration Module test failed!")
        exit(1)