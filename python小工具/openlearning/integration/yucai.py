"""
RGA集成器 - 规则治理架构统一实现
================================

RGA Integrator - Unified implementation of Rule-Governed Architecture

功能 | Features:
• 集成所有RGA核心功能 | Integrates all RGA core functionalities
• 提供完整前向传播流程 | Provides complete forward propagation pipeline
• 支持伪装保存/加载 | Supports disguise save/load
• 实时状态监控与分析 | Real-time state monitoring and analysis
• V值动态调控 | Dynamic V-value regulation
"""

import os
import sys

# ==================== 修复导入路径 ====================
# ==================== Fix Import Path ====================

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"当前目录: {current_dir}")

# 寻找包含 core 和 layers 的项目根目录
def find_project_root(start_path):
    """查找包含 core 和 layers 目录的项目根目录"""
    path = start_path
    while path != os.path.dirname(path):
        core_exists = os.path.exists(os.path.join(path, "core", "__init__.py"))
        layers_exists = os.path.exists(os.path.join(path, "layers", "__init__.py"))
        
        if core_exists and layers_exists:
            print(f"✅ 在 {path} 找到 core 和 layers 目录")
            return path
        
        core_dir = os.path.join(path, "core")
        layers_dir = os.path.join(path, "layers")
        if os.path.isdir(core_dir) and os.path.isdir(layers_dir):
            print(f"✅ 在 {path} 找到 core 和 layers 目录")
            return path
        
        path = os.path.dirname(path)
    
    return None

# 查找项目根目录
project_root = find_project_root(current_dir)

if project_root is None:
    possible_paths = [
        os.path.dirname(current_dir),
        os.path.dirname(os.path.dirname(current_dir)),
        r"D:\桌面\学习\python小工具\openlearning",
        current_dir,
    ]
    
    for path in possible_paths:
        print(f"尝试路径: {path}")
        core_exists = os.path.exists(os.path.join(path, "core", "__init__.py"))
        layers_exists = os.path.exists(os.path.join(path, "layers", "__init__.py"))
        
        if core_exists and layers_exists:
            project_root = path
            print(f"✅ 在 {path} 找到 core 和 layers 目录")
            break

if project_root is None:
    print("❌ 无法找到包含 core 和 layers 的目录")
    print("当前目录内容:")
    for item in os.listdir(current_dir):
        print(f"  {item}")
    
    print("\n父目录内容:")
    parent_dir = os.path.dirname(current_dir)
    for item in os.listdir(parent_dir):
        print(f"  {item}")
    
    sys.exit(1)

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"项目根目录: {project_root}")

# 现在可以尝试导入
try:
    from core import (
        RGAConfig, RGAEngine, CoreMetricsCalculator,
        create_rga_engine, get_default_config, validate_config
    )
    
    from layers import (
        VKQ_SubNet_WithFixedNorm, QVK_SubNet_WithFixedNorm, KQV_SubNet_WithFixedNorm,
        ChainReactionUnit_Final, TriValueBalancer, VDominantBalancer,
        DensityDrivenBalancer, AdaptiveStabilizer, EnhancedEmbeddingLayer,
        ConceptAwareEmbedding, SandwichFusion, GeologicalMemory, OneWayValve,
        SimpleOneWayValve, FixedRMSNorm, FixedGroupRMSNorm, ScaledFixedRMSNorm,
        create_attention_subnet, create_balancer_layer, create_embedding_layer,
        create_one_way_valve, LayerFactory, LayerRegistry, LayerConfig,
        LayerConfigManager, get_layer_factory, create_layer, list_available_layers
    )
    
    print("✅ 模块导入成功")
    
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import json
import warnings
from dataclasses import dataclass, field
from collections import deque

# ==================== 配置类 ====================
# ==================== Configuration Classes ====================

@dataclass
class IntegrationConfig:
    """集成配置类 | Integration Configuration Class"""
    
    # 模型架构参数
    vocab_size: int = 10000
    dim: int = 512
    num_units: int = 3
    max_cycles: int = 3
    phase_threshold: float = 0.43
    
    # 记忆配置
    geo_depth: int = 3
    memory_size: int = 1000
    history_length: int = 10
    
    # 性能优化
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    
    # 层配置
    embedding_type: str = "enhanced"
    attention_subnet_type: str = "vkq"
    balancer_type: str = "tri_value"
    valve_type: str = "learnable"
    
    # V值调控
    v_scaling_factor: float = 1.0
    min_v_mean: float = 0.3
    max_v_mean: float = 2.0
    target_v_mean: float = 1.0
    
    # 扩展参数
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
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
        config_dict.update(self.kwargs)
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IntegrationConfig':
        """从字典创建"""
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
        """验证配置"""
        errors = []
        
        # 验证正整数参数
        for param_name in ['vocab_size', 'dim', 'num_units', 'max_cycles', 
                          'geo_depth', 'memory_size', 'history_length']:
            value = getattr(self, param_name)
            if not isinstance(value, int) or value <= 0:
                errors.append(f"{param_name} 必须是正整数")
        
        # 验证范围参数
        if not (0 <= self.phase_threshold <= 1):
            errors.append("phase_threshold 必须在0-1之间")
        
        if not (0 < self.min_v_mean < self.max_v_mean):
            errors.append("min_v_mean 必须小于 max_v_mean")
        
        # 验证枚举参数
        valid_embedding_types = ['enhanced', 'concept_aware']
        if self.embedding_type not in valid_embedding_types:
            errors.append(f"embedding_type 必须是 {valid_embedding_types} 之一")
        
        valid_attention_types = ['vkq', 'qvk', 'kqv']
        if self.attention_subnet_type not in valid_attention_types:
            errors.append(f"attention_subnet_type 必须是 {valid_attention_types} 之一")
        
        valid_balancer_types = ['tri_value', 'v_dominant', 'density_driven', 'adaptive']
        if self.balancer_type not in valid_balancer_types:
            errors.append(f"balancer_type 必须是 {valid_balancer_types} 之一")
        
        valid_valve_types = ['learnable', 'simple', 'detach', 'gate']
        if self.valve_type not in valid_valve_types:
            errors.append(f"valve_type 必须是 {valid_valve_types} 之一")
        
        return len(errors) == 0, errors
    
    def __str__(self) -> str:
        """字符串表示"""
        items = [f"  {key}: {value}" for key, value in self.to_dict().items() 
                if key != 'kwargs']
        return "IntegrationConfig:\n" + "\n".join(items)


# ==================== 核心集成器类 ====================
# ==================== Core Integrator Class ====================

class RGAIntegrator(nn.Module):
    """RGA规则治理架构集成器"""
    
    def __init__(self, config: Optional[Union[IntegrationConfig, Dict]] = None):
        super().__init__()
        
        # 1. 配置初始化
        self.config = self._init_config(config)
        
        # 2. 设备设置
        self.device = self._init_device()
        
        # 3. 核心引擎
        self.engine = self._init_engine()
        
        # 4. 层工厂
        self.layer_factory = get_layer_factory()
        
        # 5. 组件初始化
        self._init_components()
        
        # 6. 性能优化
        self._setup_optimization()
        
        # 7. 状态跟踪
        self._init_state_tracking()
        
        self._print_init_summary()
    
    def _init_config(self, config: Optional[Union[IntegrationConfig, Dict]]) -> IntegrationConfig:
        """初始化配置"""
        if config is None:
            return IntegrationConfig()
        elif isinstance(config, dict):
            return IntegrationConfig.from_dict(config)
        else:
            return config
    
    def _init_device(self) -> torch.device:
        """初始化设备"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   设备: {device}")
        return device
    
    def _init_engine(self) -> RGAEngine:
        """初始化核心引擎"""
        return create_rga_engine(
            config={
                'vocab_size': self.config.vocab_size,
                'dim': self.config.dim,
                'phase_threshold': self.config.phase_threshold
            },
            enable_logging=True,
            enable_monitoring=True
        )
    
    def _init_components(self):
        """初始化所有组件"""
        print("🔧 初始化组件...")
        
        # 嵌入层
        self.embedding = create_embedding_layer(
            embed_type=self.config.embedding_type,
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.dim,
            marker_dim=self.config.dim
        )
        
        # 单向阀
        self.one_way_valve = create_one_way_valve(
            dim=self.config.dim,
            valve_type=self.config.valve_type
        )
        
        # 链式反应单元
        self.chain_units = nn.ModuleList([
            ChainReactionUnit_Final(dim=self.config.dim, unit_id=i)
            for i in range(self.config.num_units)
        ])
        
        # 平衡器
        self.tri_balancers = nn.ModuleList([
            create_balancer_layer(balancer_type=self.config.balancer_type, dim=self.config.dim)
            for _ in range(self.config.num_units)
        ])
        
        # 地质记忆
        self.geological_memory = GeologicalMemory(dim=self.config.dim)
        
        # 融合层
        self.sandwich_fusion = SandwichFusion()
        
        # 输出层
        self.layer_norm = nn.LayerNorm(self.config.dim)
        self.output_projection = nn.Linear(self.config.dim, self.config.vocab_size)
        
        # 循环投影
        self.cycle_projection = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim * 2),
            nn.ReLU(),
            nn.Linear(self.config.dim * 2, self.config.dim),
            nn.Tanh()
        )
        
        # QKV初始化
        self.init_Q = nn.Linear(self.config.dim, self.config.dim)
        self.init_K = nn.Linear(self.config.dim, self.config.dim)
        self.init_V = nn.Linear(self.config.dim, self.config.dim)
        
        self.to(self.device)
        print("✅ 组件初始化完成")
    
    def _setup_optimization(self):
        """设置性能优化"""
        print("⚡ 设置性能优化...")
        
        # 混合精度
        self.use_mixed_precision = (
            self.config.enable_mixed_precision and torch.cuda.is_available()
        )
        
        if self.use_mixed_precision:
            try:
                version = torch.__version__.split('.')
                major_version = int(version[0])
                
                if major_version >= 2:
                    self.scaler = torch.amp.GradScaler('cuda')
                else:
                    self.scaler = torch.cuda.amp.GradScaler()
                print("  ✅ 混合精度已启用")
            except Exception as e:
                print(f"  ⚠️  混合精度初始化失败: {e}")
                self.use_mixed_precision = False
                self.scaler = None
        else:
            self.scaler = None
        
        # 梯度检查点
        self.enable_gradient_checkpointing = self.config.enable_gradient_checkpointing
        
        # 梯度累积 - 修复：添加这个属性
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        
        # CUDA优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  ✅ CUDA优化已启用")
        
        print("✅ 性能优化设置完成")
    
    def _init_state_tracking(self):
        """初始化状态跟踪"""
        self.forward_history = []
        self.v_history = []
        self.phase_history = []
    
    def _print_init_summary(self):
        """打印初始化摘要"""
        print(f"✅ RGA集成器初始化完成")
        print(f"   版本: 1.0.0")
        print(f"   总参数量: {self.count_parameters():,}")
    
    def count_parameters(self) -> int:
        """计算总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    # ==================== V值调控方法 ====================
    # ==================== V-value Regulation Methods ====================
    
    def _adjust_v_by_qk_relation(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                cycle: int, unit_num: int) -> torch.Tensor:
        """基于Q-K关系调整V值"""
        # 计算Q-K相似度
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        R_QK = torch.sum(Q_norm * K_norm, dim=-1, keepdim=True)
        M_R = torch.tanh(R_QK)
        
        # 根据单元类型调整
        cycle_factor = max(0.5, 1.0 - cycle * 0.1)
        
        if unit_num == 1:
            G_output = V * (0.7 + 0.3 * M_R)
        elif unit_num == 2:
            G_output = V * (0.5 + 0.5 * M_R)
        else:
            similarity_threshold = 0.5
            adjustment = torch.where(
                R_QK > similarity_threshold,
                1.2 * M_R,
                0.8 * M_R
            )
            G_output = V * (0.6 + 0.4 * adjustment)
        
        # 应用循环因子并限制范围
        G_output = G_output * cycle_factor
        V_mean = G_output.mean().item()
        
        if V_mean > self.config.max_v_mean or V_mean < self.config.min_v_mean:
            scaling_factor = self.config.target_v_mean / (V_mean + 1e-8)
            G_output = G_output * scaling_factor
        
        # 残差连接
        return 0.7 * G_output + 0.3 * V
    
    def _post_process_v(self, V: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, 
                       cycle: int) -> torch.Tensor:
        """V值后处理"""
        # 计算相似度
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        dot_product = torch.sum(Q_norm * K_norm, dim=-1)
        avg_similarity = dot_product.mean().item()
        
        # 基于相似度调整
        adjustment = 1.05 if avg_similarity > 0.7 else 0.95 if avg_similarity < 0.3 else 1.0
        V = V * adjustment
        
        # 循环衰减
        cycle_decay = max(0.85, 1.0 - cycle * 0.03)
        V = V * cycle_decay
        
        # 范围限制
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
        """检查V值健康度"""
        if len(V_list) < 2:
            return 1.0
        
        v_means = [V.mean().item() for V in V_list]
        health_score = 0.0
        
        # 检查变化率
        changes = []
        for i in range(1, len(v_means)):
            if v_means[i-1] != 0:
                change = abs(v_means[i] - v_means[i-1]) / abs(v_means[i-1])
                changes.append(change)
        
        avg_change = sum(changes) / len(changes) if changes else 0
        if avg_change < 0.3:
            health_score += 0.3
        
        # 检查范围
        v_fused_mean = v_means[-1]
        if self.config.min_v_mean < v_fused_mean < self.config.max_v_mean:
            health_score += 0.3
        
        # 检查方差
        v_fused_std = V_list[-1].std().item()
        if v_fused_std < 1.0:
            health_score += 0.2
        
        # 检查单调性
        is_monotonic = all(v_means[i] <= v_means[i+1] for i in range(len(v_means)-1))
        if not is_monotonic:
            health_score += 0.2
        
        return health_score
    
    # ==================== 前向传播 ====================
    # ==================== Forward Propagation ====================
    
    def forward(self, input_ids: torch.Tensor, num_cycles: int = None, 
               return_details: bool = False) -> Dict[str, Any]:
        """前向传播主函数"""
        num_cycles = num_cycles or min(self.config.max_cycles, 3)
        
        # 设备对齐
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        
        # 混合精度上下文
        if self.use_mixed_precision:
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                return self._forward_impl(input_ids, num_cycles, return_details)
        else:
            return self._forward_impl(input_ids, num_cycles, return_details)
    
    def _forward_impl(self, input_ids: torch.Tensor, num_cycles: int, 
                     return_details: bool) -> Dict[str, Any]:
        """前向传播实现 - 支持动态批次大小"""
        # 1. 嵌入层
        emb_result = self.embedding(input_ids, return_details=False)
        base_emb = emb_result['base_embeddings'].float()
        
        # 2. 持续思考循环
        all_cycle_results = []
        
        for cycle in range(num_cycles):
            print(f"🔄 持续思考循环 {cycle+1}/{num_cycles}")
            
            # 3. 获取当前批次信息
            current_batch_size = input_ids.shape[0] if len(input_ids.shape) > 0 else 1
            current_seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else 1
            
            # 4. 初始化嵌入
            if cycle == 0:
                current_emb = base_emb.clone()
            else:
                prev_result = all_cycle_results[-1]
                # 使用前一循环的Q_final，但保持正确形状
                if prev_result['Q_final'].shape[:2] == (current_batch_size, current_seq_len):
                    prev_q = prev_result['Q_final']
                else:
                    # 如果形状不匹配，进行适配
                    prev_q = self._adapt_tensor_shape(
                        prev_result['Q_final'], 
                        (current_batch_size, current_seq_len)
                    )
                
                current_emb = self.cycle_projection(prev_q)
            
            # 5. 单向阀处理
            Q, K, V = self.one_way_valve(
                current_emb.clone(),
                current_emb.clone(),
                current_emb.clone()
            )
            
            # 6. 处理引擎状态（独立于批次）
            self.engine.process_state(Q, K, V)
            
            # 7. 链式反应单元处理
            V_list = []
            for unit_idx in range(self.config.num_units):
                Q, K, V = self.chain_units[unit_idx](Q, K, V)
                Q, K, V, density, connections = self.tri_balancers[unit_idx](
                    Q, K, V, return_density=True
                )
                V = self._adjust_v_by_qk_relation(Q, K, V, cycle, unit_idx + 1)
                V_list.append(V)
            
            # 8. 地质记忆存储与检索（独立于批次）
            Q_list, K_list, V_sublist_list = [Q]*3, [K]*3, [V_list]*3
            self.geological_memory.store(Q_list, K_list, V_sublist_list)
            
            depth = min(cycle, 2)
            time_layer = min(cycle, 2)
            Q_deep, K_deep, V_deep = self.geological_memory.retrieve(
                depth=depth, time_layer=time_layer
            )
            
            # 9. 地质记忆向量适配当前批次
            # 地质记忆可能返回不同形状，需要适配当前批次
            Q_deep = self._adapt_memory_tensor(Q_deep, Q.shape)
            K_deep = self._adapt_memory_tensor(K_deep, K.shape)
            V_deep = self._adapt_memory_tensor(V_deep, V.shape)
            
            # 10. 融合计算
            # 动态权重调整
            if cycle == 0:
                q_weights, k_weights, v_weights = [0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.6, 0.3, 0.1]
            else:
                historical_v_means = [r['V_stats']['V_fused_mean'] for r in all_cycle_results]
                current_v_mean = V.mean().item()
                historical_v_mean = np.mean(historical_v_means) if historical_v_means else 0
                
                if current_v_mean > historical_v_mean * 1.2:
                    v_weights = [0.7, 0.2, 0.1]
                elif current_v_mean < historical_v_mean * 0.8:
                    v_weights = [0.5, 0.4, 0.1]
                else:
                    v_weights = [0.6, 0.3, 0.1]
                q_weights, k_weights = [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]
            
            # 融合
            Q_fused = (q_weights[0] * Q_deep + 
                      q_weights[1] * Q + 
                      q_weights[2] * current_emb)
            
            K_fused = (k_weights[0] * K_deep + 
                      k_weights[1] * K + 
                      k_weights[2] * current_emb)
            
            V_fused = (v_weights[0] * V_deep + 
                      v_weights[1] * V + 
                      v_weights[2] * current_emb)
            
            V_fused = self._post_process_v(V_fused, Q_fused, K_fused, cycle)
            
            # 11. 输出层
            Q_normalized = self.layer_norm(Q_fused)
            logits = self.output_projection(Q_normalized)
            
            # 12. 保存结果
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
            
            # 健康检查
            if self._check_v_health(V_list + [V_fused]) < 0.25:
                print("⚠️ V值健康度低，提前结束思考循环")
                break
        
        # 13. 准备最终输出
        final_result = all_cycle_results[-1]
        
        if len(all_cycle_results) > 1:
            v_values = [r['V_stats']['V_fused_mean'] for r in all_cycle_results]
            final_result['V_evolution'] = self._analyze_v_evolution(v_values)
        
        if return_details:
            final_result['all_cycles'] = all_cycle_results
            final_result['engine_report'] = self.engine.get_analysis_report()
            final_result['config'] = self.config.to_dict()
            final_result['batch_info'] = {
                'batch_size': current_batch_size,
                'seq_len': current_seq_len,
                'shape_adaptations': self.shape_adaptations
            }
        
        # 更新历史
        self.forward_history.append({
            'timestamp': len(self.forward_history),
            'num_cycles': num_cycles,
            'input_shape': input_ids.shape,
            'output_keys': list(final_result.keys())
        })
        self.v_history.append(final_result['V_stats']['V_fused_mean'])
        
        return final_result
    
    # ==================== 新增的适配方法 ====================
    
    def _adapt_tensor_shape(self, tensor: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        """
        适配张量形状到目标形状
        
        参数:
            tensor: 输入张量
            target_shape: 目标形状 (batch_size, seq_len, ...)
        
        返回:
            适配后的张量
        """
        # 如果形状已经匹配，直接返回
        if tensor.shape[:2] == target_shape[:2]:
            return tensor
        
        current_batch, current_seq = tensor.shape[0], tensor.shape[1]
        target_batch, target_seq = target_shape[0], target_shape[1]
        
        # 记录适配情况
        if not hasattr(self, 'shape_adaptations'):
            self.shape_adaptations = []
        
        self.shape_adaptations.append({
            'from': tuple(tensor.shape),
            'to': target_shape,
            'method': 'unknown'
        })
        
        # 1. 适配批次大小
        if current_batch != target_batch:
            if target_batch > current_batch:
                # 扩展批次
                repeats = target_batch // current_batch + 1
                tensor = tensor.repeat(repeats, 1, 1)[:target_batch]
            else:
                # 缩减批次
                tensor = tensor[:target_batch]
        
        # 2. 适配序列长度
        if len(tensor.shape) > 1 and current_seq != target_seq:
            if target_seq > current_seq:
                # 扩展序列
                repeats = target_seq // current_seq + 1
                tensor = tensor.repeat(1, repeats, 1)[:, :target_seq]
            else:
                # 缩减序列
                tensor = tensor[:, :target_seq]
        
        return tensor
    
    def _adapt_memory_tensor(self, memory_tensor: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        """
        适配地质记忆张量到目标形状
        
        参数:
            memory_tensor: 地质记忆检索的张量
            target_shape: 目标形状
            
        返回:
            适配后的张量
        """
        # 地质记忆可能返回不同维度的张量
        # 我们需要适配到 (batch_size, seq_len, dim)
        
        # 如果已经是三维且批次大小匹配，直接返回
        if len(memory_tensor.shape) == 3:
            if memory_tensor.shape[0] == target_shape[0]:
                return memory_tensor
            else:
                # 批次大小不匹配，使用第一个样本并扩展
                if memory_tensor.shape[0] > 0:
                    single_sample = memory_tensor[0:1]  # 取第一个样本
                    return single_sample.expand(target_shape[0], -1, -1)
        
        # 如果是一维（特征向量），扩展到整个批次和序列
        if len(memory_tensor.shape) == 1:
            # memory_tensor 是 [dim]
            dim = memory_tensor.shape[0]
            # 扩展到 [batch_size, seq_len, dim]
            batch_size, seq_len = target_shape[0], target_shape[1]
            expanded = memory_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
            expanded = expanded.expand(batch_size, seq_len, -1)
            return expanded
        
        # 如果是二维 [seq_len, dim] 或 [batch_size, dim]
        if len(memory_tensor.shape) == 2:
            if memory_tensor.shape[0] == target_shape[1]:  # [seq_len, dim]
                # 序列维度匹配，扩展批次
                expanded = memory_tensor.unsqueeze(0)  # [1, seq_len, dim]
                expanded = expanded.expand(target_shape[0], -1, -1)
                return expanded
            elif memory_tensor.shape[0] == target_shape[0]:  # [batch_size, dim]
                # 批次匹配，扩展序列
                expanded = memory_tensor.unsqueeze(1)  # [batch_size, 1, dim]
                expanded = expanded.expand(-1, target_shape[1], -1)
                return expanded
        
        # 默认：创建一个全零张量
        return torch.zeros(target_shape, device=memory_tensor.device)
    
    def _calculate_adaptive_weights(self, current_cycle: int, total_cycles: int, 
                                   batch_size: int) -> Tuple[List, List, List]:
        """
        计算自适应融合权重，考虑批次大小
        
        参数:
            current_cycle: 当前循环索引
            total_cycles: 总循环数
            batch_size: 当前批次大小
            
        返回:
            (q_weights, k_weights, v_weights)
        """
        base_q_weights = [0.5, 0.3, 0.2]
        base_k_weights = [0.5, 0.3, 0.2]
        base_v_weights = [0.6, 0.3, 0.1]
        
        # 根据批次大小调整权重
        batch_factor = min(1.0, batch_size / 32.0)  # 以32为基准
        
        if batch_size < 4:
            # 小批次：增加地质记忆权重
            adjustment = 0.1 * batch_factor
            q_weights = [base_q_weights[0] + adjustment, 
                        base_q_weights[1] - adjustment/2, 
                        base_q_weights[2] - adjustment/2]
            
            k_weights = [base_k_weights[0] + adjustment, 
                        base_k_weights[1] - adjustment/2, 
                        base_k_weights[2] - adjustment/2]
            
            v_weights = [base_v_weights[0] + adjustment, 
                        base_v_weights[1] - adjustment/2, 
                        base_v_weights[2] - adjustment/2]
        elif batch_size > 64:
            # 大批次：增加当前层权重
            adjustment = 0.1 * batch_factor
            q_weights = [base_q_weights[0] - adjustment, 
                        base_q_weights[1] + adjustment, 
                        base_q_weights[2]]
            
            k_weights = [base_k_weights[0] - adjustment, 
                        base_k_weights[1] + adjustment, 
                        base_k_weights[2]]
            
            v_weights = [base_v_weights[0] - adjustment, 
                        base_v_weights[1] + adjustment, 
                        base_v_weights[2]]
        else:
            # 中等批次：使用基础权重
            q_weights, k_weights, v_weights = base_q_weights, base_k_weights, base_v_weights
        
        return q_weights, k_weights, v_weights
    
    def reset_shape_adaptations(self):
        """重置形状适配记录"""
        self.shape_adaptations = []
    
    # ==================== 分析方法 ====================
    # ==================== Analysis Methods ====================
    
    def _analyze_v_evolution(self, v_values: List[float]) -> Dict[str, Any]:
        """分析V值演化"""
        if len(v_values) < 2:
            return {'analysis': '数据不足'}
        
        # 计算趋势
        trend = '稳定'
        if len(v_values) >= 3:
            slope = (v_values[-1] - v_values[0]) / len(v_values)
            if slope > 0.1:
                trend = '上升'
            elif slope < -0.1:
                trend = '下降'
        
        # 计算波动性
        volatility = np.std(v_values) / (np.mean(v_values) + 1e-8)
        
        # 检测相变点
        phase_transitions = 0
        for i in range(1, len(v_values)):
            if abs(v_values[i] - v_values[i-1]) / (abs(v_values[i-1]) + 1e-8) > 0.3:
                phase_transitions += 1
        
        return {
            'trend': trend,
            'volatility': volatility,
            'phase_transitions': phase_transitions,
            'final_V': v_values[-1],
            'V_range': [min(v_values), max(v_values)],
            'recommendation': self._get_v_management_recommendation(v_values)
        }
    
    def _get_v_management_recommendation(self, v_values: List[float]) -> str:
        """获取V值管理建议"""
        if len(v_values) < 2:
            return "继续收集数据"
        
        if v_values[-1] > self.config.max_v_mean:
            return "V值过高，建议降低V权重或增加Q-K相似度"
        
        if all(v_values[i] > v_values[i+1] for i in range(len(v_values)-1)):
            return "V值持续下降，建议增加V权重"
        
        volatility = np.std(v_values) / (np.mean(v_values) + 1e-8)
        if volatility > 0.5:
            return "V值波动过大，建议稳定Q-K关系"
        
        return "V值健康，继续保持"
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """获取完整分析报告"""
        engine_report = self.engine.get_analysis_report()
        
        # 组件统计
        component_stats = {
            'embedding': str(self.embedding.__class__.__name__),
            'one_way_valve': str(self.one_way_valve.__class__.__name__),
            'chain_units': len(self.chain_units),
            'tri_balancers': len(self.tri_balancers),
            'geological_memory': str(self.geological_memory.__class__.__name__),
            'sandwich_fusion': str(self.sandwich_fusion.__class__.__name__),
        }
        
        # 性能统计
        performance_stats = {
            'total_parameters': self.count_parameters(),
            'mixed_precision_enabled': self.use_mixed_precision,
            'gradient_checkpointing_enabled': self.enable_gradient_checkpointing,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'device': str(self.device),
        }
        
        # V值统计
        v_stats = {}
        if self.v_history:
            v_stats = {
                'current_V': self.v_history[-1],
                'V_history_length': len(self.v_history),
                'V_mean': np.mean(self.v_history),
                'V_std': np.std(self.v_history),
                'V_trend': '上升' if len(self.v_history) > 1 and 
                          self.v_history[-1] > self.v_history[0] else '稳定',
            }
        
        return {
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
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于V值历史
        if self.v_history:
            current_v = self.v_history[-1]
            
            if current_v > self.config.max_v_mean:
                recommendations.append("当前V值过高，建议降低V权重")
            elif current_v < self.config.min_v_mean:
                recommendations.append("当前V值过低，建议增加V权重")
            
            # 检查V值趋势
            if len(self.v_history) >= 3:
                recent_trend = self.v_history[-3:]
                if all(recent_trend[i] < recent_trend[i+1] for i in range(2)):
                    recommendations.append("V值持续上升，考虑调整融合权重")
                elif all(recent_trend[i] > recent_trend[i+1] for i in range(2)):
                    recommendations.append("V值持续下降，考虑调整融合权重")
        
        # 基于引擎状态
        engine_report = self.engine.get_analysis_report()
        if engine_report.get('transition_analysis', {}).get('total_transitions', 0) > 5:
            recommendations.append("检测到多次相变，学习过程不稳定")
            recommendations.append("建议降低学习率或增加批量大小")
        
        # 基于配置
        if self.config.max_cycles > 5:
            recommendations.append("持续思考循环次数较多，可能增加计算成本")
            recommendations.append("考虑减少循环次数或启用梯度检查点")
        
        return recommendations
    
    # ==================== 保存/加载方法 ====================
    # ==================== Save/Load Methods ====================
    
    def save_pretrained(self, save_directory: str, include_config: bool = True):
        """伪装保存：将模型保存为Transformer格式"""
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. 保存模型权重
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")
        
        # 2. 创建配置文件
        config = {
            # Transformer标准字段
            "model_type": "bert",
            "architectures": ["BertForMaskedLM"],
            "hidden_size": self.config.dim,
            "num_hidden_layers": self.config.num_units,
            "vocab_size": self.config.vocab_size,
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "max_position_embeddings": 512,
            
            # RGA隐藏字段
            "_is_rga_disguised": True,
            "_rga_version": "2.0",
            "_rga_integrator": True,
            "_rga_config": self.config.to_dict()
        }
        
        with open(f"{save_directory}/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 3. 保存集成器配置
        if include_config:
            with open(f"{save_directory}/integrator_config.json", "w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 4. 创建词汇表
        with open(f"{save_directory}/vocab.txt", "w", encoding="utf-8") as f:
            f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
            for i in range(min(100, self.config.vocab_size)):
                f.write(f"[WORD{i}]\n")
        
        # 5. 创建分词器配置
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
        
        # 6. 保存引擎状态
        engine_state_path = f"{save_directory}/engine_state.pkl"
        self.engine.save_state(engine_state_path)
        
        total_params = self.count_parameters()
        file_list = os.listdir(save_directory)
        
        print(f"✅ 伪装保存完成: {save_directory}")
        print(f"   参数数量: {total_params:,}")
        print(f"   文件结构: {file_list}")
        
        return save_directory
    
    def load_pretrained(self, load_directory: str, strict: bool = True):
        """伪装加载：从Transformer格式加载模型"""
        import json
        
        # 1. 加载配置
        config_path = f"{load_directory}/config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        # 检查是否是伪装模型
        if not config_data.get("_is_rga_disguised", False):
            warnings.warn("⚠️ 警告：这可能不是RGA伪装模型，继续尝试加载...")
        
        # 2. 创建配置
        if config_data.get("_rga_integrator", False):
            rga_config = config_data.get("_rga_config", {})
            self.config = IntegrationConfig.from_dict(rga_config)
        else:
            self.config = IntegrationConfig(
                vocab_size=config_data.get("vocab_size", 10000),
                dim=config_data.get("hidden_size", 512),
                num_units=config_data.get("num_hidden_layers", 3)
            )
        
        # 3. 重新初始化组件 - 修复：调用正确的方法名
        self._init_components()
        
        # 4. 加载权重
        model_path = f"{load_directory}/pytorch_model.bin"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        
        # 5. 加载引擎状态
        engine_state_path = f"{load_directory}/engine_state.pkl"
        if os.path.exists(engine_state_path):
            try:
                self.engine.load_state(engine_state_path)
            except Exception as e:
                print(f"⚠️ 加载引擎状态失败: {e}")
        
        # 统计信息
        total_params = self.count_parameters()
        loaded_params = sum(p.numel() for p in state_dict.values())
        
        print(f"✅ 伪装加载完成: {load_directory}")
        print(f"   加载参数: {loaded_params:,}")
        print(f"   模型参数: {total_params:,}")
        
        if missing_keys:
            print(f"   ⚠️ 缺失参数: {len(missing_keys)} 个")
            for key in missing_keys[:5]:
                print(f"     - {key}")
            if len(missing_keys) > 5:
                print(f"     ... 还有 {len(missing_keys) - 5} 个")
        
        if unexpected_keys:
            print(f"   ⚠️ 意外参数: {len(unexpected_keys)} 个")
            for key in unexpected_keys[:5]:
                print(f"     - {key}")
            if len(unexpected_keys) > 5:
                print(f"     ... 还有 {len(unexpected_keys) - 5} 个")
        
        return self
    
    # ==================== 辅助方法 ====================
    # ==================== Helper Methods ====================
    
    def visualize_state_changes(self, save_path: Optional[str] = None):
        """可视化状态变化"""
        self.engine.visualize_state_changes(save_path)
    
    def reset(self):
        """重置集成器状态"""
        self.engine.reset()
        self.forward_history = []
        self.v_history = []
        self.phase_history = []
        print("✅ 集成器状态已重置")


# ==================== 工厂函数 ====================
# ==================== Factory Functions ====================

def create_integrator(config: Optional[Union[IntegrationConfig, Dict]] = None, 
                     device: Optional[str] = None) -> RGAIntegrator:
    """创建RGA集成器"""
    integrator = RGAIntegrator(config)
    
    if device is not None:
        target_device = torch.device(device)
        if target_device != integrator.device:
            integrator.to(target_device)
            integrator.device = target_device
            print(f"✅ 集成器已移动到设备: {target_device}")
    
    return integrator


def save_disguised_model(integrator: RGAIntegrator, save_directory: str, 
                        include_config: bool = True) -> str:
    """伪装保存模型"""
    return integrator.save_pretrained(save_directory, include_config)


def load_disguised_model(load_directory: str, device: Optional[str] = None, 
                        config: Optional[Union[IntegrationConfig, Dict]] = None) -> RGAIntegrator:
    """伪装加载模型"""
    import json
    
    # 检查是否是集成器保存的
    config_path = f"{load_directory}/config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {load_directory}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    
    # 创建配置
    if config is None:
        if config_data.get("_rga_integrator", False):
            rga_config = config_data.get("_rga_config", {})
            config = IntegrationConfig.from_dict(rga_config)
        else:
            config = IntegrationConfig(
                vocab_size=config_data.get("vocab_size", 10000),
                dim=config_data.get("hidden_size", 512),
                num_units=config_data.get("num_hidden_layers", 3)
            )
    
    # 创建集成器
    integrator = create_integrator(config, device)
    
    # 加载权重
    integrator.load_pretrained(load_directory)
    
    return integrator


# ==================== 便捷函数 ====================
# ==================== Convenience Functions ====================

def get_default_integration_config(**kwargs) -> IntegrationConfig:
    """获取默认集成配置"""
    return IntegrationConfig(**kwargs)


def validate_integration_config(config: Union[IntegrationConfig, Dict]) -> Tuple[bool, List[str]]:
    """验证集成配置"""
    try:
        if isinstance(config, dict):
            config = IntegrationConfig.from_dict(config)
        return config.validate()
    except Exception as e:
        return False, [f"配置验证异常: {e}"]


# ==================== 测试函数 ====================
# ==================== Test Functions ====================

def test_integrator() -> bool:
    """测试集成器"""
    print("🧪 测试RGA集成器")
    print("=" * 60)
    
    try:
        # 1. 创建集成器
        integrator = create_integrator({
            "vocab_size": 1000,
            "dim": 32,
            "num_units": 2,
            "max_cycles": 2
        })
        print("✅ 集成器创建成功")
        
        # 2. 创建测试输入
        input_ids = torch.randint(0, 1000, (2, 16))
        
        # 3. 前向传播
        output = integrator.forward(input_ids, num_cycles=2, return_details=False)
        print("✅ 前向传播成功")
        print(f"   输出键: {list(output.keys())}")
        print(f"   Logits形状: {output['logits'].shape}")
        
        # 4. 获取分析报告
        report = integrator.get_analysis_report()
        print("✅ 分析报告获取成功")
        print(f"   报告键: {list(report.keys())}")
        
        # 5. 伪装保存
        save_dir = "./test_saved_model"
        integrator.save_pretrained(save_dir, include_config=True)
        print("✅ 伪装保存成功")
        
        # 6. 伪装加载
        loaded_integrator = load_disguised_model(save_dir)
        print("✅ 伪装加载成功")
        
        # 7. 清理测试文件
        import shutil
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        print("✅ 测试文件清理完成")
        
        print("=" * 60)
        print("✅ 所有测试通过")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 主程序入口 ====================
# ==================== Main Entry Point ====================

if __name__ == "__main__":
    # 运行集成器测试
    print("🚀 启动RGA集成器测试...")
    success = test_integrator()
    
    if success:
        print("🎉 RGA集成器测试通过，可以正常使用！")
        print("\n使用示例:")
        print("1. 创建集成器: integrator = create_integrator()")
        print("2. 前向传播: output = integrator.forward(input_ids)")
        print("3. 获取报告: report = integrator.get_analysis_report()")
        print("4. 保存模型: integrator.save_pretrained('./model')")
    else:
        print("❌ RGA集成器测试失败，请检查错误信息")
    
    print("\n✨ RGA集成器模块加载完成")