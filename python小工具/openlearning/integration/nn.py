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

import sys
from torch.utils.data import DataLoader, random_split
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
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
import math

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

# ==================== RGA集成器类 ====================
# ==================== RGA Integrator Class ====================
class RGAIntegrator(nn.Module):
    """RGA集成器类 | RGA Integrator Class"""
    """
    规则治理架构主类
    核心设计理念：
    1. 基于连接点的密度公式驱动
    2. 模块化但集中管理
    3. 专注公式验证，不过度工程化
    """
    
    def __init__(self, config: RGAConfig = None):
        """
        初始化规则治理架构
    
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        super().__init__()
    
        # 1. 配置管理
        self.config = config if config else RGAConfig()

        # 新增：V值调控参数
        self.V_regulation_params = {
            'max_V_mean': 1.0,      # V值最大均值
            'min_V_mean': 0.3,      # V值最小均值
            'target_V_mean': 0.5,   # 目标V值均值
            'similarity_threshold': 0.2,  # Q-K相似度阈值
            'adjustment_strength': 0.3,   # 调整强度
            'cycle_decay_rate': 0.05,     # 循环衰减率
        }
        
        # 新增：V值历史记录器
        self.V_history = []
        self.max_V_history_length = 100
        
        # 新增：V值健康监控器
        self.V_health_scores = []

        # ========== 1. 初始化所有组件 ==========
        print(f"初始化RGA模型 (dim={self.config.dim}, units={self.config.num_units})...")

        # QKV初始化
        self.init_Q = nn.Linear(self.config.dim, self.config.dim)
        self.init_K = nn.Linear(self.config.dim, self.config.dim)
        self.init_V = nn.Linear(self.config.dim, self.config.dim)

        # 2. 核心模块实例化
        # 嵌入层 - 使用原始成功的模块
        self.embedding_layer = EnhancedEmbeddingLayer(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.dim,
            marker_dim=self.config.dim  # 统一维度
        )

        # 单向阀组合（可选）
        self.one_way_valve = OneWayValve(dim=self.config.dim)

        # 链式反应单元集合
        self.chain_units = nn.ModuleList()
        for unit_id in range(self.config.num_units):
            unit = ChainReactionUnit_Final(
                dim=self.config.dim,
                unit_id=unit_id
            )
            self.chain_units.append(unit)

        # 🆕 新增：在每个链式反应单元后添加三值平衡器
        self.tri_balancers = nn.ModuleList()
        for unit_id in range(self.config.num_units):
            balancer = TriValueBalancer(dim=self.config.dim)
            self.tri_balancers.append(balancer)    

        # 地质记忆层
        self.geological_memory = GeologicalMemory(dim=self.config.dim)

        # 三明治融合层
        self.sandwich_fusion = SandwichFusion()

        # 新增：循环投影层，用于多轮持续思考
        self.cycle_projection = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim * 2),
            nn.ReLU(),
            nn.Linear(self.config.dim * 2, self.config.dim),
            nn.Tanh()
        )

        # 输出层
        self.layer_norm = nn.LayerNorm(self.config.dim)
        self.output_projection = nn.Linear(self.config.dim, self.config.vocab_size)
 
        # 核心公式验证器
        self.metrics_calculator = CoreMetricsCalculator()

        # 状态跟踪
        self.phase_state = "初始阶段"
        self.density_history = deque(maxlen=self.config.history_length)
        self.validation_log = []

        # 关键参数（公式驱动）
        self._init_formula_parameters()

        # ==================== 自动内存优化配置 ====================
        # 自动检测并启用最优内存优化
        self._setup_memory_optimization()

    def _init_formula_parameters(self):
        """
        初始化公式相关参数
        这些参数由密度公式驱动，不是可学习参数
        """
        # 连接点判断阈值
        self.connection_threshold = 0.3  # 固定值，符合设计目的
        
        # 相变阈值
        self.phase_transition_threshold = self.config.phase_threshold
        
        # V值缩放因子
        self.v_scaling_factor = self.config.v_scaling_factor
        
        # 最小概念密度
        self.min_Q_concepts = self.config.min_Q_concepts
        
        # 注册为buffer，确保设备移动正确
        self.register_buffer('_dummy', torch.tensor(0.0))

    def _setup_memory_optimization(self):
        """
        自动设置内存优化 - 修复混合精度问题
        """
        print("="*60)
        print("💾 自动内存优化初始化")
        print("="*60)
    
        # 自动检测硬件
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # 1. 自动混合精度设置 - 改为根据PyTorch版本自动选择
        self.use_mixed_precision = True  # 启用混合精度
    
        if self.use_mixed_precision and torch.cuda.is_available():
            try:
                # 检查PyTorch版本
                
                version = torch.__version__.split('.')
                major_version = int(version[0])
            
                if major_version >= 2:
                    # PyTorch 2.0+ 使用新API
                    self.scaler = torch.amp.GradScaler('cuda')
                    print("✅ 自动混合精度已启用 (PyTorch 2.0+ API)")
                else:
                    # PyTorch 1.x 使用旧API
                    self.scaler = torch.cuda.amp.GradScaler()
                    print("✅ 自动混合精度已启用 (PyTorch 1.x API)")
            except Exception as e:
                print(f"⚠️  混合精度初始化失败: {e}")
                print("  将回退到全精度模式")
                self.use_mixed_precision = False
                self.scaler = None
        else:
            self.scaler = None
            if not torch.cuda.is_available():
                print("ℹ️  使用CPU模式，混合精度不可用")
    
        # 2. 梯度累积设置
        self.gradient_accumulation_steps = 1
    
        # 3. 内存高效模式
        self.memory_efficient_mode = True
    
        # 4. CUDA优化设置
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ CUDA高级优化已启用")
    
        # 5. 内存监控
        self.memory_history = []
    
        print(f"📊 内存优化已配置完成")
        print(f"   设备: {self.device}")
        print(f"   混合精度: {self.use_mixed_precision}")
        print(f"   梯度累积: {self.gradient_accumulation_steps}步")
        print("="*60)

    def _apply_memory_optimizations(self):
        """
        应用所有内存优化到当前模型 - 简化版
        """
        if hasattr(self, '_memory_optimizations_applied'):
            return
    
        print("🔧 正在应用内存优化...")
    
        # 1. 移动模型到正确设备
        self.to(self.device)
    
        # 2. 不要手动转换模型为半精度！
        # 让 autocast() 上下文自动处理精度转换
        print("  ✅ 模型保持全精度，由autocast自动管理混合精度")
    
        # 3. 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        self._memory_optimizations_applied = True
        print("✅ 内存优化已应用")

    def _enable_checkpointing(self, module):
        """启用梯度检查点"""
        # 这是一个简单实现，实际使用时可以更复杂
        try:
            from torch.utils.checkpoint import checkpoint
            module.forward = self._checkpoint_wrapper(module.forward)
        except:
            pass
    
    def _checkpoint_wrapper(self, original_forward):
        """包装forward方法以支持检查点"""
        def wrapped_forward(*args, **kwargs):
            from torch.utils.checkpoint import checkpoint
            # 只对训练模式启用检查点
            if self.training and torch.cuda.is_available():
                return checkpoint(original_forward, *args, **kwargs)
            else:
                return original_forward(*args, **kwargs)
        return wrapped_forward     

    def _log_validation_error(self, message: str):
        """
        记录验证错误
        """
        error_entry = {
            'timestamp': len(self.validation_log),
            'type': 'error',
            'message': message,
            'phase': self.phase_state
        }
        self.validation_log.append(error_entry)
        print(f"[验证错误] {message}")
    
    def _log_validation_warning(self, message: str):
        """
        记录验证警告
        """
        warning_entry = {
            'timestamp': len(self.validation_log),
            'type': 'warning',
            'message': message,
            'phase': self.phase_state
        }
        self.validation_log.append(warning_entry)
        print(f"[验证警告] {message}")
    
    def get_formula_stats(self) -> Dict:
        """
        获取公式统计信息
        """
        if len(self.density_history) == 0:
            density_stats = {'mean': 0.0, 'std': 0.0, 'trend': 'unknown'}
        else:
            density_list = list(self.density_history)
            density_stats = {
                'mean': np.mean(density_list),
                'std': np.std(density_list),
                'trend': '上升' if len(density_list) > 1 and density_list[-1] > density_list[0] else '稳定'
            }
        
        return {
            'phase_state': self.phase_state,
            'connection_threshold': self.connection_threshold,
            'phase_transition_threshold': self.phase_transition_threshold,
            'density_stats': density_stats,
            'validation_errors': len([e for e in self.validation_log if e['type'] == 'error']),
            'validation_warnings': len([e for e in self.validation_log if e['type'] == 'warning'])
        }
    
    def reset_formula_parameters(self, 
                                connection_threshold: float = None,
                                phase_threshold: float = None):
        """
        重置公式参数（用于调优）
        """
        if connection_threshold is not None and 0 <= connection_threshold <= 1:
            self.connection_threshold = connection_threshold
        
        if phase_threshold is not None:
            self.phase_transition_threshold = phase_threshold
        
        print(f"公式参数已更新: connection_threshold={self.connection_threshold}, "
              f"phase_threshold={self.phase_transition_threshold}")
    
    def update_phase_state(self, new_phase: str):
        """
        更新学习阶段
        """
        valid_phases = ['初始阶段', '探索期', '学习期', '稳定期', '收敛期']
        if new_phase in valid_phases:
            self.phase_state = new_phase
        else:
            self._log_validation_warning(f"无效的学习阶段: {new_phase}")
    
        # ==================== 核心前向传播 ====================    

    def forward(self, input_ids: torch.Tensor, num_cycles: int = 1) -> Dict[str, torch.Tensor]:
        """
        规则治理架构前向传播 - 修复V值增长问题，实现动态V值调控
        严格按照三值关系公式：V_emergent = G(R(Q,K), M(R(Q,K)))
        确保V值可增可减，基于Q-K关系动态调整
        """

        # 🆕 自动内存优化：检查并应用优化
        if not hasattr(self, '_memory_optimized'):
            self._apply_memory_optimizations()
            self._memory_optimized = True
    
        # ==================== 数据类型修复 ====================
        # 确保输入数据在正确设备上
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
    
        # 确保输入是long类型（这是token id的标准类型）
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        
        # ==================== 第1步：初始化原始嵌入 ====================
        emb_result = self.embedding_layer(input_ids, return_details=False)
        base_emb = emb_result['base_embeddings']  # [batch, seq, dim]

        # 确保基础嵌入是float32
        if base_emb.dtype != torch.float32:
            base_emb = base_emb.float()
        
        # ==================== 第2步：持续思考循环 ====================
        all_cycle_results = []
        
        for cycle in range(num_cycles):
            print(f"🔄 持续思考循环 {cycle+1}/{num_cycles}")
            
            # ==================== 第3步：嵌入初始化 ====================
            if cycle == 0:
                current_emb = base_emb.clone()
            else:
                prev_result = all_cycle_results[-1]
                current_emb = prev_result['Q_final']
                if hasattr(self, 'cycle_projection'):
                    current_emb = self.cycle_projection(current_emb)
            
            # ==================== 第4步：单向阀处理 ====================
            Q = current_emb.clone()
            K = current_emb.clone()
            V = current_emb.clone()
            
            Q, K, V = self.one_way_valve(Q, K, V)
            
            # 记录当前循环的初始状态
            self.metrics_calculator.record_state(Q, K, V)
            
            # ==================== 第5步：第1个链式反应单元 ====================
            unit1 = self.chain_units[0]
            Q1, K1, V1 = unit1(Q, K, V)

            # 🆕 新增：应用三值平衡器1
            balancer1 = self.tri_balancers[0]
            Q1, K1, V1, density1, connections1 = balancer1(Q1, K1, V1, return_density=True)
            
            # 修复：基于三值关系公式调整V1
            V1 = self._adjust_V_by_QK_relation(Q1, K1, V1, cycle, 1)
            
            V1_1V = V.clone()
            V1_2V = (V + V1 * 0.7) / 1.7
            V1_3V = V1.clone()
            V1_sublist = [V1_1V, V1_2V, V1_3V]
            
            # ==================== 第6步：第2个链式反应单元 ====================
            unit2 = self.chain_units[1]
            Q2, K2, V2 = unit2(Q1, K1, V1)
            
            # 🆕 新增：应用三值平衡器2
            balancer2 = self.tri_balancers[1]
            Q2, K2, V2, density2, connections2 = balancer2(Q2, K2, V2, return_density=True)

            # 修复：基于三值关系公式调整V2
            V2 = self._adjust_V_by_QK_relation(Q2, K2, V2, cycle, 2)
            
            V2_1V = V1.clone()
            V2_2V = (V1 + V2 * 0.7) / 1.7
            V2_3V = V2.clone()
            V2_sublist = [V2_1V, V2_2V, V2_3V]
            
            # ==================== 第7步：第3个链式反应单元 ====================
            unit3 = self.chain_units[2]
            Q3, K3, V3 = unit3(Q2, K2, V2)
            
            # 🆕 新增：应用三值平衡器3
            balancer3 = self.tri_balancers[2]
            Q3, K3, V3, density3, connections3 = balancer3(Q3, K3, V3, return_density=True)

            # 修复：基于三值关系公式调整V3
            V3 = self._adjust_V_by_QK_relation(Q3, K3, V3, cycle, 3)
            
            V3_1V = V2.clone()
            V3_2V = (V2 + V3 * 0.7) / 1.7
            V3_3V = V3.clone()
            V3_sublist = [V3_1V, V3_2V, V3_3V]
            
            # ==================== 第8步：地质记忆存储 ====================
            Q_list = [Q1, Q2, Q3]
            K_list = [K1, K2, K3]
            V_sublist_list = [V1_sublist, V2_sublist, V3_sublist]
            
            self.geological_memory.store(Q_list, K_list, V_sublist_list)
            
            # ==================== 第9步：地质记忆检索 ====================
            depth = min(cycle, 2)
            time_layer = min(cycle, 2)
            
            Q_deep, K_deep, V_deep = self.geological_memory.retrieve(
                depth=depth, 
                time_layer=time_layer
            )
            
            # ==================== 第10步：三明治融合 ====================
            # 使用动态权重调整，确保V值不会持续增长
            if cycle == 0:
                q_weights = [0.5, 0.3, 0.2]
                k_weights = [0.5, 0.3, 0.2]
                v_weights = [0.6, 0.3, 0.1]
            else:
                # 根据历史V值调整权重，防止V值持续增长
                historical_V_mean = torch.mean(torch.stack([
                    prev_result['V_final'] 
                    for prev_result in all_cycle_results
                ]), dim=0)
                
                current_V_mean = V3.mean().item()
                historical_V_mean_val = historical_V_mean.mean().item()
                
                # 如果当前V值明显高于历史均值，减少当前V的权重
                if current_V_mean > historical_V_mean_val * 1.2:
                    v_weights = [0.7, 0.2, 0.1]  # 增加历史权重，减少当前权重
                elif current_V_mean < historical_V_mean_val * 0.8:
                    v_weights = [0.5, 0.4, 0.1]  # 增加当前权重
                else:
                    v_weights = [0.6, 0.3, 0.1]  # 保持原权重
                
                q_weights = [0.5, 0.3, 0.2]
                k_weights = [0.5, 0.3, 0.2]
            
            Q_deep_weighted = Q_deep * q_weights[0]
            Q_current_weighted = Q3 * q_weights[1]
            Q_original_weighted = current_emb * q_weights[2]
            
            K_deep_weighted = K_deep * k_weights[0]
            K_current_weighted = K3 * k_weights[1]
            K_original_weighted = current_emb * k_weights[2]
            
            V_deep_weighted = V_deep * v_weights[0]
            V_current_weighted = V3 * v_weights[1]
            V_original_weighted = current_emb * v_weights[2]
            
            Q_fused = Q_deep_weighted + Q_current_weighted + Q_original_weighted
            K_fused = K_deep_weighted + K_current_weighted + K_original_weighted
            V_fused = V_deep_weighted + V_current_weighted + V_original_weighted
            
            # 对V_fused进行后处理，防止过度增长
            V_fused = self._post_process_V(V_fused, Q_fused, K_fused, cycle)
            
            # ==================== 第11步：输出层处理 ====================
            Q_normalized = self.layer_norm(Q_fused)
            logits = self.output_projection(Q_normalized)
            
            # ==================== 第12步：保存当前循环结果 ====================
            cycle_result = {
                'cycle_num': cycle + 1,
                'logits': logits,
                'Q_final': Q_fused,
                'K_final': K_fused,
                'V_final': V_fused,
                'current_emb': current_emb,
                'V_stats': {
                    'V1_mean': V1.mean().item(),
                    'V2_mean': V2.mean().item(),
                    'V3_mean': V3.mean().item(),
                    'V_fused_mean': V_fused.mean().item(),
                    'QK_similarity': F.cosine_similarity(Q3.flatten(), K3.flatten(), dim=0).item(),
                },
                'thought_metrics': {
                    'v_dominance_ratio': V3.mean().item() / (max(Q3.mean().item(), K3.mean().item()) + 1e-8),
                    'fusion_weights': {'Q': q_weights, 'K': k_weights, 'V': v_weights}
                }
            }
            
            all_cycle_results.append(cycle_result)
            
            # ==================== 第13步：V值健康检查 ====================
            if self._check_V_health(V1, V2, V3, V_fused) < 0.25:
                print(f"⚠️ V值健康度低，提前结束思考循环")
                break
        
        # ==================== 第14步：返回最终结果 ====================
        final_result = all_cycle_results[-1]
        
        # 添加V值变化分析
        if len(all_cycle_results) > 1:
            V_values = [r['V_stats']['V_fused_mean'] for r in all_cycle_results]
            final_result['V_evolution_analysis'] = self._analyze_V_evolution(V_values)
        
        return final_result
    
    def _adjust_V_by_QK_relation(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                cycle: int, unit_num: int) -> torch.Tensor:
        """
        基于三值关系公式调整V值: V_emergent = G(R(Q,K), M(R(Q,K)))
        确保V值基于Q-K关系动态调整，可增可减
        """
        batch_size, seq_len, dim = Q.shape
        
        # R(Q,K): 计算Q和K的关系矩阵（注意力分数）
        # 简化为计算Q和K的逐元素相似度
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        R_QK = torch.sum(Q_norm * K_norm, dim=-1, keepdim=True)  # [batch, seq, 1]
        
        # M(R(Q,K)): 对关系矩阵进行映射（这里使用简单的非线性变换）
        # 表示从关系中提取的模式或特征
        M_R = torch.tanh(R_QK)  # [batch, seq, 1]
        
        # G(R(Q,K), M(R(Q,K))): 基于关系和映射生成V值
        # 这里使用一个简单的线性组合，实际可以更复杂
        # 权重根据循环和单元调整，确保V值不会单调增长
        
        # 根据循环数调整：早期循环允许V值增长，后期循环限制增长
        cycle_factor = max(0.5, 1.0 - cycle * 0.1)
        
        # 根据单元调整：不同单元可能有不同的V值生成策略
        if unit_num == 1:
            # 单元1：允许V值有较大变化
            G_output = V * (0.7 + 0.3 * M_R)
        elif unit_num == 2:
            # 单元2：更保守的调整
            G_output = V * (0.5 + 0.5 * M_R)
        else:  # unit_num == 3
            # 单元3：基于Q-K关系的精细调整
            # 如果Q-K相似度高，V值增加；如果相似度低，V值减少
            similarity_threshold = 0.5
            adjustment = torch.where(
                R_QK > similarity_threshold,
                1.2 * M_R,  # 增加
                0.8 * M_R   # 减少
            )
            G_output = V * (0.6 + 0.4 * adjustment)
        
        # 应用循环因子
        G_output = G_output * cycle_factor
        
        # 确保V值不会过度增长：应用软限制
        V_mean = G_output.mean().item()
        if V_mean > 2.0:  # 如果V值均值过大
            scaling_factor = 1.5 / V_mean  # 缩放到1.5左右
            G_output = G_output * scaling_factor
        elif V_mean < 0.5:  # 如果V值均值过小
            scaling_factor = 0.8 / V_mean  # 适度增加
            G_output = G_output * scaling_factor
        
        # 保留部分原始V值的信息（残差连接思想）
        V_adjusted = 0.7 * G_output + 0.3 * V
        
        return V_adjusted
    
    def _post_process_V(self, V: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, 
                   cycle: int) -> torch.Tensor:
        """
        对融合后的V值进行后处理（内存优化版）
        避免计算大的相似度矩阵
        """
        batch_size, seq_len, dim = V.shape
     
        # 1. 使用简化的方式计算Q-K平均相似度
        # ❌ 原来的方法：计算 N×N 的大矩阵
        # ✅ 新方法：只计算对应位置的点积
    
        # 归一化Q和K
        Q_norm = F.normalize(Q, p=2, dim=-1)  # [batch, seq, dim]
        K_norm = F.normalize(K, p=2, dim=-1)  # [batch, seq, dim]
    
        # 计算对应位置的点积（余弦相似度）
        # 这避免了创建 [N, N] 的大矩阵
        dot_product = torch.sum(Q_norm * K_norm, dim=-1)  # [batch, seq]
    
        # 计算平均相似度
        avg_similarity = dot_product.mean().item()  # 标量
    
        # 2. 基于相似度调整V值
        # 注意：现在使用更温和的调整
        adjustment = 1.0
        if avg_similarity > 0.7:
            adjustment = 1.05  # 从1.1减小到1.05
        elif avg_similarity < 0.3:
            adjustment = 0.95  # 从0.9增加到0.95
    
        V = V * adjustment
    
        # 3. 基于循环数的动态调整
        # 使用更平缓的衰减
        cycle_decay = max(0.85, 1.0 - cycle * 0.03)  # 从0.7提高到0.85，衰减率从0.05减小到0.03
        V = V * cycle_decay
    
        # 4. 应用轻量级归一化，避免计算每个位置的RMS
        # ❌ 原来的方法：计算每个位置的RMS
        # ✅ 新方法：使用简单的缩放
    
        v_mean = V.mean().item()
        v_std = V.std().item()
    
        # 如果V值异常大或异常小，进行缩放
        if v_mean > 2.0 or v_std > 1.5:
            # 计算缩放因子，使均值为1.0
            scale_factor = 1.0 / (v_mean + 1e-8)
            # 限制缩放范围，避免过度调整
            scale_factor = max(0.5, min(2.0, scale_factor))
            V = V * scale_factor
        elif v_mean < 0.5:
            # 如果均值太小，适当放大
            scale_factor = 0.8 / v_mean
            V = V * scale_factor
    
        return V
    
    def _check_V_health(self, V1: torch.Tensor, V2: torch.Tensor, 
                       V3: torch.Tensor, V_fused: torch.Tensor) -> float:
        """
        检查V值的健康度
        返回0-1之间的分数，1表示健康
        """
        health_score = 0.0
        
        # 1. 检查V值是否过度增长
        v_means = [V1.mean().item(), V2.mean().item(), V3.mean().item(), V_fused.mean().item()]
        
        # 计算V值的变化率
        changes = []
        for i in range(1, len(v_means)):
            if v_means[i-1] != 0:
                change = abs(v_means[i] - v_means[i-1]) / abs(v_means[i-1])
                changes.append(change)
        
        avg_change = sum(changes) / len(changes) if changes else 0
        
        # 如果平均变化率小于0.3，认为是健康的
        if avg_change < 0.3:
            health_score += 0.3
        
        # 2. 检查V值是否在合理范围内
        v_fused_mean = V_fused.mean().item()
        if 0.3 < v_fused_mean < 2.0:
            health_score += 0.3
        
        # 3. 检查V值的方差（稳定性）
        v_fused_std = V_fused.std().item()
        if v_fused_std < 1.0:
            health_score += 0.2
        
        # 4. 检查V值是否单调增长（不应该）
        is_monotonic = all(v_means[i] <= v_means[i+1] for i in range(len(v_means)-1))
        if not is_monotonic:  # 如果不是单调增长，加分
            health_score += 0.2
        
        return health_score
    
    def _analyze_V_evolution(self, V_values: List[float]) -> Dict:
        """
        分析V值的演化模式
        """
        if len(V_values) < 2:
            return {'analysis': '数据不足'}
        
        # 计算趋势
        trend = '稳定'
        if len(V_values) >= 3:
            slope = (V_values[-1] - V_values[0]) / len(V_values)
            if slope > 0.1:
                trend = '上升'
            elif slope < -0.1:
                trend = '下降'
        
        # 计算波动性
        volatility = np.std(V_values) / (np.mean(V_values) + 1e-8)
        
        # 检测相变点
        phase_transitions = 0
        for i in range(1, len(V_values)):
            if abs(V_values[i] - V_values[i-1]) / (abs(V_values[i-1]) + 1e-8) > 0.3:
                phase_transitions += 1
        
        return {
            'trend': trend,
            'volatility': volatility,
            'phase_transitions': phase_transitions,
            'final_V': V_values[-1],
            'V_range': [min(V_values), max(V_values)],
            'recommendation': self._get_V_management_recommendation(V_values)
        }
    
    def _get_V_management_recommendation(self, V_values: List[float]) -> str:
        """
        根据V值演化给出管理建议
        """
        if len(V_values) < 2:
            return "继续收集数据"
        
        # 检查是否过度增长
        if V_values[-1] > 2.0:
            return "V值过高，建议降低V权重或增加Q-K相似度"
        
        # 检查是否持续下降
        if all(V_values[i] > V_values[i+1] for i in range(len(V_values)-1)):
            return "V值持续下降，建议增加V权重"
        
        # 检查波动性
        volatility = np.std(V_values) / (np.mean(V_values) + 1e-8)
        if volatility > 0.5:
            return "V值波动过大，建议稳定Q-K关系"
        
        return "V值健康，继续保持"
    
    def _validate_formula_execution(self, step_data: Dict) -> bool:
        """
        核心公式验证方法
        验证当前步骤的公式执行是否符合设计目的
    
        Args:
            step_data: 当前步骤的数据字典
        
        Returns:
            bool: 是否通过验证
        """
        try:
            # 验证1: 连接点密度公式
            if 'markers' in step_data:
                markers = step_data['markers']
                density_info = self._compute_connection_density(markers)
            
                # 检查密度值范围 [0, 1]
                density = density_info['static_density']
                if not (0 <= density <= 1):
                    self._log_validation_error(f"密度值越界: {density}")
                    return False
            
                # 检查连接数合理性
                max_connections = (density_info['nodes'] * (density_info['nodes'] - 1)) / 2
                if density_info['connections'] > max_connections:
                    self._log_validation_error("连接数异常")
                    return False
             
                step_data['density_validated'] = True
                self.density_history.append(density)
        
            # 验证2: 相变检测公式
            if 'phase_delta' in step_data:
                delta = step_data['phase_delta']
                # 相变阈值验证
                if delta > self.phase_transition_threshold:
                    step_data['phase_transition'] = True
                else:
                    step_data['phase_transition'] = False
        
            # 验证3: V值主导性验证
            if 'V_means' in step_data:
                V_means = step_data['V_means']
                # 检查V值是否显著大于Q/K（V值主导）
                if 'Q_mean' in step_data and 'K_mean' in step_data:
                    V_avg = sum(V_means) / len(V_means) if V_means else 0
                    Q_avg = step_data['Q_mean']
                    K_avg = step_data['K_mean']
                 
                    if V_avg < max(Q_avg, K_avg) * self.v_scaling_factor:
                        self._log_validation_warning("V值主导性不足")
        
            return True
        
        except Exception as e:
            self._log_validation_error(f"公式验证异常: {str(e)}")
            return False

    def _compute_connection_density(self, markers: torch.Tensor) -> Dict:
        """
        执行连接点密度公式计算
        这是整个架构的数学基础
    
        Args:
            markers: 标记向量 [B, S, D]
         
        Returns:
            密度计算结果字典
        """
        B, S, D = markers.shape
        markers_flat = markers.contiguous().view(-1, D)
        N = markers_flat.size(0)
    
        if N <= 1:
            return {'static_density': 0.0, 'connections': 0, 'nodes': N}
     
        # 核心：连接点判断（余弦相似度 > threshold）
        norm_markers = F.normalize(markers_flat, p=2, dim=1)
        similarity_matrix = torch.mm(norm_markers, norm_markers.T)
    
        # 二值化连接判断
        connections = (similarity_matrix > self.connection_threshold).float()
        M = connections.sum().item() / 2  # 无向图去重
    
        # 静态密度公式
        static_density = (2 * M) / ((N + 1) * N) if N > 1 else 0.0
    
        return {
            'static_density': static_density,
            'connections': M,
            'nodes': N,
            'similarity_mean': similarity_matrix.mean().item()
        }
    
    def detect_phase_transition(self, V_history: List[float], threshold: float = 0.43) -> int:
        """检测相变次数"""
        if len(V_history) < 2:
            return 0
        
        transitions = 0
        for i in range(1, len(V_history)):
            prev_V = V_history[i-1]
            curr_V = V_history[i]
            if prev_V > 0:
                delta = (curr_V - prev_V) / abs(prev_V)
                if delta > threshold:
                    transitions += 1
        return transitions
    
    def identify_learning_phase(self, V_history: List[float]) -> str:
        """识别学习阶段"""
        if len(V_history) < 3:
            return "初始化"
        
        recent = V_history[-3:]
        mean_v = np.mean(recent)
        std_v = np.std(recent)
        trend = np.mean(np.diff(recent))
        
        if std_v > mean_v * 0.5:
            return "震荡期"
        elif trend < -0.1:
            return "下降期"
        elif abs(trend) < 0.05:
            return "稳定期"
        else:
            return "上升期"
    
    def initialize_QKV(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """初始化Q₀、K₀、V₀"""
        batch_size, seq_len, dim = x.shape
        x_mean = x.mean(dim=1, keepdim=True).expand(batch_size, seq_len, dim)
        return self.init_Q(x_mean), self.init_K(x_mean), self.init_V(x_mean)
    
    # ==================== 伪装保存 ====================
    
    def save_pretrained(self, save_directory: str):
        """
        伪装保存：将RGA模型保存为Transformer格式
        创建标准文件：pytorch_model.bin, config.json, vocab.txt, tokenizer_config.json
        """
        import os
        import json
        
        # 创建目录
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. 保存完整状态字典（包括buffer）
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")
        
        # 2. 创建Transformer标准配置文件
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
            
            # RGA识别字段（隐藏）
            "_is_rga_disguised": True,
            "_rga_version": "1.0",
            "_rga_config": {
                "dim": self.config.dim,
                "num_units": self.config.num_units,
                "geo_depth": self.config.geo_depth,
                "phase_threshold": self.config.phase_threshold
            }
        }
        
        with open(f"{save_directory}/config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # 3. 创建词汇表文件（最小集）
        with open(f"{save_directory}/vocab.txt", "w") as f:
            f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
            for i in range(100):
                f.write(f"[WORD{i}]\n")
        
        # 4. 创建分词器配置文件
        tokenizer_config = {
            "do_lower_case": True,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "tokenizer_class": "BertTokenizer"
        }
        
        with open(f"{save_directory}/tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        
        # 5. 统计信息
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ 伪装保存完成: {save_directory}")
        print(f"   参数数量: {total_params:,}")
        print(f"   文件结构: {os.listdir(save_directory)}")
        
        return save_directory
    
    # ==================== 伪装加载 ====================
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str, config: RGAConfig = None):
        """
        伪装加载：从Transformer格式加载回RGA模型
        自动识别伪装标记，恢复完整RGA架构
        """
        import json
        
        # 1. 加载配置文件
        with open(f"{pretrained_path}/config.json", "r") as f:
            config_data = json.load(f)
        
        # 检查是否是伪装模型
        if not config_data.get("_is_rga_disguised", False):
            print("⚠️  警告：这可能不是RGA伪装模型，继续尝试加载...")
        
        # 2. 创建RGA配置
        if config is None:
            config = RGAConfig()
            config.dim = config_data["hidden_size"]
            config.vocab_size = config_data["vocab_size"]
            config.num_units = config_data["num_hidden_layers"]
        
        # 3. 创建RGA模型实例
        model = cls(config)
        
        # 4. 加载权重文件
        state_dict = torch.load(f"{pretrained_path}/pytorch_model.bin", map_location="cpu")
        
        # 5. 加载到模型（允许部分参数不匹配）
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        # 6. 统计信息
        total_params = sum(p.numel() for p in model.parameters())
        loaded_params = sum(p.numel() for p in state_dict.values())
        
        print(f"✅ 伪装加载完成: {pretrained_path}")
        print(f"   加载参数: {loaded_params:,}")
        print(f"   模型参数: {total_params:,}")
        
        if missing_keys:
            print(f"   ⚠️  缺失参数: {len(missing_keys)} 个（使用初始化值）")
        if unexpected_keys:
            print(f"   ⚠️  意外参数: {len(unexpected_keys)} 个（已忽略）")
        
        return model    
    
# ==================== 3. 智能文本数据集（结构化整理版）====================
class SmartTextDataset(Dataset):
    """智能文本数据集（结构化日志版）- 支持LCCC-base_train.json格式"""
    
    def __init__(self, data_path, seq_length=128, vocab_size=15000, max_samples=None, language='zh'):
        self.data_path = data_path
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.language = language
        
        # 打印头部信息
        self._print_header("📁 智能文本数据集初始化")
        
        # 执行数据处理流程
        self._load_data(max_samples)
        self._analyze_data()
        self._build_vocabulary()
        self._encode_for_training()
        
        # 打印尾部信息
        self._print_footer("✅ 数据集准备完成")
    
    # ==================== 日志打印方法 ====================
    def _print_header(self, title):
        """打印标题"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
    
    def _print_step(self, step, message, level=1):
        """打印步骤信息"""
        indent = "  " * level
        prefix = "├─ " if level == 1 else "│  ├─ " if level == 2 else "│  │  └─ "
        print(f"{indent}{prefix}{step}: {message}")
    
    def _print_info(self, key, value, level=2):
        """打印键值对信息"""
        indent = "  " * level
        prefix = "│  " + "  " * (level-1) + "├─ " if level >= 2 else "├─ "
        print(f"{indent}{prefix}{key}: {value}")
    
    def _print_stat(self, stats_dict, level=3):
        """打印统计信息"""
        indent = "  " * level
        prefix = "│  " + "  " * (level-1) + "├─ " if level >= 3 else "│  └─ "
        
        if isinstance(stats_dict, dict):
            for key, value in stats_dict.items():
                print(f"{indent}{prefix}{key}: {value}")
        else:
            print(f"{indent}{prefix}{stats_dict}")
    
    def _print_footer(self, message):
        """打印尾部信息"""
        print(f"\n{message}")
        print(f"{'='*60}\n")
    
    # ==================== 数据加载模块 ====================
    def _load_data(self, max_samples):
        """加载原始数据"""
        self._print_step("数据加载", "", level=1)
        self._print_info("来源", self.data_path, level=2)
        
        if not os.path.exists(self.data_path):
            self._print_info("状态", "❌ 文件不存在", level=2)
            self.texts = []
            return
        
        if not self.data_path.endswith('.json'):
            self._print_info("状态", "❌ 不是JSON格式", level=2)
            self.texts = []
            return
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_dialogues = len(data)
            self._print_info("总对话数", f"{total_dialogues:,}条", level=2)
            
            # 采样加载数据
            self.texts = []
            sample_count = 0
            max_to_load = max_samples if max_samples else total_dialogues
            
            for i, dialogue in enumerate(data):
                if max_samples and sample_count >= max_samples:
                    break
                
                if isinstance(dialogue, list):
                    dialogue_parts = []
                    
                    for sentence in dialogue:
                        if isinstance(sentence, str):
                            cleaned = self._clean_text_basic(sentence)
                            if cleaned:
                                dialogue_parts.append(cleaned)
                    
                    if dialogue_parts:
                        dialogue_text = " ".join(dialogue_parts)
                        self.texts.append(dialogue_text)
                        sample_count += 1
                
                # 每50万条显示一次进度
                if i > 0 and i % 500000 == 0:
                    self._print_info("处理进度", f"{i:,}/{total_dialogues:,}", level=2)
            
            self._print_info("采样数量", f"{len(self.texts):,}条", level=2)
            self._print_info("状态", "✅ 成功", level=2)
            
        except Exception as e:
            self._print_info("状态", f"❌ 加载失败: {e}", level=2)
            self.texts = []
    
    def _clean_text_basic(self, text):
        """基础文本清洗"""
        if not isinstance(text, str):
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除特殊符号
        text = re.sub(r'[\{\}\[\]\"]', '', text)
        # 合并多个空格
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空格
        text = text.strip()
        
        return text
    
    # ==================== 数据分析模块 ====================
    def _analyze_data(self):
        """分析数据特性并决定处理策略"""
        if not self.texts:
            self._print_step("数据分析", "❌ 无数据可分析", level=1)
            self.use_char_level = True
            return
        
        self._print_step("数据分析", "", level=1)
        
        # 采样分析
        sample_size = min(1000, len(self.texts))
        sample_texts = random.sample(self.texts, sample_size) if len(self.texts) > sample_size else self.texts
        
        self._print_info("采样数量", f"{len(sample_texts)}条", level=2)
        
        # 分析空格比例和词长
        analysis_result = self._analyze_text_properties(sample_texts)
        
        # 决定处理策略
        if analysis_result['space_ratio'] > 0.1 and ' ' in sample_texts[0]:
            self.use_char_level = False
            self._print_info("处理策略", "🎯 词级处理（检测到空格）", level=2)
        else:
            self.use_char_level = True
            self._print_info("处理策略", "🎯 字符级处理", level=2)
        
        # 文本长度统计
        self._analyze_text_lengths()
    
    def _analyze_text_properties(self, sample_texts):
        """分析文本属性"""
        total_chars = 0
        space_chars = 0
        word_lengths = []
        
        for text in sample_texts:
            total_chars += len(text)
            space_chars += text.count(' ')
            
            words = text.split()
            for word in words:
                word_lengths.append(len(word))
        
        space_ratio = space_chars / total_chars if total_chars > 0 else 0
        avg_word_len = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        self._print_info("平均空格比例", f"{space_ratio:.1%}", level=2)
        self._print_info("平均词长", f"{avg_word_len:.1f}字符", level=2)
        
        return {
            'space_ratio': space_ratio,
            'avg_word_len': avg_word_len
        }
    
    def _analyze_text_lengths(self):
        """分析文本长度"""
        if self.use_char_level:
            lengths = [len(t.replace(' ', '')) for t in self.texts]
            length_desc = "字符"
        else:
            lengths = [len(t.split()) for t in self.texts]
            length_desc = "词"
        
        # 计算统计信息
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
        
        sorted_lengths = sorted(lengths)
        p95_length = sorted_lengths[int(len(lengths) * 0.95)]
        
        self._print_step("长度统计", "", level=2)
        self._print_stat({"平均": f"{avg_length:.1f} {length_desc}"}, level=3)
        self._print_stat({"最短": f"{min_length} {length_desc}"}, level=3)
        self._print_stat({"最长": f"{max_length} {length_desc}"}, level=3)
        self._print_stat({"95%分位": f"{p95_length} {length_desc}"}, level=3)
        
        # 显示示例文本
        if self.texts:
            example = self.texts[0][:200] + ("..." if len(self.texts[0]) > 200 else "")
            self._print_info("示例文本", example, level=2)
        
        # 给出建议
        if p95_length < self.seq_length:
            self._print_info("建议", f"seq_length可从{self.seq_length}调整为{p95_length}", level=2)
        else:
            self._print_info("注意", f"{p95_length}个{length_desc}可能超过模型容量", level=2)
    
    # ==================== 词汇表构建模块 ====================
    def _build_vocabulary(self):
        """构建词汇表"""
        if not self.texts:
            self._print_step("词汇表构建", "❌ 无数据，使用默认词汇表", level=1)
            self._create_default_vocab()
            return
        
        self._print_step("词汇表构建", "", level=1)
        
        # 特殊token
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        if self.use_char_level:
            self._build_char_vocab(special_tokens)
        else:
            self._build_word_vocab(special_tokens)
        
        # 词汇表分析
        self._analyze_vocabulary()
    
    def _create_default_vocab(self):
        """创建默认词汇表"""
        self.vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.token2idx = {token: i for i, token in enumerate(self.vocab)}
        self.idx2token = {i: token for i, token in enumerate(self.vocab)}
    
    def _build_char_vocab(self, special_tokens):
        """构建字符级词汇表"""
        self._print_info("处理方式", "字符级处理", level=2)
        
        all_chars = ''.join([t.replace(' ', '') for t in self.texts])
        
        if not all_chars:
            self._print_info("状态", "❌ 文本为空，使用默认词汇表", level=2)
            self._create_default_vocab()
            return
        
        char_counter = Counter(all_chars)
        total_chars = len(all_chars)
        
        # 构建词汇表
        vocab = special_tokens.copy()
        for char, freq in char_counter.most_common():
            if char not in vocab:
                vocab.append(char)
            if len(vocab) >= self.vocab_size:
                break
        
        # 计算覆盖度
        covered_chars = sum(freq for char, freq in char_counter.items() if char in vocab)
        coverage = covered_chars / total_chars if total_chars > 0 else 0
        
        self.vocab = vocab
        self.token2idx = {token: i for i, token in enumerate(self.vocab)}
        self.idx2token = {i: token for i, token in enumerate(self.vocab)}
        
        self._print_info("总字符数", f"{total_chars:,}", level=2)
        self._print_info("不同字符数", f"{len(char_counter):,}", level=2)
        self._print_info("词汇表大小", f"{len(self.vocab)}", level=2)
        self._print_info("覆盖度", f"{coverage:.1%}", level=2)
    
    def _build_word_vocab(self, special_tokens):
        """构建词级词汇表"""
        self._print_info("处理方式", "词级处理（使用空格分词）", level=2)
        
        import jieba
        word_counter = Counter()
        
        # 统计词频
        for text in self.texts:
            words = text.split()
            word_counter.update(words)
        
        total_words = sum(word_counter.values())
        
        # 智能词汇表构建策略
        vocab = special_tokens.copy()
        
        # 策略1：先收集所有出现≥2次的词
        freq_ge_2 = [(word, freq) for word, freq in word_counter.items() if freq >= 2]
        
        if len(freq_ge_2) <= self.vocab_size - len(special_tokens):
            # 如果出现≥2次的词不多，全部加入
            for word, freq in sorted(freq_ge_2, key=lambda x: x[1], reverse=True):
                vocab.append(word)
        else:
            # 如果出现≥2次的词太多，按频率排序取前N个
            sorted_words = sorted(freq_ge_2, key=lambda x: x[1], reverse=True)
            for word, freq in sorted_words[:self.vocab_size - len(special_tokens)]:
                vocab.append(word)
        
        # 如果词汇表还没满，补充一些出现1次的词（优先选择单字词）
        if len(vocab) < self.vocab_size:
            freq_eq_1 = [(word, 1) for word, freq in word_counter.items() if freq == 1]
            
            # 优先加入单字词（对中文很重要）
            single_char_words = [(word, 1) for word, _ in freq_eq_1 if len(word) == 1]
            for word, _ in single_char_words[:self.vocab_size - len(vocab)]:
                vocab.append(word)
            
            # 如果还有空间，加入其他词
            if len(vocab) < self.vocab_size:
                remaining_words = [(word, 1) for word, _ in freq_eq_1 if len(word) > 1]
                for word, _ in remaining_words[:self.vocab_size - len(vocab)]:
                    vocab.append(word)
        
        # 计算覆盖度
        covered_words = sum(freq for word, freq in word_counter.items() if word in vocab)
        coverage = covered_words / total_words if total_words > 0 else 0
        
        self.vocab = vocab
        self.token2idx = {token: i for i, token in enumerate(self.vocab)}
        self.idx2token = {i: token for i, token in enumerate(self.vocab)}
        
        self._print_step("统计信息", "", level=2)
        self._print_stat({"总词数": f"{total_words:,}"}, level=3)
        self._print_stat({"不同词数": f"{len(word_counter):,}"}, level=3)
        self._print_stat({"高频词(≥2次)": f"{len(freq_ge_2):,}个"}, level=3)
        self._print_stat({"低频词(1次)": f"{len(word_counter) - len(freq_ge_2):,}个"}, level=3)
        
        self._print_step("词汇表详情", "", level=2)
        self._print_stat({"大小": f"{len(self.vocab)}词元"}, level=3)
        self._print_stat({"覆盖度": f"{coverage:.1%}"}, level=3)
        
        # 显示示例词汇
        if len(self.vocab) > 30:
            self._print_info("示例词汇", f"{self.vocab[:30]}...", level=2)
        else:
            self._print_info("示例词汇", self.vocab, level=2)
    
    def _analyze_vocabulary(self):
        """分析词汇表特征"""
        self._print_step("词汇分布", "", level=2)
        
        # 检查多字词
        multi_char_words = [word for word in self.vocab if len(word) > 1]
        chinese_multi_char_words = [
            word for word in multi_char_words 
            if all('\u4e00' <= ch <= '\u9fff' for ch in word)
        ]
        
        # 按词长统计
        length_dist = {}
        for word in self.vocab:
            length = len(word)
            length_dist[length] = length_dist.get(length, 0) + 1
        
        self._print_stat({"多字词": f"{len(multi_char_words)}个 ({len(multi_char_words)/len(self.vocab):.1%})"}, level=3)
        self._print_stat({"中文多字词": f"{len(chinese_multi_char_words)}个"}, level=3)
        self._print_stat({"长度分布": dict(sorted(length_dist.items()))}, level=3)
        
        # 显示中文多字词示例
        if chinese_multi_char_words:
            if len(chinese_multi_char_words) > 30:
                self._print_info("中文多字词示例", f"{chinese_multi_char_words[:30]}...", level=2)
            else:
                self._print_info("中文多字词示例", chinese_multi_char_words, level=2)
    
    # ==================== 数据编码模块 ====================
    def _encode_for_training(self):
        """编码数据为训练格式"""
        if not self.texts or not hasattr(self, 'token2idx'):
            self._print_step("数据编码", "❌ 无数据或词汇表，无法编码", level=1)
            self.encoded_segments = []
            return
        
        self._print_step("数据编码", "", level=1)
        
        self.encoded_segments = []
        total_segments = 0
        total_tokens = 0
        
        # 编码每段文本
        for text in self.texts:
            if not text:
                continue
            
            # 编码原始文本
            encoded = self._encode_text(text)
            
            # 添加特殊标记
            encoded = [self.token2idx.get('<BOS>', 2)] + encoded + [self.token2idx.get('<EOS>', 3)]
            
            # 分割为训练段
            segments = self._split_into_segments(encoded)
            
            for segment in segments:
                self.encoded_segments.append(segment)
                total_segments += 1
                total_tokens += len([t for t in segment if t != 0])
        
        self._print_info("训练段", f"{len(self.encoded_segments):,}个", level=2)
        self._print_info("有效token数", f"{total_tokens:,}", level=2)
        
        # 分析填充情况
        self._analyze_padding()
    
    def _encode_text(self, text):
        """编码单段文本"""
        if self.use_char_level:
            text_no_spaces = text.replace(' ', '')
            return [self.token2idx.get(ch, 1) for ch in text_no_spaces]
        else:
            import jieba
            continuous_text = text.replace(' ', '')
            words = jieba.lcut(continuous_text)
            return [self.token2idx.get(word, 1) for word in words]
    
    def _split_into_segments(self, encoded):
        """将编码后的文本分割为训练段"""
        segments = []
        step_size = max(1, self.seq_length // 2)
        
        for i in range(0, len(encoded), step_size):
            segment = encoded[i:i + self.seq_length]
            
            # 检查有效长度
            valid_length = len([t for t in segment if t != 0])
            min_valid = max(5, self.seq_length // 4)
            
            if valid_length >= min_valid:
                if len(segment) < self.seq_length:
                    segment = segment + [0] * (self.seq_length - len(segment))
                segments.append(segment)
        
        return segments
    
    def _analyze_padding(self):
        """分析填充情况"""
        if not self.encoded_segments:
            return
        
        padded_count = 0
        total_padding = 0
        actual_lengths = []
        
        for segment in self.encoded_segments:
            padding_len = segment.count(0)
            if padding_len > 0:
                padded_count += 1
                total_padding += padding_len
            
            # 计算实际长度
            actual_len = 0
            for token in segment:
                if token == 0:
                    break
                actual_len += 1
            actual_lengths.append(actual_len)
        
        total_segments_len = len(self.encoded_segments) * self.seq_length
        padding_ratio = total_padding / total_segments_len if total_segments_len > 0 else 0
        effective_ratio = (total_segments_len - total_padding) / total_segments_len
        avg_actual_len = sum(actual_lengths) / len(actual_lengths) if actual_lengths else 0
        
        self._print_step("填充统计", "", level=2)
        self._print_stat({"填充比例": f"{padding_ratio:.1%}"}, level=3)
        self._print_stat({"有效比例": f"{effective_ratio:.1%}"}, level=3)
        self._print_stat({"填充段比例": f"{padded_count/len(self.encoded_segments):.1%}"}, level=3)
        
        self._print_step("长度信息", "", level=2)
        self._print_stat({"平均长度": f"{avg_actual_len:.1f} tokens"}, level=3)
        self._print_stat({"长度范围": f"{min(actual_lengths)}-{max(actual_lengths)} tokens"}, level=3)
    
    # ==================== Dataset接口方法 ====================
    def __len__(self):
        """返回数据集大小"""
        return len(self.encoded_segments)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        if idx >= len(self.encoded_segments):
            raise IndexError(f"索引 {idx} 超出范围，数据集大小: {len(self.encoded_segments)}")
        
        seq = self.encoded_segments[idx]
        
        # 确保序列长度正确
        if len(seq) != self.seq_length:
            if len(seq) > self.seq_length:
                seq = seq[:self.seq_length]
            else:
                seq = seq + [0] * (self.seq_length - len(seq))
        
        # 语言建模任务：预测下一个token
        target = seq[1:] + [0]
        
        return {
            'input_ids': torch.tensor(seq, dtype=torch.long),
            'target_ids': torch.tensor(target, dtype=torch.long)
        }
    
    # ==================== 辅助方法 ====================
    def get_vocab_size(self):
        """获取词汇表大小"""
        return len(self.vocab) if hasattr(self, 'vocab') else 0
    
    def get_sample_count(self):
        """获取样本数量"""
        return len(self.encoded_segments) if hasattr(self, 'encoded_segments') else 0
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'vocab_size': self.get_vocab_size(),
            'sample_count': self.get_sample_count(),
            'seq_length': self.seq_length,
            'use_char_level': self.use_char_level if hasattr(self, 'use_char_level') else None
        }

class VisualTrainingProgress:
    """可视化训练进度管理器"""
    
    def __init__(self, total_batches, desc="训练", bar_width=50):
        self.total_batches = total_batches
        self.desc = desc
        self.bar_width = bar_width
        self.current_batch = 0
        self.start_time = time.time()
        self.metrics = {}
        
    def update(self, metrics_dict, batch_idx):
        """更新进度和指标"""
        self.current_batch = batch_idx
        self.metrics = metrics_dict
        
        # 计算进度百分比
        progress = (batch_idx + 1) / self.total_batches
        filled_length = int(self.bar_width * progress)
        bar = '█' * filled_length + '─' * (self.bar_width - filled_length)
        
        # 计算时间
        elapsed = time.time() - self.start_time
        if progress > 0:
            remaining = elapsed / progress * (1 - progress)
        else:
            remaining = 0
            
        # 构建指标字符串
        metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics_dict.items()])
        
        # 清屏并重新打印
        print("\033[F" * 3 + "\033[K")  # 回到上三行并清除
        print(f"{self.desc} [{bar}] {progress*100:5.1f}% ({elapsed:5.0f}s/{remaining:5.0f}s)")
        print(f"┌{'─'*(self.bar_width+30)}┐")
        print(f"│ {metrics_str:<{self.bar_width+28}} │")
        print(f"└{'─'*(self.bar_width+30)}┘", flush=True)
        
    def complete(self):
        """完成进度显示"""
        elapsed = time.time() - self.start_time
        print("\033[F" * 3 + "\033[K")  # 回到上三行并清除
        print(f"{self.desc} [{'█'*self.bar_width}] 100% ({elapsed:5.0f}s)")
        print(f"┌{'─'*(self.bar_width+30)}┐")
        print(f"│ {'训练完成!':^{self.bar_width+28}} │")
        print(f"└{'─'*(self.bar_width+30)}┘")


class AdvancedConstrainedArchitectureTrainer:
    """高级可视化训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        os.makedirs(config['output_dir'], exist_ok=True)
        
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'best_val_loss': float('inf'),
            'metrics': []
        }
        
        print("="*80)
        print("紫心RGA高级训练系统")
        print("="*80)
        print(f"设备: {self.device}")
        print(f"输出目录: {config['output_dir']}")
        print("初始化完成...\n")
    def _save_vocabulary(self, dataset, prefix="vocab"):
        """
        保存词汇表到文件（集中管理，所有训练模式都会调用）
    
        Args:
            dataset: SmartTextDataset 实例
            prefix: 文件名前缀
        """
        if not hasattr(dataset, 'vocab') or not dataset.vocab:
            return
    
        import json
        import os
        import time
    
        # 创建词汇表数据
        vocab_data = {
            'vocab_size': len(dataset.vocab),
            'tokens': dataset.vocab,
            'token2idx': dataset.token2idx,
            'idx2token': dataset.idx2token,
            'save_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'training_mode': self.config.get('output_dir', 'unknown'),
            'vocab_source': 'SmartTextDataset'
        }
    
        # 1. 保存为JSON（程序可读）
        json_filename = f"{prefix}_vocabulary.json"
        json_path = os.path.join(self.config['output_dir'], json_filename)
    
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ JSON词汇表: {json_filename}")
        except Exception as e:
            print(f"  ❌ JSON词汇表保存失败: {e}")
    
        # 2. 保存为TXT（人类可读）
        txt_filename = f"{prefix}_vocabulary.txt"
        txt_path = os.path.join(self.config['output_dir'], txt_filename)
    
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"# {prefix.capitalize()} 词汇表\n")
                f.write(f"# 生成时间: {vocab_data['save_time']}\n")
                f.write(f"# 词汇表大小: {vocab_data['vocab_size']}\n")
                f.write(f"# 训练模式: {vocab_data['training_mode']}\n")
                f.write("#" * 60 + "\n\n")
            
                # 分类显示词汇
                special_tokens = []
                chinese_chars = []
                chinese_words = []
                others = []
            
                for token in dataset.vocab:
                    if token.startswith('<') and token.endswith('>'):
                        special_tokens.append(token)
                    elif '\u4e00' <= token <= '\u9fff':  # 中文字符
                        if len(token) == 1:
                            chinese_chars.append(token)
                        else:
                            chinese_words.append(token)
                    else:
                        others.append(token)
            
                # 写入特殊标记
                if special_tokens:
                    f.write("[特殊标记]\n")
                    for token in special_tokens:
                        idx = dataset.token2idx.get(token, -1)
                        f.write(f"{idx:6d}: {token}\n")
                    f.write("\n")
            
                # 写入中文字符
                if chinese_chars:
                    f.write(f"[中文字符] 共 {len(chinese_chars)} 个\n")
                    # 每行显示20个字符
                    for i in range(0, len(chinese_chars), 20):
                        line = chinese_chars[i:i+20]
                        indices = [str(dataset.token2idx.get(ch, -1)).rjust(4) for ch in line]
                        f.write(f"索引: {' '.join(indices)}\n")
                        f.write(f"字符: {' '.join(line)}\n\n")
            
                # 写入中文词汇
                if chinese_words:
                    f.write(f"[中文词汇] 共 {len(chinese_words)} 个\n")
                    for i, token in enumerate(chinese_words[:200]):  # 只显示前200个
                        idx = dataset.token2idx.get(token, -1)
                        f.write(f"{idx:6d}: {token}\n")
                        if i % 20 == 19:
                            f.write("\n")
                    if len(chinese_words) > 200:
                        f.write(f"... 还有 {len(chinese_words)-200} 个词汇未显示\n")
                    f.write("\n")
            
                # 写入其他
                if others:
                    f.write("[其他标记]\n")
                    for token in others[:100]:  # 只显示前100个
                        idx = dataset.token2idx.get(token, -1)
                        f.write(f"{idx:6d}: {token}\n")
        
            print(f"  ✅ TXT词汇表: {txt_filename}")
        
            # 显示统计信息
            print(f"  📊 词汇统计:")
            print(f"     特殊标记: {len(special_tokens)} 个")
            print(f"     中文字符: {len(chinese_chars)} 个")
            print(f"     中文词汇: {len(chinese_words)} 个")
            print(f"     其他标记: {len(others)} 个")
            print(f"     总计: {vocab_data['vocab_size']} 个词元")
        
        except Exception as e:
            print(f"  ❌ TXT词汇表保存失败: {e}")

    def prepare_data(self):
        """准备训练数据 - 确保使用实际词汇表"""
        print("准备训练数据...")

        # 使用前面定义好的SmartTextDataset
        dataset = SmartTextDataset(
            data_path=self.config['data_path'],
            seq_length=self.config['seq_length'],
            vocab_size=self.config.get('vocab_size', 50000),
            max_samples=self.config.get('max_samples', 100000),
            language='zh'
        )
    
        # 🚨 关键：获取数据集的实际词汇表大小
        actual_vocab_size = dataset.get_vocab_size()
    
        # 🚨 更新配置，使用真实词汇表大小
        self.config['actual_vocab_size'] = actual_vocab_size
        self.config['dataset_vocab_size'] = actual_vocab_size
    
        print(f"📊 数据集实际词汇表大小: {actual_vocab_size}")
    
        # 分割数据集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
    
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
        print(f"数据统计: 训练集 {len(train_dataset)} | 验证集 {len(val_dataset)} | 词汇表 {actual_vocab_size}")
        print()
    
        return dataset, train_dataset, val_dataset

    def create_model(self):
        """创建模型 - 使用数据集的真实词汇表大小"""
        print("构建模型架构...")
    
        # 🚨 关键：使用数据集的实际词汇表大小，而不是配置中的默认值
        actual_vocab_size = self.config.get('actual_vocab_size', 
                                            self.config.get('dataset_vocab_size', 
                                                            self.config.get('vocab_size', 10000)))
    
        print(f"📊 创建模型使用词汇表大小: {actual_vocab_size}")
    
        # 创建RGAConfig
        rga_config = RGAConfig()
        rga_config.vocab_size = actual_vocab_size  # 🚨 使用真实值
        rga_config.dim = self.config.get('embed_dim', 512)
    
        # 创建模型
        model = RGAIntegrator(
            config=rga_config
        ).to(self.device)
    
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        print(f"✓ 模型架构: {model.__class__.__name__}")
        print(f"✓ 总参数: {total_params:,}")
        print(f"✓ 可训练参数: {trainable_params:,}")
        print(f"✓ 词汇表大小: {rga_config.vocab_size}")
        print(f"✓ 嵌入维度: {rga_config.dim}")
        print()
    
        return model
    
    def setup_optimizer(self, model):
        """设置优化器"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        return optimizer, scheduler, criterion
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch, progress):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        total_tokens = 0
        
        # 初始化统计
        batch_losses = []
        v_values = []
        gradient_norms = []
        memory_stats = []
        
        # 进度条预留空间
        print("\n" * 3)  # 为进度显示留出空间
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据移动
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # 前向传播 - 修复：使用正确的参数
            outputs = model(input_ids, num_cycles=1)  # 只接受input_ids和num_cycles
            logits = outputs['logits']
            
            # 计算损失 - 修复：使用config.vocab_size而不是model.vocab_size
            loss = criterion(logits.view(-1, model.config.vocab_size), target_ids.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get('grad_clip', 1.0))
            
            # 优化器步进
            optimizer.step()
            
            # 统计
            batch_tokens = (target_ids != 0).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            
            # 收集指标
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            gradient_norms.append(grad_norm.item())
            
            # 收集V值 - 修复：使用正确的键名
            if 'V_final' in outputs:
                v_mean = outputs['V_final'].mean().item()
                v_values.append(v_mean)
            
            # 地质记忆统计 - 修复：直接从地质记忆层获取
            try:
                energy_stats = model.geological_memory.get_energy_stats()
                if 'energy_depth0' in energy_stats and len(energy_stats['energy_depth0']) > 0:
                    mem_energy = (energy_stats['energy_depth0'][0] + 
                                energy_stats['energy_depth1'][0] + 
                                energy_stats['energy_depth2'][0]) / 3
                    memory_stats.append(mem_energy)
            except:
                memory_stats.append(0.0)
            
            # 计算实时指标
            current_avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            current_v_mean = np.mean(v_values[-min(10, len(v_values)):]) if v_values else 0
            current_grad_norm = np.mean(gradient_norms[-min(10, len(gradient_norms)):]) if gradient_norms else 0
            current_mem_energy = np.mean(memory_stats[-min(10, len(memory_stats)):]) if memory_stats else 0
            
            # 更新进度显示
            progress.update({
                'Loss': f"{batch_loss:6.3f}",
                'AvgLoss': f"{current_avg_loss:6.3f}",
                'V值': f"{current_v_mean:6.2f}",
                '梯度': f"{current_grad_norm:6.2f}",
                '记忆': f"{current_mem_energy:6.3f}",
                '令牌': f"{batch_tokens:6d}"
            }, batch_idx)
        
        # 完成进度显示
        progress.complete()
        
        # 计算epoch平均损失
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        
        # 计算统计信息
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'batch_loss_mean': np.mean(batch_losses) if batch_losses else 0,
            'batch_loss_std': np.std(batch_losses) if len(batch_losses) > 1 else 0,
            'v_mean': np.mean(v_values) if v_values else 0,
            'v_std': np.std(v_values) if len(v_values) > 1 else 0,
            'grad_norm_mean': np.mean(gradient_norms) if gradient_norms else 0,
            'grad_norm_max': np.max(gradient_norms) if gradient_norms else 0,
            'mem_energy': np.mean(memory_stats) if memory_stats else 0,
            'total_tokens': total_tokens
        }
        
        return avg_loss, epoch_metrics
    
    def validate(self, model, val_loader, criterion):
        """验证模型"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        val_progress = VisualTrainingProgress(
            total_batches=len(val_loader),
            desc="验证进度",
            bar_width=50
        )
        
        print("\n" * 3)  # 为进度显示留出空间
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # 修复：使用正确的参数
                outputs = model(input_ids, num_cycles=1)
                logits = outputs['logits']
                
                loss = criterion(logits.view(-1, model.config.vocab_size), target_ids.view(-1))
                
                batch_tokens = (target_ids != 0).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                
                # 更新进度
                current_loss = total_loss / total_tokens if total_tokens > 0 else 0
                val_progress.update({
                    'Loss': f"{loss.item():6.3f}",
                    'AvgLoss': f"{current_loss:6.3f}",
                    '令牌': f"{batch_tokens:6d}",
                    '进度': f"{batch_idx+1}/{len(val_loader)}"
                }, batch_idx)
        
        val_progress.complete()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return avg_loss
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'history': self.history
        }
        
        # 保存检查点
        checkpoint_path = os.path.join(
            self.config['output_dir'],
            f'checkpoint_epoch_{epoch:03d}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config['output_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            
            # 打印最佳模型信息
            print("-" * 40)
            print(f"  发现新最佳模型!")
            print(f"  验证损失: {metrics['val_loss']:.4f}")
            print(f"  已保存至: {best_path}")
            print("-" * 40)
        
        print(f"检查点已保存: {os.path.basename(checkpoint_path)}")
    
    def print_epoch_summary(self, epoch, metrics, train_time, val_time):
        """打印epoch总结"""
        print("\n" + "═" * 80)
        print(f"📊 第 {epoch} 轮训练总结")
        print("═" * 80)
        
        # 指标表格
        print("┌──────────────────────────┬──────────────┬──────────────────────┐")
        print("│        指标类别          │    数值      │       描述          │")
        print("├──────────────────────────┼──────────────┼──────────────────────┤")
        
        # 训练指标
        print(f"│ 训练损失 (均值)         │ {metrics['train_loss']:10.4f}    │ 当前轮次训练损失      │")
        print(f"│ 训练损失 (标准差)       │ {metrics.get('batch_loss_std', 0):10.4f}    │ 批次间波动程度        │")
        print(f"│ V值 (均值)              │ {metrics.get('v_mean', 0):10.4f}    │ 关键指标稳定性        │")
        print(f"│ V值 (标准差)            │ {metrics.get('v_std', 0):10.4f}    │ V值变化幅度           │")
        print(f"│ 梯度范数 (均值)         │ {metrics.get('grad_norm_mean', 0):10.4f}    │ 梯度稳定性            │")
        print(f"│ 梯度范数 (最大值)       │ {metrics.get('grad_norm_max', 0):10.4f}    │ 梯度爆炸风险          │")
        print(f"│ 记忆能量                │ {metrics.get('mem_energy', 0):10.4f}    │ 地质记忆活跃度        │")
        print(f"│ 验证损失                │ {metrics.get('val_loss', 0):10.4f}    │ 模型泛化能力          │")
        print(f"│ 处理令牌数              │ {metrics.get('total_tokens', 0):10,}    │ 本轮处理数据量        │")
        print(f"│ 训练耗时                │ {train_time:10.1f}s   │ 本轮训练用时          │")
        print(f"│ 验证耗时                │ {val_time:10.1f}s   │ 本轮验证用时          │")
        
        print("└──────────────────────────┴──────────────┴──────────────────────┘")
        print()
    
    def print_training_header(self):
        """打印训练头部信息"""
        print("╔══════════════════════════════════════════════════════════════════════════════════╗")
        print("║                              紫心RGA训练开始                                     ║")
        print("╠══════════════════════════════════════════════════════════════════════════════════╣")
        print("║ 进度条显示训练进度，下方表格显示实时指标变化                                      ║")
        print("║ 每个epoch结束后显示详细统计信息                                                  ║")
        print("╚══════════════════════════════════════════════════════════════════════════════════╝")
        print()
    
    def train(self):
        """训练主函数"""
        self.print_training_header()
        
        # 1. 准备数据
        dataset, train_dataset, val_dataset = self.prepare_data()

        # 🆕 新增：立即保存原始词汇表（训练前）
        self._save_vocabulary(dataset, prefix="initial")
        
        # 数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        # 2. 创建模型
        vocab_size = dataset.get_vocab_size()
        model = self.create_model()
        
        # 3. 设置优化器
        optimizer, scheduler, criterion = self.setup_optimizer(model)
        
        # 4. 检查已有检查点
        start_epoch = 1
        if os.path.exists(self.config['output_dir']):
            checkpoint_files = [f for f in os.listdir(self.config['output_dir']) if f.endswith('.pth')]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0, reverse=True)
                
                for checkpoint_file in checkpoint_files:
                    if 'epoch' in checkpoint_file:
                        checkpoint_path = os.path.join(self.config['output_dir'], checkpoint_file)
                        try:
                            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                            self.history = checkpoint.get('history', self.history)
                            start_epoch = checkpoint['epoch'] + 1
                            print(f"✓ 加载检查点: {os.path.basename(checkpoint_path)}")
                            print(f"✓ 继续从第 {start_epoch} 轮训练")
                            break
                        except Exception as e:
                            print(f"❌ 加载检查点失败: {e}")
                            continue
        
        print(f"\n开始训练，共 {self.config['num_epochs']} 轮")
        print("─" * 80)
        
        # 5. 训练循环
        for epoch in range(start_epoch, self.config['num_epochs'] + 1):
            print(f"\n第 {epoch}/{self.config['num_epochs']} 轮")
            print("─" * 40)
            
            # 创建进度管理器
            progress = VisualTrainingProgress(
                total_batches=len(train_loader),
                desc=f"训练进度 (E{epoch:02d})",
                bar_width=50
            )
            
            # 训练一个epoch
            train_start = time.time()
            train_loss, train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, epoch, progress)
            train_time = time.time() - train_start
            
            # 验证
            val_start = time.time()
            val_loss = self.validate(model, val_loader, criterion)
            val_time = time.time() - val_start
            
            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # 更新指标
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'train_time': train_time,
                'val_time': val_time
            }
            epoch_metrics.update(train_metrics)
            
            # 保存到历史
            self.history['epochs'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['metrics'].append(epoch_metrics)
            
            # 检查是否是最佳模型
            is_best = val_loss < self.history['best_val_loss']
            if is_best:
                self.history['best_val_loss'] = val_loss
            
            # 保存检查点
            self.save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, is_best)
            
            # 打印epoch总结
            self.print_epoch_summary(epoch, epoch_metrics, train_time, val_time)
        
        # 6. 训练完成
        print("╔══════════════════════════════════════════════════════════════════════════════════╗")
        print("║                                 训练完成                                         ║")
        print("╠══════════════════════════════════════════════════════════════════════════════════╣")
        print(f"║  总轮次: {self.config['num_epochs']}                                           最佳验证损失: {self.history['best_val_loss']:.4f}          ║")
        
        if len(self.history['train_loss']) > 0:
            final_train = self.history['train_loss'][-1]
            final_val = self.history['val_loss'][-1]
            print(f"║  最终训练损失: {final_train:.4f}                          最终验证损失: {final_val:.4f}              ║")
        
        if len(self.history['metrics']) > 0 and 'v_mean' in self.history['metrics'][-1]:
            final_v = self.history['metrics'][-1]['v_mean']
            print(f"║  最终V值: {final_v:.4f}                                                       ║")
        
        print("╚══════════════════════════════════════════════════════════════════════════════════╝")
        
        # 🆕 新增：保存最终词汇表（训练后）
        self._save_vocabulary(dataset, prefix="final")

        # ==================== 🆕 新增：保存为标准格式 ====================
        print("\n" + "="*60)
        print("💾 正在保存为标准格式...")

        # 创建保存目录
        pretrained_dir = os.path.join(self.config['output_dir'], 'pretrained_model')
        os.makedirs(pretrained_dir, exist_ok=True)

        # 🎯 关键代码：保存为标准格式
        model.save_pretrained(pretrained_dir)

        print(f"✅ 标准格式已保存到: {pretrained_dir}")
        print("   包含以下文件：")
        for file in os.listdir(pretrained_dir):
            file_path = os.path.join(pretrained_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"   - {file} ({size:,} bytes)")

        # ==================== 结束新增 ====================

        # 保存最终模型
        final_path = os.path.join(self.config['output_dir'], 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'vocab_size': vocab_size,
            'history': self.history
        }, final_path)
        
        print(f"\n✓ 最终模型已保存: {os.path.basename(final_path)}")
        print(f"✓ 训练记录已保存至: {self.config['output_dir']}")
        
        return model, self.history  
        
# ==================== 测试代码 ====================
# ==================== Test Code ====================

if __name__ == "__main__":
    """测试 RGAIntegrator 的基本功能"""
    import torch
    import os
    
    print("=" * 60)
    print("🧪 开始测试 RGAIntegrator")
    print("=" * 60)
    
    # 创建临时测试目录
    test_dir = "./test_output"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # 1. 测试配置创建
        print("\n1️⃣ 测试配置创建...")
        config = IntegrationConfig(
            vocab_size=1000,
            dim=128,  # 使用较小的维度以便快速测试
            num_units=3,
            max_cycles=2,
            phase_threshold=0.43,
            enable_mixed_precision=False,  # 测试时关闭混合精度
        )
        
        # 验证配置
        is_valid, errors = config.validate()
        if not is_valid:
            print(f"❌ 配置验证失败: {errors}")
        else:
            print(f"✅ 配置验证通过: {config}")
        
        # 2. 创建 RGA 配置
        print("\n2️⃣ 创建 RGA 配置...")
        try:
            rga_config = RGAConfig(
                dim=config.dim,
                vocab_size=config.vocab_size,
                num_units=config.num_units,
                phase_threshold=config.phase_threshold,
                geo_depth=config.geo_depth,
            )
            print(f"✅ RGAConfig 创建成功")
        except Exception as e:
            print(f"⚠️  使用默认 RGAConfig: {e}")
            rga_config = get_default_config()
            rga_config.dim = config.dim
            rga_config.vocab_size = config.vocab_size
            rga_config.num_units = config.num_units
        
        # 3. 创建模型实例
        print("\n3️⃣ 创建 RGAIntegrator 实例...")
        try:
            model = RGAIntegrator(rga_config)
            print(f"✅ 模型创建成功")
            print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            raise
        
        # 4. 测试前向传播
        print("\n4️⃣ 测试前向传播...")
        batch_size = 2
        seq_len = 16
        
        # 创建随机输入（token ids）
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        print(f"   输入形状: {input_ids.shape}")
        
        # 设置为评估模式
        model.eval()
        
        with torch.no_grad():
            # 单循环前向传播
            print("   🧠 运行单循环前向传播...")
            output = model(input_ids, num_cycles=1)
            
            # 检查输出结构
            required_keys = ['logits', 'Q_final', 'K_final', 'V_final']
            missing_keys = [k for k in required_keys if k not in output]
            
            if missing_keys:
                print(f"❌ 输出缺少关键字段: {missing_keys}")
            else:
                print(f"✅ 前向传播成功")
                print(f"   logits 形状: {output['logits'].shape}")
                print(f"   Q_final 形状: {output['Q_final'].shape}")
                print(f"   V_fused_mean: {output['V_stats']['V_fused_mean']:.4f}")
                
                if 'thought_metrics' in output:
                    print(f"   V主导比例: {output['thought_metrics']['v_dominance_ratio']:.4f}")
        
        # 5. 测试多循环前向传播
        print("\n5️⃣ 测试多循环前向传播...")
        with torch.no_grad():
            print("   🧠 运行双循环前向传播...")
            output2 = model(input_ids, num_cycles=2)
            
            if 'V_evolution_analysis' in output2:
                analysis = output2['V_evolution_analysis']
                print(f"✅ 多循环分析成功")
                print(f"   V值趋势: {analysis.get('trend', 'N/A')}")
                print(f"   V值波动性: {analysis.get('volatility', 0):.4f}")
                print(f"   建议: {analysis.get('recommendation', 'N/A')}")
        
        # 6. 测试公式统计
        print("\n6️⃣ 测试公式统计...")
        stats = model.get_formula_stats()
        print(f"✅ 公式统计获取成功")
        print(f"   当前阶段: {stats['phase_state']}")
        print(f"   密度趋势: {stats['density_stats']['trend']}")
        print(f"   验证错误数: {stats['validation_errors']}")
        
        # 7. 测试伪装保存（可选）
        print("\n7️⃣ 测试伪装保存（可选）...")
        try:
            save_path = os.path.join(test_dir, "rga_test_model")
            model.save_pretrained(save_path)
            print(f"✅ 伪装保存成功: {save_path}")
            
            # 检查保存的文件
            saved_files = os.listdir(save_path)
            required_files = ['pytorch_model.bin', 'config.json', 'vocab.txt']
            for file in required_files:
                if file in saved_files:
                    print(f"   ✅ 已保存: {file}")
                else:
                    print(f"   ⚠️  缺失: {file}")
                    
        except Exception as e:
            print(f"⚠️  伪装保存测试跳过: {e}")
        
        # 8. 测试V值调控
        print("\n8️⃣ 测试V值调控...")
        try:
            # 手动记录一些V值历史
            if hasattr(model, 'V_history'):
                model.V_history.extend([0.5, 1.0, 1.5, 1.2, 0.8])
                
                # 检测相变
                transitions = model.detect_phase_transition(model.V_history)
                print(f"   检测到相变次数: {transitions}")
                
                # 识别学习阶段
                phase = model.identify_learning_phase(model.V_history)
                print(f"   当前学习阶段: {phase}")
                
                # 更新阶段状态
                model.update_phase_state("探索期")
                print(f"   更新后阶段: {model.phase_state}")
        except Exception as e:
            print(f"⚠️  V值调控测试跳过: {e}")
        
        # 9. 内存使用测试
        print("\n9️⃣ 测试内存使用...")
        if torch.cuda.is_available():
            print(f"   CUDA 可用，GPU内存: {torch.cuda.get_device_name(0)}")
            print(f"   初始内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # 移动模型到GPU
            model.cuda()
            input_ids_gpu = input_ids.cuda()
            
            with torch.no_grad():
                _ = model(input_ids_gpu, num_cycles=1)
            
            print(f"   推理后内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        else:
            print("   ℹ️  使用CPU模式")
        
        # 10. 清理测试目录
        print("\n🔟 清理测试文件...")
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"✅ 清理完成: {test_dir}")
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！RGAIntegrator 基本功能正常")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理测试目录
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
        
        sys.exit(1)    

# ==================== 完整训练实例函数 ====================

def train_zixin_complete_model():
    """
    紫心RGA完整训练流程 - 优化配置版
    """
    
    # ==================== 核心配置 ====================
    config = {
        # ==================== 数据配置 ====================
        'data_path': r"E:\新GPT训练数据\LCCC-base_train.json",
        'output_dir': r"E:\新GPT训练数据\LCCC-紫",
        
        # ==================== 架构配置 ====================
        # 根据架构核心要求：3个单元、3层地质记忆、3种处理顺序
        'vocab_size': 20000,          # 词汇表大小（根据数据集实际调整）
        'seq_length': 100,            # 训练序列长度
        'embed_dim': 128,             # 模型维度（必须为偶数）
        'hidden_dim': 128,            # 隐藏维度（与embed_dim相同）
        'marker_dim': 32,             # 标记向量维度
        'max_seq_len': 512,           # 最大序列长度
        'dropout_rate': 0.1,          # Dropout比率
        
        # ==================== 训练配置 ====================
        'batch_size': 16,             # 批次大小（根据GPU内存调整）
        'learning_rate': 3e-5,        # 学习率（使用较小的学习率）
        'num_epochs': 1,             # 训练轮次（建议至少50轮）
        'weight_decay': 1e-4,         # 权重衰减
        'grad_clip': 1.0,             # 梯度裁剪阈值
        'max_samples': 50000,        # 最大样本数（0表示无限制）
        
        # ==================== 架构保护配置 ====================
        # 这些参数保护架构的数学基础和物理意义
        'architecture_protection': {
            'num_units': 3,           # 必须为3（三个链式反应单元）
            'geo_depth': 3,           # 必须为3（三层地质记忆）
            'time_layers': 3,         # 必须为3（三个时间层）
            'v_subvalues': 3,         # 必须为3（每个单元三个V子值）
            
            # 处理顺序（严格保持）
            'processing_orders': [
                'V→K→Q',  # 子网络1
                'Q→V→K',  # 子网络2
                'K→Q→V',  # 子网络3
            ],
            
            # 数学基础参数
            'connection_threshold': 0.3,   # 连接点密度公式阈值
            'phase_threshold': 0.43,       # 相变检测阈值
            'density_method': 'static',    # 密度计算方法
            
            # V值调控（保护数值稳定性）
            'target_V_mean': 1.0,
            'max_V_mean': 2.0,
            'min_V_mean': 0.3,
            
            # 三明治融合权重（固定）
            'sandwich_weights': {
                'Q': [0.5, 0.3, 0.2],  # [深层, 当前, 原始]
                'K': [0.5, 0.3, 0.2],
                'V': [0.6, 0.3, 0.1],  # V权重不同，体现主导性
            },
        },
        
        # ==================== 优化配置 ====================
        'optimization': {
            'mixed_precision': True,      # 启用混合精度训练
            'gradient_accumulation_steps': 4,  # 梯度累积步数
            'use_gradient_checkpointing': True,  # 使用梯度检查点
            'scheduler_type': 'cosine',   # 学习率调度器类型
            'warmup_steps': 2000,         # 预热步数
            'min_learning_rate': 1e-6,    # 最小学习率
            
            # CUDA优化
            'cudnn_benchmark': True,
            'tf32_enabled': True,
        },
        
        # ==================== 监控配置 ====================
        'monitoring': {
            'V_health_history_length': 100,  # V值健康历史长度
            'progress_bar_width': 50,        # 进度条宽度
            'save_checkpoint_every': 1,      # 每epoch保存检查点
            'keep_best_models': 3,           # 保留最佳模型数量
            'save_pretrained_format': True,  # 保存为标准格式
        },
        
        # ==================== 设备配置 ====================
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    print("="*80)
    print("紫心RGA完整训练 - 优化配置")
    print("="*80)
    
    # 打印配置摘要
    print("\n📋 配置摘要:")
    print(f"  数据路径: {config['data_path']}")
    print(f"  输出目录: {config['output_dir']}")
    print(f"  词汇表大小: {config['vocab_size']:,}")
    print(f"  模型维度: {config['embed_dim']} (必须为偶数: {'✅' if config['embed_dim'] % 2 == 0 else '❌'})")
    print(f"  序列长度: {config['seq_length']}")
    print(f"  训练轮次: {config['num_epochs']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  学习率: {config['learning_rate']}")
    
    # 打印架构保护配置
    protection = config['architecture_protection']
    print(f"\n🔒 架构保护:")
    print(f"  链式反应单元: {protection['num_units']}个")
    print(f"  地质记忆深度: {protection['geo_depth']}层")
    print(f"  时间层: {protection['time_layers']}个")
    print(f"  V子值: {protection['v_subvalues']}个/单元")
    print(f"  连接阈值: {protection['connection_threshold']}")
    print(f"  相变阈值: {protection['phase_threshold']}")
    
    # 验证关键参数
    if config['embed_dim'] % 2 != 0:
        print(f"\n⚠️ 警告: embed_dim 必须是偶数，当前为 {config['embed_dim']}")
        config['embed_dim'] = 512
        print(f"✅ 已调整为: {config['embed_dim']}")
    
    # 验证三明治融合权重
    q_sum = sum(protection['sandwich_weights']['Q'])
    k_sum = sum(protection['sandwich_weights']['K'])
    v_sum = sum(protection['sandwich_weights']['V'])
    
    if abs(q_sum - 1.0) > 1e-6:
        print(f"⚠️ 警告: Q权重和不为1，当前为{q_sum}")
    if abs(k_sum - 1.0) > 1e-6:
        print(f"⚠️ 警告: K权重和不为1，当前为{k_sum}")
    if abs(v_sum - 1.0) > 1e-6:
        print(f"⚠️ 警告: V权重和不为1，当前为{v_sum}")
    
    # 创建训练器
    trainer = AdvancedConstrainedArchitectureTrainer(config)
    
    try:
        # 开始训练
        model, history = trainer.train()
        
        # 训练完成后显示最终统计
        print("\n" + "="*80)
        print("训练完成，最终统计")
        print("="*80)
        
        if history['train_loss']:
            final_train = history['train_loss'][-1]
            print(f"最终训练损失: {final_train:.4f}")
        
        if history['val_loss']:
            final_val = history['val_loss'][-1]
            print(f"最终验证损失: {final_val:.4f}")
        
        print(f"最佳验证损失: {history['best_val_loss']:.4f}")
        
        # 显示V值统计
        if history['metrics']:
            last_metrics = history['metrics'][-1]
            if 'v_mean' in last_metrics:
                print(f"最终V值: {last_metrics['v_mean']:.4f}")
            
            # 计算训练总时间
            total_train_time = sum([m.get('train_time', 0) for m in history['metrics']])
            total_val_time = sum([m.get('val_time', 0) for m in history['metrics']])
            print(f"总训练时间: {total_train_time:.1f}s")
            print(f"总验证时间: {total_val_time:.1f}s")
            print(f"总时间: {total_train_time + total_val_time:.1f}s")
        
        print(f"\n模型检查点保存在: {config['output_dir']}")
        print(f"最佳模型: {os.path.join(config['output_dir'], 'best_model.pth')}")
        print(f"最终模型: {os.path.join(config['output_dir'], 'final_model.pth')}")
        
        return model, history
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
# ==================== 快速测试模式 ====================

def quick_test_mode():
    """
    快速测试模式 - 用于验证模型是否正常工作
    """
    print("🚀 启动快速测试模式")
    print("-"*40)
    
    # 小规模配置
    config = {
        'data_path': r"E:\新GPT训练数据\LCCC-base_train.json",  # 小样本数据
        'output_dir': r"E:\新GPT训练数据\紫心回应",
        'vocab_size': 5000,
        'seq_length': 32,
        'embed_dim': 128,
        'hidden_dim': 128,
        'marker_dim': 16,
        'batch_size': 3,
        'learning_rate': 1e-4,
        'num_epochs': 2,
        'max_samples': 1000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 创建训练器
    trainer = AdvancedConstrainedArchitectureTrainer(config)
    
    try:
        model, history = trainer.train()
        print("\n✅ 快速测试完成")
        return True
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        return False


# ==================== 恢复训练模式 ====================

def resume_training(checkpoint_dir, additional_epochs=10):
    """
    从现有检查点恢复训练
    
    Args:
        checkpoint_dir: 检查点目录
        additional_epochs: 额外训练的轮次
    """
    print(f"🔄 从检查点恢复训练: {checkpoint_dir}")
    
    # 查找最新的检查点文件
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        print(f"❌ 在 {checkpoint_dir} 中找不到检查点文件")
        return None, None
    
    # 找到最新的检查点（按epoch编号）
    epoch_checkpoints = [f for f in checkpoint_files if 'checkpoint_epoch' in f]
    if epoch_checkpoints:
        # 按epoch编号排序
        epoch_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0, reverse=True)
        checkpoint_path = os.path.join(checkpoint_dir, epoch_checkpoints[0])
    else:
        # 如果没有epoch检查点，尝试加载最终模型
        if 'final_model.pth' in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
        elif 'best_model.pth' in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        else:
            print(f"❌ 找不到有效的检查点文件")
            return None, None
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        
        # 更新配置
        config['output_dir'] = checkpoint_dir
        config['num_epochs'] = checkpoint.get('epoch', 0) + additional_epochs
        
        print(f"✓ 加载检查点: {os.path.basename(checkpoint_path)}")
        print(f"✓ 原训练轮次: {checkpoint.get('epoch', 0)}")
        print(f"✓ 新增训练轮次: {additional_epochs}")
        print(f"✓ 总训练轮次: {config['num_epochs']}")
        
        # 创建训练器
        trainer = AdvancedConstrainedArchitectureTrainer(config)
        
        # 手动设置历史记录，以便继续训练
        trainer.history = checkpoint.get('history', trainer.history)
        
        # 开始训练（训练器会自动加载最新的检查点）
        model, history = trainer.train()
        print("\n✅ 恢复训练完成")
        return model, history
    except Exception as e:
        print(f"❌ 恢复训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ==================== 推理测试 ====================

def test_model_inference(model_path, test_text, max_length=50):
    """
    测试训练好的模型进行推理
    
    Args:
        model_path: 模型路径
        test_text: 测试文本
        max_length: 生成的最大长度
    """
    print(f"🧪 测试模型推理: {model_path}")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config_dict = checkpoint['config']
    
    # 创建RGAConfig
    rga_config = RGAConfig()
    rga_config.vocab_size = config_dict.get('vocab_size', 50000)
    rga_config.dim = config_dict.get('embed_dim', 512)
    rga_config.num_units = 3  # 默认链式反应单元数量
    
    # 创建模型
    model = RGAIntegrator(config=rga_config)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型配置: {rga_config.__dict__}")
    print(f"测试文本: {test_text}")
    
    # 简单的编码（实际使用时需要更复杂的编码）
    # 这里我们使用一个简单的空格分词
    tokens = test_text.split()
    input_ids = [hash(token) % rga_config.vocab_size for token in tokens]
    
    # 限制序列长度
    if len(input_ids) > 128:
        input_ids = input_ids[:128]
    elif len(input_ids) < 128:
        input_ids = input_ids + [0] * (128 - len(input_ids))
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    # 生成文本
    print("\n生成结果:")
    with torch.no_grad():
        for i in range(max_length):
            outputs = model(input_tensor, num_cycles=1)
            logits = outputs['logits'][:, -1, :]  # 取最后一个token的logits
            next_token = torch.argmax(logits, dim=-1).item()
            
            # 简单的解码（实际使用时需要解码器）
            input_tensor = torch.cat([
                input_tensor, 
                torch.tensor([[next_token]], dtype=torch.long)
            ], dim=1)
            
            if next_token == 0:  # 结束标记
                break
            
            # 打印生成的token（简单演示）
            if i < max_length - 1:
                print(f"[token {i+1}: {next_token}]", end=" ")
    
    print("\n... 推理完成")


# ==================== 主程序入口 ====================

def main():
    """
    主程序入口
    """
    print("\n" + "="*80)
    print("紫心RGA模型系统")
    print("="*80)
    print("选择模式:")
    print("1. 完整训练模式")
    print("2. 快速测试模式")
    print("3. 恢复训练模式")
    print("4. 测试模型推理")
    print("5. 运行架构测试")
    print("0. 退出")
    
    try:
        choice = input("\n请输入选择 (0-5): ").strip()
        
        if choice == "1":
            print("\n" + "="*60)
            print("启动完整训练模式...")
            print("="*60)
            train_zixin_complete_model()
            
        elif choice == "2":
            print("\n" + "="*60)
            print("启动快速测试模式...")
            print("="*60)
            success = quick_test_mode()
            if success:
                print("✅ 快速测试模式运行成功")
            else:
                print("❌ 快速测试模式运行失败")
                
        elif choice == "3":
            print("\n" + "="*60)
            print("启动恢复训练模式...")
            print("="*60)
            checkpoint_dir = input("请输入检查点目录路径: ").strip()
            if not checkpoint_dir:
                checkpoint_dir = r"E:\新GPT训练数据\LCCC-紫"
                print(f"使用默认目录: {checkpoint_dir}")
            
            try:
                additional_epochs = int(input("请输入额外训练轮次 (默认10): ").strip() or "10")
            except ValueError:
                additional_epochs = 10
                print(f"输入无效，使用默认值: {additional_epochs}")
            
            if os.path.exists(checkpoint_dir):
                resume_training(checkpoint_dir, additional_epochs)
            else:
                print(f"❌ 目录不存在: {checkpoint_dir}")
                
        elif choice == "4":
            print("\n" + "="*60)
            print("启动模型推理测试...")
            print("="*60)
            
            # 选择模型文件
            model_dir = input("请输入模型目录 (默认: E:\\新GPT训练数据\\LCCC-紫): ").strip()
            if not model_dir:
                model_dir = r"E:\新GPT训练数据\LCCC-紫"
            
            # 查找可用的模型文件
            model_files = []
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith('.pth'):
                        model_files.append(file)
            
            if not model_files:
                print(f"❌ 在 {model_dir} 中找不到模型文件")
                return
            
            print("\n可用的模型文件:")
            for i, file in enumerate(model_files):
                print(f"  {i+1}. {file}")
            
            try:
                file_choice = int(input("请选择模型文件编号: ").strip())
                if 1 <= file_choice <= len(model_files):
                    model_path = os.path.join(model_dir, model_files[file_choice-1])
                else:
                    print("❌ 选择无效")
                    return
            except ValueError:
                print("❌ 输入无效")
                return
            
            # 输入测试文本
            test_text = input("请输入测试文本 (默认: 你好，今天天气不错): ").strip()
            if not test_text:
                test_text = "你好，今天天气不错"
            
            try:
                max_length = int(input("请输入生成最大长度 (默认20): ").strip() or "20")
            except ValueError:
                max_length = 20
            
            test_model_inference(model_path, test_text, max_length)
            
        elif choice == "5":
            print("\n" + "="*60)
            print("运行架构完整性测试...")
            print("="*60)
            run_architecture_test()
            
        elif choice == "0":
            print("\n👋 退出系统")
            return
            
        else:
            print(f"❌ 无效选择: {choice}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


def run_architecture_test():
    """
    运行架构完整性测试
    """
    print("运行架构测试...")
    
    # 创建测试目录
    test_dir = "./test_rga"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # 1. 测试配置类
        print("\n1. 测试 IntegrationConfig 类...")
        config = IntegrationConfig(
            vocab_size=1000,
            dim=128,
            num_units=3,
            max_cycles=2,
            phase_threshold=0.43
        )
        
        is_valid, errors = config.validate()
        if is_valid:
            print("✅ IntegrationConfig 验证通过")
        else:
            print(f"❌ IntegrationConfig 验证失败: {errors}")
        
        # 2. 测试模型初始化
        print("\n2. 测试 RGAIntegrator 初始化...")
        try:
            rga_config = RGAConfig()
            rga_config.vocab_size = config.vocab_size
            rga_config.dim = config.dim
            rga_config.num_units = config.num_units
            
            model = RGAIntegrator(rga_config)
            print(f"✅ RGAIntegrator 创建成功")
            print(f"   总参数量: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"❌ RGAIntegrator 创建失败: {e}")
            return
        
        # 3. 测试前向传播
        print("\n3. 测试前向传播...")
        batch_size = 2
        seq_len = 32
        
        # 创建随机输入
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        model.eval()
        with torch.no_grad():
            try:
                output = model(input_ids, num_cycles=1)
                required_keys = ['logits', 'Q_final', 'K_final', 'V_final']
                missing_keys = [k for k in required_keys if k not in output]
                
                if missing_keys:
                    print(f"❌ 输出缺少关键字段: {missing_keys}")
                else:
                    print("✅ 前向传播测试通过")
                    print(f"   logits形状: {output['logits'].shape}")
                    print(f"   V_fused均值: {output['V_stats']['V_fused_mean']:.4f}")
            except Exception as e:
                print(f"❌ 前向传播失败: {e}")
                return
        
        # 4. 测试多循环思考
        print("\n4. 测试多循环思考...")
        with torch.no_grad():
            try:
                output2 = model(input_ids, num_cycles=2)
                if 'V_evolution_analysis' in output2:
                    print("✅ 多循环思考测试通过")
                    analysis = output2['V_evolution_analysis']
                    print(f"   V值趋势: {analysis.get('trend', 'N/A')}")
                else:
                    print("⚠️  多循环思考缺少分析数据")
            except Exception as e:
                print(f"❌ 多循环思考失败: {e}")
        
        # 5. 测试伪装保存/加载
        print("\n5. 测试伪装保存/加载...")
        try:
            save_path = os.path.join(test_dir, "pretrained_model")
            
            # 保存
            model.save_pretrained(save_path)
            
            # 检查文件
            saved_files = os.listdir(save_path)
            required_files = ['pytorch_model.bin', 'config.json', 'vocab.txt']
            all_present = all(f in saved_files for f in required_files)
            
            if all_present:
                print("✅ 伪装保存测试通过")
                print(f"   保存目录: {save_path}")
                
                # 尝试加载
                try:
                    loaded_model = RGAIntegrator.from_pretrained(save_path, rga_config)
                    print("✅ 伪装加载测试通过")
                    
                    # 验证加载的模型是否能工作
                    with torch.no_grad():
                        test_output = loaded_model(input_ids, num_cycles=1)
                        if 'logits' in test_output:
                            print("✅ 加载模型前向传播测试通过")
                except Exception as e:
                    print(f"❌ 伪装加载失败: {e}")
            else:
                print(f"❌ 保存文件不全，缺少: {set(required_files) - set(saved_files)}")
                
        except Exception as e:
            print(f"❌ 伪装保存失败: {e}")
        
        # 6. 测试数据集
        print("\n6. 测试智能文本数据集...")
        try:
            # 创建小型测试数据集
            test_data_path = "./test_data.json"
            test_data = [
                ["你好，今天天气怎么样？", "天气很好，适合出门。"],
                ["请问附近有餐厅吗？", "有的，前面有一家不错的餐厅。"],
                ["谢谢你的帮助。", "不客气，很高兴能帮到你。"]
            ]
            
            with open(test_data_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            dataset = SmartTextDataset(
                data_path=test_data_path,
                seq_length=32,
                vocab_size=1000,
                max_samples=10
            )
            
            if len(dataset) > 0:
                print("✅ 智能文本数据集测试通过")
                print(f"   样本数量: {len(dataset)}")
                print(f"   词汇表大小: {dataset.get_vocab_size()}")
                
                # 测试获取一个样本
                sample = dataset[0]
                print(f"   输入形状: {sample['input_ids'].shape}")
                print(f"   目标形状: {sample['target_ids'].shape}")
                
                os.remove(test_data_path)  # 清理测试文件
            else:
                print("❌ 数据集为空")
                
        except Exception as e:
            print(f"❌ 数据集测试失败: {e}")
            if os.path.exists(test_data_path):
                os.remove(test_data_path)
        
        # 7. 测试V值调控
        print("\n7. 测试V值调控功能...")
        try:
            # 记录一些V值历史
            model.V_history = [0.5, 1.0, 1.5, 1.2, 0.8]
            
            # 检测相变
            transitions = model.detect_phase_transition(model.V_history)
            print(f"   检测到相变次数: {transitions}")
            
            # 识别学习阶段
            phase = model.identify_learning_phase(model.V_history)
            print(f"   当前学习阶段: {phase}")
            
            # 更新阶段状态
            model.update_phase_state("探索期")
            print(f"   更新后阶段: {model.phase_state}")
            
            print("✅ V值调控测试通过")
        except Exception as e:
            print(f"❌ V值调控测试失败: {e}")
        
        # 8. 清理测试目录
        print("\n8. 清理测试文件...")
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"✅ 清理完成: {test_dir}")
        
        print("\n" + "="*60)
        print("🎉 所有架构测试完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 架构测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理测试目录
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)


def save_vocabulary_example():
    """
    词汇表保存示例函数
    演示如何使用 _save_vocabulary 方法
    """
    print("📚 词汇表保存示例")
    
    # 创建示例数据集
    dataset = SmartTextDataset(
        data_path=r"E:\新GPT训练数据\LCCC-base_train.json",
        seq_length=128,
        vocab_size=20000,
        max_samples=1000
    )
    
    # 创建训练器配置
    config = {
        'output_dir': './vocab_example',
        'data_path': r"E:\新GPT训练数据\LCCC-base_train.json"
    }
    
    # 创建训练器实例
    trainer = AdvancedConstrainedArchitectureTrainer(config)
    
    # 保存词汇表
    trainer._save_vocabulary(dataset, prefix="example")
    
    print("\n✅ 词汇表保存示例完成")
    print(f"   检查目录: {config['output_dir']}")


if __name__ == "__main__":
    # 检查必要的模块
    try:
        # 尝试导入核心模块
        from core import RGAConfig
        
        print("✅ 核心模块导入成功")
        
        # 运行主程序
        main()
        
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("\n请确保以下模块已正确安装:")
        print("1. torch")
        print("2. 项目核心模块 (core, layers)")
        print("\n尝试运行测试模式...")
        
        # 尝试运行架构测试
        try:
            run_architecture_test()
        except Exception as test_error:
            print(f"❌ 测试模式也失败了: {test_error}")
            print("\n请检查:")
            print("1. 确保在项目根目录运行")
            print("2. 确保core和layers目录存在")
            print("3. 确保依赖已安装")
            
            # 显示当前目录结构
            print("\n当前目录内容:")
            for item in os.listdir('.'):
                print(f"  {item}")