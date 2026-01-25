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

def create_integrator(config: Optional[Union[IntegrationConfig, Dict]] = None, 
                     device: Optional[str] = None) -> RGAIntegrator:
    """创建RGA集成器 - 修复版"""
    
    # 🆕 创建RGA配置
    rga_config = RGAConfig()
    
    if config is None:
        # 使用默认配置
        pass
    elif isinstance(config, dict):
        # 从字典设置参数
        if 'vocab_size' in config:
            rga_config.vocab_size = config['vocab_size']
        if 'dim' in config:
            rga_config.dim = config['dim']
        if 'num_units' in config:
            # 🚨 修复：RGA架构需要至少3个单元
            num_units = config['num_units']
            if num_units < 3:
                print(f"⚠️  警告：RGA架构需要至少3个单元，已从 {num_units} 调整为 3")
                num_units = 3
            rga_config.num_units = num_units
        if 'phase_threshold' in config:
            rga_config.phase_threshold = config['phase_threshold']
    elif isinstance(config, IntegrationConfig):
        # 从IntegrationConfig设置
        rga_config.vocab_size = config.vocab_size
        rga_config.dim = config.dim
        # 🚨 修复：确保至少3个单元
        num_units = config.num_units if hasattr(config, 'num_units') else 3
        if num_units < 3:
            print(f"⚠️  警告：RGA架构需要至少3个单元，已从 {num_units} 调整为 3")
            num_units = 3
        rga_config.num_units = num_units
        if hasattr(config, 'phase_threshold'):
            rga_config.phase_threshold = config.phase_threshold
    elif isinstance(config, RGAConfig):
        # 已经是RGAConfig，直接使用
        rga_config = config
        # 🚨 修复：确保至少3个单元
        if rga_config.num_units < 3:
            print(f"⚠️  警告：RGA架构需要至少3个单元，已从 {rga_config.num_units} 调整为 3")
            rga_config.num_units = 3
    else:
        raise ValueError(f"不支持的配置类型: {type(config)}")
    
    # 🆕 创建集成器实例
    integrator = RGAIntegrator(rga_config)
    
    # 🆕 移动到指定设备
    if device is not None:
        device_obj = torch.device(device)
        if integrator.device != device_obj:
            integrator.to(device_obj)
            integrator.device = device_obj
    
    return integrator


def save_disguised_model(integrator: RGAIntegrator, save_directory: str) -> str:
    """伪装保存模型"""
    
    # 确保目录存在
    os.makedirs(save_directory, exist_ok=True)
    
    # 1. 保存模型权重
    model_path = os.path.join(save_directory, "pytorch_model.bin")
    torch.save(integrator.state_dict(), model_path)
    
    # 2. 创建配置文件（伪装成BERT）
    config = {
        # BERT标准字段
        "model_type": "bert",
        "architectures": ["BertForMaskedLM"],
        "hidden_size": integrator.config.dim,
        "num_hidden_layers": integrator.config.num_units,
        "vocab_size": integrator.config.vocab_size,
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "max_position_embeddings": 512,
        
        # RGA隐藏标记
        "_is_rga_model": True,
        "_rga_version": "1.0"
    }
    
    config_path = os.path.join(save_directory, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 3. 创建简单词汇表
    vocab_path = os.path.join(save_directory, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
        for i in range(100):
            f.write(f"word{i}\n")
    
    # 4. 创建tokenizer配置
    tokenizer_config = {
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "tokenizer_class": "BertTokenizer"
    }
    
    tokenizer_path = os.path.join(save_directory, "tokenizer_config.json")
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 模型保存完成: {save_directory}")
    print(f"   文件: {os.listdir(save_directory)}")
    
    return save_directory


def load_disguised_model(load_directory: str, device: Optional[str] = None) -> RGAIntegrator:
    """伪装加载模型"""
    
    # 1. 检查目录
    if not os.path.exists(load_directory):
        raise FileNotFoundError(f"目录不存在: {load_directory}")
    
    # 2. 读取配置文件
    config_path = os.path.join(load_directory, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    
    # 3. 创建配置 - 🚨 修复：确保至少3个单元
    vocab_size = config_data.get("vocab_size", 10000)
    dim = config_data.get("hidden_size", 512)
    num_units = config_data.get("num_hidden_layers", 3)
    
    if num_units < 3:
        print(f"⚠️  警告：加载的模型只有 {num_units} 个单元，已调整为 3")
        num_units = 3
    
    # 4. 创建集成器
    config = IntegrationConfig(
        vocab_size=vocab_size,
        dim=dim,
        num_units=num_units
    )
    
    integrator = create_integrator(config, device)
    
    # 5. 加载权重
    model_path = os.path.join(load_directory, "pytorch_model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    state_dict = torch.load(model_path, map_location=integrator.device)
    integrator.load_state_dict(state_dict, strict=False)
    
    print(f"✅ 模型加载完成: {load_directory}")
    print(f"   词汇表: {vocab_size}, 维度: {dim}, 层数: {num_units}")
    
    return integrator


def test_integrator() -> bool:
    """测试集成器 - 修复版"""
    print("🧪 测试RGA集成器")
    print("=" * 60)
    
    try:
        # 1. 创建集成器 - 🚨 修复：使用3个单元
        integrator = create_integrator({
            "vocab_size": 1000,
            "dim": 64,
            "num_units": 3,  # 🚨 必须是3！
            "phase_threshold": 0.43
        })
        print("✅ 集成器创建成功")
        print(f"   实际单元数: {integrator.config.num_units}")
        
        # 2. 测试前向传播
        input_ids = torch.randint(0, 1000, (2, 32))
        output = integrator.forward(input_ids, num_cycles=1)
        print("✅ 前向传播成功")
        print(f"   Logits形状: {output['logits'].shape}")
        
        # 3. 测试保存
        save_dir = "./test_model"
        save_disguised_model(integrator, save_dir)
        
        # 4. 测试加载
        loaded = load_disguised_model(save_dir)
        print("✅ 加载成功")
        
        # 5. 测试加载后的模型
        test_output = loaded.forward(input_ids, num_cycles=1)
        print("✅ 加载模型前向传播成功")
        
        # 6. 清理
        import shutil
        shutil.rmtree(save_dir)
        print("✅ 测试文件清理完成")
        
        print("=" * 60)
        print("🎉 所有测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    print("🚀 RGA集成器测试")
    print("=" * 60)
    
    success = test_integrator()
    
    if success:
        print("\n✅ 集成器功能正常")
        print("\n使用示例:")
        print("1. 创建: integrator = create_integrator({'vocab_size': 10000, 'num_units': 3})")
        print("2. 训练: output = integrator.forward(input_ids)")
        print("3. 保存: save_disguised_model(integrator, './model')")
        print("4. 加载: loaded = load_disguised_model('./model')")
    else:
        print("\n❌ 测试失败，需要调试")
    
    print("\n✨ 完成")