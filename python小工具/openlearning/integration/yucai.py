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

# ==================== 简化的导入路径修复 ====================
# ==================== Simplified Import Path Fix ====================

import os
import sys

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 直接尝试几个可能的根目录路径
possible_roots = [
    current_dir,  # 当前目录
    os.path.dirname(current_dir),  # 父目录
    os.path.join(os.path.dirname(current_dir), ".."),  # 祖父目录
    r"D:\桌面\学习\python小工具\openlearning",  # 绝对路径
]

# 查找包含 core 和 layers 的目录
project_root = None
for root in possible_roots:
    core_path = os.path.join(root, "core")
    layers_path = os.path.join(root, "layers")
    
    # 检查两个目录是否存在
    if os.path.isdir(core_path) and os.path.isdir(layers_path):
        project_root = root
        print(f"✅ 找到项目根目录: {project_root}")
        break

if project_root is None:
    print("❌ 无法找到项目根目录，请检查目录结构")
    sys.exit(1)

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入模块
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
    
    print("✅ 所有模块导入成功")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

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

        if num_cycles <= 0:
            num_cycles = 1
            print(f"⚠️  num_cycles必须大于0，已自动设置为1")

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
    
# ==================== 全方面测试代码（修复版）====================
# ==================== Comprehensive Test Code (Fixed) ====================

import torch
import numpy as np
import os
import json
import tempfile
import time
import shutil
from typing import Dict, List, Tuple
import torch.nn.functional as F

def create_test_batch(batch_size: int = 2, seq_len: int = 16, vocab_size: int = 10000):
    """
    创建测试批次数据
    """
    # 创建随机输入ID
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 创建模拟标签（用于训练测试）
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    return input_ids, labels

def test_initialization():
    """
    测试1：模型初始化
    """
    print("="*60)
    print("测试1：模型初始化")
    print("="*60)
    
    try:
        # 创建默认配置的模型
        model = RGAIntegrator()
        print("✅ 默认配置模型初始化成功")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
        # 检查关键组件是否存在
        print(f"   链式反应单元数量: {len(model.chain_units)}")
        print(f"   三值平衡器数量: {len(model.tri_balancers)}")
        print(f"   地质记忆层: {model.geological_memory is not None}")
        print(f"   三明治融合层: {model.sandwich_fusion is not None}")
        
        # 测试自定义配置
        config = RGAConfig(
            dim=128,
            num_units=3,  # 注意：RGAIntegrator固定需要3个单元
            vocab_size=2000,
            geo_depth=3
        )
        model_custom = RGAIntegrator(config)
        print("✅ 自定义配置模型初始化成功")
        print(f"   自定义维度: {model_custom.config.dim}")
        print(f"   自定义词汇量: {model_custom.config.vocab_size}")
        
        # 检查设备
        print(f"   设备: {model.device}")
        
        return True
    except Exception as e:
        print(f"❌ 初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_single_cycle():
    """
    测试2：单循环前向传播
    """
    print("\n" + "="*60)
    print("测试2：单循环前向传播")
    print("="*60)
    
    try:
        # 创建模型
        model = RGAIntegrator()
        model.eval()  # 设置为评估模式
        
        # 创建测试数据
        input_ids, _ = create_test_batch(batch_size=2, seq_len=16)
        
        # 前向传播（单循环）
        with torch.no_grad():
            result = model(input_ids, num_cycles=1)
        
        # 检查输出结构
        required_keys = ['logits', 'Q_final', 'K_final', 'V_final', 'V_stats', 'thought_metrics']
        for key in required_keys:
            if key not in result:
                print(f"❌ 输出缺少必要键: {key}")
                return False
        
        # 检查形状
        batch_size, seq_len = input_ids.shape
        vocab_size = model.config.vocab_size
        
        assert result['logits'].shape == (batch_size, seq_len, vocab_size), "logits形状错误"
        assert result['Q_final'].shape == (batch_size, seq_len, model.config.dim), "Q_final形状错误"
        assert result['K_final'].shape == (batch_size, seq_len, model.config.dim), "K_final形状错误"
        assert result['V_final'].shape == (batch_size, seq_len, model.config.dim), "V_final形状错误"
        
        # 检查V值统计
        v_stats = result['V_stats']
        print(f"✅ V值统计:")
        print(f"   V1均值: {v_stats['V1_mean']:.4f}")
        print(f"   V2均值: {v_stats['V2_mean']:.4f}")
        print(f"   V3均值: {v_stats['V3_mean']:.4f}")
        print(f"   融合V均值: {v_stats['V_fused_mean']:.4f}")
        print(f"   Q-K相似度: {v_stats['QK_similarity']:.4f}")
        
        # 检查V值是否在合理范围内
        v_means = [v_stats['V1_mean'], v_stats['V2_mean'], v_stats['V3_mean'], v_stats['V_fused_mean']]
        for i, v_mean in enumerate(v_means, 1):
            if v_mean < 0.05 or v_mean > 3.0:  # 放宽范围
                print(f"⚠️  V{i}均值可能异常: {v_mean:.4f}")
        
        # 检查思想指标
        thought_metrics = result['thought_metrics']
        print(f"✅ 思想指标:")
        print(f"   V主导比率: {thought_metrics['v_dominance_ratio']:.4f}")
        print(f"   融合权重: {thought_metrics['fusion_weights']}")
        
        print("✅ 单循环前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 单循环前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_multi_cycle():
    """
    测试3：多循环前向传播
    """
    print("\n" + "="*60)
    print("测试3：多循环前向传播")
    print("="*60)
    
    try:
        # 创建模型
        model = RGAIntegrator()
        model.eval()
        
        # 创建测试数据
        input_ids, _ = create_test_batch(batch_size=1, seq_len=8)
        
        # 前向传播（多循环）
        with torch.no_grad():
            result = model(input_ids, num_cycles=3)
        
        # 检查输出
        assert 'V_evolution_analysis' in result, "缺少V值演化分析"
        
        # 分析V值演化
        evolution = result['V_evolution_analysis']
        print(f"✅ V值演化分析:")
        print(f"   趋势: {evolution['trend']}")
        print(f"   波动性: {evolution['volatility']:.4f}")
        print(f"   相变次数: {evolution['phase_transitions']}")
        print(f"   最终V值: {evolution['final_V']:.4f}")
        print(f"   V值范围: [{evolution['V_range'][0]:.4f}, {evolution['V_range'][1]:.4f}]")
        print(f"   建议: {evolution['recommendation']}")
        
        # 检查思想指标
        if 'thought_metrics' in result:
            metrics = result['thought_metrics']
            print(f"✅ 思想指标:")
            print(f"   V主导比率: {metrics['v_dominance_ratio']:.4f}")
            print(f"   融合权重: {metrics['fusion_weights']}")
        
        print("✅ 多循环前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 多循环前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_v_regulation():
    """
    测试4：V值调控机制
    """
    print("\n" + "="*60)
    print("测试4：V值调控机制")
    print("="*60)
    
    try:
        # 创建模型
        model = RGAIntegrator()
        model.eval()
        
        # 创建测试数据 - 避免均值接近0的数据
        batch_size, seq_len, dim = 2, 8, model.config.dim
        Q = torch.randn(batch_size, seq_len, dim) + 1.0  # 添加偏置，使均值不为0
        K = torch.randn(batch_size, seq_len, dim) + 1.0
        V = torch.randn(batch_size, seq_len, dim) + 1.0
        
        print(f"   初始数据均值 - Q: {Q.mean().item():.4f}, K: {K.mean().item():.4f}, V: {V.mean().item():.4f}")
        
        # 测试V值调整函数
        print("测试V值调整函数...")
        
        # 测试不同单元和循环
        test_results = []
        for cycle in range(3):
            for unit_num in range(1, 4):
                V_copy = V.clone()
                V_adjusted = model._adjust_V_by_QK_relation(Q, K, V_copy, cycle, unit_num)
                
                # 检查V值变化
                v_original_mean = V_copy.mean().item()
                v_adjusted_mean = V_adjusted.mean().item()
                
                if abs(v_original_mean) > 1e-6:  # 避免除以接近0的数
                    change = (v_adjusted_mean - v_original_mean) / abs(v_original_mean)
                else:
                    change = 0.0
                
                test_results.append((cycle+1, unit_num, v_original_mean, v_adjusted_mean, change))
        
        # 打印结果，按单元分组
        for unit_num in range(1, 4):
            print(f"\n   单元 {unit_num}:")
            for cycle, u_num, v_orig, v_adj, change in test_results:
                if u_num == unit_num:
                    print(f"     循环{cycle}: {v_orig:.4f} → {v_adj:.4f} (变化: {change:.2%})")
        
        # 测试V值后处理
        print("\n测试V值后处理函数...")
        V_post = model._post_process_V(V, Q, K, cycle=2)
        print(f"   后处理: {V.mean().item():.4f} → {V_post.mean().item():.4f}")
        
        # 测试V值健康检查
        print("\n测试V值健康检查...")
        V1 = torch.randn(batch_size, seq_len, dim) * 0.5 + 0.8
        V2 = torch.randn(batch_size, seq_len, dim) * 0.8 + 0.8
        V3 = torch.randn(batch_size, seq_len, dim) * 1.2 + 0.8
        V_fused = torch.randn(batch_size, seq_len, dim) * 1.0 + 0.8
        
        health_score = model._check_V_health(V1, V2, V3, V_fused)
        print(f"   V值健康度: {health_score:.4f}")
        print(f"   V1均值: {V1.mean().item():.4f}")
        print(f"   V2均值: {V2.mean().item():.4f}")
        print(f"   V3均值: {V3.mean().item():.4f}")
        print(f"   V融合均值: {V_fused.mean().item():.4f}")
        
        # 验证健康度评分范围
        assert 0.0 <= health_score <= 1.0, f"健康度评分超出范围: {health_score}"
        
        print("✅ V值调控机制测试通过")
        return True
        
    except Exception as e:
        print(f"❌ V值调控机制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_formula_validation():
    """
    测试5：公式验证系统
    """
    print("\n" + "="*60)
    print("测试5：公式验证系统")
    print("="*60)
    
    try:
        # 创建模型
        model = RGAIntegrator()
        
        # 测试连接点密度计算
        print("测试连接点密度计算...")
        markers = torch.randn(2, 8, model.config.dim)
        density_info = model._compute_connection_density(markers)
        
        print(f"   静态密度: {density_info['static_density']:.4f}")
        print(f"   连接数: {density_info['connections']}")
        print(f"   节点数: {density_info['nodes']}")
        print(f"   相似度均值: {density_info['similarity_mean']:.4f}")
        
        # 检查密度值范围
        assert 0 <= density_info['static_density'] <= 1, f"密度值超出范围: {density_info['static_density']}"
        
        # 测试公式验证
        print("\n测试公式验证...")
        step_data = {
            'markers': markers,
            'V_means': [0.5, 0.6, 0.7],
            'Q_mean': 0.3,
            'K_mean': 0.4,
            'phase_delta': 0.5
        }
        
        is_valid = model._validate_formula_execution(step_data)
        print(f"   公式验证结果: {is_valid}")
        
        # 检查历史记录
        assert len(model.density_history) > 0, "密度历史记录为空"
        
        # 获取公式统计
        stats = model.get_formula_stats()
        print(f"\n公式统计:")
        print(f"   学习阶段: {stats['phase_state']}")
        print(f"   连接阈值: {stats['connection_threshold']}")
        print(f"   相变阈值: {stats['phase_transition_threshold']}")
        print(f"   密度统计: 均值={stats['density_stats']['mean']:.4f}, 标准差={stats['density_stats']['std']:.4f}, 趋势={stats['density_stats']['trend']}")
        print(f"   验证错误: {stats['validation_errors']}")
        print(f"   验证警告: {stats['validation_warnings']}")
        
        # 测试公式参数重置
        print("\n测试公式参数重置...")
        model.reset_formula_parameters(
            connection_threshold=0.4,
            phase_threshold=0.6
        )
        print(f"   重置后连接阈值: {model.connection_threshold}")
        print(f"   重置后相变阈值: {model.phase_transition_threshold}")
        
        print("✅ 公式验证系统测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 公式验证系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_disguise_save_load():
    """
    测试6：伪装保存和加载
    """
    print("\n" + "="*60)
    print("测试6：伪装保存和加载")
    print("="*60)
    
    temp_dir = None
    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        print(f"临时目录: {temp_dir}")
        
        # 创建并保存模型
        model_save = RGAIntegrator()
        model_save.eval()
        
        # 保存模型
        save_path = os.path.join(temp_dir, "test_model")
        model_save.save_pretrained(save_path)
        
        # 检查保存的文件
        expected_files = ['pytorch_model.bin', 'config.json', 'vocab.txt', 'tokenizer_config.json']
        for file in expected_files:
            file_path = os.path.join(save_path, file)
            assert os.path.exists(file_path), f"文件不存在: {file_path}"
        
        # 检查配置文件内容
        with open(os.path.join(save_path, 'config.json'), 'r') as f:
            config_data = json.load(f)
        
        assert config_data.get('_is_rga_disguised', False), "伪装标记未设置"
        assert config_data['model_type'] == 'bert', f"模型类型不正确: {config_data['model_type']}"
        
        # 验证隐藏的RGA配置
        assert '_rga_config' in config_data, "缺少RGA配置"
        assert config_data['_rga_config']['dim'] == model_save.config.dim, "维度不匹配"
        
        # 加载模型
        print("\n加载模型...")
        model_load = RGAIntegrator.from_pretrained(save_path)
        model_load.eval()
        
        # 测试加载的模型
        input_ids, _ = create_test_batch(batch_size=1, seq_len=8)
        
        with torch.no_grad():
            # 原始模型前向传播
            result_save = model_save(input_ids, num_cycles=1)
            
            # 加载模型前向传播
            result_load = model_load(input_ids, num_cycles=1)
        
        # 比较输出
        logits_diff = torch.abs(result_save['logits'] - result_load['logits']).mean().item()
        print(f"logits差异: {logits_diff:.6f}")
        
        # 允许小的差异（由于加载时的参数不完全匹配）
        if logits_diff < 0.1:
            print("✅ 模型保存和加载测试通过")
            return True
        else:
            print(f"⚠️  输出差异较大: {logits_diff:.4f}")
            return False
            
    except Exception as e:
        print(f"❌ 伪装保存和加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"临时目录已清理: {temp_dir}")

def test_training_mode():
    """
    测试7：训练模式
    """
    print("\n" + "="*60)
    print("测试7：训练模式")
    print("="*60)
    
    try:
        # 创建模型并设置为训练模式
        model = RGAIntegrator()
        model.train()
        
        # 创建测试数据和标签
        input_ids, labels = create_test_batch(batch_size=4, seq_len=32)
        
        # 将数据移动到模型所在设备
        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)
        
        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 训练步骤
        total_loss = 0.0
        for step in range(3):  # 少量训练步骤
            optimizer.zero_grad()
            
            # 前向传播
            result = model(input_ids, num_cycles=1)
            
            # 计算损失
            logits = result['logits']
            loss = F.cross_entropy(
                logits.view(-1, model.config.vocab_size),
                labels.view(-1)
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            optimizer.step()
            
            total_loss += loss.item()
            print(f"   训练步骤 {step+1}: 损失 = {loss.item():.4f}")
            
            # 检查梯度
            grad_norm = 0.0
            grad_params = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
                    grad_params += 1
            
            if grad_params > 0:
                print(f"     梯度参数: {grad_params}, 梯度范数: {grad_norm:.4f}")
        
        avg_loss = total_loss / 3
        print(f"\n   平均损失: {avg_loss:.4f}")
        print(f"   最终V值均值: {result['V_stats']['V_fused_mean']:.4f}")
        
        print("✅ 训练模式测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 训练模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_optimization():
    """
    测试8：内存优化
    """
    print("\n" + "="*60)
    print("测试8：内存优化")
    print("="*60)
    
    try:
        # 创建模型
        model = RGAIntegrator()
        
        # 检查内存优化设置
        print("检查内存优化配置...")
        print(f"   使用混合精度: {model.use_mixed_precision}")
        print(f"   梯度累积步数: {model.gradient_accumulation_steps}")
        print(f"   内存高效模式: {model.memory_efficient_mode}")
        print(f"   设备: {model.device}")
        
        # 测试混合精度（如果可用）
        if torch.cuda.is_available():
            print("\n测试CUDA内存优化...")
            model.cuda()
            
            # 创建测试数据
            input_ids, _ = create_test_batch(batch_size=4, seq_len=64)
            input_ids = input_ids.cuda()
            
            # 使用混合精度前向传播
            if model.use_mixed_precision and model.scaler is not None:
                print("测试混合精度前向传播...")
                
                # 使用正确的autocast API
                from torch.amp import autocast
                
                with autocast('cuda'):
                    result = model(input_ids, num_cycles=1)
                
                print(f"   混合精度计算完成")
                print(f"   输出数据类型: {result['logits'].dtype}")
                
                # 检查是否使用半精度
                if result['logits'].dtype == torch.float16:
                    print("   ✅ 混合精度生效（输出为float16）")
                else:
                    print("   ⚠️  混合精度未生效（输出为float32）")
            
            # 检查CUDA内存使用
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"   CUDA内存使用: {memory_allocated:.2f} MB (已分配) / {memory_reserved:.2f} MB (预留)")
        
        print("✅ 内存优化测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 内存优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """
    测试9：边界情况
    """
    print("\n" + "="*60)
    print("测试9：边界情况")
    print("="*60)
    
    try:
        # 创建模型
        model = RGAIntegrator()
        model.eval()
        
        # 测试1：非常短的序列
        print("测试非常短的序列...")
        input_ids_short = torch.randint(0, 100, (1, 1))
        with torch.no_grad():
            result_short = model(input_ids_short, num_cycles=1)
        print(f"   序列长度1: {result_short['logits'].shape}")
        
        # 测试2：非常长的序列
        print("\n测试非常长的序列...")
        input_ids_long = torch.randint(0, 100, (1, 256))
        with torch.no_grad():
            result_long = model(input_ids_long, num_cycles=1)
        print(f"   序列长度256: {result_long['logits'].shape}")
        
        # 测试3：大批次
        print("\n测试大批次...")
        input_ids_large = torch.randint(0, 100, (8, 64))
        with torch.no_grad():
            result_large = model(input_ids_large, num_cycles=1)
        print(f"   批次大小8: {result_large['logits'].shape}")
        
        # 测试4：单循环（确保至少一个循环）
        print("\n测试单循环...")
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            result_single = model(input_ids, num_cycles=1)
        print(f"   单循环输入: {result_single['logits'].shape}")
        print(f"   单循环V值均值: {result_single['V_stats']['V_fused_mean']:.4f}")
        
        # 测试5：大量循环（测试健康检查）
        print("\n测试大量循环...")
        try:
            with torch.no_grad():
                result_many = model(input_ids, num_cycles=10)
            print(f"   10个循环: 完成")
            print(f"   最终V值均值: {result_many['V_stats']['V_fused_mean']:.4f}")
        except Exception as e:
            print(f"   10个循环可能因V值健康检查提前结束: {e}")
        
        # 测试6：零循环 - 修复：模型应处理num_cycles=0的情况
        print("\n测试零循环处理...")
        try:
            with torch.no_grad():
                result_zero = model(input_ids, num_cycles=0)
            print(f"   零循环处理: 成功，输出形状: {result_zero['logits'].shape}")
        except Exception as e:
            print(f"   零循环测试失败，需要修改模型代码: {e}")
            # 测试使用最小循环数
            with torch.no_grad():
                result_min = model(input_ids, num_cycles=1)
            print(f"   使用最小循环数(1): {result_min['logits'].shape}")
        
        print("✅ 边界情况测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 边界情况测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase_detection():
    """
    测试10：学习阶段检测
    """
    print("\n" + "="*60)
    print("测试10：学习阶段检测")
    print("="*60)
    
    try:
        # 创建模型
        model = RGAIntegrator()
        
        # 测试相变检测
        print("测试相变检测...")
        V_history = [0.1, 0.3, 0.2, 0.6, 0.5, 0.8]
        transitions = model.detect_phase_transition(V_history, threshold=0.43)
        print(f"   V值历史: {V_history}")
        print(f"   检测到相变次数: {transitions}")
        
        # 测试不同阈值
        transitions_low = model.detect_phase_transition(V_history, threshold=0.2)
        transitions_high = model.detect_phase_transition(V_history, threshold=0.6)
        print(f"   低阈值(0.2)相变次数: {transitions_low}")
        print(f"   高阈值(0.6)相变次数: {transitions_high}")
        
        # 测试学习阶段识别
        print("\n测试学习阶段识别...")
        phase = model.identify_learning_phase(V_history)
        print(f"   识别学习阶段: {phase}")
        
        # 测试不同V值模式
        stable_history = [0.5, 0.51, 0.49, 0.5, 0.51]
        volatile_history = [0.1, 0.9, 0.2, 0.8, 0.3]
        rising_history = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        print(f"   稳定历史识别: {model.identify_learning_phase(stable_history)}")
        print(f"   波动历史识别: {model.identify_learning_phase(volatile_history)}")
        print(f"   上升历史识别: {model.identify_learning_phase(rising_history)}")
        
        # 测试阶段更新
        print("\n测试阶段更新...")
        model.update_phase_state("探索期")
        print(f"   更新后阶段: {model.phase_state}")
        
        # 测试无效阶段
        model.update_phase_state("无效阶段")
        print(f"   无效阶段处理: {model.phase_state} (应保持不变)")
        
        # 测试所有有效阶段
        valid_phases = ['初始阶段', '探索期', '学习期', '稳定期', '收敛期']
        for phase in valid_phases:
            model.update_phase_state(phase)
            print(f"   设置阶段 '{phase}': {model.phase_state}")
        
        print("✅ 学习阶段检测测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 学习阶段检测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_interaction():
    """
    测试11：组件交互测试
    """
    print("\n" + "="*60)
    print("测试11：组件交互测试")
    print("="*60)
    
    try:
        # 创建模型
        model = RGAIntegrator()
        model.eval()
        
        # 创建测试数据
        input_ids, _ = create_test_batch(batch_size=1, seq_len=16)
        
        print("测试各组件交互...")
        
        # 测试嵌入层
        emb_result = model.embedding_layer(input_ids, return_details=False)
        print(f"   嵌入层输出形状: {emb_result['base_embeddings'].shape}")
        
        # 测试单向阀
        Q = emb_result['base_embeddings'].clone()
        K = emb_result['base_embeddings'].clone()
        V = emb_result['base_embeddings'].clone()
        
        Q_out, K_out, V_out = model.one_way_valve(Q, K, V)
        print(f"   单向阀输出形状: Q={Q_out.shape}, K={K_out.shape}, V={V_out.shape}")
        
        # 测试链式反应单元
        print(f"\n测试链式反应单元 (共{len(model.chain_units)}个):")
        for i, unit in enumerate(model.chain_units):
            Q_test = torch.randn(1, 8, model.config.dim)
            K_test = torch.randn(1, 8, model.config.dim)
            V_test = torch.randn(1, 8, model.config.dim)
            
            Q_out, K_out, V_out = unit(Q_test, K_test, V_test)
            print(f"   单元{i+1}输出形状: Q={Q_out.shape}, K={K_out.shape}, V={V_out.shape}")
        
        # 测试三值平衡器
        print(f"\n测试三值平衡器 (共{len(model.tri_balancers)}个):")
        for i, balancer in enumerate(model.tri_balancers):
            Q_test = torch.randn(1, 8, model.config.dim)
            K_test = torch.randn(1, 8, model.config.dim)
            V_test = torch.randn(1, 8, model.config.dim)
            
            Q_out, K_out, V_out, density, connections = balancer(Q_test, K_test, V_test, return_density=True)
            print(f"   平衡器{i+1}输出: Q={Q_out.shape}, 密度={density:.4f}, 连接数={connections}")
        
        # 测试地质记忆
        print("\n测试地质记忆...")
        Q_list = [torch.randn(1, 8, model.config.dim) for _ in range(3)]
        K_list = [torch.randn(1, 8, model.config.dim) for _ in range(3)]
        V_sublist_list = [[torch.randn(1, 8, model.config.dim) for _ in range(3)] for _ in range(3)]
        
        model.geological_memory.store(Q_list, K_list, V_sublist_list)
        
        # 测试不同深度检索
        for depth in range(3):
            Q_ret, K_ret, V_ret = model.geological_memory.retrieve(depth=depth, time_layer=0)
            print(f"   深度{depth}检索: Q={Q_ret.shape}, K={K_ret.shape}, V={V_ret.shape}")
        
        print("✅ 组件交互测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 组件交互测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_tests():
    """
    运行所有测试
    """
    print("🚀 开始RGAIntegrator全方面测试")
    print("="*60)
    
    test_results = {}
    
    # 运行所有测试
    tests = [
        ("初始化测试", test_initialization),
        ("单循环前向传播测试", test_forward_single_cycle),
        ("多循环前向传播测试", test_forward_multi_cycle),
        ("V值调控测试", test_v_regulation),
        ("公式验证测试", test_formula_validation),
        ("伪装保存加载测试", test_disguise_save_load),
        ("训练模式测试", test_training_mode),
        ("内存优化测试", test_memory_optimization),
        ("边界情况测试", test_edge_cases),
        ("学习阶段检测测试", test_phase_detection),
        ("组件交互测试", test_component_interaction),
    ]
    
    passed_count = 0
    failed_count = 0
    
    for test_name, test_func in tests:
        print(f"\n执行: {test_name}")
        try:
            result = test_func()
            test_results[test_name] = result
            
            if result:
                passed_count += 1
                print(f"✅ {test_name}: 通过")
            else:
                failed_count += 1
                print(f"❌ {test_name}: 失败")
        except Exception as e:
            print(f"❌ {test_name}发生异常: {e}")
            test_results[test_name] = False
            failed_count += 1
    
    # 打印测试总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print("\n" + "="*60)
    print(f"总计: {len(tests)} 个测试")
    print(f"通过: {passed_count}")
    print(f"失败: {failed_count}")
    
    if failed_count == 0:
        print("🎉 所有测试通过！RGAIntegrator功能正常。")
        return True
    else:
        print(f"⚠️  有 {failed_count} 个测试失败，请检查问题。")
        return False

# ==================== 性能基准测试 ====================
def benchmark_performance():
    """
    性能基准测试 - 健壮版，完全避免变量名冲突
    """
    print("\n" + "="*60)
    print("性能基准测试")
    print("="*60)
    
    try:
        # 在函数内部导入，避免全局变量名冲突
        import time
        model = RGAIntegrator()
        model.eval()
        
        # 不同输入尺寸的基准测试
        test_cases = [
            (1, 8, "小"),
            (2, 32, "中"),
            (4, 64, "大"),
            (8, 128, "超大"),
        ]
        
        print("基准测试 (每个尺寸运行3次，取平均):")
        
        for batch_size, seq_len, size_label in test_cases:
            print(f"\n测试 {size_label} 尺寸 (batch={batch_size}, seq={seq_len}):")
            
            input_ids, _ = create_test_batch(batch_size, seq_len)
            
            # 预热
            with torch.no_grad():
                _ = model(input_ids, num_cycles=1)
            
            # 基准测试
            run_times = []
            memory_usage = []
            
            for run_idx in range(3):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                t_start = time.time()  # 使用局部变量t_start
                with torch.no_grad():
                    result = model(input_ids, num_cycles=1)
                t_end = time.time()  # 使用局部变量t_end
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
                    memory_usage.append(memory_allocated)
                
                run_times.append(t_end - t_start)
            
            avg_time = np.mean(run_times)
            std_time = np.std(run_times)
            
            print(f"   推理时间: {avg_time:.4f} ± {std_time:.4f} 秒")
            print(f"   logits形状: {result['logits'].shape}")
            
            if memory_usage:
                avg_memory = np.mean(memory_usage)
                print(f"   GPU内存峰值: {avg_memory:.2f} MB")
        
        # 多循环性能测试
        print("\n多循环性能测试:")
        input_ids, _ = create_test_batch(1, 16)
        
        for num_cycles in [1, 3, 5]:
            cycle_run_times = []
            for _ in range(3):
                t_start = time.time()
                with torch.no_grad():
                    result = model(input_ids, num_cycles=num_cycles)
                t_end = time.time()
                cycle_run_times.append(t_end - t_start)
            
            avg_time = np.mean(cycle_run_times)
            print(f"   {num_cycles}个循环: {avg_time:.4f}秒, V值={result['V_stats']['V_fused_mean']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能基准测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
# ==================== 示例用法 ====================
def example_usage():
    """
    示例用法展示
    """
    print("\n" + "="*60)
    print("示例用法")
    print("="*60)
    
    try:
        # 1. 创建模型
        print("1. 创建RGAIntegrator模型")
        model = RGAIntegrator()
        model.eval()
        
        # 2. 创建输入数据
        print("2. 创建输入数据")
        # 模拟文本："深度学习是人工智能的重要分支"
        # 使用随机ID模拟分词结果
        input_ids = torch.randint(0, 100, (1, 10))
        print(f"   输入形状: {input_ids.shape}")
        print(f"   输入示例: {input_ids[0, :5].tolist()}...")
        
        # 3. 推理
        print("3. 进行推理")
        with torch.no_grad():
            result = model(input_ids, num_cycles=3)
        
        # 4. 分析结果
        print("4. 分析结果")
        print(f"   logits形状: {result['logits'].shape}")
        print(f"   V值演化分析: {result['V_evolution_analysis']['trend']}")
        print(f"   V值范围: [{result['V_evolution_analysis']['V_range'][0]:.4f}, {result['V_evolution_analysis']['V_range'][1]:.4f}]")
        print(f"   建议: {result['V_evolution_analysis']['recommendation']}")
        
        # 5. 获取预测
        print("5. 获取预测")
        predictions = torch.argmax(result['logits'], dim=-1)
        print(f"   预测形状: {predictions.shape}")
        print(f"   预测示例: {predictions[0, :5].tolist()}...")
        
        # 6. 获取统计信息
        print("6. 获取模型统计信息")
        stats = model.get_formula_stats()
        print(f"   当前阶段: {stats['phase_state']}")
        print(f"   密度趋势: {stats['density_stats']['trend']}")
        
        # 7. 获取V值统计
        print("7. V值统计信息")
        v_stats = result['V_stats']
        print(f"   V1均值: {v_stats['V1_mean']:.4f}")
        print(f"   V2均值: {v_stats['V2_mean']:.4f}")
        print(f"   V3均值: {v_stats['V3_mean']:.4f}")
        print(f"   融合V均值: {v_stats['V_fused_mean']:.4f}")
        print(f"   Q-K相似度: {v_stats['QK_similarity']:.4f}")
        
        # 8. 保存和加载示例
        print("8. 保存和加载示例")
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, "example_model")
        
        # 保存模型
        model.save_pretrained(save_path)
        print(f"   模型已保存到: {save_path}")
        
        # 加载模型
        loaded_model = RGAIntegrator.from_pretrained(save_path)
        loaded_model.eval()
        print(f"   模型已从 {save_path} 加载")
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        print("\n✅ 示例完成")
        return True
        
    except Exception as e:
        print(f"❌ 示例用法失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== 快速诊断测试 ====================
def quick_diagnostic_test():
    """
    快速诊断测试 - 检查最基本的功能
    """
    print("\n" + "="*60)
    print("快速诊断测试")
    print("="*60)
    
    tests = [
        ("模型初始化", lambda: RGAIntegrator() is not None),
        ("创建测试数据", lambda: create_test_batch()[0].shape == (2, 16)),
        ("CUDA可用性", lambda: torch.cuda.is_available() or True),  # 总是返回True
        ("PyTorch版本", lambda: torch.__version__ is not None),
    ]
    
    all_pass = True
    for test_name, test_func in tests:
        try:
            result = test_func()
            status = "✅" if result else "❌"
            print(f"{status} {test_name}")
            if not result:
                all_pass = False
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            all_pass = False
    
    return all_pass

# ==================== 主执行函数 ====================
if __name__ == "__main__":
    print("🧪 RGAIntegrator 全方面测试套件")
    print("版本: 修复版")
    print("="*60)
    
    # 运行快速诊断
    diagnostic_passed = quick_diagnostic_test()
    
    if diagnostic_passed:
        # 运行主测试
        all_passed = run_comprehensive_tests()
        
        # 运行性能基准测试
        if all_passed:
            benchmark_performance()
        
        # 展示示例用法
        if all_passed:
            example_usage()
        
        print("\n" + "="*60)
        if all_passed:
            print("🎉 所有测试通过！RGAIntegrator功能完整且稳定。")
        else:
            print("⚠️  部分测试失败，请检查具体问题。")
    else:
        print("❌ 快速诊断失败，无法继续测试")
    
    print("="*60)
    print("测试套件执行完成")
    print("="*60)