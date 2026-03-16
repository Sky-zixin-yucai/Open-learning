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
from torch.utils.data import Dataset
import re
import random
from collections import Counter, deque
from torch.autograd import Function

import sys
import os

# ========== 核心修正：正确处理路径 ==========
# 获取当前运行的 rga.py 文件的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取 rga.py 所在的目录（即 openlearning包目录：/home/skyzi/openlearning/openlearning/）
current_dir = os.path.dirname(current_script_path)
# 获取 openlearning包目录的上级目录（即 openlearning根目录：/home/skyzi/openlearning/）
parent_dir = os.path.dirname(current_dir)

# 将「openlearning根目录」加入Python搜索路径（关键！）
# 这样 Python 就能找到 `openlearning` 这个包了
sys.path.append(parent_dir)

# ========== 尝试导入模块 ==========
try:
    from openlearning import (
        RGAConfig,
        VRegularizationParams,
        CoreMetricsCalculator,
        TriValueBalancer,
        OneWayValve,
        EnhancedEmbeddingLayer,
        ChainReactionUnit_Final,
        GeologicalMemory,
        SandwichFusion,
        test_complete_architecture,
    )
    
    print("✅ 所有模块导入成功")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"🔍 当前脚本路径: {current_script_path}")
    print(f"🔍 openlearning包目录: {current_dir}")
    print(f"🔍 已添加的搜索路径: {parent_dir}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 发生其他错误: {type(e).__name__}: {e}")
    sys.exit(1)

# ========== 验证导入（可选） ==========
if __name__ == "__main__":
    imported_components = [
        "RGAConfig", "VRegularizationParams", "CoreMetricsCalculator",
        "TriValueBalancer", "OneWayValve", "EnhancedEmbeddingLayer",
        "ChainReactionUnit_Final", "GeologicalMemory", "SandwichFusion",
        "test_complete_architecture"
    ]
    
    print("\n📋 已导入的组件列表:")
    for comp in imported_components:
        if comp in locals():
            print(f"  - {comp}: ✅ 存在")
        else:
            print(f"  - {comp}: ❌ 缺失")

class RuleGovernedArchitecture(nn.Module):
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
        self.V_regularization_params = VRegularizationParams()
        
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
        model = RuleGovernedArchitecture(
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
    
__all__ = {
    'RuleGovernedArchitecture',
    'SmartTextDataset',
    'VisualTrainingProgress',
    'AdvancedConstrainedArchitectureTrainer',
}
