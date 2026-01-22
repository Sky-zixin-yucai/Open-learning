"""
地质记忆模块 | Geological Memory Module
======================================

实现三层深度×三个时间层×三个V子值的地质记忆系统。
Implements a geological memory system with three depths × three time layers × three V subvalues.

设计原理:
Design Principles:
• 三层存储结构：最新层(n)、中期层(n-1)、深层(n-2)
• 三个时间层：每个深度有三个时间切片
• 三个V子值：每个时间层存储三个V值状态
• Three-layer storage structure: latest layer (n), medium-term layer (n-1), deep layer (n-2)
• Three time layers: three time slices per depth
• Three V subvalues: each time layer stores three V value states

架构图示:
Architecture Diagram:
    depth0 (最新层 | Latest): [time0, time1, time2] × (Q, K, [V1, V2, V3])
    depth1 (中期层 | Medium): [time0, time1, time2] × (Q, K, [V1, V2, V3])
    depth2 (深层 | Deep):    [time0, time1, time2] × (Q, K, [V1, V2, V3])

能量系统:
Energy System:
• 线性衰退：energy *= decay_factor
• 密度公式计算能量：D = 2m/((N+1)N)
• 阈值过滤：相似度 > 0.3的连接才计数
• Linear decay: energy *= decay_factor
• Energy calculated by density formula: D = 2m/((N+1)N)
• Threshold filtering: only connections with similarity > 0.3 are counted

移位更新机制:
Shift Update Mechanism:
    1. 深层衰退：能量最低的时间层被中期层覆盖
    2. 中期层 ← 最新层：数据下移一层
    3. 存储新数据到最新层：更新最新状态
    1. Deep layer decay: time layer with lowest energy is overwritten by medium layer
    2. Medium layer ← Latest layer: data shifts down one level
    3. Store new data to latest layer: update latest state

应用场景:
Application Scenarios:
• 长期记忆存储：保留重要模式
• 信息衰退模拟：模拟遗忘过程
• 状态恢复：从深层记忆检索稳定状态
• Long-term memory storage: retains important patterns
• Information decay simulation: simulates forgetting process
• State recovery: retrieves stable states from deep memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class GeologicalMemory(nn.Module):
    """
    地质记忆层 | Geological Memory Layer
    
    严格按照文档设计的三层深度×三个时间层×三个V子值记忆系统。
    Strictly follows the document design: three depths × three time layers × three V subvalues.
    
    核心特性:
    Core Features:
    1. 三层深度：最新层(0)、中期层(1)、深层(2)
    2. 三个时间层：每个深度存储三个时间切片
    3. 三个V子值：每个时间层存储三个V值的演变历史
    4. 能量系统：基于密度公式的活跃度评估
    5. 线性衰退：模拟记忆衰减过程
    
    Three depths: latest layer (0), medium-term layer (1), deep layer (2)
    Three time layers: each depth stores three time slices
    Three V subvalues: each time layer stores evolution history of three V values
    Energy system: activity assessment based on density formula
    Linear decay: simulates memory decay process
    """
    
    def __init__(self, dim: int, decay_factor: float = 0.7):
        """
        初始化地质记忆层 | Initialize Geological Memory Layer
        
        参数:
        Parameters:
            dim: 特征维度 | Feature dimension
            decay_factor: 线性衰退因子，默认为0.7 | Linear decay factor, default 0.7
        """
        super().__init__()
        self.dim = dim
        self.decay_factor = decay_factor
        
        # ==================== 三层存储结构 ====================
        # ==================== Three-Layer Storage Structure ====================
        
        # 最新层 (n) - 存储当前最新状态
        # Latest layer (n) - stores current latest states
        self.register_buffer('depth0_Q', torch.zeros(3, dim))  # [时间层, dim] | [time layers, dim]
        self.register_buffer('depth0_K', torch.zeros(3, dim))
        self.register_buffer('depth0_V', torch.zeros(3, 3, dim))  # [时间层, 3个子值, dim] | [time layers, 3 subvalues, dim]
        
        # 中期层 (n-1) - 存储中期状态
        # Medium-term layer (n-1) - stores medium-term states
        self.register_buffer('depth1_Q', torch.zeros(3, dim))
        self.register_buffer('depth1_K', torch.zeros(3, dim))
        self.register_buffer('depth1_V', torch.zeros(3, 3, dim))
        
        # 深层 (n-2) - 存储长期稳定状态
        # Deep layer (n-2) - stores long-term stable states
        self.register_buffer('depth2_Q', torch.zeros(3, dim))
        self.register_buffer('depth2_K', torch.zeros(3, dim))
        self.register_buffer('depth2_V', torch.zeros(3, 3, dim))
        
        # ==================== 能量系统 ====================
        # ==================== Energy System ====================
        self.register_buffer('energy_depth0', torch.ones(3))  # 最新层能量 | Latest layer energy
        self.register_buffer('energy_depth1', torch.ones(3))  # 中期层能量 | Medium-term layer energy
        self.register_buffer('energy_depth2', torch.ones(3))  # 深层能量 | Deep layer energy
        
        # ==================== 状态跟踪 ====================
        # ==================== State Tracking ====================
        self.step = 0
        self.register_buffer('timestamps_depth0', torch.zeros(3, dtype=torch.long))
        self.register_buffer('timestamps_depth1', torch.zeros(3, dtype=torch.long))
        self.register_buffer('timestamps_depth2', torch.zeros(3, dtype=torch.long))
        
        print(f"✅ 地质记忆层初始化完成 | Geological memory layer initialized")
        print(f"   维度: {dim}, 衰退因子: {decay_factor}")
        print(f"   存储结构: 3层深度 × 3个时间层 × 3个V子值")
    
    def reset(self):
        """
        重置记忆 | Reset Memory
        
        功能:
        Function:
            清空所有存储状态和能量，恢复到初始状态
            Clears all stored states and energy, returns to initial state
        
        应用场景:
        Application Scenarios:
            • 模型重新训练时
            • 测试新数据模式时
            • When retraining model
            • When testing new data patterns
        """
        for buffer in self.buffers():
            if buffer.dim() > 0:  # 跳过标量 | Skip scalars
                buffer.zero_()
        
        self.energy_depth0.fill_(1.0)
        self.energy_depth1.fill_(1.0)
        self.energy_depth2.fill_(1.0)
        self.step = 0
        
        print("✅ 地质记忆已重置 | Geological memory reset")
    
    def _compute_energy_by_density(self, values: torch.Tensor) -> float:
        """
        使用密度公式计算能量 | Compute Energy Using Density Formula
        
        核心公式:
        Core Formula:
            静态密度公式: D = 2m/((N+1)N)
            其中N=节点数，m=连接数
            Static density formula: D = 2m/((N+1)N)
            where N = number of nodes, m = number of connections
        
        参数:
        Parameters:
            values: 输入值矩阵，形状为[n_nodes, dim] | Input value matrix, shape [n_nodes, dim]
        
        返回:
        Returns:
            energy: 计算得到的能量值，范围[0,1] | Computed energy value, range [0,1]
        
        物理意义:
        Physical Meaning:
            • 节点间连接越多，密度越高，能量越大
            • 完全连接时密度最高，能量接近1
            • 无连接时密度最低，能量接近0
            • More connections between nodes → higher density → higher energy
            • Maximum density when fully connected → energy close to 1
            • Minimum density when no connections → energy close to 0
        """
        if values.numel() == 0 or values.size(0) < 2:
            return 1.0  # 单节点或无数据时返回最大能量 | Return maximum energy for single node or no data
        
        N = values.size(0)
        
        # 计算余弦相似度矩阵 | Compute cosine similarity matrix
        values_norm = F.normalize(values, p=2, dim=1)
        similarity = torch.mm(values_norm, values_norm.T)
        
        # 统计连接数 (相似度 > 0.3) | Count connections (similarity > 0.3)
        connections = (similarity > 0.3).float()
        m = connections.sum().item() / 2  # 无向图，每条边计一次 | Undirected graph, count each edge once
        
        # 计算密度 | Compute density
        if N > 1:
            density = (2 * m) / ((N + 1) * N)
        else:
            density = 0.0
        
        # 能量 = 密度归一化到[0,1] | Energy = density normalized to [0,1]
        energy = min(max(density, 0.0), 1.0)
        return energy
    
    def _linear_decay(self):
        """
        线性衰退：能量 *= 衰退因子 | Linear Decay: energy *= decay_factor
        
        物理意义:
        Physical Meaning:
            • 模拟记忆随时间衰减的过程
            • 衰退因子控制衰减速度
            • 深层记忆衰退更快（能量更低）
            • Simulates memory decay over time
            • Decay factor controls decay speed
            • Deep memory decays faster (lower energy)
        """
        with torch.no_grad():
            self.energy_depth0.mul_(self.decay_factor)
            self.energy_depth1.mul_(self.decay_factor)
            self.energy_depth2.mul_(self.decay_factor)
    
    def _shift_and_update(self, Q_list: List[torch.Tensor], K_list: List[torch.Tensor], 
                         V_list: List[List[torch.Tensor]]):
        """
        执行三层移位和更新 | Execute Three-Layer Shift and Update
        
        算法流程:
        Algorithm Flow:
            1. 检查深层衰退：能量最低的时间层被中期层覆盖
            2. 中期层 ← 最新层：数据下移一层
            3. 存储新数据到最新层：更新最新状态
        
        1. Check deep layer decay: time layer with lowest energy is overwritten by medium layer
        2. Medium layer ← Latest layer: data shifts down one level
        3. Store new data to latest layer: update latest state
        
        参数:
        Parameters:
            Q_list: Q值列表，包含3个链式反应单元的Q输出
            K_list: K值列表，包含3个链式反应单元的K输出
            V_list: V值列表，三维结构: [3个单元] × [3个时间层] × [3个V子值]
            
            Q_list: List of Q values, contains Q outputs from 3 chain reaction units
            K_list: List of K values, contains K outputs from 3 chain reaction units
            V_list: List of V values, 3D structure: [3 units] × [3 time layers] × [3 V subvalues]
        """
        self.step += 1
        
        # ==================== 1. 检查深层衰退 ====================
        # ==================== 1. Check Deep Layer Decay ====================
        # 找到能量最低的时间层（最可能衰退）
        # Find time layer with lowest energy (most likely to decay)
        min_energy_idx = torch.argmin(self.energy_depth2).item()
        min_energy = self.energy_depth2[min_energy_idx].item()
        
        # 如果能量低于阈值，用中期层数据覆盖
        # If energy below threshold, overwrite with medium layer data
        if min_energy < 0.1:
            if self.energy_depth1[min_energy_idx] > 0.1:  # 中期层有有效数据 | Medium layer has valid data
                self.depth2_Q[min_energy_idx] = self.depth1_Q[min_energy_idx].clone()
                self.depth2_K[min_energy_idx] = self.depth1_K[min_energy_idx].clone()
                self.depth2_V[min_energy_idx] = self.depth1_V[min_energy_idx].clone()
                self.energy_depth2[min_energy_idx] = self.energy_depth1[min_energy_idx].clone() * 0.8
                self.timestamps_depth2[min_energy_idx] = self.step
                print(f"  深层衰退: 时间层{min_energy_idx}被中期层覆盖 | Deep decay: time layer {min_energy_idx} overwritten by medium layer")
        
        # ==================== 2. 中期层 ← 最新层 ====================
        # ==================== 2. Medium Layer ← Latest Layer ====================
        for i in range(3):
            if self.energy_depth0[i] > 0.1:  # 最新层有有效数据 | Latest layer has valid data
                self.depth1_Q[i] = self.depth0_Q[i].clone()
                self.depth1_K[i] = self.depth0_K[i].clone()
                self.depth1_V[i] = self.depth0_V[i].clone()
                self.energy_depth1[i] = self.energy_depth0[i].clone() * 0.9
                self.timestamps_depth1[i] = self.step
        
        # ==================== 3. 存储新数据到最新层 ====================
        # ==================== 3. Store New Data to Latest Layer ====================
        for time_layer in range(min(3, len(Q_list))):
            if Q_list[time_layer] is not None:
                # 计算批次和序列的均值
                # Compute mean across batch and sequence
                Q_mean = Q_list[time_layer].mean(dim=(0, 1))  # [dim]
                K_mean = K_list[time_layer].mean(dim=(0, 1))
                
                # 存储三个V子值
                # Store three V subvalues
                V_means = []
                for v_idx in range(3):
                    if v_idx < len(V_list[time_layer]):
                        V_mean = V_list[time_layer][v_idx].mean(dim=(0, 1))  # [dim]
                        V_means.append(V_mean)
                    else:
                        V_means.append(torch.zeros_like(Q_mean))
                
                # 存储到最新层
                # Store to latest layer
                self.depth0_Q[time_layer] = Q_mean
                self.depth0_K[time_layer] = K_mean
                self.depth0_V[time_layer] = torch.stack(V_means, dim=0)  # [3, dim]
                
                # 更新时间戳
                # Update timestamp
                self.timestamps_depth0[time_layer] = self.step
                
                # 计算新能量（使用Q, K, V的平均状态）
                # Compute new energy (using average states of Q, K, V)
                combined = torch.stack([Q_mean, K_mean] + V_means, dim=0)  # [5, dim]
                new_energy = self._compute_energy_by_density(combined)
                self.energy_depth0[time_layer] = new_energy
                
                print(f"  最新层更新: 时间层{time_layer}, 能量={new_energy:.3f}")
    
    def store(self, 
              Q_list: List[torch.Tensor],      # [Q1, Q2, Q3] - 3个链式反应单元的Q输出
              K_list: List[torch.Tensor],      # [K1, K2, K3] - 3个链式反应单元的K输出
              V_list: List[List[torch.Tensor]]):  # [[1V1,2V1,3V1], [1V2,2V2,3V2], [1V3,2V3,3V3]]
        """
        存储三个链式反应单元的输出 | Store Outputs of Three Chain Reaction Units
        
        参数:
        Parameters:
            Q_list: Q值列表，包含3个链式反应单元的Q输出
            K_list: K值列表，包含3个链式反应单元的K输出
            V_list: V值列表，三维结构: [3个单元] × [3个时间层] × [3个V子值]
            
            Q_list: List of Q values, contains Q outputs from 3 chain reaction units
            K_list: List of K values, contains K outputs from 3 chain reaction units
            V_list: List of V values, 3D structure: [3 units] × [3 time layers] × [3 V subvalues]
        
        流程:
        Process:
            1. 应用线性衰退
            2. 执行移位和更新
            1. Apply linear decay
            2. Execute shift and update
        """
        # 应用线性衰退
        # Apply linear decay
        self._linear_decay()
        
        # 执行移位和更新
        # Execute shift and update
        self._shift_and_update(Q_list, K_list, V_list)
        
        print(f"✅ 地质记忆存储完成 (步骤{self.step}) | Geological memory stored (step {self.step})")
    
    def retrieve(self, 
                depth: int = 2,        # 0:最新, 1:中期, 2:深层
                time_layer: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        检索地质记忆 | Retrieve Geological Memory
        
        默认检索深层(depth=2)的第3个时间层(time_layer=2)
        By default retrieves the 3rd time layer (time_layer=2) of the deep layer (depth=2)
        
        参数:
        Parameters:
            depth: 深度索引，0=最新层，1=中期层，2=深层
            time_layer: 时间层索引，0/1/2对应三个时间层
            
            depth: Depth index, 0=latest layer, 1=medium layer, 2=deep layer
            time_layer: Time layer index, 0/1/2 correspond to three time layers
        
        返回:
        Returns:
            (Q_ret, K_ret, V_ret): 检索到的Q, K, V值
            (Q_ret, K_ret, V_ret): Retrieved Q, K, V values
        
        物理意义:
        Physical Meaning:
            • 最新层：当前活跃状态，适合短期记忆检索
            • 中期层：近期稳定状态，适合中期模式识别
            • 深层：长期稳定状态，适合核心记忆检索
            • Latest layer: current active states, suitable for short-term memory retrieval
            • Medium layer: recent stable states, suitable for medium-term pattern recognition
            • Deep layer: long-term stable states, suitable for core memory retrieval
        """
        # 选择深度层
        # Select depth layer
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
        # Retrieve and detach to prevent gradient backflow
        Q_det = Q_mem[time_layer_idx].detach().clone()
        K_ret = K_mem[time_layer_idx].detach().clone()
        V_ret = V_mem[time_layer_idx, 2].detach().clone()  # 第3个V子值（最新演变） | 3rd V subvalue (latest evolution)
        
        # 获取能量信息
        # Get energy information
        if depth == 0:
            energy = self.energy_depth0[time_layer_idx].item()
        elif depth == 1:
            energy = self.energy_depth1[time_layer_idx].item()
        else:
            energy = self.energy_depth2[time_layer_idx].item()
        
        print(f"✅ 地质记忆检索: 深度{depth}, 时间层{time_layer_idx}, 能量={energy:.3f}")
        
        return Q_det, K_ret, V_ret
    
    def forward(self, 
               Q_list: List[torch.Tensor],
               K_list: List[torch.Tensor],
               V_list: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：存储并检索 | Forward Propagation: Store and Retrieve
        
        参数:
        Parameters:
            Q_list: Q值列表
            K_list: K值列表
            V_list: V值列表
        
        返回:
        Returns:
            检索到的深层稳定状态（默认）
            Retrieved deep stable states (default)
        """
        self.store(Q_list, K_list, V_list)
        return self.retrieve(depth=2, time_layer=2)
    
    def get_energy_stats(self) -> Dict:
        """
        获取能量统计 | Get Energy Statistics
        
        返回:
        Returns:
            energy_stats: 各层能量统计信息
            energy_stats: Energy statistics for each layer
        """
        return {
            'step': self.step,
            'energy_depth0': self.energy_depth0.cpu().numpy().tolist(),
            'energy_depth1': self.energy_depth1.cpu().numpy().tolist(),
            'energy_depth2': self.energy_depth2.cpu().numpy().tolist(),
            'energy_summary': {
                'depth0_mean': self.energy_depth0.mean().item(),
                'depth1_mean': self.energy_depth1.mean().item(),
                'depth2_mean': self.energy_depth2.mean().item(),
                'total_energy': self.energy_depth0.sum().item() + 
                               self.energy_depth1.sum().item() + 
                               self.energy_depth2.sum().item(),
            }
        }
    
    def get_v_stats(self, depth: int = 0, time_layer: int = 0) -> Dict:
        """
        获取V值统计 | Get V Value Statistics
        
        参数:
        Parameters:
            depth: 深度索引 | Depth index
            time_layer: 时间层索引 | Time layer index
        
        返回:
        Returns:
            v_stats: V值统计信息
            v_stats: V value statistics
        """
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
            'V_differences': {
                'V1_V2': torch.norm(V_tensor[0] - V_tensor[1], p=2).item(),
                'V2_V3': torch.norm(V_tensor[1] - V_tensor[2], p=2).item(),
                'V3_V1': torch.norm(V_tensor[2] - V_tensor[0], p=2).item(),
            },
            'V_all_different': torch.all(V_tensor[0] != V_tensor[1]).item() and 
                              torch.all(V_tensor[1] != V_tensor[2]).item()
        }
    
    def get_memory_health(self) -> Dict:
        """
        获取记忆健康度 | Get Memory Health Status
        
        返回:
        Returns:
            health_stats: 记忆健康度统计
            health_stats: Memory health statistics
        
        评估指标:
        Evaluation Metrics:
            1. 能量均匀性：各层能量是否均衡
            2. 时间新鲜度：最新数据是否及时更新
            3. V值差异性：三个V子值是否足够不同
            1. Energy uniformity: whether energy is balanced across layers
            2. Time freshness: whether latest data is updated timely
            3. V value differentiation: whether three V subvalues are sufficiently different
        """
        energy_stats = self.get_energy_stats()
        
        # 计算能量均匀性
        # Calculate energy uniformity
        energies = self.energy_depth0.tolist() + self.energy_depth1.tolist() + self.energy_depth2.tolist()
        energy_std = np.std(energies)
        energy_uniformity = 1.0 / (energy_std + 1e-6)
        
        # 计算时间新鲜度
        # Calculate time freshness
        current_step = self.step
        time_gaps = []
        for ts in self.timestamps_depth0:
            if ts > 0:
                time_gaps.append(current_step - ts)
        
        avg_time_gap = np.mean(time_gaps) if time_gaps else 0
        freshness = 1.0 / (avg_time_gap + 1e-6)
        
        # 计算V值健康度
        # Calculate V value health
        v_health_scores = []
        for depth in range(3):
            for time_layer in range(3):
                v_stats = self.get_v_stats(depth, time_layer)
                if v_stats['V_all_different']:
                    v_health_scores.append(1.0)
                else:
                    v_health_scores.append(0.5)
        
        v_health = np.mean(v_health_scores) if v_health_scores else 0
        
        # 总体健康度
        # Overall health
        overall_health = (energy_uniformity * 0.3 + freshness * 0.3 + v_health * 0.4)
        
        return {
            'overall_health': overall_health,
            'energy_uniformity': energy_uniformity,
            'freshness': freshness,
            'v_health': v_health,
            'health_status': self._get_health_status(overall_health),
            'recommendation': self._get_health_recommendation(overall_health, energy_uniformity, freshness, v_health)
        }
    
    def _get_health_status(self, health_score: float) -> str:
        """根据健康评分获取状态描述 | Get status description based on health score"""
        if health_score > 0.8:
            return "优秀 | Excellent"
        elif health_score > 0.6:
            return "良好 | Good"
        elif health_score > 0.4:
            return "一般 | Fair"
        else:
            return "需要关注 | Needs Attention"
    
    def _get_health_recommendation(self, overall: float, energy: float, freshness: float, v_health: float) -> str:
        """获取健康建议 | Get health recommendations"""
        recommendations = []
        
        if energy < 0.5:
            recommendations.append("能量分布不均，建议调整衰退因子 | Uneven energy distribution, suggest adjusting decay factor")
        
        if freshness < 0.5:
            recommendations.append("记忆更新不及时，建议增加存储频率 | Memory not updated timely, suggest increasing storage frequency")
        
        if v_health < 0.5:
            recommendations.append("V值区分度不足，建议检查V值生成逻辑 | Insufficient V value differentiation, suggest checking V value generation logic")
        
        if overall < 0.4:
            recommendations.append("总体健康度低，建议重置记忆 | Overall health low, suggest resetting memory")
        
        return "; ".join(recommendations) if recommendations else "状态正常 | Status normal"
    
    def visualize_memory_structure(self) -> str:
        """
        可视化记忆结构 | Visualize Memory Structure
        
        返回:
        Returns:
            visualization: 记忆结构的文本可视化
            visualization: Text visualization of memory structure
        """
        energy_stats = self.get_energy_stats()
        health_stats = self.get_memory_health()
        
        visualization = []
        visualization.append("=" * 60)
        visualization.append("🧠 地质记忆结构可视化 | Geological Memory Structure Visualization")
        visualization.append("=" * 60)
        
        # 深度层信息
        # Depth layer information
        depths = ["最新层 (Latest)", "中期层 (Medium)", "深层 (Deep)"]
        for i, depth_name in enumerate(depths):
            if i == 0:
                Q_mem = self.depth0_Q
                K_mem = self.depth0_K
                V_mem = self.depth0_V
                energies = self.energy_depth0
                timestamps = self.timestamps_depth0
            elif i == 1:
                Q_mem = self.depth1_Q
                K_mem = self.depth1_K
                V_mem = self.depth1_V
                energies = self.energy_depth1
                timestamps = self.timestamps_depth1
            else:
                Q_mem = self.depth2_Q
                K_mem = self.depth2_K
                V_mem = self.depth2_V
                energies = self.energy_depth2
                timestamps = self.timestamps_depth2
            
            visualization.append(f"\n{depth_name}:")
            for tl in range(3):
                energy = energies[tl].item()
                timestamp = timestamps[tl].item()
                age = self.step - timestamp if timestamp > 0 else "N/A"
                
                v_stats = self.get_v_stats(i, tl)
                
                visualization.append(f"  时间层 {tl}: 能量={energy:.3f}, 年龄={age}, "
                                   f"V均值=[{v_stats['V1_mean']:.3f}, {v_stats['V2_mean']:.3f}, {v_stats['V3_mean']:.3f}]")
        
        # 健康度信息
        # Health information
        visualization.append("\n" + "-" * 60)
        visualization.append("📊 记忆健康度报告 | Memory Health Report")
        visualization.append(f"  总体健康度: {health_stats['overall_health']:.3f} ({health_stats['health_status']})")
        visualization.append(f"  能量均匀性: {health_stats['energy_uniformity']:.3f}")
        visualization.append(f"  时间新鲜度: {health_stats['freshness']:.3f}")
        visualization.append(f"  V值健康度: {health_stats['v_health']:.3f}")
        visualization.append(f"  建议: {health_stats['recommendation']}")
        visualization.append("=" * 60)
        
        return "\n".join(visualization)


# ==================== 工厂函数 ====================
# ==================== Factory Functions ====================

def create_geological_memory(dim: int, **kwargs) -> GeologicalMemory:
    """
    创建地质记忆工厂函数 | Geological Memory Factory Function
    
    参数:
    Parameters:
        dim: 特征维度 | Feature dimension
        **kwargs: 其他参数传递给GeologicalMemory
        **kwargs: Additional parameters passed to GeologicalMemory
    
    返回:
    Returns:
        memory: 地质记忆实例 | Geological memory instance
    """
    return GeologicalMemory(dim=dim, **kwargs)


# ==================== 测试函数 ====================
# ==================== Test Functions ====================

def test_geological_memory():
    """
    测试地质记忆功能 | Test Geological Memory Functionality
    """
    print("🧪 测试地质记忆... | Testing geological memory...")
    
    # 创建测试参数
    # Create test parameters
    dim = 64
    batch_size = 2
    seq_len = 8
    
    # 创建地质记忆实例
    # Create geological memory instance
    memory = GeologicalMemory(dim=dim, decay_factor=0.8)
    
    # 创建测试数据
    # Create test data
    Q_list = [
        torch.randn(batch_size, seq_len, dim) * 0.5,
        torch.randn(batch_size, seq_len, dim) * 0.7,
        torch.randn(batch_size, seq_len, dim) * 0.9,
    ]
    
    K_list = [
        torch.randn(batch_size, seq_len, dim) * 0.4,
        torch.randn(batch_size, seq_len, dim) * 0.6,
        torch.randn(batch_size, seq_len, dim) * 0.8,
    ]
    
    V_list = [
        [  # 单元1的3个V子值 | 3 V subvalues for unit 1
            torch.randn(batch_size, seq_len, dim) * 0.3,
            torch.randn(batch_size, seq_len, dim) * 0.5,
            torch.randn(batch_size, seq_len, dim) * 0.7,
        ],
        [  # 单元2的3个V子值 | 3 V subvalues for unit 2
            torch.randn(batch_size, seq_len, dim) * 0.4,
            torch.randn(batch_size, seq_len, dim) * 0.6,
            torch.randn(batch_size, seq_len, dim) * 0.8,
        ],
        [  # 单元3的3个V子值 | 3 V subvalues for unit 3
            torch.randn(batch_size, seq_len, dim) * 0.5,
            torch.randn(batch_size, seq_len, dim) * 0.7,
            torch.randn(batch_size, seq_len, dim) * 0.9,
        ],
    ]
    
    # 测试存储
    # Test storage
    print("1. 测试存储... | Testing storage...")
    memory.store(Q_list, K_list, V_list)
    
    # 验证能量更新
    # Verify energy update
    energy_stats = memory.get_energy_stats()
    assert len(energy_stats['energy_depth0']) == 3, "能量深度0应为3个元素"
    print(f"✅ 能量统计: {energy_stats}")
    
    # 测试检索
    # Test retrieval
    print("\n2. 测试检索... | Testing retrieval...")
    Q_ret, K_ret, V_ret = memory.retrieve(depth=0, time_layer=0)
    
    assert Q_ret.shape == (dim,), f"Q检索形状错误: {Q_ret.shape}"
    assert K_ret.shape == (dim,), f"K检索形状错误: {K_ret.shape}"
    assert V_ret.shape == (dim,), f"V检索形状错误: {V_ret.shape}"
    print(f"✅ 检索形状验证通过 | Retrieval shape verification passed")
    
    # 测试多次存储（模拟记忆更新）
    # Test multiple storage (simulate memory updates)
    print("\n3. 测试多次存储... | Testing multiple storage...")
    for i in range(3):
        # 更新数据
        # Update data
        Q_list = [q * (1.0 + i * 0.1) for q in Q_list]
        memory.store(Q_list, K_list, V_list)
    
    print(f"✅ 多次存储完成，当前步骤: {memory.step}")
    
    # 测试健康度检查
    # Test health check
    print("\n4. 测试健康度检查... | Testing health check...")
    health_stats = memory.get_memory_health()
    assert 'overall_health' in health_stats, "健康度统计应包含overall_health"
    print(f"✅ 健康度统计: {health_stats}")
    
    # 测试可视化
    # Test visualization
    print("\n5. 测试可视化... | Testing visualization...")
    visualization = memory.visualize_memory_structure()
    print(visualization)
    
    # 测试重置
    # Test reset
    print("\n6. 测试重置... | Testing reset...")
    memory.reset()
    energy_after_reset = memory.get_energy_stats()
    assert memory.step == 0, f"重置后step应为0，得到{memory.step}"
    assert all(e == 1.0 for e in energy_after_reset['energy_depth0']), "重置后能量应为1.0"
    print("✅ 重置测试通过 | Reset test passed")
    
    print("\n" + "="*60)
    print("✅ 所有地质记忆测试通过 | All geological memory tests passed")
    print("="*60)
    
    return True


# ==================== 模块主入口 ====================
# ==================== Module Main Entry ====================

if __name__ == "__main__":
    # 当直接运行此文件时，执行测试
    # Execute tests when this file is run directly
    test_geological_memory()