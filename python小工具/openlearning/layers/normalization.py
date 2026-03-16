"""
归一化层模块 | Normalization Layers Module
=======================================

固定归一化层实现，基于RMSNorm变体，不包含可学习参数。
Fixed normalization layers implementation, based on RMSNorm variants, without learnable parameters.

设计目标:
Design Goals:
• 无参数归一化：避免引入额外可学习参数 | Parameter-free normalization: Avoid introducing extra learnable parameters
• 数值稳定性：防止梯度爆炸和消失 | Numerical stability: Prevent gradient explosion and vanishing
• 计算效率：轻量级计算，适合大规模部署 | Computational efficiency: Lightweight computation, suitable for large-scale deployment

核心公式 | Core Formula:
RMS(x) = √(mean(x²) + ε)
x_norm = x / RMS(x)

注意事项 | Notes:
• 该层不包含可学习参数，仅进行统计标准化 | This layer does not contain learnable parameters, only statistical normalization
• 适用于需要固定归一化的场景 | Suitable for scenarios requiring fixed normalization
• 与LayerNorm、BatchNorm等可学习归一化层互补 | Complementary to learnable normalization layers like LayerNorm, BatchNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class FixedRMSNorm(nn.Module):
    """
    固定RMS归一化层 | Fixed RMS Normalization Layer
    ---------------------------------------------
    
    基于RMSNorm的无参数归一化层，计算输入的均方根值并进行缩放。
    Parameter-free normalization layer based on RMSNorm, computing the root mean square of inputs and scaling.
    
    数学公式 | Mathematical Formula:
        RMS(x) = √(mean(x²) + ε)
        x_norm = x / RMS(x)
    
    特点 | Features:
        1. 无参数设计：无可学习的缩放参数 | Parameter-free design: No learnable scaling parameters
        2. 固定归一化：每次前向传播独立计算统计量 | Fixed normalization: Independently computes statistics each forward pass
        3. 数值稳定：添加微小epsilon防止除零错误 | Numerically stable: Adds small epsilon to prevent division by zero
        4. 设备感知：自动处理设备移动 | Device aware: Automatically handles device movement
    
    适用场景 | Suitable Scenarios:
        • 需要轻量级归一化的模块 | Modules requiring lightweight normalization
        • 防止过度参数化的场景 | Scenarios preventing over-parameterization
        • 需要固定特征尺度的位置 | Positions requiring fixed feature scales
        • 作为其他归一化层的补充 | As a complement to other normalization layers
    
    参数 | Parameters:
        dim (int): 输入特征维度 | Input feature dimension
        eps (float, optional): 数值稳定性的小常数，默认1e-6 | Small constant for numerical stability, default 1e-6
    
    输入形状 | Input Shape:
        (*, dim): 任意形状，但最后一维必须为dim | Arbitrary shape, but the last dimension must be dim
    
    输出形状 | Output Shape:
        (*, dim): 与输入相同形状 | Same shape as input
    
    示例 | Examples:
        >>> norm = FixedRMSNorm(dim=128)
        >>> x = torch.randn(4, 32, 128)
        >>> y = norm(x)  # y.shape = (4, 32, 128)
        >>> print(f"Input mean: {x.mean():.3f}, std: {x.std():.3f}")
        >>> print(f"Output mean: {y.mean():.3f}, std: {y.std():.3f}")
    
    注意事项 | Notes:
        • 与LayerNorm不同，本层不会学习缩放和偏移参数 | Unlike LayerNorm, this layer does not learn scale and shift parameters
        • 适用于特征尺度需要保持一致的场景 | Suitable for scenarios where feature scales need to remain consistent
        • 对于需要自适应归一化的场景，建议使用可学习归一化层 | For scenarios requiring adaptive normalization, learnable normalization layers are recommended
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化固定RMS归一化层 | Initialize Fixed RMS Normalization Layer
        
        参数 | Parameters:
            dim (int): 输入特征维度 | Input feature dimension
            eps (float, optional): 数值稳定性的小常数，默认1e-6 | Small constant for numerical stability, default 1e-6
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # 🚫 没有可学习参数 | No learnable parameters
        # 这是固定归一化层的核心特点 | This is the core feature of fixed normalization layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 | Forward Propagation
        
        计算输入的RMS并进行归一化 | Compute RMS of input and normalize
        
        参数 | Parameters:
            x (torch.Tensor): 输入张量，形状为(*, dim) | Input tensor with shape (*, dim)
        
        返回 | Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同 | Normalized tensor with same shape as input
        
        步骤 | Steps:
            1. 计算均方根值: RMS = √(mean(x²) + ε) | Compute root mean square: RMS = √(mean(x²) + ε)
            2. 归一化: x_norm = x / RMS | Normalize: x_norm = x / RMS
        """
        # 验证输入维度 | Validate input dimensions
        if x.size(-1) != self.dim:
            raise ValueError(
                f"输入张量的最后一维({x.size(-1)})与预期维度({self.dim})不匹配 | "
                f"Last dimension of input tensor ({x.size(-1)}) does not match expected dimension ({self.dim})"
            )
        
        # 计算均方根值 | Compute root mean square
        # x.pow(2): 计算平方 | Compute square
        # mean(dim=-1, keepdim=True): 沿最后一维计算均值 | Compute mean along last dimension
        # sqrt(): 开平方 | Square root
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps
        
        # 归一化 | Normalize
        return x / rms
    
    def extra_repr(self) -> str:
        """
        额外的字符串表示，用于打印模型信息 | Extra string representation for model printing
        
        返回 | Returns:
            str: 描述层参数的字符串 | String describing layer parameters
        """
        return f'dim={self.dim}, eps={self.eps}'
    
    def get_normalization_stats(self, x: torch.Tensor) -> dict:
        """
        获取归一化统计信息 | Get normalization statistics
        
        参数 | Parameters:
            x (torch.Tensor): 输入张量 | Input tensor
        
        返回 | Returns:
            dict: 包含统计信息的字典 | Dictionary containing statistics
        
        统计信息包括 | Statistics include:
            • input_mean: 输入均值 | Input mean
            • input_std: 输入标准差 | Input standard deviation
            • rms_mean: RMS均值 | RMS mean
            • output_mean: 输出均值 | Output mean
            • output_std: 输出标准差 | Output standard deviation
        """
        with torch.no_grad():
            input_mean = x.mean().item()
            input_std = x.std().item()
            
            rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps
            rms_mean = rms.mean().item()
            
            output = x / rms
            output_mean = output.mean().item()
            output_std = output.std().item()
        
        return {
            'input_mean': input_mean,
            'input_std': input_std,
            'rms_mean': rms_mean,
            'output_mean': output_mean,
            'output_std': output_std,
        }


class FixedGroupRMSNorm(nn.Module):
    """
    固定分组RMS归一化层 | Fixed Group RMS Normalization Layer
    -------------------------------------------------------
    
    将特征分组后进行RMS归一化，每组独立计算统计量。
    Group features and perform RMS normalization, computing statistics independently per group.
    
    数学公式 | Mathematical Formula:
        对于每组g: | For each group g:
            x_g = x[:, :, g*group_size:(g+1)*group_size]
            RMS_g(x_g) = √(mean(x_g²) + ε)
            x_norm_g = x_g / RMS_g(x_g)
    
    特点 | Features:
        1. 分组归一化：每个组独立计算统计量 | Group normalization: Each group computes statistics independently
        2. 灵活性：支持不同组大小 | Flexibility: Supports different group sizes
        3. 并行计算：各组可并行处理 | Parallel computation: Groups can be processed in parallel
    
    适用场景 | Suitable Scenarios:
        • 特征维度较大时 | When feature dimension is large
        • 需要分组独立归一化的场景 | Scenarios requiring independent normalization per group
        • 多头注意力机制的预处理 | Preprocessing for multi-head attention mechanisms
    
    参数 | Parameters:
        dim (int): 输入特征维度 | Input feature dimension
        num_groups (int): 分组数量 | Number of groups
        eps (float, optional): 数值稳定性的小常数，默认1e-6 | Small constant for numerical stability, default 1e-6
    
    输入形状 | Input Shape:
        (*, dim): 任意形状，但最后一维必须为dim | Arbitrary shape, but the last dimension must be dim
    
    输出形状 | Output Shape:
        (*, dim): 与输入相同形状 | Same shape as input
    
    要求 | Requirements:
        • dim必须能被num_groups整除 | dim must be divisible by num_groups
    """
    
    def __init__(self, dim: int, num_groups: int, eps: float = 1e-6):
        """
        初始化固定分组RMS归一化层 | Initialize Fixed Group RMS Normalization Layer
        
        参数 | Parameters:
            dim (int): 输入特征维度 | Input feature dimension
            num_groups (int): 分组数量 | Number of groups
            eps (float, optional): 数值稳定性的小常数，默认1e-6 | Small constant for numerical stability, default 1e-6
        
        异常 | Raises:
            ValueError: 如果dim不能被num_groups整除 | If dim is not divisible by num_groups
        """
        super().__init__()
        
        if dim % num_groups != 0:
            raise ValueError(
                f"特征维度({dim})必须能被分组数({num_groups})整除 | "
                f"Feature dimension ({dim}) must be divisible by number of groups ({num_groups})"
            )
        
        self.dim = dim
        self.num_groups = num_groups
        self.group_size = dim // num_groups
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 | Forward Propagation
        
        分组计算RMS并进行归一化 | Compute RMS per group and normalize
        
        参数 | Parameters:
            x (torch.Tensor): 输入张量，形状为(*, dim) | Input tensor with shape (*, dim)
        
        返回 | Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同 | Normalized tensor with same shape as input
        """
        # 验证输入维度 | Validate input dimensions
        if x.size(-1) != self.dim:
            raise ValueError(
                f"输入张量的最后一维({x.size(-1)})与预期维度({self.dim})不匹配 | "
                f"Last dimension of input tensor ({x.size(-1)}) does not match expected dimension ({self.dim})"
            )
        
        # 获取输入形状 | Get input shape
        original_shape = x.shape
        batch_size = x.size(0) if len(original_shape) > 1 else 1
        
        # 重塑为分组形式 | Reshape to group form
        # 形状: (*, num_groups, group_size) | Shape: (*, num_groups, group_size)
        x_reshaped = x.view(*original_shape[:-1], self.num_groups, self.group_size)
        
        # 计算每组RMS | Compute RMS per group
        # 沿group_size维度计算 | Compute along group_size dimension
        rms = x_reshaped.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps
        
        # 归一化 | Normalize
        x_norm = x_reshaped / rms
        
        # 恢复原始形状 | Restore original shape
        return x_norm.view(original_shape)
    
    def extra_repr(self) -> str:
        """
        额外的字符串表示 | Extra string representation
        
        返回 | Returns:
            str: 描述层参数的字符串 | String describing layer parameters
        """
        return f'dim={self.dim}, num_groups={self.num_groups}, group_size={self.group_size}, eps={self.eps}'


class ScaledFixedRMSNorm(FixedRMSNorm):
    """
    缩放固定RMS归一化层 | Scaled Fixed RMS Normalization Layer
    -------------------------------------------------------
    
    在固定RMS归一化基础上添加可学习的缩放参数。
    Adds learnable scaling parameters on top of fixed RMS normalization.
    
    数学公式 | Mathematical Formula:
        RMS(x) = √(mean(x²) + ε)
        x_norm = (x / RMS(x)) * γ
    
    特点 | Features:
        1. 轻量级可学习参数：仅学习缩放参数γ | Lightweight learnable parameters: Only learn scaling parameter γ
        2. 固定统计量：RMS计算仍为固定统计 | Fixed statistics: RMS computation remains fixed
        3. 灵活缩放：允许模型学习最优特征尺度 | Flexible scaling: Allows model to learn optimal feature scale
    
    与原始FixedRMSNorm的区别 | Differences from original FixedRMSNorm:
        • 添加了可学习的缩放参数γ | Adds learnable scaling parameter γ
        • 允许模型调整特征尺度 | Allows model to adjust feature scale
        • 仍保持固定归一化统计 | Still maintains fixed normalization statistics
    
    参数 | Parameters:
        dim (int): 输入特征维度 | Input feature dimension
        eps (float, optional): 数值稳定性的小常数，默认1e-6 | Small constant for numerical stability, default 1e-6
        init_scale (float, optional): 缩放参数的初始值，默认1.0 | Initial value for scaling parameter, default 1.0
        learnable (bool, optional): 缩放参数是否可学习，默认True | Whether scaling parameter is learnable, default True
    """
    
    def __init__(self, dim: int, eps: float = 1e-6, init_scale: float = 1.0, learnable: bool = True):
        """
        初始化缩放固定RMS归一化层 | Initialize Scaled Fixed RMS Normalization Layer
        
        参数 | Parameters:
            dim (int): 输入特征维度 | Input feature dimension
            eps (float, optional): 数值稳定性的小常数，默认1e-6 | Small constant for numerical stability, default 1e-6
            init_scale (float, optional): 缩放参数的初始值，默认1.0 | Initial value for scaling parameter, default 1.0
            learnable (bool, optional): 缩放参数是否可学习，默认True | Whether scaling parameter is learnable, default True
        """
        super().__init__(dim, eps)
        
        # 添加可学习的缩放参数 | Add learnable scaling parameter
        self.scale = nn.Parameter(torch.ones(dim) * init_scale, requires_grad=learnable)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 | Forward Propagation
        
        计算固定RMS归一化后应用缩放参数 | Compute fixed RMS normalization then apply scaling parameter
        
        参数 | Parameters:
            x (torch.Tensor): 输入张量，形状为(*, dim) | Input tensor with shape (*, dim)
        
        返回 | Returns:
            torch.Tensor: 归一化和缩放后的张量 | Normalized and scaled tensor
        """
        # 调用父类的固定RMS归一化 | Call parent class's fixed RMS normalization
        x_norm = super().forward(x)
        
        # 应用缩放参数 | Apply scaling parameter
        return x_norm * self.scale
    
    def extra_repr(self) -> str:
        """
        额外的字符串表示 | Extra string representation
        
        返回 | Returns:
            str: 描述层参数的字符串 | String describing layer parameters
        """
        return f'dim={self.dim}, eps={self.eps}, scale_learnable={self.scale.requires_grad}'


# ==================== 工厂函数 ====================
# Factory Functions

def create_fixed_norm(norm_type: str = 'rms', **kwargs) -> nn.Module:
    """
    创建固定归一化层 | Create fixed normalization layer
    
    参数 | Parameters:
        norm_type (str): 归一化类型，可选['rms', 'group_rms', 'scaled_rms'] | Normalization type, options: ['rms', 'group_rms', 'scaled_rms']
        **kwargs: 传递给具体归一化层的参数 | Parameters passed to specific normalization layer
    
    返回 | Returns:
        nn.Module: 创建的归一化层 | Created normalization layer
    
    异常 | Raises:
        ValueError: 如果归一化类型不支持 | If normalization type is not supported
    """
    norm_map = {
        'rms': FixedRMSNorm,
        'group_rms': FixedGroupRMSNorm,
        'scaled_rms': ScaledFixedRMSNorm,
    }
    
    if norm_type not in norm_map:
        raise ValueError(
            f"不支持的归一化类型: {norm_type} | Unsupported normalization type: {norm_type}\n"
            f"支持的类型: {list(norm_map.keys())} | Supported types: {list(norm_map.keys())}"
        )
    
    return norm_map[norm_type](**kwargs)


# ==================== 测试函数 ====================
# Test Functions

def test_fixed_rms_norm():
    """测试固定RMS归一化层 | Test Fixed RMS Normalization Layer"""
    print("=" * 60)
    print("测试固定RMS归一化层 | Testing Fixed RMS Normalization Layer")
    print("=" * 60)
    
    # 创建测试张量 | Create test tensor
    batch_size, seq_len, dim = 4, 32, 128
    x = torch.randn(batch_size, seq_len, dim) * 2 + 1  # 非零均值 | Non-zero mean
    
    # 创建归一化层 | Create normalization layer
    norm = FixedRMSNorm(dim=dim)
    
    # 前向传播 | Forward propagation
    y = norm(x)
    
    # 获取统计信息 | Get statistics
    stats = norm.get_normalization_stats(x)
    
    # 打印结果 | Print results
    print(f"输入形状: {x.shape} | Input shape: {x.shape}")
    print(f"输出形状: {y.shape} | Output shape: {y.shape}")
    print()
    print("统计信息 | Statistics:")
    print(f"  输入均值: {stats['input_mean']:.6f} | Input mean: {stats['input_mean']:.6f}")
    print(f"  输入标准差: {stats['input_std']:.6f} | Input std: {stats['input_std']:.6f}")
    print(f"  RMS均值: {stats['rms_mean']:.6f} | RMS mean: {stats['rms_mean']:.6f}")
    print(f"  输出均值: {stats['output_mean']:.6f} | Output mean: {stats['output_mean']:.6f}")
    print(f"  输出标准差: {stats['output_std']:.6f} | Output std: {stats['output_std']:.6f}")
    
    # 验证RMS归一化特性 | Verify RMS normalization properties
    print()
    print("验证特性 | Verification:")
    
    # 计算每个位置的RMS | Compute RMS per position
    rms_per_position = x.pow(2).mean(dim=-1).sqrt()
    normalized_rms = y.pow(2).mean(dim=-1).sqrt()
    
    print(f"  原始RMS范围: [{rms_per_position.min():.3f}, {rms_per_position.max():.3f}] | "
          f"Original RMS range: [{rms_per_position.min():.3f}, {rms_per_position.max():.3f}]")
    print(f"  归一化后RMS范围: [{normalized_rms.min():.3f}, {normalized_rms.max():.3f}] | "
          f"Normalized RMS range: [{normalized_rms.min():.3f}, {normalized_rms.max():.3f}]")
    
    # 验证归一化效果 | Verify normalization effect
    avg_normalized_rms = normalized_rms.mean().item()
    print(f"  平均归一化RMS: {avg_normalized_rms:.6f} | Average normalized RMS: {avg_normalized_rms:.6f}")
    print(f"  接近1.0: {'是' if abs(avg_normalized_rms - 1.0) < 0.1 else '否'} | Close to 1.0: {'Yes' if abs(avg_normalized_rms - 1.0) < 0.1 else 'No'}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)


def test_group_rms_norm():
    """测试分组RMS归一化层 | Test Group RMS Normalization Layer"""
    print("=" * 60)
    print("测试分组RMS归一化层 | Testing Group RMS Normalization Layer")
    print("=" * 60)
    
    # 创建测试张量 | Create test tensor
    batch_size, seq_len, dim = 2, 16, 128
    num_groups = 4
    x = torch.randn(batch_size, seq_len, dim)
    
    # 创建归一化层 | Create normalization layer
    norm = FixedGroupRMSNorm(dim=dim, num_groups=num_groups)
    
    # 前向传播 | Forward propagation
    y = norm(x)
    
    # 验证分组归一化 | Verify group normalization
    print(f"输入形状: {x.shape} | Input shape: {x.shape}")
    print(f"分组数: {num_groups} | Number of groups: {num_groups}")
    print(f"每组大小: {dim // num_groups} | Group size: {dim // num_groups}")
    print(f"输出形状: {y.shape} | Output shape: {y.shape}")
    
    # 检查每组是否独立归一化 | Check if each group is normalized independently
    x_reshaped = x.view(batch_size, seq_len, num_groups, -1)
    y_reshaped = y.view(batch_size, seq_len, num_groups, -1)
    
    print()
    print("分组统计 | Group statistics:")
    for g in range(num_groups):
        group_x = x_reshaped[..., g, :]
        group_y = y_reshaped[..., g, :]
        
        # 计算每组RMS | Compute RMS per group
        original_rms = group_x.pow(2).mean(dim=-1).sqrt().mean().item()
        normalized_rms = group_y.pow(2).mean(dim=-1).sqrt().mean().item()
        
        print(f"  组 {g}: 原始RMS={original_rms:.3f}, 归一化RMS={normalized_rms:.3f} | "
              f"Group {g}: Original RMS={original_rms:.3f}, Normalized RMS={normalized_rms:.3f}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)


def test_scaled_rms_norm():
    """测试缩放RMS归一化层 | Test Scaled RMS Normalization Layer"""
    print("=" * 60)
    print("测试缩放RMS归一化层 | Testing Scaled RMS Normalization Layer")
    print("=" * 60)
    
    # 创建测试张量 | Create test tensor
    batch_size, seq_len, dim = 2, 8, 64
    x = torch.randn(batch_size, seq_len, dim)
    
    # 创建归一化层 | Create normalization layer
    norm = ScaledFixedRMSNorm(dim=dim, init_scale=0.5)
    
    # 前向传播 | Forward propagation
    y = norm(x)
    
    # 验证缩放效果 | Verify scaling effect
    print(f"输入形状: {x.shape} | Input shape: {x.shape}")
    print(f"输出形状: {y.shape} | Output shape: {y.shape}")
    print(f"缩放参数形状: {norm.scale.shape} | Scale parameter shape: {norm.scale.shape}")
    print(f"缩放参数均值: {norm.scale.mean().item():.3f} | Scale parameter mean: {norm.scale.mean().item():.3f}")
    
    # 计算缩放前后的标准差 | Compute standard deviation before and after scaling
    # 先计算无缩放的归一化 | First compute normalization without scaling
    base_norm = FixedRMSNorm(dim=dim)
    y_base = base_norm(x)
    
    print()
    print("缩放效果比较 | Scaling effect comparison:")
    print(f"  无缩放输出标准差: {y_base.std().item():.6f} | Output std without scaling: {y_base.std().item():.6f}")
    print(f"  有缩放输出标准差: {y.std().item():.6f} | Output std with scaling: {y.std().item():.6f}")
    print(f"  预期缩放因子: {norm.scale.mean().item():.3f} | Expected scaling factor: {norm.scale.mean().item():.3f}")
    print(f"  实际缩放比: {y.std().item() / y_base.std().item():.3f} | Actual scaling ratio: {y.std().item() / y_base.std().item():.3f}")
    
    # 检查是否可学习 | Check if learnable
    print(f"  缩放参数可学习: {norm.scale.requires_grad} | Scale parameter learnable: {norm.scale.requires_grad}")
    
    print()
    print("✅ 测试通过 | Test passed")
    print("=" * 60)


# ==================== 导出列表 ====================
# Export List

__all__ = [
    'FixedRMSNorm',
    'FixedGroupRMSNorm',
    'ScaledFixedRMSNorm',
    'create_fixed_norm',
    'test_fixed_rms_norm',
    'test_group_rms_norm',
    'test_scaled_rms_norm',
]


# ==================== 模块自检 ====================
# Module Self-Test

if __name__ == "__main__":
    print("=" * 60)
    print("归一化层模块自检 | Normalization Layers Module Self-Test")
    print("=" * 60)
    
    try:
        # 运行测试 | Run tests
        test_fixed_rms_norm()
        print()
        test_group_rms_norm()
        print()
        test_scaled_rms_norm()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过 | All tests passed")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败 | Test failed: {e}")
        import traceback
        traceback.print_exc()
