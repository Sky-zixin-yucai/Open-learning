"""
配置管理模块 | Configuration Management Module
============================================

管理RGA架构的所有配置参数和验证逻辑。
Manages all configuration parameters and validation logic for RGA architecture.

主要功能:
Main Functions:
1. 定义配置数据结构 | Define configuration data structures
2. 参数验证和默认值设置 | Parameter validation and default value setting
3. 配置序列化和反序列化 | Configuration serialization and deserialization

设计原则:
Design Principles:
• 配置优先于硬编码 | Configuration over hardcoding
• 运行时验证避免错误 | Runtime validation to prevent errors
• 向后兼容的配置变更 | Backward-compatible configuration changes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

@dataclass
class RGAConfig:
    """
    RGA配置类 - 规则治理架构的配置管理中心
    RGA Configuration Class - Configuration management center for Rule-Governed Architecture
    ==============================================================================
    
    功能: 管理RGA架构的所有核心参数，确保配置的一致性和有效性。
    Purpose: Manage all core parameters of RGA architecture to ensure configuration consistency and validity.
    
    核心特性:
    Core Features:
    - 无硬编码默认值，允许外部传入 | No hardcoded defaults, allow external input
    - 严格的参数验证 | Strict parameter validation
    - 自动维度调整 | Automatic dimension adjustment
    - 公式参数管理 | Formula parameter management
    
    使用示例:
    Usage Example:
        # 创建配置实例 | Create configuration instance
        config = RGAConfig(vocab_size=20000, dim=512)
        
        # 验证配置 | Validate configuration
        config.validate()
        
        # 序列化配置 | Serialize configuration
        config_dict = config.to_dict()
    """
    
    # ==================== 基础配置 | Basic Configuration ====================
    vocab_size: int = 10000       # 词汇表大小 | Vocabulary size
    dim: int = 32                 # 模型维度（必须为偶数）| Model dimension (must be even)
    
    # ==================== 架构配置 | Architecture Configuration ====================
    num_units: int = 3           # 链式反应单元数量（必须为3）| Number of chain reaction units (must be 3)
    geo_depth: int = 3           # 地质记忆深度（必须为3）| Geological memory depth (must be 3)
    max_cycles: int = 3          # 最大持续思考循环数 | Maximum continuous thinking cycles
    
    # ==================== 公式参数 | Formula Parameters ====================
    phase_threshold: float = 0.43    # 相变检测阈值 | Phase transition detection threshold
    v_scaling_factor: float = 1.0    # V值缩放因子 | V-value scaling factor
    min_Q_concepts: float = 0.01     # 最小Q概念密度 | Minimum Q concept density
    history_length: int = 10         # 历史记录长度 | History record length
    density_method: str = 'static'   # 密度计算方法 | Density calculation method
    
    def __post_init__(self):
        """
        配置初始化后验证
        Post-initialization configuration validation
        """
        self.validate()
    
    def validate(self):
        """
        验证配置参数的有效性
        Validate the validity of configuration parameters
        
        验证规则:
        Validation Rules:
        1. 维度必须为偶数（便于位置编码） | Dimensions must be even (for positional encoding)
        2. 架构核心参数必须为3 | Core architecture parameters must be 3
        3. 公式参数必须在合理范围内 | Formula parameters must be within reasonable range
        
        异常 | Raises:
        ------------
        ValueError
            当配置参数无效时抛出 | Raised when configuration parameters are invalid
        """
        # 验证维度 | Validate dimension
        if self.dim % 2 != 0:
            raise ValueError(f"维度必须为偶数，当前为{self.dim} | Dimension must be even, current: {self.dim}")
        
        # 验证架构参数 | Validate architecture parameters
        if self.num_units != 3:
            raise ValueError(f"链式反应单元数量必须为3，当前为{self.num_units} | Number of chain reaction units must be 3, current: {self.num_units}")
        
        if self.geo_depth != 3:
            raise ValueError(f"地质记忆深度必须为3，当前为{self.geo_depth} | Geological memory depth must be 3, current: {self.geo_depth}")
        
        # 验证公式参数范围 | Validate formula parameter range
        if not 0 <= self.phase_threshold <= 1:
            raise ValueError(f"相变阈值必须在0-1之间，当前为{self.phase_threshold} | Phase threshold must be between 0-1, current: {self.phase_threshold}")
        
        if self.v_scaling_factor <= 0:
            raise ValueError(f"V值缩放因子必须大于0，当前为{self.v_scaling_factor} | V scaling factor must be greater than 0, current: {self.v_scaling_factor}")
        
        if self.history_length <= 0:
            raise ValueError(f"历史记录长度必须大于0，当前为{self.history_length} | History length must be greater than 0, current: {self.history_length}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典（用于序列化）
        Convert configuration to dictionary (for serialization)
        
        返回 | Returns:
        ------------
        Dict[str, Any]
            配置字典，包含所有公共属性 | Configuration dictionary containing all public attributes
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RGAConfig':
        """
        从字典创建配置实例
        Create configuration instance from dictionary
        
        参数 | Parameters:
        ------------
        config_dict : Dict[str, Any]
            配置字典 | Configuration dictionary
        
        返回 | Returns:
        ------------
        RGAConfig
            新的配置实例 | New configuration instance
        
        示例 | Example:
        ------------
            config_dict = {"vocab_size": 20000, "dim": 512}
            config = RGAConfig.from_dict(config_dict)
        """
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """
        更新配置参数
        Update configuration parameters
        
        参数 | Parameters:
        ------------
        **kwargs : dict
            要更新的参数键值对 | Key-value pairs of parameters to update
        
        异常 | Raises:
        ------------
        ValueError
            当参数不存在时抛出 | Raised when parameter does not exist
        
        示例 | Example:
        ------------
            config.update(vocab_size=30000, dim=256)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"配置参数'{key}'不存在 | Configuration parameter '{key}' does not exist")
        
        # 验证更新后的配置
        self.validate()
    
    def __str__(self) -> str:
        """
        返回配置的字符串表示
        Return string representation of configuration
        
        返回 | Returns:
        ------------
        str
            格式化的配置信息 | Formatted configuration information
        """
        lines = ["RGA Configuration | RGA 配置", "=" * 40]
        for key, value in self.to_dict().items():
            # 添加中文说明
            descriptions = {
                "vocab_size": "词汇表大小 | Vocabulary size",
                "dim": "模型维度 | Model dimension",
                "num_units": "链式反应单元 | Chain reaction units",
                "geo_depth": "地质记忆深度 | Geological memory depth",
                "max_cycles": "最大循环数 | Maximum cycles",
                "phase_threshold": "相变阈值 | Phase threshold",
                "v_scaling_factor": "V值缩放因子 | V scaling factor",
                "min_Q_concepts": "最小Q概念 | Minimum Q concepts",
                "history_length": "历史长度 | History length",
                "density_method": "密度方法 | Density method"
            }
            
            desc = descriptions.get(key, key)
            lines.append(f"{desc}: {value}")
        
        return "\n".join(lines)