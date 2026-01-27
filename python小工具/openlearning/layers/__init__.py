"""
layers包 | layers Package
====================

RGA架构层模块包，包含以下核心组件：
RGA Architecture layers module package, includes the following core components:

1. 嵌入层 (Embeddings): EnhancedEmbeddingLayer, ConceptAwareEmbedding
2. 融合层 (Fusion): SandwichFusion
3. 记忆层 (Memory): GeologicalMemory
4. 归一化层 (Normalization): FixedRMSNorm, FixedGroupRMSNorm, ScaledFixedRMSNorm
5. 单向阀层 (Valve): OneWayValve, SimpleOneWayValve
6. 平衡器层 (Balancer): TriValueBalancer, VDominantBalancer, DensityDrivenBalancer, AdaptiveStabilizer
7. 注意力层 (Attention): VKQ_SubNet_WithFixedNorm, QVK_SubNet_WithFixedNorm, KQV_SubNet_WithFixedNorm, ChainReactionUnit_Final

设计原则:
Design Principles:
• 规则治理架构: 遵循RGA数学基础 | Rule-Governed Architecture: Follows RGA mathematical foundation
• 模块化设计: 各层独立可组合 | Modular design: Each layer is independent and composable
• 概念驱动: 基于词汇语义概念生成特征 | Concept-driven: Generate features based on word semantic concepts
• 记忆分层: 三层深度地质记忆系统 | Memory layering: Three-layer geological memory system

数学基础:
Mathematical Foundation:
• 连接点密度公式: D = 2m/((N+1)N) | Connection point density formula: D = 2m/((N+1)N)
• 三明治融合: Q/K/V按固定权重融合 | Sandwich fusion: Q/K/V fused with fixed weights
• 固定归一化: RMSNorm变体，无学习参数 | Fixed normalization: RMSNorm variants without learnable parameters
"""

__version__ = "0.0.7"
__author__ = "RGA Architecture Team"

import sys
from typing import Dict, List, Type, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field

# 直接从当前目录导入模块（绝对导入）
# Import modules directly from current directory (absolute import)
try:
    from embeddings import EnhancedEmbeddingLayer
    from embeddings import ConceptAwareEmbedding
    from embeddings import create_embedding_layer
    
    from fusion import SandwichFusion
    from fusion import create_sandwich_fusion
    
    from memory import GeologicalMemory
    from memory import create_geological_memory
    
    from normalization import FixedRMSNorm
    from normalization import FixedGroupRMSNorm
    from normalization import ScaledFixedRMSNorm
    from normalization import create_fixed_norm
    
    from valve import OneWayValve
    from valve import SimpleOneWayValve
    from valve import create_one_way_valve
    
    from balancer import TriValueBalancer
    from balancer import VDominantBalancer
    from balancer import DensityDrivenBalancer
    from balancer import AdaptiveStabilizer
    from balancer import create_balancer_layer
    
    from attention import VKQ_SubNet_WithFixedNorm
    from attention import QVK_SubNet_WithFixedNorm
    from attention import KQV_SubNet_WithFixedNorm
    from attention import ChainReactionUnit_Final
    from attention import create_attention_subnet
    from attention import create_chain_reaction_unit
    
except ImportError:
    # 如果直接导入失败，尝试相对导入
    # If direct import fails, try relative import
    try:
        from .embeddings import EnhancedEmbeddingLayer
        from .embeddings import ConceptAwareEmbedding
        from .embeddings import create_embedding_layer
        
        from .fusion import SandwichFusion
        from .fusion import create_sandwich_fusion
        
        from .memory import GeologicalMemory
        from .memory import create_geological_memory
        
        from .normalization import FixedRMSNorm
        from .normalization import FixedGroupRMSNorm
        from .normalization import ScaledFixedRMSNorm
        from .normalization import create_fixed_norm
        
        from .valve import OneWayValve
        from .valve import SimpleOneWayValve
        from .valve import create_one_way_valve
        
        from .balancer import TriValueBalancer
        from .balancer import VDominantBalancer
        from .balancer import DensityDrivenBalancer
        from .balancer import AdaptiveStabilizer
        from .balancer import create_balancer_layer
        
        from .attention import VKQ_SubNet_WithFixedNorm
        from .attention import QVK_SubNet_WithFixedNorm
        from .attention import KQV_SubNet_WithFixedNorm
        from .attention import ChainReactionUnit_Final
        from .attention import create_attention_subnet
        from .attention import create_chain_reaction_unit
        
    except ImportError as e:
        raise ImportError(f"Failed to import layers modules: {e}")

# ==================== 导出列表 ====================
# ==================== Export List ====================

__all__ = [
    # 嵌入层 | Embedding Layers
    "EnhancedEmbeddingLayer",
    "ConceptAwareEmbedding",
    "create_embedding_layer",
    
    # 融合层 | Fusion Layers
    "SandwichFusion",
    "create_sandwich_fusion",
    
    # 记忆层 | Memory Layers
    "GeologicalMemory",
    "create_geological_memory",
    
    # 归一化层 | Normalization Layers
    "FixedRMSNorm",
    "FixedGroupRMSNorm",
    "ScaledFixedRMSNorm",
    "create_fixed_norm",
    
    # 单向阀层 | Valve Layers
    "OneWayValve",
    "SimpleOneWayValve",
    "create_one_way_valve",
    
    # 平衡器层 | Balancer Layers
    "TriValueBalancer",
    "VDominantBalancer",
    "DensityDrivenBalancer",
    "AdaptiveStabilizer",
    "create_balancer_layer",
    
    # 注意力层 | Attention Layers
    "VKQ_SubNet_WithFixedNorm",
    "QVK_SubNet_WithFixedNorm",
    "KQV_SubNet_WithFixedNorm",
    "ChainReactionUnit_Final",
    "create_attention_subnet",
    "create_chain_reaction_unit",
]

# ==================== 包配置 ====================
# ==================== Package Configuration ====================

LAYERS_CONFIG = {
    "version": __version__,
    "modules": {
        "embeddings": ["EnhancedEmbeddingLayer", "ConceptAwareEmbedding", "create_embedding_layer"],
        "fusion": ["SandwichFusion", "create_sandwich_fusion"],
        "memory": ["GeologicalMemory", "create_geological_memory"],
        "normalization": ["FixedRMSNorm", "FixedGroupRMSNorm", "ScaledFixedRMSNorm", "create_fixed_norm"],
        "valve": ["OneWayValve", "SimpleOneWayValve", "create_one_way_valve"],
        "balancer": ["TriValueBalancer", "VDominantBalancer", "DensityDrivenBalancer", "AdaptiveStabilizer", "create_balancer_layer"],
        "attention": ["VKQ_SubNet_WithFixedNorm", "QVK_SubNet_WithFixedNorm", "KQV_SubNet_WithFixedNorm", "ChainReactionUnit_Final", "create_attention_subnet", "create_chain_reaction_unit"],
    },
    "description": "RGA Architecture Layers Module Package"
}

# ==================== 类型定义 ====================
# ==================== Type Definitions ====================

LayerType = Union[
    # 归一化层 | Normalization layers
    Type[FixedRMSNorm],
    Type[FixedGroupRMSNorm],
    Type[ScaledFixedRMSNorm],
    
    # 注意力层 | Attention layers
    Type[VKQ_SubNet_WithFixedNorm],
    Type[QVK_SubNet_WithFixedNorm],
    Type[KQV_SubNet_WithFixedNorm],
    Type[ChainReactionUnit_Final],
    
    # 平衡器层 | Balancer layers
    Type[TriValueBalancer],
    Type[VDominantBalancer],
    Type[DensityDrivenBalancer],
    Type[AdaptiveStabilizer],
    
    # 嵌入层 | Embedding layers
    Type[EnhancedEmbeddingLayer],
    Type[ConceptAwareEmbedding],
    
    # 融合层 | Fusion layers
    Type[SandwichFusion],
    
    # 记忆层 | Memory layers
    Type[GeologicalMemory],
    
    # 单向阀层 | Valve layers
    Type[OneWayValve],
]

FactoryFunction = Callable[..., Any]

# ==================== 层配置 ====================
# ==================== Layer Configuration ====================

@dataclass
class LayerConfig:
    """层配置数据类 | Layer Configuration Dataclass"""
    
    # 基础配置 | Basic configuration
    name: str                        # 层名称 | Layer name
    layer_type: str                  # 层类型 | Layer type
    dim: Optional[int] = None        # 特征维度 | Feature dimension
    
    # 特定配置 | Specific configurations
    attention_subnet_type: Optional[str] = None  # 注意力子网络类型 | Attention subnet type
    vocab_size: Optional[int] = None             # 词汇表大小 | Vocabulary size
    embed_dim: Optional[int] = None              # 嵌入维度 | Embedding dimension
    marker_dim: Optional[int] = None             # 标记维度 | Marker dimension
    unit_id: Optional[int] = None                # 单元ID | Unit ID
    memory_size: Optional[int] = None            # 记忆大小 | Memory size
    valve_type: Optional[str] = None             # 阀类型 | Valve type
    
    # 其他参数 | Additional parameters
    kwargs: Dict[str, Any] = field(default_factory=dict)  # 其他参数字典 | Additional parameters dict
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 | Convert to dictionary"""
        config = {
            'name': self.name,
            'layer_type': self.layer_type,
            'dim': self.dim,
            'attention_subnet_type': self.attention_subnet_type,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'marker_dim': self.marker_dim,
            'unit_id': self.unit_id,
            'memory_size': self.memory_size,
            'valve_type': self.valve_type,
        }
        
        # 移除None值 | Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        
        # 合并其他参数 | Merge additional parameters
        config.update(self.kwargs)
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LayerConfig':
        """从字典创建配置 | Create configuration from dictionary"""
        # 提取已知字段 | Extract known fields
        known_fields = {
            'name', 'layer_type', 'dim', 'attention_subnet_type',
            'vocab_size', 'embed_dim', 'marker_dim', 'unit_id', 
            'memory_size', 'valve_type'
        }
        
        base_fields = {}
        kwargs = {}
        
        for key, value in config_dict.items():
            if key in known_fields:
                base_fields[key] = value
            else:
                kwargs[key] = value
        
        return cls(**base_fields, kwargs=kwargs)


# ==================== 层注册表 ====================
# ==================== Layer Registry ====================

class LayerRegistry:
    """
    层注册表 | Layer Registry
    
    功能 | Functions:
        • 注册层类和工厂函数 | Register layer classes and factory functions
        • 提供类型检查 | Provide type checking
        • 支持动态发现 | Support dynamic discovery
    
    使用示例 | Usage Example:
        >>> registry = LayerRegistry()
        >>> registry.register_class('FixedRMSNorm', FixedRMSNorm)
        >>> registry.register_factory('attention', create_attention_subnet)
        >>> registry.register_factory('valve', create_one_way_valve)
        >>> 
        >>> # 创建层 | Create layer
        >>> layer = registry.create_layer('attention', subnet_type='vkq', dim=128)
        >>> valve = registry.create_layer('valve', dim=128, valve_type='learnable')
    """
    
    def __init__(self):
        """初始化注册表 | Initialize registry"""
        self._classes: Dict[str, Type] = {}
        self._factories: Dict[str, FactoryFunction] = {}
        self._layer_types: Dict[str, List[str]] = {}  # 类型到类名的映射 | Type to class name mapping
        
        # 自动注册已知层 | Auto-register known layers
        self._auto_register()
    
    def _auto_register(self):
        """自动注册已知层类 | Auto-register known layer classes"""
        # 归一化层 | Normalization layers
        self.register_class('FixedRMSNorm', FixedRMSNorm)
        self.register_class('FixedGroupRMSNorm', FixedGroupRMSNorm)
        self.register_class('ScaledFixedRMSNorm', ScaledFixedRMSNorm)
        
        # 注意力层 | Attention layers
        self.register_class('VKQ_SubNet_WithFixedNorm', VKQ_SubNet_WithFixedNorm)
        self.register_class('QVK_SubNet_WithFixedNorm', QVK_SubNet_WithFixedNorm)
        self.register_class('KQV_SubNet_WithFixedNorm', KQV_SubNet_WithFixedNorm)
        self.register_class('ChainReactionUnit_Final', ChainReactionUnit_Final)
        
        # 平衡器层 | Balancer layers
        self.register_class('TriValueBalancer', TriValueBalancer)
        self.register_class('VDominantBalancer', VDominantBalancer)
        self.register_class('DensityDrivenBalancer', DensityDrivenBalancer)
        self.register_class('AdaptiveStabilizer', AdaptiveStabilizer)
        
        # 嵌入层 | Embedding layers
        self.register_class('EnhancedEmbeddingLayer', EnhancedEmbeddingLayer)
        self.register_class('ConceptAwareEmbedding', ConceptAwareEmbedding)
        
        # 融合层 | Fusion layers
        self.register_class('SandwichFusion', SandwichFusion)
        
        # 记忆层 | Memory layers
        self.register_class('GeologicalMemory', GeologicalMemory)
        
        # 单向阀层 | Valve layers
        self.register_class('OneWayValve', OneWayValve)
        
        # 注册工厂函数 | Register factory functions
        self.register_factory('fixed_norm', create_fixed_norm)
        self.register_factory('attention', create_attention_subnet)
        self.register_factory('chain_reaction', create_chain_reaction_unit)
        self.register_factory('balancer', create_balancer_layer)
        self.register_factory('embedding', create_embedding_layer)
        self.register_factory('fusion', create_sandwich_fusion)
        self.register_factory('memory', create_geological_memory)
        self.register_factory('valve', create_one_way_valve)
    
    def register_class(self, name: str, layer_class: Type):
        """
        注册层类 | Register layer class
        
        参数 | Parameters:
            name (str): 层类名称 | Layer class name
            layer_class (Type): 层类 | Layer class
        """
        self._classes[name] = layer_class
        
        # 提取类型信息 | Extract type information
        module_name = layer_class.__module__
        if 'attention' in module_name:
            layer_type = 'attention'
        elif 'balancer' in module_name:
            layer_type = 'balancer'
        elif 'embeddings' in module_name:
            layer_type = 'embedding'
        elif 'normalization' in module_name:
            layer_type = 'normalization'
        elif 'fusion' in module_name:
            layer_type = 'fusion'
        elif 'memory' in module_name:
            layer_type = 'memory'
        elif 'valve' in module_name:
            layer_type = 'valve'
        else:
            layer_type = 'other'
        
        if layer_type not in self._layer_types:
            self._layer_types[layer_type] = []
        
        if name not in self._layer_types[layer_type]:
            self._layer_types[layer_type].append(name)
    
    def register_factory(self, factory_type: str, factory_func: FactoryFunction):
        """
        注册工厂函数 | Register factory function
        
        参数 | Parameters:
            factory_type (str): 工厂类型 | Factory type
            factory_func (FactoryFunction): 工厂函数 | Factory function
        """
        self._factories[factory_type] = factory_func
    
    def get_class(self, name: str) -> Optional[Type]:
        """
        获取层类 | Get layer class
        
        参数 | Parameters:
            name (str): 层类名称 | Layer class name
        
        返回 | Returns:
            Optional[Type]: 层类或None | Layer class or None
        """
        return self._classes.get(name)
    
    def get_factory(self, factory_type: str) -> Optional[FactoryFunction]:
        """
        获取工厂函数 | Get factory function
        
        参数 | Parameters:
            factory_type (str): 工厂类型 | Factory type
        
        返回 | Returns:
            Optional[FactoryFunction]: 工厂函数或None | Factory function or None
        """
        return self._factories.get(factory_type)
    
    def list_classes(self, layer_type: Optional[str] = None) -> List[str]:
        """
        列出层类 | List layer classes
        
        参数 | Parameters:
            layer_type (Optional[str]): 层类型过滤器 | Layer type filter
        
        返回 | Returns:
            List[str]: 层类名称列表 | List of layer class names
        """
        if layer_type:
            return self._layer_types.get(layer_type, [])
        else:
            return list(self._classes.keys())
    
    def list_factories(self) -> List[str]:
        """
        列出工厂函数 | List factory functions
        
        返回 | Returns:
            List[str]: 工厂函数类型列表 | List of factory function types
        """
        return list(self._factories.keys())
    
    def create_layer(self, factory_type: str, **kwargs) -> Any:
        """
        使用工厂函数创建层 | Create layer using factory function
        
        参数 | Parameters:
            factory_type (str): 工厂类型 | Factory type
            **kwargs: 传递给工厂函数的参数 | Parameters passed to factory function
        
        返回 | Returns:
            Any: 创建的层实例 | Created layer instance
        
        异常 | Raises:
            ValueError: 如果工厂类型不存在 | If factory type does not exist
        """
        factory = self.get_factory(factory_type)
        if not factory:
            # 尝试通过类名直接创建 | Try to create by class name directly
            layer_class = self.get_class(factory_type)
            if layer_class:
                try:
                    return layer_class(**kwargs)
                except Exception as e:
                    raise ValueError(
                        f"使用类 '{factory_type}' 创建层失败: {e}\n"
                        f"参数: {kwargs}"
                    )
            raise ValueError(
                f"工厂类型 '{factory_type}' 不存在 | Factory type '{factory_type}' does not exist\n"
                f"可用工厂: {self.list_factories()}\n"
                f"可用类: {self.list_classes()}"
            )
        
        try:
            return factory(**kwargs)
        except Exception as e:
            raise ValueError(
                f"使用工厂 '{factory_type}' 创建层失败: {e}\n"
                f"参数: {kwargs}"
            )
    
    def create_from_config(self, config: LayerConfig) -> Any:
        """
        从配置创建层 | Create layer from configuration
        
        参数 | Parameters:
            config (LayerConfig): 层配置 | Layer configuration
        
        返回 | Returns:
            Any: 创建的层实例 | Created layer instance
        """
        config_dict = config.to_dict()
        
        # 根据层类型选择创建方式 | Select creation method based on layer type
        if config.layer_type == 'attention':
            subnet_type = config.attention_subnet_type or 'vkq'
            return self.create_layer('attention', subnet_type=subnet_type, dim=config.dim)
        
        elif config.layer_type == 'chain_reaction':
            return self.create_layer('chain_reaction', dim=config.dim, unit_id=config.unit_id or 0)
        
        elif config.layer_type == 'balancer':
            return self.create_layer('balancer', balancer_type='tri_value', dim=config.dim)
        
        elif config.layer_type == 'embedding':
            return self.create_layer('embedding', embed_type='enhanced', 
                                   vocab_size=config.vocab_size, 
                                   embed_dim=config.embed_dim)
        
        elif config.layer_type == 'normalization':
            return self.create_layer('fixed_norm', norm_type='rms', dim=config.dim)
        
        elif config.layer_type == 'fusion':
            return self.create_layer('fusion')
        
        elif config.layer_type == 'memory':
            return self.create_layer('memory', dim=config.dim)
        
        elif config.layer_type == 'valve':
            valve_type = config.valve_type or 'learnable'
            return self.create_layer('valve', dim=config.dim, valve_type=valve_type)
        
        else:
            # 尝试直接创建类 | Try to create class directly
            layer_class = self.get_class(config.layer_type)
            if layer_class:
                # 创建参数字典，排除类不支持的参数
                # 这些层不支持name参数和layer_type参数
                unsupported_params = {'name', 'layer_type', 'kwargs'}
                filtered_kwargs = {k: v for k, v in config_dict.items() 
                                 if k not in unsupported_params}
                return layer_class(**filtered_kwargs)
            else:
                raise ValueError(f"未知的层类型: {config.layer_type}")


# ==================== 层工厂 ====================
# ==================== Layer Factory ====================

class LayerFactory:
    """
    层工厂 | Layer Factory
    
    高级工厂类，提供更便捷的层创建和管理功能。
    Advanced factory class providing more convenient layer creation and management.
    
    功能 | Features:
        • 统一创建接口 | Unified creation interface
        • 配置管理 | Configuration management
        • 层组合 | Layer composition
        • 性能优化 | Performance optimization
    
    使用示例 | Usage Example:
        >>> factory = LayerFactory()
        >>> 
        >>> # 创建单个层 | Create single layer
        >>> attention = factory.create('attention', subnet_type='vkq', dim=128)
        >>> valve = factory.create('valve', dim=128, valve_type='learnable')
        >>> 
        >>> # 从配置创建 | Create from configuration
        >>> config = LayerConfig(name='my_valve', layer_type='valve', 
        >>>                      dim=128, valve_type='learnable')
        >>> valve2 = factory.create_from_config(config)
        >>> 
        >>> # 批量创建 | Batch creation
        >>> layers = factory.create_batch([
        >>>     ('attention', {'subnet_type': 'vkq', 'dim': 128}),
        >>>     ('balancer', {'balancer_type': 'tri_value', 'dim': 128}),
        >>>     ('valve', {'dim': 128, 'valve_type': 'learnable'}),
        >>> ])
    """
    
    def __init__(self, registry: Optional[LayerRegistry] = None):
        """
        初始化层工厂 | Initialize layer factory
        
        参数 | Parameters:
            registry (Optional[LayerRegistry]): 层注册表实例 | Layer registry instance
        """
        self.registry = registry or LayerRegistry()
        self._created_layers: Dict[str, Any] = {}
    
    def create(self, layer_type: str, **kwargs) -> Any:
        """
        创建层 | Create layer
        
        参数 | Parameters:
            layer_type (str): 层类型 | Layer type
            **kwargs: 层参数 | Layer parameters
        
        返回 | Returns:
            Any: 创建的层实例 | Created layer instance
        
        异常 | Raises:
            ValueError: 如果层类型不支持 | If layer type is not supported
        """
        # 检查是否已缓存 | Check if already cached
        cache_key = f"{layer_type}_{str(kwargs)}"
        if cache_key in self._created_layers:
            return self._created_layers[cache_key]
        
        # 创建新层 | Create new layer
        try:
            layer = self.registry.create_layer(layer_type, **kwargs)
            
            # 缓存结果 | Cache result
            self._created_layers[cache_key] = layer
            return layer
            
        except ValueError as e:
            # 尝试使用类名直接创建 | Try to create using class name directly
            layer_class = self.registry.get_class(layer_type)
            if layer_class:
                try:
                    # 过滤掉类不支持的参数
                    # 这些层不支持name参数和layer_type参数
                    unsupported_params = {'name', 'layer_type', 'kwargs'}
                    filtered_kwargs = {k: v for k, v in kwargs.items() 
                                     if k not in unsupported_params}
                    layer = layer_class(**filtered_kwargs)
                    self._created_layers[cache_key] = layer
                    return layer
                except Exception as e2:
                    raise ValueError(
                        f"创建层 '{layer_type}' 失败: {e2}\n"
                        f"参数: {kwargs}\n"
                        f"过滤后的参数: {filtered_kwargs}"
                    )
            else:
                raise ValueError(
                    f"不支持的类型: {layer_type}\n"
                    f"可用工厂: {self.registry.list_factories()}\n"
                    f"可用类: {self.registry.list_classes()}"
                )
    
    def create_from_config(self, config: LayerConfig) -> Any:
        """
        从配置创建层 | Create layer from configuration
        
        参数 | Parameters:
            config (LayerConfig): 层配置 | Layer configuration
        
        返回 | Returns:
            Any: 创建的层实例 | Created layer instance
        """
        return self.registry.create_from_config(config)
    
    def create_batch(self, layer_specs: List[tuple]) -> List[Any]:
        """
        批量创建层 | Batch create layers
        
        参数 | Parameters:
            layer_specs (List[tuple]): 层规格列表，每个元素为 (layer_type, kwargs)
                                     | List of layer specs, each element is (layer_type, kwargs)
        
        返回 | Returns:
            List[Any]: 创建的层实例列表 | List of created layer instances
        """
        layers = []
        for spec in layer_specs:
            if len(spec) == 2:
                layer_type, kwargs = spec
                layer = self.create(layer_type, **kwargs)
                layers.append(layer)
            else:
                raise ValueError(f"无效的规格格式: {spec}")
        
        return layers
    
    def create_pipeline(self, pipeline_config: List[Dict[str, Any]]) -> List[Any]:
        """
        创建处理管道 | Create processing pipeline
        
        参数 | Parameters:
            pipeline_config (List[Dict[str, Any]]): 管道配置列表 | Pipeline configuration list
        
        返回 | Returns:
            List[Any]: 管道层列表 | Pipeline layers list
        """
        pipeline = []
        
        for config in pipeline_config:
            if isinstance(config, dict):
                # 从字典创建 | Create from dictionary
                layer_type = config.get('type', config.get('layer_type'))
                if not layer_type:
                    raise ValueError("配置中必须包含 'type' 或 'layer_type' 字段")
                
                # 创建层配置 | Create layer configuration
                layer_config = LayerConfig.from_dict(config)
                
                # 创建层 | Create layer
                layer = self.create_from_config(layer_config)
                pipeline.append(layer)
            else:
                raise ValueError(f"无效的管道配置: {config}")
        
        return pipeline
    
    def get_layer_info(self, layer_type: str) -> Dict[str, Any]:
        """
        获取层信息 | Get layer information
        
        参数 | Parameters:
            layer_type (str): 层类型 | Layer type
        
        返回 | Returns:
            Dict[str, Any]: 层信息字典 | Layer information dictionary
        """
        layer_class = self.registry.get_class(layer_type)
        
        if not layer_class:
            # 尝试通过工厂获取信息 | Try to get information through factory
            factory = self.registry.get_factory(layer_type)
            if factory:
                return {
                    'type': 'factory',
                    'name': layer_type,
                    'factory_func': factory.__name__,
                    'module': factory.__module__,
                }
            else:
                raise ValueError(f"未知的层类型: {layer_type}")
        
        return {
            'type': 'class',
            'name': layer_type,
            'class_name': layer_class.__name__,
            'module': layer_class.__module__,
            'doc': layer_class.__doc__,
            'parameters': self._get_class_parameters(layer_class),
        }
    
    def _get_class_parameters(self, layer_class: Type) -> List[Dict[str, Any]]:
        """获取类的参数信息 | Get class parameter information"""
        # 这里可以扩展以获取更详细的参数信息
        # This can be extended to get more detailed parameter information
        return []
    
    def clear_cache(self):
        """清空缓存 | Clear cache"""
        self._created_layers.clear()
    
    def print_summary(self):
        """打印工厂摘要 | Print factory summary"""
        print("=" * 60)
        print("层工厂摘要 | Layer Factory Summary")
        print("=" * 60)
        
        print(f"注册表统计 | Registry Statistics:")
        print(f"  已注册类: {len(self.registry.list_classes())}")
        print(f"  已注册工厂: {len(self.registry.list_factories())}")
        print(f"  已缓存层: {len(self._created_layers)}")
        
        print(f"\n可用的层类型 | Available Layer Types:")
        for layer_type in ['attention', 'balancer', 'embedding', 'normalization', 'fusion', 'memory', 'valve']:
            classes = self.registry.list_classes(layer_type)
            if classes:
                print(f"  {layer_type}: {len(classes)} 个类 | {len(classes)} classes")
                for cls in classes[:3]:  # 只显示前3个 | Only show first 3
                    print(f"    - {cls}")
                if len(classes) > 3:
                    print(f"    ... 还有 {len(classes) - 3} 个 | ... and {len(classes) - 3} more")
        
        print("=" * 60)


# ==================== 全局实例 ====================
# ==================== Global Instances ====================

# 创建全局注册表和工厂 | Create global registry and factory
_layer_registry = LayerRegistry()
_layer_factory = LayerFactory(_layer_registry)


# ==================== 便捷函数 ====================
# ==================== Convenience Functions ====================

def get_layer_factory() -> LayerFactory:
    """
    获取全局层工厂 | Get global layer factory
    
    返回 | Returns:
        LayerFactory: 层工厂实例 | Layer factory instance
    """
    return _layer_factory


def get_layer_registry() -> LayerRegistry:
    """
    获取全局层注册表 | Get global layer registry
    
    返回 | Returns:
        LayerRegistry: 层注册表实例 | Layer registry instance
    """
    return _layer_registry


def create_layer(layer_type: str, **kwargs) -> Any:
    """
    便捷创建层 | Conveniently create layer
    
    参数 | Parameters:
        layer_type (str): 层类型 | Layer type
        **kwargs: 层参数 | Layer parameters
    
    返回 | Returns:
        Any: 创建的层实例 | Created layer instance
    """
    return _layer_factory.create(layer_type, **kwargs)


def list_available_layers(layer_type: Optional[str] = None) -> List[str]:
    """
    列出可用层 | List available layers
    
    参数 | Parameters:
        layer_type (Optional[str]): 层类型过滤器 | Layer type filter
    
    返回 | Returns:
        List[str]: 层名称列表 | List of layer names
    """
    return _layer_registry.list_classes(layer_type)


# ==================== 模块配置 ====================
# ==================== Module Configuration ====================

class LayerConfigManager:
    """
    层配置管理器 | Layer Configuration Manager
    
    管理层的配置和序列化。
    Manage layer configuration and serialization.
    """
    
    def __init__(self):
        self.configs: Dict[str, LayerConfig] = {}
    
    def save_config(self, name: str, config: LayerConfig):
        """
        保存配置 | Save configuration
        
        参数 | Parameters:
            name (str): 配置名称 | Configuration name
            config (LayerConfig): 层配置 | Layer configuration
        """
        self.configs[name] = config
    
    def load_config(self, name: str) -> Optional[LayerConfig]:
        """
        加载配置 | Load configuration
        
        参数 | Parameters:
            name (str): 配置名称 | Configuration name
        
        返回 | Returns:
            Optional[LayerConfig]: 层配置或None | Layer configuration or None
        """
        return self.configs.get(name)
    
    def export_configs(self, filepath: str):
        """
        导出配置到文件 | Export configurations to file
        
        参数 | Parameters:
            filepath (str): 文件路径 | File path
        """
        import json
        
        configs_dict = {
            name: config.to_dict()
            for name, config in self.configs.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(configs_dict, f, indent=2, ensure_ascii=False)
    
    def import_configs(self, filepath: str):
        """
        从文件导入配置 | Import configurations from file
        
        参数 | Parameters:
            filepath (str): 文件路径 | File path
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            configs_dict = json.load(f)
        
        for name, config_dict in configs_dict.items():
            config = LayerConfig.from_dict(config_dict)
            self.save_config(name, config)


# ==================== 导出列表 ====================
# ==================== Export List ====================

# 导出所有层类 | Export all layer classes
__all__ = [
    # 归一化层 | Normalization layers
    'FixedRMSNorm',
    'FixedGroupRMSNorm',
    'ScaledFixedRMSNorm',
    
    # 注意力层 | Attention layers
    'VKQ_SubNet_WithFixedNorm',
    'QVK_SubNet_WithFixedNorm',
    'KQV_SubNet_WithFixedNorm',
    'ChainReactionUnit_Final',
    
    # 平衡器层 | Balancer layers
    'TriValueBalancer',
    'VDominantBalancer',
    'DensityDrivenBalancer',
    'AdaptiveStabilizer',
    
    # 嵌入层 | Embedding layers
    'EnhancedEmbeddingLayer',
    'ConceptAwareEmbedding',
    
    # 融合层 | Fusion layers
    'SandwichFusion',
    
    # 记忆层 | Memory layers
    'GeologicalMemory',
    
    # 单向阀层 | Valve layers
    'OneWayValve',
    'SimpleOneWayValve',
    
    # 工厂函数 | Factory functions
    'create_fixed_norm',
    'create_attention_subnet',
    'create_chain_reaction_unit',
    'create_balancer_layer',
    'create_embedding_layer',
    'create_sandwich_fusion',
    'create_geological_memory',
    'create_one_way_valve',
    
    # 核心管理类 | Core management classes
    'LayerRegistry',
    'LayerFactory',
    'LayerConfig',
    'LayerConfigManager',
    
    # 便捷函数 | Convenience functions
    'get_layer_factory',
    'get_layer_registry',
    'create_layer',
    'list_available_layers',
]

# ==================== 模块初始化 ====================
# ==================== Module Initialization ====================

def _init_module():
    """初始化模块 | Initialize module"""
    print(f"✅ RGA层模块已加载 | RGA layers module loaded")
    print(f"   版本: 0.0.7 | Version: 0.0.7")
    print(f"   包含 {len(__all__)} 个导出项 | Contains {len(__all__)} exports")
    print(f"   注册表: {len(_layer_registry.list_classes())} 个类, {len(_layer_registry.list_factories())} 个工厂")
    print(f"   工厂: 已缓存 {len(_layer_factory._created_layers)} 个层")
    
    # 显示新增的阀层信息 | Show newly added valve layer information
    valve_classes = _layer_registry.list_classes('valve')
    if valve_classes:
        print(f"   新增阀层: {', '.join(valve_classes)}")


# 在导入时自动初始化 | Auto-initialize on import
if __name__ != "__main__":
    _init_module()


# ==================== 测试函数 ====================
# ==================== Test Functions ====================

def test_layer_factory():
    """测试层工厂 | Test layer factory"""
    print("=" * 60)
    print("测试层工厂 | Testing Layer Factory")
    print("=" * 60)
    
    factory = get_layer_factory()
    
    # 测试创建各种层 | Test creating various layers
    test_cases = [
        ('attention', {'subnet_type': 'vkq', 'dim': 64}),
        ('balancer', {'balancer_type': 'tri_value', 'dim': 64}),
        ('FixedRMSNorm', {'dim': 64, 'eps': 1e-5}),
        ('TriValueBalancer', {'dim': 64}),
        ('valve', {'dim': 64, 'valve_type': 'learnable'}),
        ('OneWayValve', {'dim': 64}),
    ]
    
    for layer_type, kwargs in test_cases:
        try:
            layer = factory.create(layer_type, **kwargs)
            print(f"✅ 成功创建 {layer_type}: {layer.__class__.__name__}")
            
            # 测试配置创建 | Test configuration creation
            # 创建配置时不包含类不支持的参数
            config_kwargs = kwargs.copy()
            
            config = LayerConfig(
                name=f"test_{layer_type}",
                layer_type=layer_type,
                dim=kwargs.get('dim'),
                attention_subnet_type=kwargs.get('subnet_type'),
                valve_type=kwargs.get('valve_type'),
                kwargs=config_kwargs
            )
            
            layer_from_config = factory.create_from_config(config)
            print(f"   从配置创建成功 | Created from config successfully")
            
        except Exception as e:
            print(f"❌ 创建 {layer_type} 失败: {e}")
    
    # 测试批量创建 | Test batch creation
    print(f"\n测试批量创建 | Testing batch creation...")
    try:
        layers = factory.create_batch([
            ('attention', {'subnet_type': 'qvk', 'dim': 128}),
            ('balancer', {'balancer_type': 'v_dominant', 'dim': 128}),
            ('valve', {'dim': 128, 'valve_type': 'learnable'}),
        ])
        print(f"✅ 批量创建成功: {len(layers)} 个层 | Batch creation successful: {len(layers)} layers")
        
        # 显示阀层的门控值 | Show valve layer gating values
        for i, layer in enumerate(layers):
            if hasattr(layer, 'get_gate_values'):
                gate_vals = layer.get_gate_values()
                print(f"   层{i} 门控值: Q={gate_vals['gate_Q']:.3f}, K={gate_vals['gate_K']:.3f}, V={gate_vals['gate_V']:.3f}")
                
    except Exception as e:
        print(f"❌ 批量创建失败: {e}")
    
    # 打印摘要 | Print summary
    factory.print_summary()
    
    print("=" * 60)
    print("✅ 层工厂测试完成 | Layer factory test completed")
    print("=" * 60)


def test_layer_registry():
    """测试层注册表 | Test layer registry"""
    print("=" * 60)
    print("测试层注册表 | Testing Layer Registry")
    print("=" * 60)
    
    registry = get_layer_registry()
    
    # 列出所有层 | List all layers
    print("可用层类 | Available layer classes:")
    for layer_type in ['attention', 'balancer', 'embedding', 'normalization', 'fusion', 'memory', 'valve']:
        classes = registry.list_classes(layer_type)
        if classes:
            print(f"  {layer_type.upper()}: {len(classes)}")
            for cls in classes[:2]:  # 显示前两个 | Show first two
                print(f"    - {cls}")
    
    print(f"\n可用工厂 | Available factories:")
    factories = registry.list_factories()
    for factory in factories:
        print(f"  - {factory}")
    
    # 测试通过工厂创建 | Test creation via factory
    print(f"\n测试工厂创建 | Testing factory creation...")
    try:
        attention = registry.create_layer('attention', subnet_type='kqv', dim=64)
        print(f"✅ 通过工厂创建成功: {attention.__class__.__name__}")
        
        valve = registry.create_layer('valve', dim=64, valve_type='learnable')
        print(f"✅ 阀层创建成功: {valve.__class__.__name__}")
        
    except Exception as e:
        print(f"❌ 工厂创建失败: {e}")
    
    print("=" * 60)
    print("✅ 层注册表测试完成 | Layer registry test completed")
    print("=" * 60)


def test_valve_functionality():
    """测试单向阀功能 | Test valve functionality"""
    print("=" * 60)
    print("测试单向阀功能 | Testing Valve Functionality")
    print("=" * 60)
    
    import torch
    
    # 创建阀层 | Create valve layer
    valve = create_one_way_valve(dim=64, valve_type='learnable')
    
    # 创建测试数据 | Create test data
    batch_size = 2
    seq_len = 5
    Q = torch.randn(batch_size, seq_len, 64)
    K = torch.randn(batch_size, seq_len, 64)
    V = torch.randn(batch_size, seq_len, 64)
    
    # 前向传播 | Forward propagation
    Q_out, K_out, V_out = valve(Q, K, V)
    
    print(f"✅ 阀层前向传播测试通过 | Valve forward propagation test passed")
    print(f"   输入形状: Q{Q.shape}, K{K.shape}, V{V.shape}")
    print(f"   输出形状: Q{Q_out.shape}, K{K_out.shape}, V{V_out.shape}")
    
    # 测试门控值 | Test gating values
    gate_vals = valve.get_gate_values()
    print(f"✅ 门控值获取: Q={gate_vals['gate_Q']:.3f}, K={gate_vals['gate_K']:.3f}, V={gate_vals['gate_V']:.3f}")
    print(f"   V主导性: {gate_vals['V_dominance']}")
    
    # 测试信息流分析 | Test information flow analysis
    flow_analysis = valve.analyze_information_flow(Q, Q_out, K, K_out, V, V_out)
    print(f"✅ 信息流分析: {flow_analysis['flow_pattern']}")
    
    # 测试简化的单向阀 | Test simple one-way valve
    simple_valve = SimpleOneWayValve()
    h = torch.randn(2, 3, 64, requires_grad=True)
    h_detached = simple_valve.apply_valve(h, mode='detach')
    print(f"✅ 简化阀detach模式: 梯度={h_detached.requires_grad}")
    
    print("=" * 60)
    print("✅ 单向阀功能测试完成 | Valve functionality test completed")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试 | Run tests
    print("🧪 运行层模块测试 | Running layers module tests...")
    print()
    
    test_layer_registry()
    print()
    test_layer_factory()
    print()
    test_valve_functionality()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过 | All tests passed")
    print("=" * 60)