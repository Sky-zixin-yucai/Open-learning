# Open-learning / RGA 规则治理架构

[English](#english) | [中文](#中文)

---

## English

### Project Description

**Open-learning** is an open-source RGA (Rule-Governed Architecture) integration framework designed for intelligent text processing and deep learning model training. This project implements a novel rule-governed architecture that combines traditional deep learning with rule-based reasoning, providing a unique approach to neural network design.

**Key Features**:
- **RGA Integrator**: Core implementation of Rule-Governed Architecture with dynamic V-value regulation
- **Smart Text Dataset**: Intelligent text preprocessing with automatic vocabulary building
- **Advanced Training System**: Visual training progress monitoring with comprehensive metrics
- **Disguise Save/Load**: Save models in Transformer-compatible format for interoperability
- **Memory Optimization**: Automatic mixed precision and gradient checkpointing

### Installation

#### Option 1: Install from PyPI (Recommended)

```bash
pip install openlearning
```

#### Option 2: Clone from GitHub

```bash
git clone https://github.com/Sky-zixin-yucai/Open-learning.git
cd Open-learning
pip install -e .
```

### Quick Start

#### Basic Usage

```python
from openlearning import RGAIntegrator, RGAConfig
import torch

# Create model configuration
config = RGAConfig(
    vocab_size=10000,
    dim=256,
    num_units=3  # RGA requires exactly 3 chain reaction units
)

# Initialize model
model = RGAIntegrator(config)

# Create sample input
input_ids = torch.randint(0, 10000, (1, 32))

# Forward pass
output = model(input_ids, num_cycles=3)
print(f"Logits shape: {output['logits'].shape}")
print(f"V value mean: {output['V_stats']['V_fused_mean']:.4f}")
```

#### Training Example

```python
from openlearning import train_zixin_complete_model

# Standard training
model, history = train_zixin_complete_model(config_mode='standard')

# Quick testing mode
from openlearning import quick_test_mode
quick_test_mode()
```

### Project Structure

```
openlearning/
├── __init__.py          # Module initialization and exports
├── yucai.py            # RGA integrator core implementation
├── nn.py               # Neural network components and trainers
└── pyproject.toml      # Project configuration and dependencies
```

### Main Components

1. **RGAIntegrator** - Core RGA implementation with:
   - Chain reaction units with V-value regulation
   - Geological memory for multi-layer storage
   - Sandwich fusion for deep information integration
   - Formula-based validation system

2. **SmartTextDataset** - Intelligent dataset with:
   - Automatic character/word level detection
   - Vocabulary building with coverage statistics
   - Chinese text processing support

3. **AdvancedConstrainedArchitectureTrainer** - Training system with:
   - Visual progress monitoring
   - V-value health checking
   - Automatic vocabulary saving
   - Pretrained model format export

### Examples

Check the `example_usage()` function in `yucai.py` for comprehensive examples including:
- Model initialization and inference
- Performance benchmarking
- Model saving and loading
- Comprehensive testing suite

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.19.0

### Development

Install development dependencies:
```bash
pip install openlearning[dev]
```

### License

Apache 2.0 License - See LICENSE file for details.

### Contact

- Author: Open-learning Team
- Email: skyzixinyucai@126.com
- GitHub: [https://github.com/Sky-zixin-yucai/Open-learning.git](https://github.com/Sky-zixin-yucai/Open-learning.git)

---

## 中文

### 项目描述

**Open-learning** 是一个开源的 RGA（规则治理架构）集成框架，专门用于智能文本处理和深度学习模型训练。本项目实现了一种新颖的规则治理架构，将传统深度学习与基于规则的推理相结合，提供了独特的神经网络设计方法。

**核心特性**：
- **RGA 集成器**：规则治理架构核心实现，支持动态V值调控
- **智能文本数据集**：自动词汇表构建的智能文本预处理
- **高级训练系统**：可视化训练进度监控，包含全面指标
- **伪装保存/加载**：以Transformer兼容格式保存模型，实现互操作性
- **内存优化**：自动混合精度和梯度检查点技术

### 安装方法

#### 方案一：通过PyPI安装（推荐）

```bash
pip install openlearning
```

#### 方案二：从GitHub克隆

```bash
git clone https://github.com/Sky-zixin-yucai/Open-learning.git
cd Open-learning
pip install -e .
```

### 快速开始

#### 基础使用

```python
from openlearning import RGAIntegrator, RGAConfig
import torch

# 创建模型配置
config = RGAConfig(
    vocab_size=10000,
    dim=256,
    num_units=3  # RGA需要恰好3个链式反应单元
)

# 初始化模型
model = RGAIntegrator(config)

# 创建示例输入
input_ids = torch.randint(0, 10000, (1, 32))

# 前向传播
output = model(input_ids, num_cycles=3)
print(f"Logits形状: {output['logits'].shape}")
print(f"V值均值: {output['V_stats']['V_fused_mean']:.4f}")
```

#### 训练示例

```python
from openlearning import train_zixin_complete_model

# 标准训练
model, history = train_zixin_complete_model(config_mode='standard')

# 快速测试模式
from openlearning import quick_test_mode
quick_test_mode()
```

### 项目结构

```
openlearning/
├── __init__.py          # 模块初始化和导出
├── yucai.py            # RGA集成器核心实现
├── nn.py               # 神经网络组件和训练器
└── pyproject.toml      # 项目配置和依赖管理
```

### 主要组件

1. **RGAIntegrator** - RGA核心实现包含：
   - 带V值调控的链式反应单元
   - 多层级存储的地质记忆系统
   - 深度信息融合的三明治融合层
   - 基于公式的验证系统

2. **SmartTextDataset** - 智能数据集包含：
   - 自动字符/词级别检测
   - 带覆盖统计的词汇表构建
   - 中文文本处理支持

3. **AdvancedConstrainedArchitectureTrainer** - 训练系统包含：
   - 可视化进度监控
   - V值健康检查
   - 自动词汇表保存
   - 预训练模型格式导出

### 示例

查看 `yucai.py` 中的 `example_usage()` 函数获取全面示例，包括：
- 模型初始化和推理
- 性能基准测试
- 模型保存和加载
- 全面测试套件

### 系统要求

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.19.0

### 开发环境

安装开发依赖：
```bash
pip install openlearning[dev]
```

### 许可证

Apache 2.0 许可证 - 详见 LICENSE 文件。

### 联系信息

- 作者：Open-learning 团队
- 邮箱：skyzixinyucai@126.com
- GitHub：[https://github.com/Sky-zixin-yucai/Open-learning.git](https://github.com/Sky-zixin-yucai/Open-learning.git)