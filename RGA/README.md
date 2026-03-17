# OpenBoat – 规则治理架构 (RGA)

[![PyPI version](https://img.shields.io/pypi/v/openboat.svg)](https://pypi.org/project/openboat/)
[![Python Versions](https://img.shields.io/pypi/pyversions/openboat.svg)](https://pypi.org/project/openboat/)
[![License](https://img.shields.io/pypi/l/openboat.svg)](https://github.com/Sky-zixin-yucai/Open-learning/blob/main/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c?logo=pytorch)](https://pytorch.org)

**OpenBoat** 是**规则治理架构 (RGA)** 的官方实现。RGA 是一个创新的神经网络框架，其核心设计基于 **三值（Q, K, V）动态平衡** 与 **地质记忆** 机制，旨在模拟认知过程中的**相变**、**持续思考**和**长期记忆**。

---

## ✨ 核心特性

- **密度公式驱动**：所有计算基于连接点密度公式，不依赖传统归一化。
- **V值主导**：V（Value）在Q、K、V三值中起主导作用，调控信息流动。
- **地质记忆**：三层深度 × 三层时间 × 三个V子值的记忆结构，模拟地层的沉积与衰退。
- **链式反应单元**：三个子网络按不同顺序处理Q、K、V，实现多视角融合。
- **单向阀**：精确控制信息流，保护核心记忆。
- **三明治融合**：将深层地质记忆、当前状态与原始输入加权融合，实现持续学习。
- **伪装保存**：支持将模型保存为标准Transformer格式（如BERT），便于集成到现有Pipeline。
- **命令行接口**：提供 `openboat` 命令，一键训练、测试和查看环境信息。

---

## 📦 安装

### 从 PyPI 安装（推荐）
```bash
pip install openboat
```

### 从源码安装
```bash
git clone https://github.com/Sky-zixin-yucai/Open-learning.git
cd Open-learning
pip install .
```

### 安装开发版本（包含测试与格式化工具）
```bash
pip install openboat[dev]
```

---

## 🚀 快速开始

### 1. 准备数据（LCCC格式）
```json
[
  ["你好", "你好呀！"],
  ["今天天气怎么样？", "晴天", "适合出去玩"]
]
```

### 2. 训练模型
```bash
openboat train --data_path ./data.json --output_dir ./rga_output
```

您也可以使用配置文件：
```bash
openboat train --config ./config.json
```

配置文件示例 (`config.json`)：
```json
{
    "data_path": "./data.json",
    "output_dir": "./rga_output",
    "seq_length": 128,
    "batch_size": 32,
    "embed_dim": 512,
    "learning_rate": 3e-4,
    "num_epochs": 20,
    "device": "cuda"
}
```

### 3. 运行测试
```bash
openboat test
```

### 4. 查看环境信息
```bash
openboat info
```

### 5. 在Python脚本中使用
```python
from openboat.rga import RuleGovernedArchitecture

model = RuleGovernedArchitecture.from_pretrained('./rga_output/pretrained_model')
# ... 进行推理
```

---

## 🧩 模块概览

| 模块文件          | 主要类                          | 功能描述                     |
|-------------------|----------------------------------|------------------------------|
| `config.py`       | `RGAConfig`, `TriValueBalancer` | 配置管理与三值平衡器          |
| `memory.py`       | `GeologicalMemory`, `SandwichFusion` | 地质记忆存储与融合层      |
| `metrics.py`      | `OneWayValve`, `EnhancedEmbeddingLayer` | 单向阀与增强嵌入          |
| `valve.py`        | `ChainReactionUnit_Final`       | 链式反应单元（含固定RMSNorm） |
| `rga.py`          | `RuleGovernedArchitecture`, `SmartTextDataset`, `AdvancedConstrainedArchitectureTrainer` | 主架构、数据集与训练器 |
| `zixin.py`        | `test_complete_architecture`     | 完整架构测试与演示            |
| `cli.py`          | `main`                           | 命令行入口                    |

---

## ⚙️ 配置参数（`RGAConfig`）

| 参数名               | 默认值 | 说明                         |
|----------------------|--------|------------------------------|
| `vocab_size`         | 10000  | 词汇表大小                   |
| `dim`                | 32     | 模型维度（必须为偶数）       |
| `num_units`          | 3      | 链式反应单元数量             |
| `geo_depth`          | 3      | 地质记忆深度                 |
| `max_loop`           | 3      | 持续思考循环次数             |
| `phase_threshold`    | 0.80   | 相变检测阈值                 |
| `v_scaling_factor`   | 1.0    | V值缩放因子                  |

---

## 💾 保存与加载（伪装Transformer格式）

```python
# 保存为BERT兼容格式
model.save_pretrained('./my_boat_model')

# 从BERT兼容格式加载
model = RuleGovernedArchitecture.from_pretrained('./my_boat_model')
```

---

## 📝 数据格式说明

`SmartTextDataset` 默认支持 LCCC 格式的 JSON 文件，即包含多个对话的列表，每个对话是一个字符串列表。数据集会自动检测文本类型（字符级/词级）并构建词汇表。

---

## ⚠️ 注意事项

- `dim` 必须为偶数。
- 训练器默认启用自动混合精度（AMP），如需关闭可在配置中添加 `'use_mixed_precision': False`。
- 若显存不足，可减小 `batch_size` 或手动修改 `gradient_accumulation_steps`（代码中默认=1）。
- 从标准格式加载时，仅恢复权重，地质记忆等状态需通过前向传播重新填充。

---

## 🤝 贡献

欢迎提交 Issue 或 Pull Request。请确保代码通过 `black` 和 `isort` 格式化，并添加相应测试。

---

## 📄 许可证

本项目基于 Apache 2.0 许可证开源，详情请参阅 [LICENSE](LICENSE) 文件。

---

## 📬 联系我们

- **邮箱**：[skyzixinyucai@126.com](mailto:skyzixinyucai@126.com)
- **项目主页**：[GitHub](https://github.com/Sky-zixin-yucai/Open-learning)

**🎉 祝您使用 OpenBoat 探索认知智能的新边界！**
```