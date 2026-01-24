```markdown
# 规则治理架构 | Rule-Governed Architecture (RGA)
==============================================

一个创新的神经网络架构，提供规则驱动的学习和推理能力。
An innovative neural network architecture providing rule-driven learning and reasoning capabilities.

## 模块结构 | Module Structure
```
openlearning/
├── core/           - 核心引擎和配置管理 | Core engine and configuration management
├── layers/         - 神经网络层实现 | Neural network layer implementations
└── integration/    - 集成和伪装系统 | Integration and disguise system
```

## 主要功能 | Key Features
- **规则驱动的模型更新** | Rule-driven model updates
  - 基于可配置规则的动态参数调整
  - 实时响应输入特征的规则触发

- **实时状态监控和相变检测** | Real-time state monitoring and phase transition detection
  - 持续追踪16个关键性能指标
  - 动态阈值相变检测与预警

- **伪装保存/加载系统** | Disguise save/load system
  - 加密模型参数存储
  - 安全模型分发与验证

- **多网络融合和信息流控制** | Multi-network fusion and information flow control
  - 支持三种网络融合策略：串联、并联、分层
  - 基于密度的智能信息阀门控制

- **完整的中英文双语接口** | Complete bilingual Chinese-English interface
  - API文档双语支持
  - 错误信息与日志双语输出

## 设计原则 | Design Principles
- **模块化**：每个组件独立可替换 | Modular: Each component independently replaceable
- **可扩展**：易于添加新功能 | Extensible: Easy to add new features
- **高性能**：优化内存和计算 | High performance: Optimized memory and computation
- **易用性**：简洁的API接口 | Usability: Clean API interface

## 使用示例 | Usage Examples
```python
# 导入集成器 | Import integrator
from openlearning import create_integrator

# 创建集成器 | Create integrator
integrator = create_integrator(vocab_size=20000, dim=512)

# 前向传播 | Forward propagation
import torch
input_ids = torch.randint(0, 20000, (2, 16))
output = integrator.forward(input_ids, num_cycles=3)

# 伪装保存 | Disguise save
integrator.save_pretrained("./saved_model")

# 获取分析报告 | Get analysis report
report = integrator.get_analysis_report()
```

## 安装方式 | Installation
```bash
# 从项目根目录安装
pip install -e .

# 验证安装
python -c "import openlearning; print(f'版本: {openlearning.__version__}')"
```

## 版本历史 | Version History
- **0.0.3** - 集成测试通过版本
  - 基础RGA架构实现
  - 核心引擎与配置系统
  - 神经网络层模块
  - 集成伪装系统
  - 一站式演示脚本 (`python -m openlearning`)
```