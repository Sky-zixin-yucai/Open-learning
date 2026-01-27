# OpenLearning RGA - 规则治理架构 | Rule Governance Architecture

基于测试运行的架构系统。 | An architecture system based on test runs.

## 文件结构 | File Structure

**核心文件** | **Core Files**:
- `__init__.py` - 统一接口模块，暴露所有子包功能 | Unified interface module, exposing all sub-package functions
- `__main__.py` - 主演示入口，运行完整演示或测试 | Main demo entry, runs complete demos or tests
- `cli.py` - 命令行接口，支持所有模块调用 | Command line interface, supports all module calls

**子模块** | **Submodules**:
- `core/` - 核心引擎、配置、度量计算 | Core engine, configuration, metric calculation
- `layers/` - 专用神经网络层（注意力、平衡器、记忆、阀等） | Specialized neural network layers (attention, balancer, memory, valve, etc.)
- `integration/` - 集成训练、推理、数据集管理 | Integrated training, inference, dataset management

## 命令行调用 | Command Line Calls

### 主包命令 | Main Package Commands
```bash
# 基本功能 | Basic functions
python -m openlearning                      # 运行完整演示 | Run full demo
python -m openlearning --demo              # 演示模式 | Demo mode
python -m openlearning --test              # 测试模式 | Test mode
python -m openlearning --check-modules     # 检查模块状态 | Check module status
python -m openlearning --fast              # 快速演示 | Fast demo
python -m openlearning --no-visualization  # 跳过可视化 | Skip visualization

# 通过CLI | Via CLI
openlearning help                          # 显示帮助 | Show help
openlearning check                         # 检查模块状态 | Check module status
openlearning demo --fast                   # 快速演示 | Fast demo
openlearning test                          # 运行测试 | Run tests
```

### 模块级调用 | Module Level Calls
```bash
# 核心模块 | Core module
openlearning core                          # 运行核心模块 | Run core module
openlearning core-metrics                  # 核心度量计算器 | Core metrics calculator
openlearning core-registry                 # 核心注册表 | Core registry

# 层模块 | Layers module
openlearning layers                        # 运行层模块测试 | Run layers module tests
openlearning layers-attention              # 注意力层测试 | Attention layers test
openlearning layers-balancer               # 平衡器层测试 | Balancer layers test
openlearning layers-memory                 # 地质记忆层测试 | Geological memory layers test
openlearning layers-normalization          # 归一化层测试 | Normalization layers test
openlearning layers-valve                  # 单向阀层测试 | One-way valve layers test
openlearning layers-embeddings             # 嵌入层测试 | Embedding layers test
openlearning layers-fusion                 # 融合层测试 | Fusion layers test

# 集成模块 | Integration module
openlearning integration                   # 运行集成模块测试 | Run integration module tests
openlearning train                         # 启动训练菜单 | Start training menu
openlearning infer                         # 启动推理测试 | Start inference test
```

### 完整测试套件 | Complete Test Suite
```bash
# 运行所有测试 | Run all tests
openlearning test-all

# 或分步测试 | Or step-by-step tests
openlearning check                         # 环境检查 | Environment check
openlearning core-metrics                  # 核心度量测试 | Core metrics test
openlearning layers                        # 所有层模块测试 | All layers module tests
openlearning integration                   # 集成模块测试 | Integration module test
```

## 模块功能 | Module Functions

### 核心模块 (core/) | Core Module (core/)
- `RGAConfig` - 配置管理 | Configuration management
- `CoreMetricsCalculator` - 状态监控和相变检测 | State monitoring and phase transition detection
- `RGAEngine` - QKV三元组处理引擎 | QKV triplet processing engine
- 状态变化计算、相变检测、三网堆叠、单向阀控制 | State change calculation, phase transition detection, three-network stacking, one-way valve control

### 层模块 (layers/) | Layers Module (layers/)
- **注意力子系统** | **Attention Subsystem**:
  - `VKQ_SubNet_WithFixedNorm` - V→K→Q路径 | V→K→Q path
  - `QVK_SubNet_WithFixedNorm` - Q→V→K路径 | Q→V→K path
  - `KQV_SubNet_WithFixedNorm` - K→Q→V路径 | K→Q→V path
  - `ChainReactionUnit_Final` - 三网合并单元 | Three-network merge unit

- **平衡器系统** | **Balancer System**:
  - `TriValueBalancer` - Q、K、V三值平衡 | Q, K, V three-value balance
  - `VDominantBalancer` - V值主导平衡 | V-dominant balance
  - `DensityDrivenBalancer` - 密度驱动平衡 | Density-driven balance
  - `AdaptiveStabilizer` - 自适应稳定器 | Adaptive stabilizer

- **记忆系统** | **Memory System**:
  - `GeologicalMemory` - 三层地质记忆（浅层、中层、深层） | Three-layer geological memory (shallow, medium, deep)

- **控制系统** | **Control System**:
  - `OneWayValve` - 单向信息流控制阀 | One-way information flow control valve
  - `FixedRMSNorm` - 固定RMS归一化 | Fixed RMS normalization
  - `SandwichFusion` - 三明治融合层 | Sandwich fusion layer

### 集成模块 (integration/) | Integration Module (integration/)
- `RGAIntegrator` - 完整模型集成器 | Complete model integrator
- `SmartTextDataset` - 智能文本数据集 | Smart text dataset
- `AdvancedConstrainedArchitectureTrainer` - 高级训练器 | Advanced trainer
- `VisualTrainingProgress` - 可视化训练进度 | Visual training progress

## 运行示例 | Running Examples

### 环境验证 | Environment Verification
```bash
# 检查所有模块 | Check all modules
openlearning check

# 输出示例 | Example output:
# ✅ core: 已导入 | ✅ core: imported
# ✅ layers: 已导入 | ✅ layers: imported
# ✅ integration: 已导入 | ✅ integration: imported
```

### 核心功能测试 | Core Function Test
```bash
# 测试核心度量计算器 | Test core metrics calculator
openlearning core-metrics

# 输出示例 | Example output:
# ✅ L2范数计算成功: 7.3363 | ✅ L2 norm calculation successful: 7.3363
# ✅ 大变化检测为相变: Δ=757.2769 | ✅ Large change detected as phase transition: Δ=757.2769
# ✅ 状态管理测试通过 | ✅ State management test passed
```

### 层模块测试 | Layers Module Test
```bash
# 测试注意力层 | Test attention layers
openlearning layers-attention

# 输出示例 | Example output:
# ✅ VKQ子网络: 参数98,883，处理顺序: V→K→Q | ✅ VKQ subnetwork: 98,883 parameters, processing order: V→K→Q
# ✅ QVK子网络: 参数98,883，处理顺序: Q→V→K | ✅ QVK subnetwork: 98,883 parameters, processing order: Q→V→K
# ✅ KQV子网络: 参数98,883，处理顺序: K→Q→V | ✅ KQV subnetwork: 98,883 parameters, processing order: K→Q→V
# ✅ 链式反应单元参数: 296,653 | ✅ Chain reaction unit parameters: 296,653
```

```bash
# 测试地质记忆 | Test geological memory
openlearning layers-memory

# 输出示例 | Example output:
# ✅ 地质记忆结构： | ✅ Geological memory structure:
#   浅层: 能量0.167，年龄0，V均值[-0.007, -0.023, 0.036] |   Shallow: energy 0.167, age 0, V mean [-0.007, -0.023, 0.036]
#   中层: 能量0.720，年龄0，V均值[0.000, 0.021, -0.007] |   Medium: energy 0.720, age 0, V mean [0.000, 0.021, -0.007]
#   深层: 能量0.800，年龄N/A，V均值[0.000, 0.000, 0.000] |   Deep: energy 0.800, age N/A, V mean [0.000, 0.000, 0.000]
```

### 训练和推理 | Training and Inference
```bash
# 启动训练菜单 | Start training menu
openlearning train

# 选择模式： | Choose mode:
# 1. 快速测试模式 (测试/调试) | 1. Quick test mode (testing/debugging)
# 2. 标准训练模式 (推荐) | 2. Standard training mode (recommended)
# 3. 完整训练模式 (需要大量资源) | 3. Complete training mode (requires significant resources)
# 4. 自定义训练模式 | 4. Custom training mode
# 5. 恢复训练模式 | 5. Resume training mode
# 6. 推理测试模式 | 6. Inference test mode

# 启动推理测试 | Start inference test
openlearning infer
# 输入模型路径: E:\新GPT训练数据\紫心测试\best_model.pth | Enter model path: E:\新GPT训练数据\紫心测试\best_model.pth
# 输入测试文本: 你好 | Enter test text: 你好
```

## 架构特性 | Architecture Features

### 三网并行注意力 | Three-Network Parallel Attention
- **VKQ路径**: 值信息影响键，再影响查询 | **VKQ Path**: Value information affects key, then affects query
- **QVK路径**: 查询信息影响值，再影响键 | **QVK Path**: Query information affects value, then affects key
- **KQV路径**: 键信息影响查询，再影响值 | **KQV Path**: Key information affects query, then affects value

### 地质记忆系统 | Geological Memory System
- **浅层记忆**: 能量0.167，存储最近状态，易被覆盖 | **Shallow Memory**: Energy 0.167, stores recent state, easily overwritten
- **中层记忆**: 能量0.720，存储中期状态，半持久 | **Medium Memory**: Energy 0.720, stores medium-term state, semi-persistent
- **深层记忆**: 能量0.800，存储长期状态，持久记忆 | **Deep Memory**: Energy 0.800, stores long-term state, persistent memory

### V主导设计 | V-Dominant Design
- V权重0.6 > Q/K权重0.5 | V weight 0.6 > Q/K weight 0.5
- V安全范围：[0.5, 2.0] | V safety range: [0.5, 2.0]
- V健康度监控和自动调整 | V health monitoring and automatic adjustment

### 相变检测 | Phase Transition Detection
- 阈值：0.83（状态变化超过83%为相变） | Threshold: 0.83 (state change >83% is phase transition)
- 触发保护机制：激活单向阀、调整平衡器 | Trigger protection mechanism: activate one-way valve, adjust balancer

## 训练流程 | Training Process

### 数据准备 | Data Preparation
```
📁 智能文本数据集初始化 | Smart text dataset initialization
├─ 数据来源: LCCC-base_train.json | Data source: LCCC-base_train.json
├─ 总对话数: 6,820,506条 | Total dialogues: 6,820,506
├─ 采样数量: 1,000条 | Sample size: 1,000
├─ 处理策略: 词级处理（检测到空格） | Processing strategy: word-level processing (space detected)
├─ 词汇表大小: 5000词元 | Vocabulary size: 5,000 tokens
└─ 覆盖度: 96.2% | Coverage: 96.2%
```

### 模型配置 | Model Configuration
```python
{
    'vocab_size': 5000,
    'dim': 64,
    'units': 3,
    'geo_depth': 3,
    'max_cycles': 3,
    'phase_threshold': 0.83,
    'v_scaling_factor': 1.0
}
```

### 训练输出 | Training Output
```
验证进度 [███████████───────────────────────] 47.3% (1s/1s) | Validation progress [███████████───────────────────────] 47.3% (1s/1s)
┌─────────────────────────────────────────────────────────────┐
│ Loss: 6.354 | AvgLoss: 6.483 | 令牌: 97 | 进度: 26/55       │
│ Loss: 6.354 | AvgLoss: 6.483 | Tokens: 97 | Progress: 26/55 │
└─────────────────────────────────────────────────────────────┘
🔄 持续思考循环 1/1 | 🔄 Continuous thinking loop 1/1
  深层衰退: 时间层0被中期层覆盖 | Deep decay: time layer 0 overwritten by medium layer
  最新层更新: 时间层0, 能量=0.833 | Latest layer update: time layer 0, energy=0.833
  最新层更新: 时间层1, 能量=0.633 | Latest layer update: time layer 1, energy=0.633
```

### 模型保存 | Model Saving
```
保存的模型文件 | Saved model files:
- best_model.pth           # 最佳模型 | Best model
- final_model.pth          # 最终模型 | Final model
- pretrained_model/        # 标准格式 | Standard format
  ├─ pytorch_model.bin     # 模型参数 | Model parameters
  ├─ config.json           # 配置文件 | Configuration file
  ├─ vocab.txt            # 词汇表 | Vocabulary
  └─ tokenizer_config.json # 分词器配置 | Tokenizer configuration
```

## 性能基准 | Performance Benchmark

### 推理速度 | Inference Speed
```
测试 小 尺寸 (batch=1, seq=8):    0.0170 ± 0.0035 秒 | Test small size (batch=1, seq=8):    0.0170 ± 0.0035 seconds
测试 中 尺寸 (batch=2, seq=32):   0.0209 ± 0.0024 秒 | Test medium size (batch=2, seq=32):   0.0209 ± 0.0024 seconds
测试 大 尺寸 (batch=4, seq=64):   0.0302 ± 0.0043 秒 | Test large size (batch=4, seq=64):   0.0302 ± 0.0043 seconds
测试 超大 尺寸 (batch=8, seq=128): 0.0674 ± 0.0057 秒 | Test extra large size (batch=8, seq=128): 0.0674 ± 0.0057 seconds
```

### 内存使用 | Memory Usage
```
GPU内存使用 | GPU memory usage:
- 小尺寸：21.80 MB | Small size: 21.80 MB
- 中尺寸：27.95 MB | Medium size: 27.95 MB
- 大尺寸：49.77 MB | Large size: 49.77 MB
- 超大尺寸：137.60 MB | Extra large size: 137.60 MB
```

### 训练稳定性 | Training Stability
```
训练稳定性指标 | Training stability metrics:
- 损失波动范围：5.598-7.197 | Loss fluctuation range: 5.598-7.197
- 损失标准差：0.3182 | Loss standard deviation: 0.3182
- V值稳定性：1.0000 ± 0.0000 | V value stability: 1.0000 ± 0.0000
- 梯度范数均值：1.1928 | Gradient norm mean: 1.1928
- 梯度范数最大值：2.1058 | Gradient norm maximum: 2.1058
```

## 故障排除 | Troubleshooting

### 常见问题 | Common Issues
1. **模块导入失败** | **Module import failed**
   ```bash
   # 检查Python路径 | Check Python path
   python -c "import sys; print(sys.path)"
   
   # 添加项目路径 | Add project path
   export PYTHONPATH="/path/to/openlearning:$PYTHONPATH"
   ```

2. **CUDA内存不足** | **Insufficient CUDA memory**
   ```bash
   # 启用内存优化 | Enable memory optimization
   # 自动混合精度已启用 (PyTorch 2.0+ API) | Auto mixed precision enabled (PyTorch 2.0+ API)
   # 梯度累积：1步 | Gradient accumulation: 1 step
   ```

3. **地质记忆可视化失败** | **Geological memory visualization failed**
   ```bash
   # 跳过可视化 | Skip visualization
   openlearning demo --no-visualization
   
   # 或安装matplotlib | Or install matplotlib
   pip install matplotlib
   ```

### 验证步骤 | Verification Steps
```bash
# 分步验证 | Step-by-step verification
openlearning check                     # 步骤1：环境检查 | Step 1: Environment check
openlearning core-metrics              # 步骤2：核心功能 | Step 2: Core function
openlearning layers-attention          # 步骤3：注意力层 | Step 3: Attention layers
openlearning layers-memory             # 步骤4：地质记忆 | Step 4: Geological memory
openlearning integration               # 步骤5：集成模块 | Step 5: Integration module
```

## 架构设计原则 | Architecture Design Principles

### 三层架构 | Three-Layer Architecture
1. **核心层**：状态监控、相变检测、基本运算 | **Core Layer**: State monitoring, phase transition detection, basic operations
2. **层系统**：专用神经网络组件（注意力、平衡器、记忆、阀） | **Layer System**: Specialized neural network components (attention, balancer, memory, valve)
3. **集成层**：训练、推理、数据集管理、可视化 | **Integration Layer**: Training, inference, dataset management, visualization

### 核心机制 | Core Mechanisms
- **持续思考循环**：多轮次信息处理 | **Continuous Thinking Loop**: Multi-round information processing
- **地质记忆衰退**：能量衰减因子0.7 | **Geological Memory Decay**: Energy decay factor 0.7
- **V主导平衡**：确保值信息的主导地位 | **V-Dominant Balance**: Ensure the dominance of value information
- **相变保护**：状态突变时激活安全机制 | **Phase Transition Protection**: Activate safety mechanism when state changes abruptly

## 技术规格 | Technical Specifications

### 模型参数 | Model Parameters
- 词汇表大小：5000词元 | Vocabulary size: 5,000 tokens
- 嵌入维度：64 | Embedding dimension: 64
- 链式反应单元：3个 | Chain reaction units: 3
- 地质记忆层：3层深度 × 3时间层 | Geological memory layers: 3 depth layers × 3 time layers
- 总参数：1,623,800个 | Total parameters: 1,623,800
- 可训练参数：1,623,800个 | Trainable parameters: 1,623,800

### 训练配置 | Training Configuration
- 训练轮次：3轮 | Training epochs: 3
- 验证损失：6.4667 | Validation loss: 6.4667
- 训练损失：6.2680 | Training loss: 6.2680
- 训练时间：70.5秒 | Training time: 70.5 seconds
- 验证时间：3.5秒 | Validation time: 3.5 seconds
- 总时间：74.0秒 | Total time: 74.0 seconds

### 文件大小 | File Sizes
- `pytorch_model.bin`：6,795,407字节 | `pytorch_model.bin`: 6,795,407 bytes
- `config.json`：420字节 | `config.json`: 420 bytes
- `vocab.txt`：1,026字节 | `vocab.txt`: 1,026 bytes
- `tokenizer_config.json`：195字节 | `tokenizer_config.json`: 195 bytes

---

**版本**: 0.0.7  
**作者**: RGA Architecture Team  
**许可**: Apache 2.0  
**GitHub**: https://github.com/Sky-zixin-yucai/Open-learning