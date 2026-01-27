# OpenLearning RGA - 规则治理架构

基于测试运行的架构系统。

## 文件结构

**核心文件**：
- `__init__.py` - 统一接口模块，暴露所有子包功能
- `__main__.py` - 主演示入口，运行完整演示或测试
- `cli.py` - 命令行接口，支持所有模块调用

**子模块**：
- `core/` - 核心引擎、配置、度量计算
- `layers/` - 专用神经网络层（注意力、平衡器、记忆、阀等）
- `integration/` - 集成训练、推理、数据集管理

## 命令行调用

### 主包命令
```bash
# 基本功能
python -m openlearning                      # 运行完整演示
python -m openlearning --demo              # 演示模式
python -m openlearning --test              # 测试模式
python -m openlearning --check-modules     # 检查模块状态
python -m openlearning --fast              # 快速演示
python -m openlearning --no-visualization  # 跳过可视化

# 通过CLI
openlearning help                          # 显示帮助
openlearning check                         # 检查模块状态
openlearning demo --fast                   # 快速演示
openlearning test                          # 运行测试
```

### 模块级调用
```bash
# 核心模块
openlearning core                          # 运行核心模块
openlearning core-metrics                  # 核心度量计算器
openlearning core-registry                 # 核心注册表

# 层模块
openlearning layers                        # 运行层模块测试
openlearning layers-attention              # 注意力层测试
openlearning layers-balancer               # 平衡器层测试
openlearning layers-memory                 # 地质记忆层测试
openlearning layers-normalization          # 归一化层测试
openlearning layers-valve                  # 单向阀层测试
openlearning layers-embeddings             # 嵌入层测试
openlearning layers-fusion                 # 融合层测试

# 集成模块
openlearning integration                   # 运行集成模块测试
openlearning train                         # 启动训练菜单
openlearning infer                         # 启动推理测试
```

### 完整测试套件
```bash
# 运行所有测试
openlearning test-all

# 或分步测试
openlearning check                         # 环境检查
openlearning core-metrics                  # 核心度量测试
openlearning layers                        # 所有层模块测试
openlearning integration                   # 集成模块测试
```

## 模块功能

### 核心模块 (core/)
- `RGAConfig` - 配置管理
- `CoreMetricsCalculator` - 状态监控和相变检测
- `RGAEngine` - QKV三元组处理引擎
- 状态变化计算、相变检测、三网堆叠、单向阀控制

### 层模块 (layers/)
- **注意力子系统**：
  - `VKQ_SubNet_WithFixedNorm` - V→K→Q路径
  - `QVK_SubNet_WithFixedNorm` - Q→V→K路径
  - `KQV_SubNet_WithFixedNorm` - K→Q→V路径
  - `ChainReactionUnit_Final` - 三网合并单元

- **平衡器系统**：
  - `TriValueBalancer` - Q、K、V三值平衡
  - `VDominantBalancer` - V值主导平衡
  - `DensityDrivenBalancer` - 密度驱动平衡
  - `AdaptiveStabilizer` - 自适应稳定器

- **记忆系统**：
  - `GeologicalMemory` - 三层地质记忆（浅层、中层、深层）

- **控制系统**：
  - `OneWayValve` - 单向信息流控制阀
  - `FixedRMSNorm` - 固定RMS归一化
  - `SandwichFusion` - 三明治融合层

### 集成模块 (integration/)
- `RGAIntegrator` - 完整模型集成器
- `SmartTextDataset` - 智能文本数据集
- `AdvancedConstrainedArchitectureTrainer` - 高级训练器
- `VisualTrainingProgress` - 可视化训练进度

## 运行示例

### 环境验证
```bash
# 检查所有模块
openlearning check

# 输出示例：
# ✅ core: 已导入
# ✅ layers: 已导入  
# ✅ integration: 已导入
```

### 核心功能测试
```bash
# 测试核心度量计算器
openlearning core-metrics

# 输出示例：
# ✅ L2范数计算成功: 7.3363
# ✅ 大变化检测为相变: Δ=757.2769
# ✅ 状态管理测试通过
```

### 层模块测试
```bash
# 测试注意力层
openlearning layers-attention

# 输出示例：
# ✅ VKQ子网络: 参数98,883，处理顺序: V→K→Q
# ✅ QVK子网络: 参数98,883，处理顺序: Q→V→K
# ✅ KQV子网络: 参数98,883，处理顺序: K→Q→V
# ✅ 链式反应单元参数: 296,653
```

```bash
# 测试地质记忆
openlearning layers-memory

# 输出示例：
# ✅ 地质记忆结构：
#   浅层: 能量0.167，年龄0，V均值[-0.007, -0.023, 0.036]
#   中层: 能量0.720，年龄0，V均值[0.000, 0.021, -0.007]
#   深层: 能量0.800，年龄N/A，V均值[0.000, 0.000, 0.000]
```

### 训练和推理
```bash
# 启动训练菜单
openlearning train

# 选择模式：
# 1. 快速测试模式 (测试/调试)
# 2. 标准训练模式 (推荐)
# 3. 完整训练模式 (需要大量资源)
# 4. 自定义训练模式
# 5. 恢复训练模式
# 6. 推理测试模式

# 启动推理测试
openlearning infer
# 输入模型路径: E:\新GPT训练数据\紫心测试\best_model.pth
# 输入测试文本: 你好
```

## 架构特性

### 三网并行注意力
- **VKQ路径**: 值信息影响键，再影响查询
- **QVK路径**: 查询信息影响值，再影响键
- **KQV路径**: 键信息影响查询，再影响值

### 地质记忆系统
- **浅层记忆**: 能量0.167，存储最近状态，易被覆盖
- **中层记忆**: 能量0.720，存储中期状态，半持久
- **深层记忆**: 能量0.800，存储长期状态，持久记忆

### V主导设计
- V权重0.6 > Q/K权重0.5
- V安全范围：[0.5, 2.0]
- V健康度监控和自动调整

### 相变检测
- 阈值：0.83（状态变化超过83%为相变）
- 触发保护机制：激活单向阀、调整平衡器

## 训练流程

### 数据准备
```
📁 智能文本数据集初始化
├─ 数据来源: LCCC-base_train.json
├─ 总对话数: 6,820,506条
├─ 采样数量: 1,000条
├─ 处理策略: 词级处理（检测到空格）
├─ 词汇表大小: 5000词元
└─ 覆盖度: 96.2%
```

### 模型配置
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

### 训练输出
```
验证进度 [███████████───────────────────────] 47.3% (1s/1s)
┌─────────────────────────────────────────────────────────────┐
│ Loss: 6.354 | AvgLoss: 6.483 | 令牌: 97 | 进度: 26/55       │
└─────────────────────────────────────────────────────────────┘
🔄 持续思考循环 1/1
  深层衰退: 时间层0被中期层覆盖
  最新层更新: 时间层0, 能量=0.833
  最新层更新: 时间层1, 能量=0.633
```

### 模型保存
```
保存的模型文件：
- best_model.pth           # 最佳模型
- final_model.pth          # 最终模型
- pretrained_model/        # 标准格式
  ├─ pytorch_model.bin     # 模型参数
  ├─ config.json           # 配置文件
  ├─ vocab.txt            # 词汇表
  └─ tokenizer_config.json # 分词器配置
```

## 性能基准

### 推理速度
```
测试 小 尺寸 (batch=1, seq=8):    0.0170 ± 0.0035 秒
测试 中 尺寸 (batch=2, seq=32):   0.0209 ± 0.0024 秒
测试 大 尺寸 (batch=4, seq=64):   0.0302 ± 0.0043 秒
测试 超大 尺寸 (batch=8, seq=128): 0.0674 ± 0.0057 秒
```

### 内存使用
```
GPU内存使用：
- 小尺寸：21.80 MB
- 中尺寸：27.95 MB
- 大尺寸：49.77 MB
- 超大尺寸：137.60 MB
```

### 训练稳定性
```
训练稳定性指标：
- 损失波动范围：5.598-7.197
- 损失标准差：0.3182
- V值稳定性：1.0000 ± 0.0000
- 梯度范数均值：1.1928
- 梯度范数最大值：2.1058
```

## 故障排除

### 常见问题
1. **模块导入失败**
   ```bash
   # 检查Python路径
   python -c "import sys; print(sys.path)"
   
   # 添加项目路径
   export PYTHONPATH="/path/to/openlearning:$PYTHONPATH"
   ```

2. **CUDA内存不足**
   ```bash
   # 启用内存优化
   # 自动混合精度已启用 (PyTorch 2.0+ API)
   # 梯度累积：1步
   ```

3. **地质记忆可视化失败**
   ```bash
   # 跳过可视化
   openlearning demo --no-visualization
   
   # 或安装matplotlib
   pip install matplotlib
   ```

### 验证步骤
```bash
# 分步验证
openlearning check                     # 步骤1：环境检查
openlearning core-metrics              # 步骤2：核心功能
openlearning layers-attention          # 步骤3：注意力层
openlearning layers-memory             # 步骤4：地质记忆
openlearning integration               # 步骤5：集成模块
```

## 架构设计原则

### 三层架构
1. **核心层**：状态监控、相变检测、基本运算
2. **层系统**：专用神经网络组件（注意力、平衡器、记忆、阀）
3. **集成层**：训练、推理、数据集管理、可视化

### 核心机制
- **持续思考循环**：多轮次信息处理
- **地质记忆衰退**：能量衰减因子0.7
- **V主导平衡**：确保值信息的主导地位
- **相变保护**：状态突变时激活安全机制

## 技术规格

### 模型参数
- 词汇表大小：5000词元
- 嵌入维度：64
- 链式反应单元：3个
- 地质记忆层：3层深度 × 3时间层
- 总参数：1,623,800个
- 可训练参数：1,623,800个

### 训练配置
- 训练轮次：3轮
- 验证损失：6.4667
- 训练损失：6.2680
- 训练时间：70.5秒
- 验证时间：3.5秒
- 总时间：74.0秒

### 文件大小
- `pytorch_model.bin`：6,795,407字节
- `config.json`：420字节
- `vocab.txt`：1,026字节
- `tokenizer_config.json`：195字节

---

**版本**: 0.0.7  
**作者**: RGA Architecture Team  
**许可**: Apache 2.0  
**GitHub**: https://github.com/Sky-zixin-yucai/Open-learning