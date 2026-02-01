# OpenLearning RGA - 规则治理架构 | Rule Governance Architecture

基于测试运行的智能导入与架构系统 | An intelligent import and architecture system based on test runs.

## 快速开始 | Quick Start

### 两种调用方式 | Two Calling Methods

**方式1：模块调用** | **Method 1: Module Call**
```bash
python -m openlearning [命令] [选项]
python -m openlearning help        # 显示帮助
python -m openlearning demo        # 运行演示
python -m openlearning test        # 运行测试
```

**方式2：直接命令** | **Method 2: Direct Command**
```bash
openlearning [命令] [选项]
openlearning help                  # 显示帮助
openlearning demo                  # 运行演示
openlearning test                  # 运行测试
```

### 安装与配置 | Installation & Configuration

1. **添加环境变量** | **Add Environment Variable**
   ```bash
   # 将项目目录添加到PYTHONPATH
   export PYTHONPATH="/path/to/your/project:$PYTHONPATH"
   ```

2. **安装依赖** | **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **验证安装** | **Verify Installation**
   ```bash
   openlearning check
   ```

## 核心命令 | Core Commands

### 基本命令 | Basic Commands
```bash
# 交互模式（默认）| Interactive mode (default)
openlearning
openlearning interactive

# 运行完整演示 | Run full demo
openlearning demo

# 运行测试模式 | Run test mode
openlearning test

# 检查模块状态 | Check module status
openlearning check

# 显示系统状态 | Show system status
openlearning --status
```

### 测试相关 | Testing Related
```bash
# 运行完整测试套件 | Run complete test suite
openlearning test-all

# 列出所有测试文件 | List all test files
openlearning list

# 运行所有测试 | Run all tests
openlearning run-all

# 运行指定测试 | Run specific test
openlearning --run <文件名/部分文件名>
openlearning --run test_core     # 运行包含'test_core'的测试
```

### 模块测试 | Module Testing
```bash
# 核心模块 | Core module
openlearning core                # 运行核心模块
openlearning core-metrics        # 核心度量计算器
openlearning core-registry       # 核心注册表

# 层模块 | Layers module
openlearning layers              # 运行所有层测试
openlearning layers-attention    # 注意力层
openlearning layers-balancer     # 平衡器层
openlearning layers-memory       # 地质记忆层
openlearning layers-normalization # 归一化层
openlearning layers-valve        # 单向阀层
openlearning layers-embeddings   # 嵌入层
openlearning layers-fusion       # 融合层

# 集成模块 | Integration module
openlearning integration         # 集成模块测试
```

### 训练与推理 | Training & Inference
```bash
# 启动训练菜单 | Start training menu
openlearning train

# 启动推理测试 | Start inference test
openlearning infer
```

## 实用命令示例 | Practical Command Examples

### 日常开发 | Daily Development
```bash
# 快速验证环境 | Quick environment verification
openlearning check

# 运行核心功能测试 | Run core function tests
openlearning core-metrics

# 测试特定模块 | Test specific module
openlearning layers-memory

# 运行所有测试并生成报告 | Run all tests and generate report
openlearning run-all
```

### 调试与诊断 | Debugging & Diagnosis
```bash
# 显示系统状态 | Show system status
openlearning --status

# 显示发现的所有模块 | Show all discovered modules
openlearning --modules

# 检查模块导入状态 | Check module import status
openlearning check

# 运行指定测试文件 | Run specific test file
openlearning --run test_attention
```

### 完整工作流程 | Complete Workflow
```bash
# 1. 检查环境 | Check environment
openlearning check

# 2. 运行核心测试 | Run core tests
openlearning core-metrics
openlearning core-registry

# 3. 运行层模块测试 | Run layers tests
openlearning layers

# 4. 运行集成测试 | Run integration tests
openlearning integration

# 5. 启动训练 | Start training
openlearning train

# 6. 测试推理 | Test inference
openlearning infer
```

## 测试案例 | Test Cases

### 环境验证测试 | Environment Verification Test
```bash
# 运行环境检查
openlearning check
# 输出示例:
# ✅ openlearning - 已导入
# ✅ openlearning.core - 已导入
# ✅ openlearning.layers - 已导入
# ✅ openlearning.integration - 已导入
```

### 核心功能测试 | Core Function Tests
```bash
# 核心度量计算器测试
openlearning core-metrics
# 输出示例:
# ✅ L2范数计算成功: 7.3363
# ✅ 大变化检测为相变: Δ=757.2769
# ✅ 状态管理测试通过
```

### 层模块测试 | Layers Module Tests
```bash
# 注意力层测试
openlearning layers-attention
# 输出示例:
# ✅ VKQ子网络: 参数98,883，处理顺序: V→K→Q
# ✅ QVK子网络: 参数98,883，处理顺序: Q→V→K
# ✅ KQV子网络: 参数98,883，处理顺序: K→Q→V
# ✅ 链式反应单元参数: 296,653

# 地质记忆测试
openlearning layers-memory
# 输出示例:
# ✅ 地质记忆结构：
#   浅层: 能量0.167，年龄0，V均值[-0.007, -0.023, 0.036]
#   中层: 能量0.720，年龄0，V均值[0.000, 0.021, -0.007]
#   深层: 能量0.800，年龄N/A，V均值[0.000, 0.000, 0.000]
```

### 训练与推理测试 | Training & Inference Tests
```bash
# 启动训练
openlearning train
# 菜单示例:
# 1. 快速测试模式 (测试/调试)
# 2. 标准训练模式 (推荐)
# 3. 完整训练模式 (需要大量资源)
# 4. 自定义训练模式
# 5. 恢复训练模式
# 6. 推理测试模式

# 启动推理
openlearning infer
# 交互示例:
# 输入模型路径: E:\新GPT训练数据\紫心测试\best_model.pth
# 输入测试文本: 你好
# 推理配置:
#   模型路径: best_model.pth
#   测试文本: 你好，这是一个测试。
```

## 命令行选项参考 | Command Line Options Reference

### 通用选项 | General Options
```bash
# 显示帮助
openlearning help
openlearning --help

# 显示版本
openlearning --version

# 显示系统状态
openlearning --status

# 显示所有模块
openlearning --modules
```

### 测试选项 | Testing Options
```bash
# 运行指定测试文件
openlearning --run <文件名或部分名称>
openlearning --run test           # 运行所有包含'test'的文件
openlearning --run attention      # 运行所有包含'attention'的测试

# 交互式运行模式（默认）
openlearning interactive
openlearning                     # 简写，效果相同
```

### 批量操作 | Batch Operations
```bash
# 运行所有测试
openlearning run-all

# 运行完整测试套件
openlearning test-all
# 包含: 环境检查、核心度量、所有层模块、集成模块

# 列出所有测试文件
openlearning list
```

## 实用技巧 | Practical Tips

### 1. 快速诊断 | Quick Diagnosis
```bash
# 一次性检查所有问题
openlearning check --status --modules
```

### 2. 自动化测试 | Automated Testing
```bash
# 创建测试脚本
#!/bin/bash
openlearning check
openlearning core-metrics
openlearning layers-attention
openlearning layers-memory
openlearning integration
```

### 3. 持续集成 | Continuous Integration
```bash
# CI/CD中的使用示例
- name: Run OpenLearning Tests
  run: |
    python -m openlearning test-all
    python -m openlearning --status
```

### 4. 开发工作流 | Development Workflow
```bash
# 开发时的典型流程
1. openlearning check              # 检查环境
2. openlearning --run test_core    # 运行当前开发模块测试
3. openlearning layers             # 运行相关层测试
4. openlearning integration        # 集成测试
5. openlearning demo              # 完整演示验证
```

## 故障排除 | Troubleshooting

### 常见问题 | Common Issues
```bash
# 问题: 命令未找到
# 解决: 确保在项目目录中，或正确设置PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# 问题: 模块导入失败
# 解决: 运行环境检查
openlearning check

# 问题: 测试失败
# 解决: 查看详细输出
openlearning --run <测试名>  # 查看具体错误
openlearning --status        # 查看系统状态
```

### 调试模式 | Debug Mode
```bash
# 详细输出模式
python -m openlearning --run test_core -v

# 通过Python调试
python -c "import openlearning; print(openlearning.__version__)"
```

## 架构特性 | Architecture Features

OpenLearning RGA 提供以下核心特性：

### 智能导入系统 | Intelligent Import System
- 环境智能感知（开发/生产/测试）
- 多层导入策略（静态/动态/延迟/混合）
- 完整错误处理和恢复机制
- 性能监控和缓存优化

### 三网并行注意力 | Three-Network Parallel Attention
- VKQ路径：值→键→查询
- QVK路径：查询→值→键
- KQV路径：键→查询→值

### 地质记忆系统 | Geological Memory System
- 三层记忆结构（浅层/中层/深层）
- 能量衰减机制
- 持久化存储

### V主导设计 | V-Dominant Design
- V权重0.6 > Q/K权重0.5
- V安全范围：[0.5, 2.0]
- 健康度监控和自动调整

## 高级用法 | Advanced Usage

### 配置文件 | Configuration Files
系统支持多种配置方式：
1. 命令行参数
2. 环境变量
3. 配置文件
4. 代码配置

### 自定义测试 | Custom Testing
```python
# 自定义测试示例
from openlearning import ImportSystem

system = ImportSystem()
system.configure(
    strategy="SMART",
    enable_cache=True,
    log_level="INFO"
)

# 导入模块
module = system.import_module("openlearning.core.metrics")
```

### 性能监控 | Performance Monitoring
```bash
# 查看导入系统指标
python -c "from openlearning import generate_report; print(generate_report())"
```

## 贡献指南 | Contribution Guide

### 开发环境设置 | Development Environment Setup
```bash
# 克隆仓库
git clone <repository-url>
cd openlearning

# 设置环境
export PYTHONPATH=$(pwd):$PYTHONPATH

# 验证安装
openlearning check
openlearning core-metrics
```

### 测试贡献 | Test Contribution
```bash
# 运行所有测试确保无回归
openlearning test-all

# 添加新测试后验证
openlearning --run <新测试名>
```

### 代码规范 | Code Standards
- 遵循PEP 8规范
- 提供完整文档字符串
- 包含中英双语注释
- 确保所有测试通过

---

**版本**: 0.0.9  
**作者**: RGA Architecture Team  
**许可**: Apache 2.0  
**GitHub**: https://github.com/Sky-zixin-yucai/Open-learning