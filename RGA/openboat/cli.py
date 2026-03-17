#!/usr/bin/env python
"""
OpenLearning CLI - 命令行接口，用于管理和训练 RGA 模型。
"""

import argparse
import json
import sys
from typing import Dict, Any

# 尝试导入必要的模块，并给出友好提示
try:
    import torch
except ImportError:
    print("错误: 未找到 PyTorch。请先安装 torch>=1.9.0")
    sys.exit(1)

try:
    from openboat import __version__
    from openboat.rga import AdvancedConstrainedArchitectureTrainer
    from openboat.zixin import test_complete_architecture
except ImportError as e:
    print(f"错误: 无法导入 openlearning 模块: {e}")
    print("请确保已正确安装 openlearning 包或在 PYTHONPATH 中包含项目根目录。")
    sys.exit(1)


def train_command(args):
    """处理 'train' 子命令：训练 RGA 模型"""
    # 从配置文件加载基础配置（如果提供）
    config = {}
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config.update(json.load(f))
            print(f"已加载配置文件: {args.config}")
        except Exception as e:
            print(f"错误: 无法读取配置文件 {args.config}: {e}")
            sys.exit(1)

    # 命令行参数覆盖配置文件中的同名项
    cli_overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k not in ['func', 'config', 'command']
    }
    config.update(cli_overrides)

    # 检查必需参数
    required = ['data_path', 'output_dir']
    missing = [r for r in required if r not in config]
    if missing:
        print(f"错误: 缺少必需参数: {', '.join(missing)}")
        print("请通过命令行指定或提供包含这些字段的配置文件。")
        sys.exit(1)

    # 设置默认值（如果未指定）
    defaults = {
        'seq_length': 128,
        'batch_size': 16,
        'embed_dim': 256,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            print(f"使用默认值 {key}={value}")

    # 创建训练器并开始训练
    print("开始训练...")
    trainer = AdvancedConstrainedArchitectureTrainer(config)
    trainer.train()
    print("训练完成。")


def test_command(args):
    """处理 'test' 子命令：运行完整架构测试"""
    print("正在运行完整架构测试...")
    success = test_complete_architecture()
    if success:
        print("✅ 测试通过。")
        sys.exit(0)
    else:
        print("❌ 测试失败。")
        sys.exit(1)


def info_command(args):
    """处理 'info' 子命令：显示系统信息"""
    print(f"OpenLearning 版本: {__version__}")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    try:
        import numpy
        print(f"NumPy 版本: {numpy.__version__}")
    except ImportError:
        print("NumPy 未安装")
    try:
        import jieba
        print(f"Jieba 版本: {jieba.__version__}")
    except ImportError:
        print("Jieba 未安装")
    try:
        import tqdm
        print(f"tqdm 版本: {tqdm.__version__}")
    except ImportError:
        print("tqdm 未安装")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="OpenLearning CLI - 管理和训练 RGA 模型。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  openlearning train --data_path ./data.json --output_dir ./output
  openlearning train --config ./config.json
  openlearning test
  openlearning info
        """
    )
    parser.add_argument('--version', action='version', version=f'OpenLearning {__version__}')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # train 子命令
    train_parser = subparsers.add_parser('train', help='训练 RGA 模型')
    train_parser.add_argument('--config', type=str, help='JSON 配置文件路径')
    train_parser.add_argument('--data_path', type=str, help='训练数据路径（JSON 格式）')
    train_parser.add_argument('--output_dir', type=str, help='输出目录')
    train_parser.add_argument('--seq_length', type=int, help='序列长度')
    train_parser.add_argument('--batch_size', type=int, help='批次大小')
    train_parser.add_argument('--embed_dim', type=int, help='嵌入维度')
    train_parser.add_argument('--learning_rate', type=float, help='学习率')
    train_parser.add_argument('--num_epochs', type=int, help='训练轮数')
    train_parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='使用的设备')
    train_parser.set_defaults(func=train_command)

    # test 子命令
    test_parser = subparsers.add_parser('test', help='运行架构测试')
    test_parser.set_defaults(func=test_command)

    # info 子命令
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    info_parser.set_defaults(func=info_command)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == '__main__':
    main()