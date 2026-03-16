#!/usr/bin/env python
"""
RGA 完整使用示例
包含：模拟数据生成、模型训练、保存、加载和推理
"""

import os
import torch
import numpy as np

# ==================== 1. 导入 RGA 组件 ====================
from openlearning.rga import (
    AdvancedConstrainedArchitectureTrainer,
    RuleGovernedArchitecture,
    SmartTextDataset
)
from openlearning.config import RGAConfig

# ==================== 2. 生成模拟训练数据 ====================
# 为了演示，我们创建一个简单的 LCCC 格式 JSON 文件
def create_dummy_data(file_path="dummy_data.json", num_dialogues=100):
    import json
    import random
    
    dialogues = []
    # 简单词汇
    words = ["你好", "今天", "天气", "不错", "RGA", "规则", "治理", "架构", 
             "学习", "模型", "PyTorch", "深度", "记忆", "相变", "测试"]
    
    for _ in range(num_dialogues):
        # 随机生成 3-7 轮的对话
        turns = random.randint(3, 7)
        dialogue = []
        for _ in range(turns):
            # 随机生成 3-8 个词的句子
            sentence = "".join(random.choices(words, k=random.randint(3, 8)))
            dialogue.append(sentence)
        dialogues.append(dialogue)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)
    print(f"✅ 模拟数据已生成: {file_path}")

create_dummy_data("demo_data.json")

# ==================== 3. 配置训练参数 ====================
config = {
    'data_path': 'demo_data.json',           # 训练数据路径
    'output_dir': './demo_output',            # 输出目录
    'seq_length': 32,                         # 序列长度
    'batch_size': 4,                          # 批次大小（小批量以便快速演示）
    'embed_dim': 64,                          # 嵌入维度
    'learning_rate': 1e-3,                    # 学习率
    'num_epochs': 2,                          # 训练轮数（仅演示，实际应更多）
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==================== 4. 创建训练器并开始训练 ====================
print("\n🚀 开始训练...")
trainer = AdvancedConstrainedArchitectureTrainer(config)
model, history = trainer.train()
print("✅ 训练完成！")

# ==================== 5. 模型保存（伪装 BERT 格式） ====================
save_path = './demo_model'
model.save_pretrained(save_path)
print(f"✅ 模型已保存到 {save_path}")

# ==================== 6. 加载模型进行推理 ====================
print("\n🔮 加载模型进行推理...")
loaded_model = RuleGovernedArchitecture.from_pretrained(save_path)
loaded_model.eval()

# 准备一个示例输入（使用数据集中的第一个样本）
dataset = SmartTextDataset('demo_data.json', seq_length=32)
sample = dataset[0]
input_ids = sample['input_ids'].unsqueeze(0)  # 增加 batch 维度

# 移到相同设备
input_ids = input_ids.to(config['device'])
loaded_model = loaded_model.to(config['device'])

# 推理
with torch.no_grad():
    outputs = loaded_model(input_ids, num_cycles=1)
    logits = outputs['logits']                # [1, seq_len, vocab_size]
    predictions = logits.argmax(dim=-1)       # [1, seq_len]

print("✅ 推理成功！")
print(f"   输入形状: {input_ids.shape}")
print(f"   输出形状: {logits.shape}")
print(f"   预测示例 (前10个token): {predictions[0, :10].tolist()}")

# 可选：查看模型内部状态
if 'thought_metrics' in outputs:
    print("\n💭 思考过程指标:")
    for k, v in outputs['thought_metrics'].items():
        print(f"   {k}: {v}")

# ==================== 7. 清理临时文件（可选） ====================
# 如需保留文件，请注释以下代码
import shutil
shutil.rmtree('./demo_output', ignore_errors=True)
shutil.rmtree('./demo_model', ignore_errors=True)
os.remove('demo_data.json')
print("\n🧹 临时文件已清理")