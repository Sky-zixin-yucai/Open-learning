#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RGA 对话演示脚本（修正词汇表加载）
使用训练好的模型进行交互式对话生成
"""

import os
import sys
import torch
import json
from openboat.rga import RuleGovernedArchitecture

# ==================== 配置 ====================
MODEL_PATH = "/mnt/e/新GPT训练数据/LCCC-紫/pretrained_model"  # 伪装BERT格式的模型目录
TRAIN_OUTPUT_DIR = "/mnt/e/新GPT训练数据/LCCC-紫"            # 训练输出目录（包含真实词汇表）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_GEN_LEN = 30          # 最大生成长度
TEMPERATURE = 0.8         # 采样温度
TOP_K = 40                # Top-k 采样
TOP_P = 0.9               # Top-p (nucleus) 采样
REPETITION_PENALTY = 1.2  # 重复惩罚

# ==================== 加载真实词汇表 ====================
def load_real_vocab(train_dir):
    """从训练输出目录加载 final_vocabulary.json，返回 token2idx 和 idx2token"""
    json_path = os.path.join(train_dir, "final_vocabulary.json")
    if not os.path.exists(json_path):
        # 尝试查找其他可能的词汇表文件
        json_path = os.path.join(train_dir, "initial_vocabulary.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到词汇表文件，请确保 {train_dir} 下有 final_vocabulary.json 或 initial_vocabulary.json")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    # 词汇表 tokens 列表
    tokens = vocab_data.get('tokens', [])
    if not tokens:
        # 兼容不同格式
        token2idx = vocab_data.get('token2idx', {})
        idx2token = [None] * len(token2idx)
        for token, idx in token2idx.items():
            idx2token[idx] = token
        tokens = idx2token
    
    # 构建映射
    token2idx = {token: i for i, token in enumerate(tokens)}
    print(f"📚 真实词汇表大小: {len(tokens)}")
    return token2idx, tokens

# ==================== 分词器（字符级，与训练一致） ====================
class CharTokenizer:
    """字符级分词器，与训练时 SmartTextDataset 的字符级处理保持一致"""
    def __init__(self, token2idx, idx2token):
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.vocab_size = len(idx2token)
        self.unk_idx = token2idx.get('<UNK>', 1)  # 默认1为UNK

    def encode(self, text, max_len=None):
        """将文本按字符切分并转换为 token ID"""
        # 移除空格（字符级训练时通常忽略空格，或空格作为字符）
        # 根据训练日志，空格比例 5.7%，说明空格被当作字符处理？但为了保险，我们保留原样，只按字符切分
        chars = list(text)  # 每个字符作为 token
        ids = []
        for ch in chars:
            if ch in self.token2idx:
                ids.append(self.token2idx[ch])
            else:
                ids.append(self.unk_idx)
        # 截断（保留最后 max_len 个，因为对话生成通常关注近期内容）
        if max_len is not None and len(ids) > max_len:
            ids = ids[-max_len:]
        return ids

    def decode(self, ids):
        """将 token ID 列表转换回文本，忽略特殊标记"""
        chars = []
        for tid in ids:
            if tid < len(self.idx2token):
                token = self.idx2token[tid]
                # 过滤特殊标记（可根据需要保留或移除）
                if token in ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<CLS>', '<SEP>', '<MASK>'] or token.startswith('[WORD'):
                    continue
                chars.append(token)
            else:
                chars.append('�')  # 替换未知字符
        return ''.join(chars)

# ==================== 生成回复 ====================
@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_len=MAX_GEN_LEN):
    """给定 prompt，生成回复"""
    model.eval()
    # 编码 prompt
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        # 如果 prompt 为空或全部未知，至少给个起始符（如 <BOS>）
        input_ids = [tokenizer.token2idx.get('<BOS>', 2)]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    generated = []
    for _ in range(max_len):
        # 前向传播
        outputs = model(input_tensor, num_cycles=1)
        logits = outputs['logits']  # [1, seq_len, vocab_size]
        next_token_logits = logits[0, -1, :]  # 最后一个位置的 logits

        # 重复惩罚
        if generated:
            for token_id in set(generated[-10:]):
                next_token_logits[token_id] /= REPETITION_PENALTY

        # 温度调整
        next_token_logits = next_token_logits / TEMPERATURE

        # Top-k 过滤
        if TOP_K > 0:
            top_k_logits, top_k_indices = torch.topk(next_token_logits, TOP_K)
            next_token_logits = torch.full_like(next_token_logits, float('-inf'))
            next_token_logits[top_k_indices] = top_k_logits

        # Top-p (nucleus) 过滤
        if TOP_P < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs > TOP_P
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

        # 采样
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        # 检查结束符（假设 <EOS> 是结束标记，索引为3）
        if next_token == 3:  # <EOS>
            break

        generated.append(next_token)

        # 将新 token 拼接到输入
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=DEVICE)], dim=1)

    # 解码生成的 token 序列
    response = tokenizer.decode(generated)
    return response

# ==================== 主对话循环 ====================
def main():
    print("🚀 正在加载模型，请稍候...")
    # 加载模型
    model = RuleGovernedArchitecture.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("✅ 模型加载成功！")

    # 加载真实词汇表
    try:
        token2idx, idx2token = load_real_vocab(TRAIN_OUTPUT_DIR)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("请检查 TRAIN_OUTPUT_DIR 路径是否正确，或手动指定词汇表文件位置。")
        sys.exit(1)

    tokenizer = CharTokenizer(token2idx, idx2token)

    print("\n" + "="*50)
    print("💬 RGA 对话机器人已启动！输入 'quit' 或 'exit' 退出。")
    print("="*50 + "\n")

    while True:
        user_input = input("👤 你: ").strip()
        if user_input.lower() in ['quit', 'exit', '再见']:
            print("👋 再见！")
            break
        if not user_input:
            continue

        # 生成回复
        response = generate_response(model, tokenizer, user_input)

        print(f"🤖 RGA: {response}\n")

if __name__ == "__main__":
    main()