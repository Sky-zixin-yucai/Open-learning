import sys
import os

# ==================== 修复导入路径 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义检查项目根目录的函数
def is_project_root(path):
    """检查给定路径是否是项目根目录"""
    core_exists = os.path.exists(os.path.join(path, "core", "__init__.py"))
    layers_exists = os.path.exists(os.path.join(path, "layers", "__init__.py"))
    return core_exists and layers_exists

# 尝试导入路径设置
try:
    from yucai import (
        RGAConfig, 
        RGAIntegrator,
        
    )
    print("✅ 从 yucai.py 导入成功")
    
except ImportError as e:
    print(f"尝试从 yucai.py 导入失败: {e}")
    print("尝试手动设置路径...")
    
    project_root = None
    
    # 1. 向上查找项目根目录
    path = current_dir
    while path != os.path.dirname(path):  # 直到根目录
        if is_project_root(path):
            project_root = path
            print(f"✅ 在 {path} 找到项目根目录")
            break
        path = os.path.dirname(path)
    
    # 2. 如果没找到，尝试常见路径
    if project_root is None:
        possible_paths = [
            os.path.dirname(current_dir),
            os.path.dirname(os.path.dirname(current_dir)),
            current_dir,
        ]
        
        for path in possible_paths:
            if is_project_root(path):
                project_root = path
                print(f"✅ 在 {path} 找到项目根目录")
                break
    
    # 3. 最终检查
    if project_root is None:
        print("❌ 无法找到包含 core 和 layers 的目录")
        sys.exit(1)
    
    # 将项目根目录添加到 sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"项目根目录: {project_root}")

# =================== 导入所需模块 ====================
import sys
from torch.utils.data import DataLoader, random_split
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
import json
from collections import deque
from torch.utils.data import Dataset, DataLoader
import re
import random
from collections import Counter, deque
from torch.autograd import Function
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
import math


# ==================== 3. 智能文本数据集（结构化整理版）====================
class SmartTextDataset(Dataset):
    """智能文本数据集（结构化日志版）- 支持LCCC-base_train.json格式"""
    
    def __init__(self, data_path, seq_length=128, vocab_size=15000, max_samples=None, language='zh'):
        self.data_path = data_path
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.language = language
        
        # 打印头部信息
        self._print_header("📁 智能文本数据集初始化")
        
        # 执行数据处理流程
        self._load_data(max_samples)
        self._analyze_data()
        self._build_vocabulary()
        self._encode_for_training()
        
        # 打印尾部信息
        self._print_footer("✅ 数据集准备完成")
    
    # ==================== 日志打印方法 ====================
    def _print_header(self, title):
        """打印标题"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
    
    def _print_step(self, step, message, level=1):
        """打印步骤信息"""
        indent = "  " * level
        prefix = "├─ " if level == 1 else "│  ├─ " if level == 2 else "│  │  └─ "
        print(f"{indent}{prefix}{step}: {message}")
    
    def _print_info(self, key, value, level=2):
        """打印键值对信息"""
        indent = "  " * level
        prefix = "│  " + "  " * (level-1) + "├─ " if level >= 2 else "├─ "
        print(f"{indent}{prefix}{key}: {value}")
    
    def _print_stat(self, stats_dict, level=3):
        """打印统计信息"""
        indent = "  " * level
        prefix = "│  " + "  " * (level-1) + "├─ " if level >= 3 else "│  └─ "
        
        if isinstance(stats_dict, dict):
            for key, value in stats_dict.items():
                print(f"{indent}{prefix}{key}: {value}")
        else:
            print(f"{indent}{prefix}{stats_dict}")
    
    def _print_footer(self, message):
        """打印尾部信息"""
        print(f"\n{message}")
        print(f"{'='*60}\n")
    
    # ==================== 数据加载模块 ====================
    def _load_data(self, max_samples):
        """加载原始数据"""
        self._print_step("数据加载", "", level=1)
        self._print_info("来源", self.data_path, level=2)
        
        if not os.path.exists(self.data_path):
            self._print_info("状态", "❌ 文件不存在", level=2)
            self.texts = []
            return
        
        if not self.data_path.endswith('.json'):
            self._print_info("状态", "❌ 不是JSON格式", level=2)
            self.texts = []
            return
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_dialogues = len(data)
            self._print_info("总对话数", f"{total_dialogues:,}条", level=2)
            
            # 采样加载数据
            self.texts = []
            sample_count = 0
            max_to_load = max_samples if max_samples else total_dialogues
            
            for i, dialogue in enumerate(data):
                if max_samples and sample_count >= max_samples:
                    break
                
                if isinstance(dialogue, list):
                    dialogue_parts = []
                    
                    for sentence in dialogue:
                        if isinstance(sentence, str):
                            cleaned = self._clean_text_basic(sentence)
                            if cleaned:
                                dialogue_parts.append(cleaned)
                    
                    if dialogue_parts:
                        dialogue_text = " ".join(dialogue_parts)
                        self.texts.append(dialogue_text)
                        sample_count += 1
                
                # 每50万条显示一次进度
                if i > 0 and i % 500000 == 0:
                    self._print_info("处理进度", f"{i:,}/{total_dialogues:,}", level=2)
            
            self._print_info("采样数量", f"{len(self.texts):,}条", level=2)
            self._print_info("状态", "✅ 成功", level=2)
            
        except Exception as e:
            self._print_info("状态", f"❌ 加载失败: {e}", level=2)
            self.texts = []
    
    def _clean_text_basic(self, text):
        """基础文本清洗"""
        if not isinstance(text, str):
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除特殊符号
        text = re.sub(r'[\{\}\[\]\"]', '', text)
        # 合并多个空格
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空格
        text = text.strip()
        
        return text
    
    # ==================== 数据分析模块 ====================
    def _analyze_data(self):
        """分析数据特性并决定处理策略"""
        if not self.texts:
            self._print_step("数据分析", "❌ 无数据可分析", level=1)
            self.use_char_level = True
            return
        
        self._print_step("数据分析", "", level=1)
        
        # 采样分析
        sample_size = min(1000, len(self.texts))
        sample_texts = random.sample(self.texts, sample_size) if len(self.texts) > sample_size else self.texts
        
        self._print_info("采样数量", f"{len(sample_texts)}条", level=2)
        
        # 分析空格比例和词长
        analysis_result = self._analyze_text_properties(sample_texts)
        
        # 决定处理策略
        if analysis_result['space_ratio'] > 0.1 and ' ' in sample_texts[0]:
            self.use_char_level = False
            self._print_info("处理策略", "🎯 词级处理（检测到空格）", level=2)
        else:
            self.use_char_level = True
            self._print_info("处理策略", "🎯 字符级处理", level=2)
        
        # 文本长度统计
        self._analyze_text_lengths()
    
    def _analyze_text_properties(self, sample_texts):
        """分析文本属性"""
        total_chars = 0
        space_chars = 0
        word_lengths = []
        
        for text in sample_texts:
            total_chars += len(text)
            space_chars += text.count(' ')
            
            words = text.split()
            for word in words:
                word_lengths.append(len(word))
        
        space_ratio = space_chars / total_chars if total_chars > 0 else 0
        avg_word_len = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        self._print_info("平均空格比例", f"{space_ratio:.1%}", level=2)
        self._print_info("平均词长", f"{avg_word_len:.1f}字符", level=2)
        
        return {
            'space_ratio': space_ratio,
            'avg_word_len': avg_word_len
        }
    
    def _analyze_text_lengths(self):
        """分析文本长度"""
        if self.use_char_level:
            lengths = [len(t.replace(' ', '')) for t in self.texts]
            length_desc = "字符"
        else:
            lengths = [len(t.split()) for t in self.texts]
            length_desc = "词"
        
        # 计算统计信息
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
        
        sorted_lengths = sorted(lengths)
        p95_length = sorted_lengths[int(len(lengths) * 0.95)]
        
        self._print_step("长度统计", "", level=2)
        self._print_stat({"平均": f"{avg_length:.1f} {length_desc}"}, level=3)
        self._print_stat({"最短": f"{min_length} {length_desc}"}, level=3)
        self._print_stat({"最长": f"{max_length} {length_desc}"}, level=3)
        self._print_stat({"95%分位": f"{p95_length} {length_desc}"}, level=3)
        
        # 显示示例文本
        if self.texts:
            example = self.texts[0][:200] + ("..." if len(self.texts[0]) > 200 else "")
            self._print_info("示例文本", example, level=2)
        
        # 给出建议
        if p95_length < self.seq_length:
            self._print_info("建议", f"seq_length可从{self.seq_length}调整为{p95_length}", level=2)
        else:
            self._print_info("注意", f"{p95_length}个{length_desc}可能超过模型容量", level=2)
    
    # ==================== 词汇表构建模块 ====================
    def _build_vocabulary(self):
        """构建词汇表"""
        if not self.texts:
            self._print_step("词汇表构建", "❌ 无数据，使用默认词汇表", level=1)
            self._create_default_vocab()
            return
        
        self._print_step("词汇表构建", "", level=1)
        
        # 特殊token
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        if self.use_char_level:
            self._build_char_vocab(special_tokens)
        else:
            self._build_word_vocab(special_tokens)
        
        # 词汇表分析
        self._analyze_vocabulary()
    
    def _create_default_vocab(self):
        """创建默认词汇表"""
        self.vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.token2idx = {token: i for i, token in enumerate(self.vocab)}
        self.idx2token = {i: token for i, token in enumerate(self.vocab)}
    
    def _build_char_vocab(self, special_tokens):
        """构建字符级词汇表"""
        self._print_info("处理方式", "字符级处理", level=2)
        
        all_chars = ''.join([t.replace(' ', '') for t in self.texts])
        
        if not all_chars:
            self._print_info("状态", "❌ 文本为空，使用默认词汇表", level=2)
            self._create_default_vocab()
            return
        
        char_counter = Counter(all_chars)
        total_chars = len(all_chars)
        
        # 构建词汇表
        vocab = special_tokens.copy()
        for char, freq in char_counter.most_common():
            if char not in vocab:
                vocab.append(char)
            if len(vocab) >= self.vocab_size:
                break
        
        # 计算覆盖度
        covered_chars = sum(freq for char, freq in char_counter.items() if char in vocab)
        coverage = covered_chars / total_chars if total_chars > 0 else 0
        
        self.vocab = vocab
        self.token2idx = {token: i for i, token in enumerate(self.vocab)}
        self.idx2token = {i: token for i, token in enumerate(self.vocab)}
        
        self._print_info("总字符数", f"{total_chars:,}", level=2)
        self._print_info("不同字符数", f"{len(char_counter):,}", level=2)
        self._print_info("词汇表大小", f"{len(self.vocab)}", level=2)
        self._print_info("覆盖度", f"{coverage:.1%}", level=2)
    
    def _build_word_vocab(self, special_tokens):
        """构建词级词汇表"""
        self._print_info("处理方式", "词级处理（使用空格分词）", level=2)
        
        import jieba
        word_counter = Counter()
        
        # 统计词频
        for text in self.texts:
            words = text.split()
            word_counter.update(words)
        
        total_words = sum(word_counter.values())
        
        # 智能词汇表构建策略
        vocab = special_tokens.copy()
        
        # 策略1：先收集所有出现≥2次的词
        freq_ge_2 = [(word, freq) for word, freq in word_counter.items() if freq >= 2]
        
        if len(freq_ge_2) <= self.vocab_size - len(special_tokens):
            # 如果出现≥2次的词不多，全部加入
            for word, freq in sorted(freq_ge_2, key=lambda x: x[1], reverse=True):
                vocab.append(word)
        else:
            # 如果出现≥2次的词太多，按频率排序取前N个
            sorted_words = sorted(freq_ge_2, key=lambda x: x[1], reverse=True)
            for word, freq in sorted_words[:self.vocab_size - len(special_tokens)]:
                vocab.append(word)
        
        # 如果词汇表还没满，补充一些出现1次的词（优先选择单字词）
        if len(vocab) < self.vocab_size:
            freq_eq_1 = [(word, 1) for word, freq in word_counter.items() if freq == 1]
            
            # 优先加入单字词（对中文很重要）
            single_char_words = [(word, 1) for word, _ in freq_eq_1 if len(word) == 1]
            for word, _ in single_char_words[:self.vocab_size - len(vocab)]:
                vocab.append(word)
            
            # 如果还有空间，加入其他词
            if len(vocab) < self.vocab_size:
                remaining_words = [(word, 1) for word, _ in freq_eq_1 if len(word) > 1]
                for word, _ in remaining_words[:self.vocab_size - len(vocab)]:
                    vocab.append(word)
        
        # 计算覆盖度
        covered_words = sum(freq for word, freq in word_counter.items() if word in vocab)
        coverage = covered_words / total_words if total_words > 0 else 0
        
        self.vocab = vocab
        self.token2idx = {token: i for i, token in enumerate(self.vocab)}
        self.idx2token = {i: token for i, token in enumerate(self.vocab)}
        
        self._print_step("统计信息", "", level=2)
        self._print_stat({"总词数": f"{total_words:,}"}, level=3)
        self._print_stat({"不同词数": f"{len(word_counter):,}"}, level=3)
        self._print_stat({"高频词(≥2次)": f"{len(freq_ge_2):,}个"}, level=3)
        self._print_stat({"低频词(1次)": f"{len(word_counter) - len(freq_ge_2):,}个"}, level=3)
        
        self._print_step("词汇表详情", "", level=2)
        self._print_stat({"大小": f"{len(self.vocab)}词元"}, level=3)
        self._print_stat({"覆盖度": f"{coverage:.1%}"}, level=3)
        
        # 显示示例词汇
        if len(self.vocab) > 30:
            self._print_info("示例词汇", f"{self.vocab[:30]}...", level=2)
        else:
            self._print_info("示例词汇", self.vocab, level=2)
    
    def _analyze_vocabulary(self):
        """分析词汇表特征"""
        self._print_step("词汇分布", "", level=2)
        
        # 检查多字词
        multi_char_words = [word for word in self.vocab if len(word) > 1]
        chinese_multi_char_words = [
            word for word in multi_char_words 
            if all('\u4e00' <= ch <= '\u9fff' for ch in word)
        ]
        
        # 按词长统计
        length_dist = {}
        for word in self.vocab:
            length = len(word)
            length_dist[length] = length_dist.get(length, 0) + 1
        
        self._print_stat({"多字词": f"{len(multi_char_words)}个 ({len(multi_char_words)/len(self.vocab):.1%})"}, level=3)
        self._print_stat({"中文多字词": f"{len(chinese_multi_char_words)}个"}, level=3)
        self._print_stat({"长度分布": dict(sorted(length_dist.items()))}, level=3)
        
        # 显示中文多字词示例
        if chinese_multi_char_words:
            if len(chinese_multi_char_words) > 30:
                self._print_info("中文多字词示例", f"{chinese_multi_char_words[:30]}...", level=2)
            else:
                self._print_info("中文多字词示例", chinese_multi_char_words, level=2)
    
    # ==================== 数据编码模块 ====================
    def _encode_for_training(self):
        """编码数据为训练格式"""
        if not self.texts or not hasattr(self, 'token2idx'):
            self._print_step("数据编码", "❌ 无数据或词汇表，无法编码", level=1)
            self.encoded_segments = []
            return
        
        self._print_step("数据编码", "", level=1)
        
        self.encoded_segments = []
        total_segments = 0
        total_tokens = 0
        
        # 编码每段文本
        for text in self.texts:
            if not text:
                continue
            
            # 编码原始文本
            encoded = self._encode_text(text)
            
            # 添加特殊标记
            encoded = [self.token2idx.get('<BOS>', 2)] + encoded + [self.token2idx.get('<EOS>', 3)]
            
            # 分割为训练段
            segments = self._split_into_segments(encoded)
            
            for segment in segments:
                self.encoded_segments.append(segment)
                total_segments += 1
                total_tokens += len([t for t in segment if t != 0])
        
        self._print_info("训练段", f"{len(self.encoded_segments):,}个", level=2)
        self._print_info("有效token数", f"{total_tokens:,}", level=2)
        
        # 分析填充情况
        self._analyze_padding()
    
    def _encode_text(self, text):
        """编码单段文本"""
        if self.use_char_level:
            text_no_spaces = text.replace(' ', '')
            return [self.token2idx.get(ch, 1) for ch in text_no_spaces]
        else:
            import jieba
            continuous_text = text.replace(' ', '')
            words = jieba.lcut(continuous_text)
            return [self.token2idx.get(word, 1) for word in words]
    
    def _split_into_segments(self, encoded):
        """将编码后的文本分割为训练段"""
        segments = []
        step_size = max(1, self.seq_length // 2)
        
        for i in range(0, len(encoded), step_size):
            segment = encoded[i:i + self.seq_length]
            
            # 检查有效长度
            valid_length = len([t for t in segment if t != 0])
            min_valid = max(5, self.seq_length // 4)
            
            if valid_length >= min_valid:
                if len(segment) < self.seq_length:
                    segment = segment + [0] * (self.seq_length - len(segment))
                segments.append(segment)
        
        return segments
    
    def _analyze_padding(self):
        """分析填充情况"""
        if not self.encoded_segments:
            return
        
        padded_count = 0
        total_padding = 0
        actual_lengths = []
        
        for segment in self.encoded_segments:
            padding_len = segment.count(0)
            if padding_len > 0:
                padded_count += 1
                total_padding += padding_len
            
            # 计算实际长度
            actual_len = 0
            for token in segment:
                if token == 0:
                    break
                actual_len += 1
            actual_lengths.append(actual_len)
        
        total_segments_len = len(self.encoded_segments) * self.seq_length
        padding_ratio = total_padding / total_segments_len if total_segments_len > 0 else 0
        effective_ratio = (total_segments_len - total_padding) / total_segments_len
        avg_actual_len = sum(actual_lengths) / len(actual_lengths) if actual_lengths else 0
        
        self._print_step("填充统计", "", level=2)
        self._print_stat({"填充比例": f"{padding_ratio:.1%}"}, level=3)
        self._print_stat({"有效比例": f"{effective_ratio:.1%}"}, level=3)
        self._print_stat({"填充段比例": f"{padded_count/len(self.encoded_segments):.1%}"}, level=3)
        
        self._print_step("长度信息", "", level=2)
        self._print_stat({"平均长度": f"{avg_actual_len:.1f} tokens"}, level=3)
        self._print_stat({"长度范围": f"{min(actual_lengths)}-{max(actual_lengths)} tokens"}, level=3)
    
    # ==================== Dataset接口方法 ====================
    def __len__(self):
        """返回数据集大小"""
        return len(self.encoded_segments)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        if idx >= len(self.encoded_segments):
            raise IndexError(f"索引 {idx} 超出范围，数据集大小: {len(self.encoded_segments)}")
        
        seq = self.encoded_segments[idx]
        
        # 确保序列长度正确
        if len(seq) != self.seq_length:
            if len(seq) > self.seq_length:
                seq = seq[:self.seq_length]
            else:
                seq = seq + [0] * (self.seq_length - len(seq))
        
        # 语言建模任务：预测下一个token
        target = seq[1:] + [0]
        
        return {
            'input_ids': torch.tensor(seq, dtype=torch.long),
            'target_ids': torch.tensor(target, dtype=torch.long)
        }
    
    # ==================== 辅助方法 ====================
    def get_vocab_size(self):
        """获取词汇表大小"""
        return len(self.vocab) if hasattr(self, 'vocab') else 0
    
    def get_sample_count(self):
        """获取样本数量"""
        return len(self.encoded_segments) if hasattr(self, 'encoded_segments') else 0
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'vocab_size': self.get_vocab_size(),
            'sample_count': self.get_sample_count(),
            'seq_length': self.seq_length,
            'use_char_level': self.use_char_level if hasattr(self, 'use_char_level') else None
        }

class VisualTrainingProgress:
    """可视化训练进度管理器"""
    
    def __init__(self, total_batches, desc="训练", bar_width=50):
        self.total_batches = total_batches
        self.desc = desc
        self.bar_width = bar_width
        self.current_batch = 0
        self.start_time = time.time()
        self.metrics = {}
        
    def update(self, metrics_dict, batch_idx):
        """更新进度和指标"""
        self.current_batch = batch_idx
        self.metrics = metrics_dict
        
        # 计算进度百分比
        progress = (batch_idx + 1) / self.total_batches
        filled_length = int(self.bar_width * progress)
        bar = '█' * filled_length + '─' * (self.bar_width - filled_length)
        
        # 计算时间
        elapsed = time.time() - self.start_time
        if progress > 0:
            remaining = elapsed / progress * (1 - progress)
        else:
            remaining = 0
            
        # 构建指标字符串
        metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics_dict.items()])
        
        # 清屏并重新打印
        print("\033[F" * 3 + "\033[K")  # 回到上三行并清除
        print(f"{self.desc} [{bar}] {progress*100:5.1f}% ({elapsed:5.0f}s/{remaining:5.0f}s)")
        print(f"┌{'─'*(self.bar_width+30)}┐")
        print(f"│ {metrics_str:<{self.bar_width+28}} │")
        print(f"└{'─'*(self.bar_width+30)}┘", flush=True)
        
    def complete(self):
        """完成进度显示"""
        elapsed = time.time() - self.start_time
        print("\033[F" * 3 + "\033[K")  # 回到上三行并清除
        print(f"{self.desc} [{'█'*self.bar_width}] 100% ({elapsed:5.0f}s)")
        print(f"┌{'─'*(self.bar_width+30)}┐")
        print(f"│ {'训练完成!':^{self.bar_width+28}} │")
        print(f"└{'─'*(self.bar_width+30)}┘")


class AdvancedConstrainedArchitectureTrainer:
    """高级可视化训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        os.makedirs(config['output_dir'], exist_ok=True)
        
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'best_val_loss': float('inf'),
            'metrics': []
        }
        
        print("="*80)
        print("紫心RGA高级训练系统")
        print("="*80)
        print(f"设备: {self.device}")
        print(f"输出目录: {config['output_dir']}")
        print("初始化完成...\n")
    def _save_vocabulary(self, dataset, prefix="vocab"):
        """
        保存词汇表到文件（集中管理，所有训练模式都会调用）
    
        Args:
            dataset: SmartTextDataset 实例
            prefix: 文件名前缀
        """
        if not hasattr(dataset, 'vocab') or not dataset.vocab:
            return
    
        import json
        import os
        import time
    
        # 创建词汇表数据
        vocab_data = {
            'vocab_size': len(dataset.vocab),
            'tokens': dataset.vocab,
            'token2idx': dataset.token2idx,
            'idx2token': dataset.idx2token,
            'save_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'training_mode': self.config.get('output_dir', 'unknown'),
            'vocab_source': 'SmartTextDataset'
        }
    
        # 1. 保存为JSON（程序可读）
        json_filename = f"{prefix}_vocabulary.json"
        json_path = os.path.join(self.config['output_dir'], json_filename)
    
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ JSON词汇表: {json_filename}")
        except Exception as e:
            print(f"  ❌ JSON词汇表保存失败: {e}")
    
        # 2. 保存为TXT（人类可读）
        txt_filename = f"{prefix}_vocabulary.txt"
        txt_path = os.path.join(self.config['output_dir'], txt_filename)
    
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"# {prefix.capitalize()} 词汇表\n")
                f.write(f"# 生成时间: {vocab_data['save_time']}\n")
                f.write(f"# 词汇表大小: {vocab_data['vocab_size']}\n")
                f.write(f"# 训练模式: {vocab_data['training_mode']}\n")
                f.write("#" * 60 + "\n\n")
            
                # 分类显示词汇
                special_tokens = []
                chinese_chars = []
                chinese_words = []
                others = []
            
                for token in dataset.vocab:
                    if token.startswith('<') and token.endswith('>'):
                        special_tokens.append(token)
                    elif '\u4e00' <= token <= '\u9fff':  # 中文字符
                        if len(token) == 1:
                            chinese_chars.append(token)
                        else:
                            chinese_words.append(token)
                    else:
                        others.append(token)
            
                # 写入特殊标记
                if special_tokens:
                    f.write("[特殊标记]\n")
                    for token in special_tokens:
                        idx = dataset.token2idx.get(token, -1)
                        f.write(f"{idx:6d}: {token}\n")
                    f.write("\n")
            
                # 写入中文字符
                if chinese_chars:
                    f.write(f"[中文字符] 共 {len(chinese_chars)} 个\n")
                    # 每行显示20个字符
                    for i in range(0, len(chinese_chars), 20):
                        line = chinese_chars[i:i+20]
                        indices = [str(dataset.token2idx.get(ch, -1)).rjust(4) for ch in line]
                        f.write(f"索引: {' '.join(indices)}\n")
                        f.write(f"字符: {' '.join(line)}\n\n")
            
                # 写入中文词汇
                if chinese_words:
                    f.write(f"[中文词汇] 共 {len(chinese_words)} 个\n")
                    for i, token in enumerate(chinese_words[:200]):  # 只显示前200个
                        idx = dataset.token2idx.get(token, -1)
                        f.write(f"{idx:6d}: {token}\n")
                        if i % 20 == 19:
                            f.write("\n")
                    if len(chinese_words) > 200:
                        f.write(f"... 还有 {len(chinese_words)-200} 个词汇未显示\n")
                    f.write("\n")
            
                # 写入其他
                if others:
                    f.write("[其他标记]\n")
                    for token in others[:100]:  # 只显示前100个
                        idx = dataset.token2idx.get(token, -1)
                        f.write(f"{idx:6d}: {token}\n")
        
            print(f"  ✅ TXT词汇表: {txt_filename}")
        
            # 显示统计信息
            print(f"  📊 词汇统计:")
            print(f"     特殊标记: {len(special_tokens)} 个")
            print(f"     中文字符: {len(chinese_chars)} 个")
            print(f"     中文词汇: {len(chinese_words)} 个")
            print(f"     其他标记: {len(others)} 个")
            print(f"     总计: {vocab_data['vocab_size']} 个词元")
        
        except Exception as e:
            print(f"  ❌ TXT词汇表保存失败: {e}")

    def prepare_data(self):
        """准备训练数据 - 确保使用实际词汇表"""
        print("准备训练数据...")

        # 使用前面定义好的SmartTextDataset
        dataset = SmartTextDataset(
            data_path=self.config['data_path'],
            seq_length=self.config['seq_length'],
            vocab_size=self.config.get('vocab_size', 50000),
            max_samples=self.config.get('max_samples', 100000),
            language='zh'
        )
    
        # 🚨 关键：获取数据集的实际词汇表大小
        actual_vocab_size = dataset.get_vocab_size()
    
        # 🚨 更新配置，使用真实词汇表大小
        self.config['actual_vocab_size'] = actual_vocab_size
        self.config['dataset_vocab_size'] = actual_vocab_size
    
        print(f"📊 数据集实际词汇表大小: {actual_vocab_size}")
    
        # 分割数据集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
    
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
        print(f"数据统计: 训练集 {len(train_dataset)} | 验证集 {len(val_dataset)} | 词汇表 {actual_vocab_size}")
        print()
    
        return dataset, train_dataset, val_dataset

    def create_model(self):
        """创建模型 - 使用数据集的真实词汇表大小"""
        print("构建模型架构...")
    
        # 🚨 关键：使用数据集的实际词汇表大小，而不是配置中的默认值
        actual_vocab_size = self.config.get('actual_vocab_size', 
                                            self.config.get('dataset_vocab_size', 
                                                            self.config.get('vocab_size', 10000)))
    
        print(f"📊 创建模型使用词汇表大小: {actual_vocab_size}")
    
        # 创建RGAConfig
        rga_config = RGAConfig()
        rga_config.vocab_size = actual_vocab_size  # 🚨 使用真实值
        rga_config.dim = self.config.get('embed_dim', 512)
    
        # 创建模型
        model = RGAIntegrator(
            config=rga_config
        ).to(self.device)
    
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        print(f"✓ 模型架构: {model.__class__.__name__}")
        print(f"✓ 总参数: {total_params:,}")
        print(f"✓ 可训练参数: {trainable_params:,}")
        print(f"✓ 词汇表大小: {rga_config.vocab_size}")
        print(f"✓ 嵌入维度: {rga_config.dim}")
        print()
    
        return model
    
    def setup_optimizer(self, model):
        """设置优化器"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        return optimizer, scheduler, criterion
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch, progress):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        total_tokens = 0
        
        # 初始化统计
        batch_losses = []
        v_values = []
        gradient_norms = []
        memory_stats = []
        
        # 进度条预留空间
        print("\n" * 3)  # 为进度显示留出空间
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据移动
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # 前向传播 - 修复：使用正确的参数
            outputs = model(input_ids, num_cycles=1)  # 只接受input_ids和num_cycles
            logits = outputs['logits']
            
            # 计算损失 - 修复：使用config.vocab_size而不是model.vocab_size
            loss = criterion(logits.view(-1, model.config.vocab_size), target_ids.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get('grad_clip', 1.0))
            
            # 优化器步进
            optimizer.step()
            
            # 统计
            batch_tokens = (target_ids != 0).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            
            # 收集指标
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            gradient_norms.append(grad_norm.item())
            
            # 收集V值 - 修复：使用正确的键名
            if 'V_final' in outputs:
                v_mean = outputs['V_final'].mean().item()
                v_values.append(v_mean)
            
            # 地质记忆统计 - 修复：直接从地质记忆层获取
            try:
                energy_stats = model.geological_memory.get_energy_stats()
                if 'energy_depth0' in energy_stats and len(energy_stats['energy_depth0']) > 0:
                    mem_energy = (energy_stats['energy_depth0'][0] + 
                                energy_stats['energy_depth1'][0] + 
                                energy_stats['energy_depth2'][0]) / 3
                    memory_stats.append(mem_energy)
            except:
                memory_stats.append(0.0)
            
            # 计算实时指标
            current_avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            current_v_mean = np.mean(v_values[-min(10, len(v_values)):]) if v_values else 0
            current_grad_norm = np.mean(gradient_norms[-min(10, len(gradient_norms)):]) if gradient_norms else 0
            current_mem_energy = np.mean(memory_stats[-min(10, len(memory_stats)):]) if memory_stats else 0
            
            # 更新进度显示
            progress.update({
                'Loss': f"{batch_loss:6.3f}",
                'AvgLoss': f"{current_avg_loss:6.3f}",
                'V值': f"{current_v_mean:6.2f}",
                '梯度': f"{current_grad_norm:6.2f}",
                '记忆': f"{current_mem_energy:6.3f}",
                '令牌': f"{batch_tokens:6d}"
            }, batch_idx)
        
        # 完成进度显示
        progress.complete()
        
        # 计算epoch平均损失
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        
        # 计算统计信息
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'batch_loss_mean': np.mean(batch_losses) if batch_losses else 0,
            'batch_loss_std': np.std(batch_losses) if len(batch_losses) > 1 else 0,
            'v_mean': np.mean(v_values) if v_values else 0,
            'v_std': np.std(v_values) if len(v_values) > 1 else 0,
            'grad_norm_mean': np.mean(gradient_norms) if gradient_norms else 0,
            'grad_norm_max': np.max(gradient_norms) if gradient_norms else 0,
            'mem_energy': np.mean(memory_stats) if memory_stats else 0,
            'total_tokens': total_tokens
        }
        
        return avg_loss, epoch_metrics
    
    def validate(self, model, val_loader, criterion):
        """验证模型"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        val_progress = VisualTrainingProgress(
            total_batches=len(val_loader),
            desc="验证进度",
            bar_width=50
        )
        
        print("\n" * 3)  # 为进度显示留出空间
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # 修复：使用正确的参数
                outputs = model(input_ids, num_cycles=1)
                logits = outputs['logits']
                
                loss = criterion(logits.view(-1, model.config.vocab_size), target_ids.view(-1))
                
                batch_tokens = (target_ids != 0).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                
                # 更新进度
                current_loss = total_loss / total_tokens if total_tokens > 0 else 0
                val_progress.update({
                    'Loss': f"{loss.item():6.3f}",
                    'AvgLoss': f"{current_loss:6.3f}",
                    '令牌': f"{batch_tokens:6d}",
                    '进度': f"{batch_idx+1}/{len(val_loader)}"
                }, batch_idx)
        
        val_progress.complete()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return avg_loss
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'history': self.history
        }
        
        # 保存检查点
        checkpoint_path = os.path.join(
            self.config['output_dir'],
            f'checkpoint_epoch_{epoch:03d}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config['output_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            
            # 打印最佳模型信息
            print("-" * 40)
            print(f"  发现新最佳模型!")
            print(f"  验证损失: {metrics['val_loss']:.4f}")
            print(f"  已保存至: {best_path}")
            print("-" * 40)
        
        print(f"检查点已保存: {os.path.basename(checkpoint_path)}")
    
    def print_epoch_summary(self, epoch, metrics, train_time, val_time):
        """打印epoch总结"""
        print("\n" + "═" * 80)
        print(f"📊 第 {epoch} 轮训练总结")
        print("═" * 80)
        
        # 指标表格
        print("┌──────────────────────────┬──────────────┬──────────────────────┐")
        print("│        指标类别          │    数值      │       描述          │")
        print("├──────────────────────────┼──────────────┼──────────────────────┤")
        
        # 训练指标
        print(f"│ 训练损失 (均值)         │ {metrics['train_loss']:10.4f}    │ 当前轮次训练损失      │")
        print(f"│ 训练损失 (标准差)       │ {metrics.get('batch_loss_std', 0):10.4f}    │ 批次间波动程度        │")
        print(f"│ V值 (均值)              │ {metrics.get('v_mean', 0):10.4f}    │ 关键指标稳定性        │")
        print(f"│ V值 (标准差)            │ {metrics.get('v_std', 0):10.4f}    │ V值变化幅度           │")
        print(f"│ 梯度范数 (均值)         │ {metrics.get('grad_norm_mean', 0):10.4f}    │ 梯度稳定性            │")
        print(f"│ 梯度范数 (最大值)       │ {metrics.get('grad_norm_max', 0):10.4f}    │ 梯度爆炸风险          │")
        print(f"│ 记忆能量                │ {metrics.get('mem_energy', 0):10.4f}    │ 地质记忆活跃度        │")
        print(f"│ 验证损失                │ {metrics.get('val_loss', 0):10.4f}    │ 模型泛化能力          │")
        print(f"│ 处理令牌数              │ {metrics.get('total_tokens', 0):10,}    │ 本轮处理数据量        │")
        print(f"│ 训练耗时                │ {train_time:10.1f}s   │ 本轮训练用时          │")
        print(f"│ 验证耗时                │ {val_time:10.1f}s   │ 本轮验证用时          │")
        
        print("└──────────────────────────┴──────────────┴──────────────────────┘")
        print()
    
    def print_training_header(self):
        """打印训练头部信息"""
        print("╔══════════════════════════════════════════════════════════════════════════════════╗")
        print("║                              紫心RGA训练开始                                     ║")
        print("╠══════════════════════════════════════════════════════════════════════════════════╣")
        print("║ 进度条显示训练进度，下方表格显示实时指标变化                                      ║")
        print("║ 每个epoch结束后显示详细统计信息                                                  ║")
        print("╚══════════════════════════════════════════════════════════════════════════════════╝")
        print()
    
    def train(self):
        """训练主函数"""
        self.print_training_header()
        
        # 1. 准备数据
        dataset, train_dataset, val_dataset = self.prepare_data()

        # 🆕 新增：立即保存原始词汇表（训练前）
        self._save_vocabulary(dataset, prefix="initial")
        
        # 数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        # 2. 创建模型
        vocab_size = dataset.get_vocab_size()
        model = self.create_model()
        
        # 3. 设置优化器
        optimizer, scheduler, criterion = self.setup_optimizer(model)
        
        # 4. 检查已有检查点
        start_epoch = 1
        if os.path.exists(self.config['output_dir']):
            checkpoint_files = [f for f in os.listdir(self.config['output_dir']) if f.endswith('.pth')]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0, reverse=True)
                
                for checkpoint_file in checkpoint_files:
                    if 'epoch' in checkpoint_file:
                        checkpoint_path = os.path.join(self.config['output_dir'], checkpoint_file)
                        try:
                            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                            self.history = checkpoint.get('history', self.history)
                            start_epoch = checkpoint['epoch'] + 1
                            print(f"✓ 加载检查点: {os.path.basename(checkpoint_path)}")
                            print(f"✓ 继续从第 {start_epoch} 轮训练")
                            break
                        except Exception as e:
                            print(f"❌ 加载检查点失败: {e}")
                            continue
        
        print(f"\n开始训练，共 {self.config['num_epochs']} 轮")
        print("─" * 80)
        
        # 5. 训练循环
        for epoch in range(start_epoch, self.config['num_epochs'] + 1):
            print(f"\n第 {epoch}/{self.config['num_epochs']} 轮")
            print("─" * 40)
            
            # 创建进度管理器
            progress = VisualTrainingProgress(
                total_batches=len(train_loader),
                desc=f"训练进度 (E{epoch:02d})",
                bar_width=50
            )
            
            # 训练一个epoch
            train_start = time.time()
            train_loss, train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, epoch, progress)
            train_time = time.time() - train_start
            
            # 验证
            val_start = time.time()
            val_loss = self.validate(model, val_loader, criterion)
            val_time = time.time() - val_start
            
            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # 更新指标
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'train_time': train_time,
                'val_time': val_time
            }
            epoch_metrics.update(train_metrics)
            
            # 保存到历史
            self.history['epochs'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['metrics'].append(epoch_metrics)
            
            # 检查是否是最佳模型
            is_best = val_loss < self.history['best_val_loss']
            if is_best:
                self.history['best_val_loss'] = val_loss
            
            # 保存检查点
            self.save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, is_best)
            
            # 打印epoch总结
            self.print_epoch_summary(epoch, epoch_metrics, train_time, val_time)
        
        # 6. 训练完成
        print("╔══════════════════════════════════════════════════════════════════════════════════╗")
        print("║                                 训练完成                                         ║")
        print("╠══════════════════════════════════════════════════════════════════════════════════╣")
        print(f"║  总轮次: {self.config['num_epochs']}                                           最佳验证损失: {self.history['best_val_loss']:.4f}          ║")
        
        if len(self.history['train_loss']) > 0:
            final_train = self.history['train_loss'][-1]
            final_val = self.history['val_loss'][-1]
            print(f"║  最终训练损失: {final_train:.4f}                          最终验证损失: {final_val:.4f}              ║")
        
        if len(self.history['metrics']) > 0 and 'v_mean' in self.history['metrics'][-1]:
            final_v = self.history['metrics'][-1]['v_mean']
            print(f"║  最终V值: {final_v:.4f}                                                       ║")
        
        print("╚══════════════════════════════════════════════════════════════════════════════════╝")
        
        # 🆕 新增：保存最终词汇表（训练后）
        self._save_vocabulary(dataset, prefix="final")

        # ==================== 🆕 新增：保存为标准格式 ====================
        print("\n" + "="*60)
        print("💾 正在保存为标准格式...")

        # 创建保存目录
        pretrained_dir = os.path.join(self.config['output_dir'], 'pretrained_model')
        os.makedirs(pretrained_dir, exist_ok=True)

        # 🎯 关键代码：保存为标准格式
        model.save_pretrained(pretrained_dir)

        print(f"✅ 标准格式已保存到: {pretrained_dir}")
        print("   包含以下文件：")
        for file in os.listdir(pretrained_dir):
            file_path = os.path.join(pretrained_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"   - {file} ({size:,} bytes)")

        # ==================== 结束新增 ====================

        # 保存最终模型
        final_path = os.path.join(self.config['output_dir'], 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'vocab_size': vocab_size,
            'history': self.history
        }, final_path)
        
        print(f"\n✓ 最终模型已保存: {os.path.basename(final_path)}")
        print(f"✓ 训练记录已保存至: {self.config['output_dir']}")
        
        return model, self.history  

# ==================== 完整训练实例函数 ====================

class RGAConfigManager:
    """RGA训练元配置管理器 - 统一管理所有配置"""
    
    # ==================== 默认配置 ====================
    DEFAULT_CONFIG = {
        # ==================== 数据配置 ====================
        'data': {
            'data_path': './data/LCCC-base_train.json',
            'output_dir': './output/zixin_model',
            'seq_length': 128,
            'vocab_size': 20000,
            'max_samples': None,
            'language': 'zh',
            'train_ratio': 0.9,
        },
        
        # ==================== 模型架构配置 ====================
        'model': {
            'embed_dim': 256,           # 嵌入维度（必须为偶数）
            'hidden_dim': 256,          # 隐藏维度
            'marker_dim': 32,           # 标记向量维度
            'max_seq_len': 512,         # 最大序列长度
            'dropout_rate': 0.1,        # Dropout比率
            'num_units': 3,             # 链式反应单元数量（必须为3）
            'geo_depth': 3,             # 地质记忆深度（必须为3）
            'time_layers': 3,           # 时间层层数（必须为3）
            'v_subvalues': 3,           # V子值数量（必须为3）
        },
        
        # ==================== 训练配置 ====================
        'training': {
            'batch_size': 16,
            'num_epochs': 50,
            'learning_rate': 3e-5,
            'weight_decay': 1e-4,
            'grad_clip': 1.0,
            'warmup_steps': 2000,
            'min_learning_rate': 1e-6,
            'gradient_accumulation_steps': 1,
        },
        
        # ==================== 架构保护配置 ====================
        'architecture_protection': {
            'connection_threshold': 0.3,
            'phase_threshold': 0.43,
            'density_method': 'static',
            
            # V值调控
            'target_V_mean': 1.0,
            'max_V_mean': 2.0,
            'min_V_mean': 0.3,
            
            # 处理顺序（严格保持）
            'processing_orders': [
                'V→K→Q',  # 子网络1
                'Q→V→K',  # 子网络2
                'K→Q→V',  # 子网络3
            ],
            
            # 三明治融合权重（固定）
            'sandwich_weights': {
                'Q': [0.5, 0.3, 0.2],  # [深层, 当前, 原始]
                'K': [0.5, 0.3, 0.2],
                'V': [0.6, 0.3, 0.1],  # V权重不同，体现主导性
            },
        },
        
        # ==================== 优化配置 ====================
        'optimization': {
            'mixed_precision': True,
            'use_gradient_checkpointing': True,
            'scheduler_type': 'cosine',
            'cudnn_benchmark': True,
            'tf32_enabled': True,
        },
        
        # ==================== 监控配置 ====================
        'monitoring': {
            'V_health_history_length': 100,
            'progress_bar_width': 50,
            'save_checkpoint_every': 1,
            'keep_best_models': 3,
            'save_pretrained_format': True,
        },
        
        # ==================== 设备配置 ====================
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # ==================== 预设配置模式 ====================
    PRESET_CONFIGS = {
        'quick': {
            'data': {'max_samples': 1000, 'seq_length': 32, 'vocab_size': 5000},
            'model': {'embed_dim': 64, 'hidden_dim': 64, 'marker_dim': 16},
            'training': {'batch_size': 4, 'num_epochs': 3, 'learning_rate': 1e-4},
        },
        'standard': {
            'data': {'seq_length': 128, 'vocab_size': 15000},
            'model': {'embed_dim': 256, 'hidden_dim': 256, 'marker_dim': 32},
            'training': {'batch_size': 16, 'num_epochs': 30, 'learning_rate': 2e-5},
        },
        'full': {
            'data': {'seq_length': 256, 'vocab_size': 50000},
            'model': {'embed_dim': 512, 'hidden_dim': 512, 'marker_dim': 64},
            'training': {'batch_size': 32, 'num_epochs': 100, 'learning_rate': 1e-5},
        }
    }
    
    def __init__(self, mode='custom', custom_config=None):
        """
        初始化配置管理器
        
        Args:
            mode: 预设模式 ('quick', 'standard', 'full', 'custom')
            custom_config: 自定义配置字典（当mode='custom'时使用）
        """
        self.mode = mode
        self.config = self._build_config(mode, custom_config)
        
    def _build_config(self, mode, custom_config):
        """构建配置字典"""
        import copy
        
        # 从默认配置开始
        config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        # 如果使用预设模式，应用预设配置
        if mode in self.PRESET_CONFIGS:
            self._deep_update(config, self.PRESET_CONFIGS[mode])
        
        # 应用自定义配置（最高优先级）
        if custom_config:
            self._deep_update(config, custom_config)
        
        return config
    
    def _deep_update(self, target, source):
        """深度更新字典（递归）"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def get_config(self):
        """获取配置字典（用于训练器）"""
        # 将嵌套的配置展平为训练器需要的格式
        flat_config = {}
        
        # 1. 直接复制顶级配置
        flat_config['device'] = self.config['device']
        
        # 2. 展平数据配置
        for key, value in self.config['data'].items():
            flat_config[key] = value
        
        # 3. 展平模型配置（部分需要重命名）
        model_keys = ['embed_dim', 'hidden_dim', 'marker_dim', 'max_seq_len', 'dropout_rate']
        for key in model_keys:
            if key in self.config['model']:
                flat_config[key] = self.config['model'][key]
        
        # 4. 展平训练配置
        for key, value in self.config['training'].items():
            flat_config[key] = value
        
        # 5. 展平架构保护配置
        flat_config['architecture_protection'] = self.config['architecture_protection']
        
        # 6. 添加其他配置
        flat_config['optimization'] = self.config['optimization']
        flat_config['monitoring'] = self.config['monitoring']
        
        return flat_config
    
    def print_summary(self):
        """打印配置摘要"""
        print("="*80)
        print(f"紫心RGA训练配置 - {self.mode.upper()}模式")
        print("="*80)
        
        config = self.config
        
        print("\n📁 数据配置:")
        data = config['data']
        print(f"  数据路径: {data['data_path']}")
        print(f"  输出目录: {data['output_dir']}")
        print(f"  序列长度: {data['seq_length']}")
        print(f"  词汇表大小: {data['vocab_size']:,}")
        
        # 正确处理max_samples的格式化
        max_samples = data['max_samples']
        if max_samples is None:
            max_samples_str = '无限制'
        else:
            max_samples_str = f"{max_samples:,}"
        print(f"  最大样本数: {max_samples_str}")
        
        print("\n🤖 模型配置:")
        model = config['model']
        print(f"  嵌入维度: {model['embed_dim']} (偶数: {'✅' if model['embed_dim'] % 2 == 0 else '❌'})")
        print(f"  隐藏维度: {model['hidden_dim']}")
        print(f"  标记维度: {model['marker_dim']}")
        print(f"  链式单元: {model['num_units']}个")
        print(f"  地质深度: {model['geo_depth']}层")
        print(f"  时间层数: {model['time_layers']}个")
        print(f"  V子值数: {model['v_subvalues']}个/单元")
        
        print("\n⚙️ 训练配置:")
        training = config['training']
        print(f"  批次大小: {training['batch_size']}")
        print(f"  训练轮次: {training['num_epochs']}")
        print(f"  学习率: {training['learning_rate']}")
        print(f"  权重衰减: {training['weight_decay']}")
        print(f"  梯度裁剪: {training['grad_clip']}")
        
        print("\n🔒 架构保护:")
        protection = config['architecture_protection']
        print(f"  连接阈值: {protection['connection_threshold']}")
        print(f"  相变阈值: {protection['phase_threshold']}")
        print(f"  处理顺序: {protection['processing_orders']}")
        
        print("\n⚡ 优化配置:")
        optimization = config['optimization']
        print(f"  混合精度: {'✅' if optimization['mixed_precision'] else '❌'}")
        print(f"  梯度检查点: {'✅' if optimization['use_gradient_checkpointing'] else '❌'}")
        
        print("\n📊 监控配置:")
        monitoring = config['monitoring']
        print(f"  保存检查点: 每{monitoring['save_checkpoint_every']}轮次")
        print(f"  保留最佳模型: {monitoring['keep_best_models']}个")
        print(f"  保存标准格式: {'✅' if monitoring['save_pretrained_format'] else '❌'}")
        
        print(f"\n🖥️  设备: {config['device']}")
        print("="*80 + "\n")
        
        # 验证关键配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置有效性"""
        config = self.config
        warnings = []
        
        # 验证模型维度为偶数
        if config['model']['embed_dim'] % 2 != 0:
            warnings.append(f"⚠️ embed_dim应该是偶数，当前为{config['model']['embed_dim']}")
        
        # 验证架构保护参数
        if config['model']['num_units'] != 3:
            warnings.append(f"⚠️ num_units必须为3，当前为{config['model']['num_units']}")
        
        if config['model']['geo_depth'] != 3:
            warnings.append(f"⚠️ geo_depth必须为3，当前为{config['model']['geo_depth']}")
        
        if config['model']['time_layers'] != 3:
            warnings.append(f"⚠️ time_layers必须为3，当前为{config['model']['time_layers']}")
        
        if config['model']['v_subvalues'] != 3:
            warnings.append(f"⚠️ v_subvalues必须为3，当前为{config['model']['v_subvalues']}")
        
        # 验证三明治权重和为1
        weights = config['architecture_protection']['sandwich_weights']
        for key, weight_list in weights.items():
            total = sum(weight_list)
            if abs(total - 1.0) > 0.001:
                warnings.append(f"⚠️ {key}权重和不为1，当前为{total:.3f}")
        
        if warnings:
            print("\n🔍 配置验证警告:")
            for warning in warnings:
                print(f"  {warning}")
            print()


def train_zixin_complete_model(config_mode='standard', custom_config=None):
    """
    紫心RGA完整训练流程 - 基于元配置管理器
    
    Args:
        config_mode: 配置模式 ('quick', 'standard', 'full', 'custom')
        custom_config: 自定义配置字典（当config_mode='custom'时使用）
    
    Returns:
        model: 训练完成的模型
        history: 训练历史记录
    """
    
    # ==================== 初始化配置管理器 ====================
    print("初始化配置管理器...")
    config_manager = RGAConfigManager(mode=config_mode, custom_config=custom_config)
    
    # 打印配置摘要
    config_manager.print_summary()
    
    # 获取展平后的配置（用于训练器）
    config = config_manager.get_config()
    
    print("🚀 开始训练...")
    
    # ==================== 创建训练器并开始训练 ====================
    trainer = AdvancedConstrainedArchitectureTrainer(config)
    
    try:
        # 开始训练
        model, history = trainer.train()
        
        # 训练完成后显示最终统计
        print("\n" + "="*80)
        print("训练完成，最终统计")
        print("="*80)
        
        if history['train_loss']:
            final_train = history['train_loss'][-1]
            print(f"最终训练损失: {final_train:.4f}")
        
        if history['val_loss']:
            final_val = history['val_loss'][-1]
            print(f"最终验证损失: {final_val:.4f}")
        
        print(f"最佳验证损失: {history['best_val_loss']:.4f}")
        
        # 显示V值统计
        if history['metrics']:
            last_metrics = history['metrics'][-1]
            if 'v_mean' in last_metrics:
                print(f"最终V值: {last_metrics['v_mean']:.4f}")
            
            # 计算训练总时间
            total_train_time = sum([m.get('train_time', 0) for m in history['metrics']])
            total_val_time = sum([m.get('val_time', 0) for m in history['metrics']])
            print(f"总训练时间: {total_train_time:.1f}s")
            print(f"总验证时间: {total_val_time:.1f}s")
            print(f"总时间: {total_train_time + total_val_time:.1f}s")
        
        print(f"\n模型检查点保存在: {config['output_dir']}")
        print(f"最佳模型: {os.path.join(config['output_dir'], 'best_model.pth')}")
        print(f"最终模型: {os.path.join(config['output_dir'], 'final_model.pth')}")
        
        return model, history
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ==================== 快速测试模式 ====================

def quick_test_mode(data_path=None, output_dir=None):
    """
    快速测试模式 - 用于验证模型是否正常工作
    
    Args:
        data_path: 数据路径（可选）
        output_dir: 输出目录（可选）
    
    Returns:
        bool: 训练是否成功
    """
    print("🚀 启动快速测试模式")
    print("-"*40)
    
    # 构建自定义配置
    custom_config = {
        'data': {
            'data_path': data_path or './data/sample_data.json',
            'output_dir': output_dir or './output/quick_test',
            'max_samples': 1000,
            'seq_length': 32,
            'vocab_size': 5000,
        },
        'model': {
            'embed_dim': 64,
            'hidden_dim': 64,
            'marker_dim': 16,
        },
        'training': {
            'batch_size': 4,
            'num_epochs': 3,
            'learning_rate': 1e-4,
        }
    }
    
    # 使用快速模式训练
    result = train_zixin_complete_model(config_mode='quick', custom_config=custom_config)
    
    if result[0] is not None:
        print("\n✅ 快速测试完成")
        return True
    else:
        print("\n❌ 快速测试失败")
        return False


# ==================== 恢复训练模式 ====================

def resume_training(checkpoint_dir, additional_epochs=10, custom_config=None):
    """
    从现有检查点恢复训练
    
    Args:
        checkpoint_dir: 检查点目录
        additional_epochs: 额外训练的轮次
        custom_config: 自定义配置（可选）
    
    Returns:
        model: 恢复训练后的模型
        history: 训练历史记录
    """
    print(f"🔄 从检查点恢复训练: {checkpoint_dir}")
    
    # 查找最新的检查点文件
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        print(f"❌ 在 {checkpoint_dir} 中找不到检查点文件")
        return None, None
    
    # 找到最新的检查点（按epoch编号）
    epoch_checkpoints = [f for f in checkpoint_files if 'checkpoint_epoch' in f]
    if epoch_checkpoints:
        # 按epoch编号排序
        epoch_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0, reverse=True)
        checkpoint_path = os.path.join(checkpoint_dir, epoch_checkpoints[0])
    else:
        # 如果没有epoch检查点，尝试加载最终模型
        if 'final_model.pth' in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
        elif 'best_model.pth' in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        else:
            print(f"❌ 找不到有效的检查点文件")
            return None, None
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint_config = checkpoint['config']
        
        # 获取检查点中的训练轮次
        checkpoint_epoch = checkpoint.get('epoch', 0)
        
        # 构建恢复训练的配置
        resume_config = custom_config or {}
        resume_config['data'] = resume_config.get('data', {})
        resume_config['data']['output_dir'] = checkpoint_dir
        resume_config['training'] = resume_config.get('training', {})
        resume_config['training']['num_epochs'] = checkpoint_epoch + additional_epochs
        
        print(f"✓ 加载检查点: {os.path.basename(checkpoint_path)}")
        print(f"✓ 原训练轮次: {checkpoint_epoch}")
        print(f"✓ 新增训练轮次: {additional_epochs}")
        print(f"✓ 总训练轮次: {resume_config['training']['num_epochs']}")
        
        # 创建训练器
        trainer = AdvancedConstrainedArchitectureTrainer(checkpoint_config)
        
        # 手动设置历史记录，以便继续训练
        trainer.history = checkpoint.get('history', trainer.history)
        
        # 开始训练（训练器会自动加载最新的检查点）
        model, history = trainer.train()
        print("\n✅ 恢复训练完成")
        return model, history
    except Exception as e:
        print(f"❌ 恢复训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ==================== 推理测试 ====================

def test_model_inference(model_path, test_text, max_length=50):
    """
    测试训练好的模型进行推理
    
    Args:
        model_path: 模型路径
        test_text: 测试文本
        max_length: 生成的最大长度
    """
    print(f"🧪 测试模型推理: {model_path}")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config_dict = checkpoint['config']
    
    # 创建RGAConfig
    rga_config = RGAConfig()
    rga_config.vocab_size = config_dict.get('vocab_size', 50000)
    rga_config.dim = config_dict.get('embed_dim', 512)
    rga_config.num_units = 3  # 默认链式反应单元数量
    
    # 创建模型
    model = RGAIntegrator(config=rga_config)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型配置: {rga_config.__dict__}")
    print(f"测试文本: {test_text}")
    
    # 简单的编码（实际使用时需要更复杂的编码）
    # 这里我们使用一个简单的空格分词
    tokens = test_text.split()
    input_ids = [hash(token) % rga_config.vocab_size for token in tokens]
    
    # 限制序列长度
    if len(input_ids) > 128:
        input_ids = input_ids[:128]
    elif len(input_ids) < 128:
        input_ids = input_ids + [0] * (128 - len(input_ids))
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    # 生成文本
    print("\n生成结果:")
    with torch.no_grad():
        for i in range(max_length):
            outputs = model(input_tensor, num_cycles=1)
            logits = outputs['logits'][:, -1, :]  # 取最后一个token的logits
            next_token = torch.argmax(logits, dim=-1).item()
            
            # 简单的解码（实际使用时需要解码器）
            input_tensor = torch.cat([
                input_tensor, 
                torch.tensor([[next_token]], dtype=torch.long)
            ], dim=1)
            
            if next_token == 0:  # 结束标记
                break
            
            # 打印生成的token（简单演示）
            if i < max_length - 1:
                print(f"[token {i+1}: {next_token}]", end=" ")
    
    print("\n... 推理完成")


# ==================== 交互式训练界面 ====================

def interactive_training():
    """交互式训练界面"""
    print("紫心RGA模型训练系统")
    print("="*60)
    print("请选择模式:")
    print("1. 快速测试模式 (测试/调试)")
    print("2. 标准训练模式 (推荐)")
    print("3. 完整训练模式 (需要大量资源)")
    print("4. 自定义训练模式")
    print("5. 恢复训练模式")
    print("6. 推理测试模式")
    
    choice = input("\n请选择 (1-6): ").strip()
    
    if choice == "1":
        # 快速测试模式
        data_path = input("数据文件路径 (留空使用默认): ").strip()
        output_dir = input("输出目录 (留空使用默认): ").strip()
        quick_test_mode(data_path if data_path else None, output_dir if output_dir else None)
        
    elif choice == "2":
        # 标准训练模式
        data_path = input("数据文件路径 (留空使用默认): ").strip()
        output_dir = input("输出目录 (留空使用默认): ").strip()
        
        custom_config = {}
        if data_path:
            custom_config['data'] = {'data_path': data_path}
        if output_dir:
            if 'data' not in custom_config:
                custom_config['data'] = {}
            custom_config['data']['output_dir'] = output_dir
        
        train_zixin_complete_model(config_mode='standard', custom_config=custom_config if custom_config else None)
        
    elif choice == "3":
        # 完整训练模式
        data_path = input("数据文件路径 (留空使用默认): ").strip()
        output_dir = input("输出目录 (留空使用默认): ").strip()
        
        custom_config = {}
        if data_path:
            custom_config['data'] = {'data_path': data_path}
        if output_dir:
            if 'data' not in custom_config:
                custom_config['data'] = {}
            custom_config['data']['output_dir'] = output_dir
        
        train_zixin_complete_model(config_mode='full', custom_config=custom_config if custom_config else None)
        
    elif choice == "4":
        # 自定义训练模式
        print("\n自定义训练配置:")
        
        # 收集配置
        custom_config = {}
        
        # 数据配置
        print("\n📁 数据配置:")
        data_path = input("数据文件路径 (默认: ./data/LCCC-base_train.json): ").strip()
        output_dir = input("输出目录 (默认: ./output/custom_model): ").strip()
        seq_length = input("序列长度 (默认: 128): ").strip()
        vocab_size = input("词汇表大小 (默认: 20000): ").strip()
        max_samples = input("最大样本数 (默认: 无限制): ").strip()
        
        custom_config['data'] = {}
        if data_path: custom_config['data']['data_path'] = data_path
        if output_dir: custom_config['data']['output_dir'] = output_dir
        if seq_length: custom_config['data']['seq_length'] = int(seq_length)
        if vocab_size: custom_config['data']['vocab_size'] = int(vocab_size)
        if max_samples: 
            try:
                if max_samples.lower() in ['none', '无限制', '']:
                    custom_config['data']['max_samples'] = None
                else:
                    custom_config['data']['max_samples'] = int(max_samples)
            except:
                custom_config['data']['max_samples'] = None
        
        # 模型配置
        print("\n🤖 模型配置:")
        embed_dim = input("嵌入维度 (默认: 256): ").strip()
        hidden_dim = input("隐藏维度 (默认: 256): ").strip()
        
        custom_config['model'] = {}
        if embed_dim: custom_config['model']['embed_dim'] = int(embed_dim)
        if hidden_dim: custom_config['model']['hidden_dim'] = int(hidden_dim)
        
        # 训练配置
        print("\n⚙️ 训练配置:")
        batch_size = input("批次大小 (默认: 16): ").strip()
        num_epochs = input("训练轮次 (默认: 50): ").strip()
        learning_rate = input("学习率 (默认: 3e-5): ").strip()
        
        custom_config['training'] = {}
        if batch_size: custom_config['training']['batch_size'] = int(batch_size)
        if num_epochs: custom_config['training']['num_epochs'] = int(num_epochs)
        if learning_rate: custom_config['training']['learning_rate'] = float(learning_rate)
        
        # 开始训练
        train_zixin_complete_model(config_mode='custom', custom_config=custom_config)
        
    elif choice == "5":
        # 恢复训练模式
        checkpoint_dir = input("检查点目录: ").strip()
        additional_epochs = input("额外训练轮次 (默认: 10): ").strip()
        additional_epochs = int(additional_epochs) if additional_epochs else 10
        
        resume_training(checkpoint_dir, additional_epochs)
        
    elif choice == "6":
        # 推理测试模式
        print("\n🧪 推理测试模式:")
        model_path = input("模型文件路径 (如: ./output/model/final_model.pth): ").strip()
        test_text = input("测试文本: ").strip()
        max_length = input("生成最大长度 (默认: 50): ").strip()
        max_length = int(max_length) if max_length else 50
        
        if not model_path or not test_text:
            print("❌ 模型路径和测试文本不能为空")
        else:
            test_model_inference(model_path, test_text, max_length)
        
    else:
        print("无效选择，退出程序")


# ==================== 直接调用函数 ====================

def run_standard_training(data_path=None, output_dir=None):
    """
    直接运行标准训练（用于脚本调用）
    
    Args:
        data_path: 数据路径
        output_dir: 输出目录
    
    Returns:
        model, history
    """
    custom_config = {}
    if data_path:
        custom_config['data'] = {'data_path': data_path}
    if output_dir:
        if 'data' not in custom_config:
            custom_config['data'] = {}
        custom_config['data']['output_dir'] = output_dir
    
    return train_zixin_complete_model(config_mode='standard', custom_config=custom_config)


def run_quick_training(data_path=None, output_dir=None):
    """
    直接运行快速训练（用于脚本调用）
    
    Args:
        data_path: 数据路径
        output_dir: 输出目录
    
    Returns:
        model, history
    """
    custom_config = {}
    if data_path:
        custom_config['data'] = {'data_path': data_path}
    if output_dir:
        if 'data' not in custom_config:
            custom_config['data'] = {}
        custom_config['data']['output_dir'] = output_dir
    
    return train_zixin_complete_model(config_mode='quick', custom_config=custom_config)


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='紫心RGA模型训练系统')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'quick', 'standard', 'full', 'resume', 'inference'],
                       help='运行模式')
    parser.add_argument('--data_path', type=str, help='数据文件路径')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--checkpoint_dir', type=str, help='检查点目录（用于恢复训练）')
    parser.add_argument('--model_path', type=str, help='模型路径（用于推理）')
    parser.add_argument('--test_text', type=str, help='测试文本（用于推理）')
    parser.add_argument('--max_length', type=int, default=50, help='生成最大长度（用于推理）')
    parser.add_argument('--additional_epochs', type=int, default=10, help='额外训练轮次（用于恢复训练）')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        # 交互式模式
        interactive_training()
    elif args.mode == 'quick':
        # 快速训练模式
        run_quick_training(args.data_path, args.output_dir)
    elif args.mode == 'standard':
        # 标准训练模式
        run_standard_training(args.data_path, args.output_dir)
    elif args.mode == 'full':
        # 完整训练模式
        train_zixin_complete_model(config_mode='full')
    elif args.mode == 'resume':
        # 恢复训练模式
        if not args.checkpoint_dir:
            print("错误: 恢复训练需要指定 --checkpoint_dir 参数")
        else:
            resume_training(args.checkpoint_dir, args.additional_epochs)
    elif args.mode == 'inference':
        # 推理测试模式
        if not args.model_path:
            print("错误: 推理测试需要指定 --model_path 参数")
        elif not args.test_text:
            print("错误: 推理测试需要指定 --test_text 参数")
        else:
            test_model_inference(args.model_path, args.test_text, args.max_length)
    else:
        print(f"未知模式: {args.mode}")