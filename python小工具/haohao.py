#!/usr/bin/env python3
"""
OpenLearning RGA框架 - 最终版完全正确的使用示例
确认：所有组件都需要3个参数 (Q, K, V)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

print("="*70)
print("🚀 OpenLearning RGA框架 - 最终版完全正确的使用示例")
print("="*70)

# ============================================================================
# 1. 导入框架
# ============================================================================

from openlearning.core import RGAConfig, create_rga_engine
from openlearning.layers import FixedRMSNorm, create_attention_subnet

print("✅ 框架导入成功")

# ============================================================================
# 2. 核心发现：所有组件都需要3个参数！
# ============================================================================

print("\n" + "="*70)
print("🔍 核心发现：所有组件都需要3个参数 (Q, K, V)")
print("="*70)

# 创建配置
config = RGAConfig(vocab_size=10000, dim=256)

# 创建RGA引擎
engine = create_rga_engine(config=config)
print("✅ RGA引擎创建成功")

# 创建测试数据
Q = torch.randn(2, 10, 256)
K = torch.randn(2, 10, 256)
V = torch.randn(2, 10, 256)

# 对于自注意力，Q、K、V可以是相同的
x = torch.randn(2, 10, 256)

print(f"\n📊 测试数据形状:")
print(f"  Q: {Q.shape}")
print(f"  K: {K.shape}")
print(f"  V: {V.shape}")
print(f"  x (用于自注意力): {x.shape}")

# ============================================================================
# 3. 测试所有组件都需要3个参数
# ============================================================================

print("\n" + "="*70)
print("🧪 测试所有组件都需要3个参数")
print("="*70)

# 测试1: RGAEngine.process_state()
print("\n1️⃣  测试 RGAEngine.process_state():")
try:
    # 错误：engine.process_state(x)  ❌ 缺少2个参数
    # 正确：engine.process_state(Q, K, V) ✅
    output = engine.process_state(Q, K, V)
    print(f"   ✅ 成功: 使用3个参数")
    print(f"      输出类型: {type(output)}")
    if isinstance(output, torch.Tensor):
        print(f"      输出形状: {output.shape}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试2: 注意力层
print("\n2️⃣  测试注意力层:")
attention_types = ["vkq", "qvk", "kqv"]
for attn_type in attention_types:
    try:
        attn = create_attention_subnet(attn_type, dim=256)
        output = attn(Q, K, V)
        print(f"   ✅ {attn_type.upper()}注意力: 成功")
        if isinstance(output, tuple):
            print(f"      返回元组, 长度={len(output)}")
            print(f"      第一个元素形状: {output[0].shape}")
    except Exception as e:
        print(f"   ❌ {attn_type.upper()}注意力失败: {e}")

# 测试3: FixedRMSNorm（只需要1个参数）
print("\n3️⃣  测试 FixedRMSNorm（只需要1个参数）:")
try:
    norm = FixedRMSNorm(dim=256)
    output = norm(x)  # 只需要1个参数
    print(f"   ✅ 成功: {x.shape} -> {output.shape}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# ============================================================================
# 4. 创建完全正确的模型类
# ============================================================================

print("\n" + "="*70)
print("🏗️  创建完全正确的模型类")
print("="*70)

class FinalRGAModel(nn.Module):
    """
    最终版RGA模型
    关键：所有RGA组件都需要3个参数 (Q, K, V)
    """
    
    def __init__(self, dim=256, use_engine=True, num_heads=3):
        super().__init__()
        self.dim = dim
        self.use_engine = use_engine
        self.num_heads = num_heads
        
        if use_engine:
            # 使用RGAEngine
            self.config = RGAConfig(vocab_size=10000, dim=dim)
            self.engine = create_rga_engine(config=self.config)
            print(f"  ✅ 使用RGAEngine (dim={dim})")
        else:
            # 使用多个注意力头
            self.attention_heads = nn.ModuleList()
            attention_types = ["vkq", "qvk", "kqv"][:num_heads]
            
            for attn_type in attention_types:
                try:
                    attn = create_attention_subnet(attn_type, dim=dim)
                    self.attention_heads.append(attn)
                    print(f"  ✅ 添加{attn_type.upper()}注意力头")
                except Exception as e:
                    print(f"  ❌ 创建{attn_type.upper()}注意力头失败: {e}")
            
            if len(self.attention_heads) == 0:
                raise ValueError("没有成功创建任何注意力头")
        
        # 公共层
        self.norm1 = FixedRMSNorm(dim=dim)
        self.norm2 = FixedRMSNorm(dim=dim)
        self.output_proj = nn.Linear(dim, dim)
        
    def forward(self, Q, K=None, V=None):
        """
        前向传播
        参数:
            Q: Query张量 [batch, seq_len, dim]
            K: Key张量 (可选，默认使用Q)
            V: Value张量 (可选，默认使用Q)
        """
        # 如果没有提供K和V，使用Q（自注意力）
        if K is None:
            K = Q
        if V is None:
            V = Q
        
        if self.use_engine:
            # 使用RGAEngine - 需要3个参数
            x = self.engine.process_state(Q, K, V)
        else:
            # 使用多个注意力头
            head_outputs = []
            for attn_head in self.attention_heads:
                attn_output = attn_head(Q, K, V)
                
                # 注意力层返回元组，取第一个元素
                if isinstance(attn_output, tuple):
                    attn_output = attn_output[0]
                
                head_outputs.append(attn_output)
            
            # 合并所有头的输出（简单平均）
            x = torch.stack(head_outputs, dim=0).mean(dim=0)
        
        # 后续处理
        x = self.norm1(x)
        x = self.output_proj(x)
        x = self.norm2(x)
        
        return x

# ============================================================================
# 5. 创建和使用模型
# ============================================================================

print("\n" + "="*70)
print("🎯 创建和使用模型")
print("="*70)

# 创建两种模型
print("1. 创建使用RGAEngine的模型:")
try:
    model_engine = FinalRGAModel(dim=256, use_engine=True)
    print(f"   ✅ 创建成功")
    print(f"   📊 参数数量: {sum(p.numel() for p in model_engine.parameters()):,}")
except Exception as e:
    print(f"   ❌ 创建失败: {e}")
    model_engine = None

print("\n2. 创建使用注意力头的模型:")
try:
    model_attention = FinalRGAModel(dim=256, use_engine=False, num_heads=3)
    print(f"   ✅ 创建成功")
    print(f"   📊 参数数量: {sum(p.numel() for p in model_attention.parameters()):,}")
except Exception as e:
    print(f"   ❌ 创建失败: {e}")
    model_attention = None

# 测试前向传播
print("\n3. 测试前向传播:")
test_Q = torch.randn(2, 8, 256)
test_K = torch.randn(2, 8, 256)
test_V = torch.randn(2, 8, 256)

if model_engine:
    try:
        with torch.no_grad():
            output = model_engine(test_Q, test_K, test_V)
        print(f"   ✅ Engine模型: Q{test_Q.shape} -> {output.shape}")
    except Exception as e:
        print(f"   ❌ Engine模型失败: {e}")

if model_attention:
    try:
        with torch.no_grad():
            output = model_attention(test_Q, test_K, test_V)
        print(f"   ✅ Attention模型: Q{test_Q.shape} -> {output.shape}")
    except Exception as e:
        print(f"   ❌ Attention模型失败: {e}")

# ============================================================================
# 6. 完整训练示例
# ============================================================================

print("\n" + "="*70)
print("🎓 完整训练示例")
print("="*70)

class TextDataset(Dataset):
    """文本风格数据集，模拟NLP任务"""
    
    def __init__(self, num_samples=1000, seq_len=20, dim=256):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dim = dim
        
        # 生成随机数据（模拟文本嵌入）
        np.random.seed(42)
        self.Q = np.random.randn(num_samples, seq_len, dim).astype(np.float32)
        self.K = np.random.randn(num_samples, seq_len, dim).astype(np.float32)
        self.V = np.random.randn(num_samples, seq_len, dim).astype(np.float32)
        
        # 生成目标（简单变换）
        self.targets = 0.5 * self.Q + 0.3 * self.K + 0.2 * self.V
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'Q': torch.FloatTensor(self.Q[idx]),
            'K': torch.FloatTensor(self.K[idx]),
            'V': torch.FloatTensor(self.V[idx]),
            'target': torch.FloatTensor(self.targets[idx])
        }

def train_final_model():
    """训练最终模型"""
    print("🚀 开始训练最终模型...")
    
    # 创建数据集
    dataset = TextDataset(num_samples=500, seq_len=12, dim=256)
    
    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # 创建模型（使用注意力头，因为RGAEngine可能需要更多调参）
    model = FinalRGAModel(dim=256, use_engine=False, num_heads=3)
    
    # 设置训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"📊 训练样本: {train_size}, 验证样本: {val_size}")
    print(f"📊 批次大小: 4, 设备: {device}")
    
    # 训练循环
    num_epochs = 3
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            Q = batch['Q'].to(device)
            K = batch['K'].to(device)
            V = batch['V'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(Q, K, V)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  批次 {batch_idx}: 损失 = {loss.item():.6f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                Q = batch['Q'].to(device)
                K = batch['K'].to(device)
                V = batch['V'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(Q, K, V)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.6f}")
        print(f"  验证损失: {avg_val_loss:.6f}")
        
        # 保存每个epoch的模型
        torch.save(model.state_dict(), f'final_model_epoch_{epoch+1}.pth')
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'dim': 256, 'use_engine': False, 'num_heads': 3},
        'epoch': num_epochs
    }, 'final_rga_model_complete.pth')
    
    print(f"\n💾 模型已保存: final_rga_model_complete.pth")
    
    return model

# 运行训练
try:
    trained_model = train_final_model()
    print("✅ 训练完成！")
except Exception as e:
    print(f"❌ 训练失败: {e}")

# ============================================================================
# 7. 推理和部署示例
# ============================================================================

print("\n" + "="*70)
print("🔮 推理和部署示例")
print("="*70)

def inference_example():
    """推理示例"""
    print("🎯 推理示例:")
    
    # 创建模型（用于推理）
    model = FinalRGAModel(dim=256, use_engine=False, num_heads=3)
    
    # 加载训练好的权重（如果存在）
    try:
        checkpoint = torch.load('final_rga_model_complete.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 加载预训练模型成功")
    except:
        print("⚠️  使用随机初始化的模型")
    
    model.eval()
    
    # 创建测试输入
    batch_size = 1
    seq_len = 15
    dim = 256
    
    Q_test = torch.randn(batch_size, seq_len, dim)
    K_test = torch.randn(batch_size, seq_len, dim)
    V_test = torch.randn(batch_size, seq_len, dim)
    
    print(f"\n📊 测试输入:")
    print(f"  Q: {Q_test.shape}")
    print(f"  K: {K_test.shape}")
    print(f"  V: {V_test.shape}")
    
    # 进行推理
    with torch.no_grad():
        output = model(Q_test, K_test, V_test)
    
    print(f"\n🎯 推理结果:")
    print(f"  输出形状: {output.shape}")
    print(f"  输出统计:")
    print(f"    最小值: {output.min():.4f}")
    print(f"    最大值: {output.max():.4f}")
    print(f"    均值: {output.mean():.4f}")
    print(f"    标准差: {output.std():.4f}")
    
    return output

# 运行推理
try:
    inference_result = inference_example()
except Exception as e:
    print(f"❌ 推理失败: {e}")

# ============================================================================
# 8. 创建最终使用指南
# ============================================================================

print("\n" + "="*70)
print("📚 最终使用指南")
print("="*70)

final_guide = """
# OpenLearning RGA框架 - 最终使用指南

## 重要发现：
所有RGA组件都需要3个参数：Query(Q), Key(K), Value(V)

## 1. 正确导入
```python
from openlearning.core import RGAConfig, create_rga_engine
from openlearning.layers import FixedRMSNorm, create_attention_subnet"""