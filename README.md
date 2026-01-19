# luming - 轻量级语言模型训练框架

luming 是一个简洁高效的语言模型训练和推理框架，专为快速入门和实践语言模型而设计。支持预训练、指令微调(SFT)、评估和部署的完整生命周期。

## 🚀 快速开始

### 1. 预训练
```bash
# 从头开始预训练 Small-26M 模型
./run.sh pretrain

# 预训练 Base-104M 模型
python train.py --hidden_size 768 --num_hidden_layers 16 --save_weight pretrain
```

### 2. 指令微调 (SFT)
```bash
# 基于预训练权重进行指令微调
./run.sh sft

# 手动指定参数微调
python train.py --from_weight pretrain --sft 1 --hidden_size 512 --num_hidden_layers 8
```

### 3. 推理测试
```bash
# 交互式对话
./run.sh eval

# 流式输出
python eval.py --weight sft --stream 1 --eval_mode sft
```

## 📊 模型规格

| 模型 | hidden_size | num_hidden_layers | 参数量 | 适用场景 |
|------|-------------|-------------------|--------|----------|
| Small-26M | 512 | 8 | ~26M | 快速实验、学习 |
| Base-104M | 768 | 16 | ~104M | 基础应用 |
| MoE-145M | 640 | 8 | ~145M | 高效推理 | luming:24-28 

## 🏗️ 核心架构

### 模型组件
- **MiniMindForCausalLM**: 主要模型类，包含语言建模头 luming:365-373 
- **Attention**: 多头注意力机制，支持 RoPE 位置编码 luming:82-145 
- **FeedForward/MOEFeedForward**: 前馈网络，支持 MoE 架构 luming:147-278 

### 训练系统
- **分布式训练**: 支持 DDP 多 GPU 训练 luming:115-117 
- **混合精度**: 自动混合精度训练，节省显存 luming:89-92 
- **检查点管理**: 自动保存和恢复训练状态 luming:66-119 

## 📁 项目结构

```
├── model_luming.py    # 模型架构定义
├── train.py          # 训练脚本
├── eval.py           # 推理评估脚本
├── utils.py          # 工具函数
├── config.py         # 配置类
├── dataloader.py     # 数据加载器
├── run.sh           # 快速启动脚本
└── tokenizer/       # 分词器文件
```

## ⚙️ 主要特性

### 1. 灵活的模型配置
通过 `MiniMindConfig` 类配置模型参数 luming:15-48 ：
- 支持标准 Transformer 和 MoE 架构
- 可配置注意力头数、层数、隐藏层维度
- 支持 Flash Attention 和 RoPE 位置编码

### 2. 完整的训练流程
- **预训练模式**: 使用 `PretrainDataset` 处理原始文本 luming:99-100 
- **SFT模式**: 使用 `SFTDataset` 处理对话数据，支持聊天模板
- **梯度累积**: 支大批量训练 luming:39-50 

### 3. 高效推理
- **KV缓存**: 加速生成过程 luming:386-398 
- **流式生成**: 实时输出生成内容 luming:425-478 
- **采样策略**: 支持温度调节和 nucleus sampling luming:400-413 

## 🛠️ 使用示例

### 自定义训练
```bash
python train.py \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --epochs 6 \
    --data_path ./dataset/pretrain_hq.jsonl \
    --save_weight my_model
```

### 模型推理
```python
from model_luming import MiniMindForCausalLM
from config import MiniMindConfig
from transformers import AutoTokenizer

# 加载模型
config = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
model = MiniMindForCausalLM(config)
tokenizer = AutoTokenizer.from_pretrained('./tokenizer')

# 生成文本
inputs = tokenizer("你好，", return_tensors="pt")
outputs = model.generate(inputs["input_ids"], temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📝 注意事项

1. **数据格式**: 预训练数据使用 JSONL 格式，每行一个文本样本
2. **显存需求**: Small-26M 模型约需 2GB 显存，Base-104M 约 4GB
3. **分词器**: 使用自定义分词器，位于 `tokenizer/` 目录
4. **MoE训练**: 使用 MoE 时需要调整 `--num_experts_per_tok` 等参数

## Notes

这个 README 专注于 MiniMind 框架的核心功能和快速使用方法。框架还包含许多高级特性，如：
- 详细的日志记录和 WandB 集成
- 灵活的学习率调度 luming:147-148 
- 完整的参数统计和分析工具 luming:121-131 

更多详细信息请参考各源码文件的注释和文档。