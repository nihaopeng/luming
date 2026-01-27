# luming - 轻量级语言模型训练框架

luming 是一个简洁高效的语言模型训练和推理框架, 主要参考[minimind](https://github.com/jingyaogong/minimind)项目（大部分代码是直接迁移的），专为快速入门和实践语言模型而设计。支持预训练、指令微调(SFT)、评估和部署的完整生命周期。支持qwen0.6B微调与评测。

## 环境

train
---

+ 1，torch

+ 2，transformer

web
---
`pip install starlette toml uvicorn`

## 🚀 快速开始

### 1. 预训练

从头开始预训练 minimind-104M 模型
```bash
CUDA_VISIBLE_DEVICES=1 python train.py --dtype=float16 --data_path "./dataset/pretrain_hq.jsonl" --save_weight "./out/minimind_pretrain" --from_weight "none" --hidden_size 768 --num_hidden_layers 16 --use_compile 1 --epochs 6 --sep "<|im_start|>,<|im_end|>,<|endoftext|>"
```

### 2. 指令微调 (SFT)

基于minimind-104M预训练模型微调，使用[sft_mini_512](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)数据集
```bash
CUDA_VISIBLE_DEVICES=1 python train.py --dtype=float16 --data_path "./dataset/sft_mini_512.jsonl" --tokenizer_path "./tokenizer/minimind" --train_mode "sft" --save_weight "./out/minimind_sft" --from_weight "./out/minimind_pretrain" --hidden_size 768 --num_hidden_layers 16 --use_compile 0 --epochs 2 --sep "<|im_start|>assistant,<|im_end|>,<|endoftext|>"
```

基于qwen0.6B模型微调，使用[sft_mini_512](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)数据集
```bash
# 缩小微调数据集的规模，由于qwen0.6B已经具有很好的生成能力，所以这里主要做对齐工作，不需要太大的数据集。让模型学到回答问题的能力即可。
head -n 50000 dataset/sft_mini_512.jsonl  > dataset/sft_mini_512_head_50000.jsonl

CUDA_VISIBLE_DEVICES=1 python train.py --dtype=float16 --data_path "./dataset/sft_mini_512_head_50000.jsonl" --tokenizer_path "./tokenizer/qwen0.6Bbase" --train_mode "sft" --save_weight "./out/qwen_sft" --from_weight "./out/qwen0.6Bbase" --use_compile 0 --epochs 2 --sep "<|im_start|>assistant,<|im_end|>,<|endoftext|>"
```

### 3. 推理测试
```bash
# pretrain minimind eval
CUDA_VISIBLE_DEVICES=2 python eval.py --tokenizer_path "./tokenizer/minimind" --from_weight "./out/minimind_pretrain" --sep "<|im_start|>,<|im_end|>,<|endoftext|>" --eval_mode "pretrain" --stream 1

# pretrain qwen0.6B eval
CUDA_VISIBLE_DEVICES=2 python eval.py --tokenizer_path "./tokenizer/qwen0.6Bbase" --from_weight "./out/qwen0.6Bbase" --sep "<|im_start|>,<|im_end|>,<|endoftext|>" --eval_mode "pretrain" --stream 0

# sft minimind eval
CUDA_VISIBLE_DEVICES=2 python eval.py --tokenizer_path "./tokenizer/qwen0.6Bbase" --from_weight "./out/qwen_sft" --sep "<|im_start|>assistant,<|im_end|>,<|endoftext|>" --eval_mode "sft" --stream 1

# sft qwen0.6B eval
CUDA_VISIBLE_DEVICES=2 python eval.py --tokenizer_path "./tokenizer/qwen0.6Bbase" --from_weight "./out/qwen_sft" --sep "<|im_start|>assistant,<|im_end|>,<|endoftext|>" --eval_mode "sft" --stream 1
```

## 🚀 web

`pip install starlette toml uvicorn`

`python -m web.main`

## 🏗️ 核心架构

### 模型组件
- **MiniMindForCausalLM**: 主要模型类，包含语言建模头 luming:365-373 
- **Attention**: 多头注意力机制，支持 RoPE 位置编码 luming:82-145 
- **FeedForward/MOEFeedForward**: 前馈网络，支持 MoE 架构 luming:147-278 

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
- 灵活的学习率调度 luming:147-148 
- 完整的参数统计和分析工具 luming:121-131 

更多详细信息请参考各源码文件的注释和文档。