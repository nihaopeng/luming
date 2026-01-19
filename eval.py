# chat.py
import os
import torch
from transformers import AutoTokenizer
from config import MiniMindConfig
from model_luming import MiniMindForCausalLM  # 假设你的模型定义在 model.py 中
from utils import setup_seed

def main():
    # === 配置 ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = 768          # 根据你训练时的配置调整
    num_hidden_layers = 16     # 同上
    use_moe = False            # 根据实际设置
    save_dir = "./out"
    weight_name = "pretrain"   # 与训练时 args.save_weight 一致
    moe_suffix = '_moe' if use_moe else ''
    ckpt_path = os.path.join(save_dir, f"{weight_name}_{hidden_size}{moe_suffix}.pth")

    # === 初始化模型和 tokenizer ===
    config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, use_moe=use_moe)
    model = MiniMindForCausalLM(config)
    
    # 加载权重（注意：训练时保存的是 .half()，所以需 map_location + float16）
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # tokenizer（假设你用的是类似 LLaMA 的 tokenizer，或自定义的）
    # 如果你没有用 transformers tokenizer，而是自定义的，请替换为你的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=False)  # 仅用于 tokenization，不加载模型

    print("✅ 模型加载成功！开始对话（输入 'quit' 退出）\n")

    # === 多轮对话历史 ===
    history = []

    while True:
        user_input = input("👤 用户: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break

        # 构造对话上下文（按你的训练格式）
        # 训练数据格式：<|im_start|>...<|im_end|>
        history.append(f"<|im_start|>{user_input}")
        prompt = " ".join(history)

        # 编码
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id  # 关键：用 <|im_end|> 作为结束符
            )

        # 解码生成部分
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()

        # 移除可能的 <|im_end|> 及之后内容
        if tokenizer.eos_token in response:
            response = response.split(tokenizer.eos_token)[0].strip()

        print(f"🤖 助手: {response}")

        # 将助手回复加入历史（用于下一轮）
        history.append(f"{response}{tokenizer.eos_token}")

        # 可选：限制历史长度防止过长
        if len(history) > 6:  # 保留最近3轮对话
            history = history[-4:]

    print("👋 再见！")

if __name__ == "__main__":
    setup_seed(42)
    main()