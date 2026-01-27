print("✅ 正在加载依赖库...")
import argparse
import time
import torch
import torch.nn.functional as F
from utils import TokenConfig, setup_seed, Logger
from transformers import AutoTokenizer,AutoModelForCausalLM

def eval_args():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--tokenizer_path', default='./tokenizer/qwen0.6Bbase', type=str, help="tokenizer数据加载路径")
    parser.add_argument('--from_weight', default='./out/qwen_sft', type=str, help="权重路径，包含文件名")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=2, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument("--sep", type=str, default="<|im_start|>assistant,<|im_end|>,<|endoftext|>", help="微调使用的起始token，结束token和填充token")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--eval_mode', default='sft', type=str, choices=["pretrain","sft"], help="测试类型[pretrain/sft]")
    parser.add_argument('--stream', default=1, type=int, choices=[0,1], help="是否流式输出?(是/否)[1]/[0]")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    args = parser.parse_args()
    return args

def init_model(args):
    Logger("✅ 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    Logger("✅ 加载参数...")
    model = AutoModelForCausalLM.from_pretrained(
        args.from_weight,  # 或本地路径包含 model.safetensors
    )
    Logger("✅ 全部加载成功！")
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(args.device), tokenizer

def stream_generate(
    model,
    tokenizer,
    token_config: TokenConfig,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop_on_eos: bool = True,
):
    """
    流式生成，但基于完整 token 序列解码后按字符逐步 yield。
    行为：
      - 每次生成新 token 后，将所有 new_token_ids 整体 decode 为 output_text。
      - 当 output_text 长度 >= 3 时，开始 yield 字符（从第0个开始）。
      - 之后每步 yield 一个新字符（即 output_text[len(yielded_chars)]）。
      - 最后确保所有字符都被 yield（包括末尾可能因 token 边界延迟出现的部分）。
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_key_values = None
    new_token_ids: list[int] = []
    yielded_char_count = 0  # 已经 yield 的字符数
    with torch.no_grad():
        for step in range(max_new_tokens):
            if step == 0:
                current_input = input_ids
            else:
                current_input = new_token_id.unsqueeze(0)
            outputs = model(
                input_ids=current_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            # Temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            token_id = next_token.item()
            # Check EOS
            if stop_on_eos and token_id == tokenizer.convert_tokens_to_ids(token_config.response_end_token):
                break
            new_token_ids.append(token_id)
            # Decode the full sequence of new tokens
            output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            # Yield characters one by one that haven't been yielded yet
            while yielded_char_count < len(output_text)-3:
                yield output_text[yielded_char_count]
                yielded_char_count += 1
            # Update state for next iteration
            new_token_id = next_token
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)
        # Final flush: in case decoding after loop adds more characters (e.g., due to BPE merging)
        final_output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        while yielded_char_count < len(final_output_text):
            yield final_output_text[yielded_char_count]
            yielded_char_count += 1

def eval(args,prompts):
    model,tokenizer = init_model(args)
    sep = args.sep.split(",")
    Logger(sep)
    assert len(sep)==3,"sep参数需要三个token，用逗号分隔"
    token_config = TokenConfig(sep[0],sep[1],sep[2])
    input_mode = int(input('[0] 自动测试 [1] 手动输入 : '))
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    conversation = []
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        if prompt == "quit" or prompt == "exit": break
        setup_seed(2026) # or setup_seed(random.randint(0, 2048))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        input_full_prompt = tokenizer.apply_chat_template(**templates) if args.eval_mode=="sft" else token_config.response_start_token + prompt
        response = ""
        print(f'🤖: ',end="")
        for token_str in stream_generate(model,tokenizer,token_config,input_full_prompt,max_new_tokens=args.max_seq_len):
            print(token_str,end="",flush=True)
            response += token_str
        print()
        conversation.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    
    print("✅ 正在解析参数...")
    setup_seed(42)
    args = eval_args()
    print("✅ 参数解析完成")
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的？',
        '请写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程。',
        '如果明天下雨，我应该如何出门？',
        '比较一下猫和狗作为宠物的优缺点。',
        '解释什么是机器学习？',
        '推荐一些中国的美食。'
    ]
    eval(args,prompts)