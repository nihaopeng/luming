# chat.py
def init_model(args):
    print("✅ 加载模型...")
    moe_suffix = '_moe' if args.use_moe else ''
    ckpt_path = os.path.join(args.save_dir, f"{args.weight}_{args.hidden_size}{moe_suffix}.pth")
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    model = MiniMindForCausalLM(config)
    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state_dict)
    print("✅ 加载参数...")
    model.to(args.device).eval()
    print("✅ 加载编码器...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    print("✅ 全部加载成功！开始对话（输入 'quit' 退出）")
    return model,tokenizer

def eval(args,prompts):
    model,tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试 [1] 手动输入 : '))
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    
    conversation = []
    for prompt in prompt_iter:
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        # 使用tokenizer_config中的chat_template构造输入prompt
        inputs = tokenizer.apply_chat_template(**templates) if args.eval_mode == "sft" else tokenizer.bos_token + prompt
        inputs = tokenizer(inputs, return_tensors="pt").to(args.device)
        if input_mode ==0: print(f'💬: {prompt}')
        st = time.time()
        response = ""
        outputs_id = [[]] # [1,input_len]
        out_text_len = 0
        print(f"prompt:{prompt}")
        with torch.no_grad():
            if args.stream:
                print(f'🤖: ',end="")
                for token_id in model.generate_stream(
                    inputs["input_ids"],
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id
                ):
                    outputs_id[0].append(token_id)
                    # ✅ 关键：decode 整个序列，不是单个 token
                    response = tokenizer.decode(outputs_id[0], skip_special_tokens=True)
                    # 只输出倒数第三个字符，避免新输出token由于不完整导致乱码。
                    print(response[out_text_len:-3], end="", flush=True)
                    out_text_len = out_text_len + len(response[out_text_len:-3])
                print(response[-3:], end="", flush=True)
            else:
                outputs_id = model.generate(
                    input_ids=inputs["input_ids"],
                    # attention_mask=inputs["attention_mask"],
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id  # 关键：用 <|im_end|> 作为结束符
                )
                response = tokenizer.decode(outputs_id[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                print(f'🤖: {response}')
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(outputs_id[0]) - len(inputs["input_ids"][0])
        print(f'[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    print("✅ 正在加载依赖库...")
    import argparse
    import os
    import time
    import torch
    from transformers import AutoTokenizer
    from config import MiniMindConfig
    from model_luming import MiniMindForCausalLM  # 假设你的模型定义在 model.py 中
    from utils import setup_seed
    print("✅ 正在解析参数...")
    setup_seed(42)
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--tokenizer_path', default='tokenizer/minimind', type=str, help="tokenizer数据加载路径")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft', type=str, help="权重名称前缀")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=256, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=3, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--eval_mode', default='pretrain', type=str, choices=["pretrain","sft"], help="测试类型[pretrain/sft]")
    parser.add_argument('--stream', default=0, type=int, choices=[0,1], help="是否流式输出?(是/否)[1]/[0]")
    args = parser.parse_args()
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