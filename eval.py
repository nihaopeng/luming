print("✅ 正在加载依赖库...")
import argparse
import time
import torch
from utils import TokenConfig, setup_seed, Logger
from transformers import AutoTokenizer,AutoModelForCausalLM

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

class miniStreamer:
    def __init__(self,args,tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.output_ids = []
        self.output_text_idx = 0
        self.response = ""
        self.is_first_output = True
    
    def put(self, value):
        ids = value.tolist()
        if self.is_first_output:
            print(f'🤖: ',end="")
            self.is_first_output = False
            ids = [ids[0][-1]] # 第一次会将prompt也传进来，此时取最后一个token即可。
        self.output_ids.extend(ids)
        text = self.tokenizer.decode(self.output_ids, skip_special_tokens=True)
        if self.args.stream and self.output_text_idx < len(self.response)-3:
            print(self.response[self.output_text_idx], end='', flush=True)
            self.output_text_idx += 1
        self.response = text
    
    def end(self):
        # 根据 stream 开关决定是否输出
        if self.args.stream:
            print(self.response[self.output_text_idx+1:])  # 流式模式：结束时换行
        else:
            print(f'{self.response}')  # 非流式：一次性输出完整 response
        self.output_ids = []
        self.output_text_idx = 0
        self.response = ""
        self.is_first_output = True

def eval(args,prompts):
    model,tokenizer = init_model(args)
    sep = args.sep.split(",")
    print(sep)
    assert len(sep)==3,"sep参数需要三个token，用逗号分隔"
    token_config = TokenConfig(sep[0],sep[1],sep[2])
    input_mode = int(input('[0] 自动测试 [1] 手动输入 : '))
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    
    conversation = []
    streamer = miniStreamer(args,tokenizer)
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        if prompt == "quit" or prompt == "exit": break
        setup_seed(2026) # or setup_seed(random.randint(0, 2048))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        inputs = tokenizer.apply_chat_template(**templates) if args.eval_mode=="sft" else token_config.response_start_token + prompt
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        st = time.time()
        generated_ids = model.generate(
            inputs=inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=args.max_seq_len, streamer=streamer,
            pad_token_id=tokenizer.convert_tokens_to_ids(token_config.pad_token),
            eos_token_id=tokenizer.convert_tokens_to_ids(token_config.response_end_token),
            top_p=args.top_p, temperature=args.temperature
        )
        conversation.append({"role": "assistant", "content": streamer.response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    
    print("✅ 正在解析参数...")
    setup_seed(42)
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--tokenizer_path', default='tokenizer/minimind', type=str, help="tokenizer数据加载路径")
    parser.add_argument('--from_weight', default='sft', type=str, help="权重路径，包含文件名")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=256, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument("--sep", type=str, default="<|im_start|>assistant,<|im_end|>,<|endoftext|>", help="微调使用的起始token，结束token和填充token")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--eval_mode', default='pretrain', type=str, choices=["pretrain","sft"], help="测试类型[pretrain/sft]")
    parser.add_argument('--stream', default=0, type=int, choices=[0,1], help="是否流式输出?(是/否)[1]/[0]")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
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