import argparse
import json
import os
import sqlite3
from starlette.routing import Router
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse, FileResponse
import torch
from transformers import AutoTokenizer
from web.core.config import configuration
from config import MiniMindConfig
from model_luming import MiniMindForCausalLM

parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
parser.add_argument('--tokenizer_path', default=configuration["LLM"]["TOKENIZER_PATH"], type=str, help="tokenizer数据加载路径")
parser.add_argument('--save_dir', default=configuration["LLM"]["WEIGHT_PATH"], type=str, help="模型权重目录")
parser.add_argument('--weight', default='sft', type=str, help="权重名称前缀")
parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
parser.add_argument('--max_new_tokens', default=256, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
parser.add_argument('--historys', default=1, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
parser.add_argument('--eval_mode', default='sft', type=str, choices=["pretrain","sft"], help="测试类型[pretrain/sft]")
args = parser.parse_args()

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

model,tokenizer = init_model(args)

async def chat_stream(conversation):
    conversation = conversation[-args.historys:] if args.historys else []
    templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
    # 使用tokenizer_config中的chat_template构造输入prompt
    inputs = tokenizer.apply_chat_template(**templates)
    inputs = tokenizer(inputs, return_tensors="pt").to(args.device)
    response = ""
    outputs_id = [[]] # [1,input_len]
    out_text_len = 0
    with torch.no_grad():
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
            yield response[out_text_len:-3]
            out_text_len = out_text_len + len(response[out_text_len:-3])
        yield response[-3:]

chat_router = Router()

@chat_router.route("/update_content",methods=["POST"])
async def update_content(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    body = None
    if "application/json" in content_type:
        # 处理 JSON body
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    username = body["username"]
    content_to_save = json.dumps(body["conversation"], ensure_ascii=False)
    db_name = configuration["DB"]["DB_NAME"]
    os.makedirs(os.path.dirname(db_name), exist_ok=True)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # 创建表（如果不存在）
    cursor.execute("UPDATE chats SET content = ? WHERE username = ?", (content_to_save, username))
    conn.commit()
    conn.close()
    return JSONResponse(content={"status":"updated"},status_code=200)

@chat_router.route("/chat_stream",methods=["POST"])
async def chat_stream_interface(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    body = None
    if "application/json" in content_type:
        # 处理 JSON body
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    conversation = body["conversation"]
    # print(body["username"],conversation[:-1])
    # print(conversation)
    return StreamingResponse(chat_stream(conversation), media_type="text/event-stream")