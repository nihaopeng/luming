import json
import os
import sqlite3
from starlette.routing import Router
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse, FileResponse
from web.core.config import configuration
from utils import Logger, TokenConfig
from eval import eval_args,init_model, stream_generate

args = eval_args()
args.tokenizer_path=configuration["LLM"]["TOKENIZER_PATH"]
args.from_weight=configuration["LLM"]["FROM_WEIGHT"]
args.historys=int(configuration["LLM"]["HISTORYS"])
args.sep=configuration["LLM"]["SEP"]
args.device=configuration["LLM"]["DEVICE"]
args.max_seq_len=int(configuration["LLM"]["MAX_SEQ_LEN"])

model,tokenizer = init_model(args)

sep = args.sep.split(",")
Logger(sep)
assert len(sep)==3,"sep参数需要三个token，用逗号分隔"
token_config = TokenConfig(sep[0],sep[1],sep[2])

chat_router = Router()

async def chat_stream(conversation):
    conversation = conversation[-args.historys-1:]
    print(f"conversation[-args.historys-1:]:{conversation}")
    templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
    input_full_prompt = tokenizer.apply_chat_template(**templates)
    for token_str in stream_generate(model,tokenizer,token_config,input_full_prompt,max_new_tokens=args.max_seq_len):
        yield token_str

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
    print(f"conversation:{conversation}")
    return StreamingResponse(chat_stream(conversation), media_type="text/event-stream")