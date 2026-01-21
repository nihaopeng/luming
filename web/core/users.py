import json
import os
import sqlite3
from starlette.routing import Router
from starlette.requests import Request
from starlette.responses import JSONResponse
from web.core.config import configuration

def create_db():
    db_name = configuration["DB"]["DB_NAME"]
    os.makedirs(os.path.dirname(db_name), exist_ok=True)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # 创建表（如果不存在）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,  -- 唯一用户名
            content TEXT NOT NULL           -- 聊天内容（直接存储文本）
        )
    """)
    conn.commit()
    conn.close()
    
def check_user(user_name):
    db_name = configuration["DB"]["DB_NAME"]
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # 尝试查询用户
    cursor.execute("SELECT content FROM chats WHERE username = ?", (user_name,))
    row = cursor.fetchone()
    if row is not None:
        conn.close()
        return row[0]
    else:
        # 插入新用户（空内容）
        cursor.execute("INSERT INTO chats (username, content) VALUES (?, ?)", (user_name, ""))
        conn.commit()
        conn.close()
        return ""
    
users_router = Router()

@users_router.route("/login",methods=["POST"])
async def user_login(request: Request):
    create_db()
    content_type = request.headers.get("content-type", "").lower()
    body = None
    if "application/json" in content_type:
        # 处理 JSON body
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    chat_content = check_user(body["username"])
    return JSONResponse({"chat_content": chat_content}, status_code=200)


    
    
    