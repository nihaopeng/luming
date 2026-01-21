# app.py
from web.core.pages import maingape
from web.core.chat import chat_router
from web.core.users import users_router
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.routing import Route, Mount
import uvicorn

# --- 路由配置 ---

routes = [
    Route("/", maingape, methods=["GET"]),
    Mount("/chat", app=chat_router, name="chat"),
    Mount("/users",app=users_router),
    # 可选：挂载整个 static 目录（用于其他静态资源）
    Mount("/static", app=StaticFiles(directory="./web/static"), name="static"),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=59067, reload=False)