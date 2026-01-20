# app.py
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, FileResponse
from starlette.requests import Request
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
import json
import uvicorn

# --- 路由配置 ---

routes = [
    Route("/json", json_endpoint),
    Route("/html", html_file),
    Route("/get", get_params, methods=["GET"]),
    Route("/post", post_params, methods=["POST"]),
    # 可选：挂载整个 static 目录（用于其他静态资源）
    Mount("/static", app=StaticFiles(directory="static"), name="static"),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)