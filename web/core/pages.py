from starlette.responses import JSONResponse, HTMLResponse, FileResponse
from starlette.routing import Router

async def maingape(request):
    return FileResponse("./web/static/web.html")