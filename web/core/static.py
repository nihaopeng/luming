from starlette.responses import HTMLResponse
from starlette.routing import Router

async def dashboard(request):
    return HTMLResponse("<h1>Admin Dashboard</h1><p>Welcome!</p>")