import os
import django

# 1. Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'camera_streaming.settings')
django.setup()

# 2. Django & Channels imports
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import camera.routing

# 3. Starlette static files
from starlette.staticfiles import StaticFiles
from starlette.applications import Starlette

# 4. Static path
# static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')

# 5. Django ASGI app
django_asgi_app = get_asgi_application()

# 6. Wrap with Starlette for static + Django app
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'staticfiles')

wrapped_app = Starlette()
wrapped_app.mount("/static", StaticFiles(directory=static_path), name="static")
wrapped_app.mount("/", django_asgi_app)


# 7. ASGI application
application = ProtocolTypeRouter({
    "http": wrapped_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(camera.routing.websocket_urlpatterns)
    ),
})


# # 🚀 Start APScheduler here
# from camera_streaming.scheduler import start as start_scheduler
# start_scheduler()