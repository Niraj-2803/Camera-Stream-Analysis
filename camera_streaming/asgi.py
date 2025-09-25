import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import camera.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'camera_streaming.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(camera.routing.websocket_urlpatterns)
    ),
})
