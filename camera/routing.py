from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/stream/<int:user_id>/<int:camera_id>/', consumers.CameraStreamConsumer.as_asgi()),
]
