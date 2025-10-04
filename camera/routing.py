from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/ai-worker/<int:user_id>/<int:camera_id>/', consumers.AiWorkerConsumer.as_asgi()),
]
