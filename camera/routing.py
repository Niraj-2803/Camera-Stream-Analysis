from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/stream/<int:user_id>/<int:camera_id>/<str:mode>/', consumers.CameraStreamConsumer.as_asgi()),
    path('ws/analytics/', consumers.AnalyticsStreamConsumer.as_asgi()),
    path('ws/live-seat-stats/', consumers.LiveSeatStatsConsumer.as_asgi()),
]
