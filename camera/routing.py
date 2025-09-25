from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/stream/<int:user_id>/<int:camera_id>/<str:mode>/', consumers.CameraStreamConsumer.as_asgi()),
    path('ws/live-inout-stats/', consumers.LiveSeatStatsConsumer.as_asgi()),  # renamed for clarity
    path('ws/db-stats/', consumers.LiveDatabaseStatsConsumer.as_asgi()),
]
