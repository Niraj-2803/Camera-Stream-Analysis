from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r"ws/stream/(?P<camera_id>\d+)/$", consumers.CameraStreamConsumer.as_asgi()),
]
