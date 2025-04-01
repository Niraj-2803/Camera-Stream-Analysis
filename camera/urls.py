
from django.urls import path
from .views import (
    CameraListCreateView,
    CameraStreamFeedView,
    CameraStreamViewer
)

urlpatterns = [
    path('', CameraListCreateView.as_view(), name='camera-list-create'),
    path('<int:pk>/stream/feed/', CameraStreamFeedView.as_view(), name='camera-stream-feed'),
    path('<int:pk>/stream/', CameraStreamViewer.as_view(), name='camera-stream-viewer'),  
]
