
from django.urls import path
from .views import (
    ActivateAiModelView,
    AiModelListView,
    CameraListCreateView,
    CameraStreamFeedView,
    CameraStreamViewer
)

urlpatterns = [
    path('', CameraListCreateView.as_view(), name='camera-list-create'),
    path('<int:pk>/stream/feed/', CameraStreamFeedView.as_view(), name='camera-stream-feed'),
    path('<int:pk>/stream/', CameraStreamViewer.as_view(), name='camera-stream-viewer'),  
    path('aimodels/', AiModelListView.as_view(), name='ai-model-list'),
    path('aimodels/activate/', ActivateAiModelView.as_view(), name='activate-ai-model'),
]
