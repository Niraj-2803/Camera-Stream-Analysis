from django.urls import path
from .views import *

urlpatterns = [
    path('', CameraListCreateView.as_view(), name='camera-list-create'),
    path('api/camera/add-to-group/', AddCameraToGroupAPIView.as_view()),
    path('api/live_wall/', LiveWallAPIView.as_view(), name='live_wall'),
    path('api/group/<int:group_id>/cameras/', ListCamerasInGroupAPIView.as_view()),
    path('aimodels/activate/', ActivateAiModelView.as_view(), name='activate-ai-model'),
    # path('dashboard/', camera_dashboard, name='camera-dashboard'),
    path('api/ai-models/create/', CreateAiModelView.as_view(), name='create-ai-model'),

]
