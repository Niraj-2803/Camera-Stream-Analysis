from django.urls import path
from .views import *

urlpatterns = [
    path('', CameraListCreateView.as_view(), name='camera-list-create'),
    path('api/camera/add-to-group/', AddCameraToGroupAPIView.as_view()),
    path('api/live_wall/', LiveWallAPIView.as_view(), name='live_wall'),
    path('api/group/<int:group_id>/cameras/', ListCamerasInGroupAPIView.as_view()),
    path('aimodels/activate/', ActivateAiModelView.as_view(), name='activate-ai-model'),
    path('dashboard/', camera_dashboard, name='camera-dashboard'),
    path('api/ai-models/', CreateAiModelView.as_view(), name='create-ai-model'),
    path('user-aimodels/', UserAiModelView.as_view(), name='user-aimodels'),
    path('user-aimodels/zones/', UserAiModelZoneView.as_view(), name='user-aimodels-zones'),
    path('api/expiry/', ExpiryConfigView.as_view(), name="expiry"),
]
