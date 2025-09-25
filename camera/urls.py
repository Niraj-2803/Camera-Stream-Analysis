from django.urls import path
from .views import *
from django.conf.urls.static import static
# from . import consumers

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

    path("api/camera/start-stream/", StartStreamView.as_view(), name="start-stream"),

    path('api/inout-stats/', InOutStatsAPIView.as_view(), name='inout-stats'),
    path('api/camera/delete/', CameraDeleteAPIView.as_view(), name='camera-delete'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
