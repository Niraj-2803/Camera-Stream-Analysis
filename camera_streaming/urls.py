from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse

# Swagger imports
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi


schema_view = get_schema_view(
    openapi.Info(
        title="Camex API",
        default_version='v1',
        description="API documentation for Camex",
        contact=openapi.Contact(email="support@camex.com"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
    authentication_classes=[],
)

# Test API view
def test_ping(request):
    return JsonResponse({"message": "pong"})

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/camera/', include('camera.urls')),
    path('api/users/', include('users.urls')),
    path('api/event/', include('event.urls')),

    # Swagger & Redoc
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),

    # Test route
    path('ping/', test_ping),
]
