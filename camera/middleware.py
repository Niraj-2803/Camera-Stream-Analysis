import datetime
from django.http import JsonResponse
from camera.models import TrialConfig

EXCLUDED_PATHS = [
    "/swagger/",
    "/swagger.json",
    "/redoc/",
    "/admin/login/",
    "/api/camera/api/expiry/",  # ✔️ your actual expiry endpoint
]

def trial_expiry_middleware(get_response):
    def middleware(request):
        if request.path in EXCLUDED_PATHS:
            return get_response(request)

        config = TrialConfig.objects.first()
        if config and datetime.date.today() > config.expiry_date:
            return JsonResponse({
                "status": "expired",
                "expiry_date": config.expiry_date.strftime('%Y-%m-%d')  # ISO format
            }, status=400)

        return get_response(request)

    return middleware
