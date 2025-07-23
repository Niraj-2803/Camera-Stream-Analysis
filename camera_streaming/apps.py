from django.apps import AppConfig

class CameraStreamingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'camera_streaming'

    def ready(self):
        from .scheduler import start
        start()
