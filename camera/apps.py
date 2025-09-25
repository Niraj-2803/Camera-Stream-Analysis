from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class CameraConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "camera"

    def ready(self):
        # Avoid side effects in migrations
        try:
            from django.db import connection
            if connection.settings_dict:  # basic guard
                from camera.tasks import load_in_out_stats_from_file
                load_in_out_stats_from_file()
                logger.info("📦 camera.apps.CameraConfig.ready → counters restored on startup.")
        except Exception as e:
            logger.warning(f"⚠️ Startup restore skipped: {e}")
