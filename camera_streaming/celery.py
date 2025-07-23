from celery import Celery
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'camera_streaming.settings')
app = Celery('camera_streaming')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# ⬇️ Enable periodic task discovery from DB
app.conf.beat_scheduler = 'django_celery_beat.schedulers.DatabaseScheduler'
