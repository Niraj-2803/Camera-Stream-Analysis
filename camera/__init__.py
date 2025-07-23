from __future__ import absolute_import, unicode_literals

# Import Celery application so it gets registered when Django starts
from camera_streaming.celery import app as celery_app

__all__ = ('celery_app',)
