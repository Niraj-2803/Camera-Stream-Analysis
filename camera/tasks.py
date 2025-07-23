# camera/tasks.py

from celery import shared_task
from datetime import datetime

@shared_task
def test_task():
    print(f"[{datetime.now()}] ✅ Celery test task ran successfully!")
    return "Done"
