# camera_streaming/scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
from ..camera.tasks import daily_camera_check
import os

scheduler = BackgroundScheduler()

def start():
    if os.environ.get('RUN_MAIN') and scheduler.running:
        return  # Prevent multiple instances

    print("🚀 Starting APScheduler...")  # Debug

    scheduler.add_job(
        daily_camera_check,
        'cron',
        hour='17',              # 5 PM
        minute='*',             # Every minute
        second='*/10',          # Every 10 seconds
        id='daily_camera_check',
        replace_existing=True
    )
    scheduler.start()
