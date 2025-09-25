# camera_streaming/scheduler.py

import os
from apscheduler.schedulers.background import BackgroundScheduler
from camera.tasks import daily_camera_check

scheduler = BackgroundScheduler()

def start():
    if os.environ.get('RUN_MAIN') and scheduler.running:
        return  # Prevent multiple instances

    print("🚀 Starting APScheduler...")

    scheduler.add_job(
        daily_camera_check,
        'cron',
        hour=17,   # Run once daily at 17:00
        minute=0,
        id='daily_camera_check',
        replace_existing=True
    )
    scheduler.start()
