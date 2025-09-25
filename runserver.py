import os
import sys
from pathlib import Path
import uvicorn
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import logging

# --------------------------
# Logging setup
# --------------------------
logging.basicConfig(
    level=logging.WARNING,  # 🔄 Change to INFO if you want periodic logs
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("camex")

# --------------------------
# Environment setup
# --------------------------
os.environ["YOLO_DISABLE_PIP"] = "1"  # stop ultralytics pip installs
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "camera_streaming.settings")
os.environ.setdefault("SECRET_KEY", "your-secret-key")
os.environ.setdefault("DEBUG", "True")
os.environ.setdefault("ALLOWED_HOSTS", "localhost,127.0.0.1")
os.environ.setdefault("WS_BASE_URL", "ws://localhost:8000")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:5173")

# Path resolve (PyInstaller or dev mode)
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Model path envs
os.environ["YOLO_INOUT_MODEL"] = str(BASE_DIR / "yolo11n.pt")

# Persistent dir
PERSISTENT_DIR = os.path.expanduser("~/.camex")
os.makedirs(PERSISTENT_DIR, exist_ok=True)

# --------------------------
# Django startup
# --------------------------
def run_startup():
    """Apply fresh migrations and init DB if missing."""
    from django.conf import settings
    import django
    from django.core.management import call_command
    from django.db import connection

    settings.DATABASES["default"]["NAME"] = os.path.join(PERSISTENT_DIR, "data.sqlite3")
    django.setup()

    try:
        call_command("makemigrations", "camera", interactive=False, verbosity=0)
        call_command("migrate", interactive=False, run_syncdb=True, verbosity=0, fake_initial=True)
        logger.info("✅ Migrations applied.")
    except Exception as e:
        logger.error(f"❌ migrate failed: {e}")

    # Ensure TrialConfig table exists
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS camera_trialconfig (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expiry_date DATE NOT NULL,
                    created_at DATETIME NOT NULL
                )
            """)
        logger.info("✅ Ensured camera_trialconfig table exists.")
    except Exception as e:
        logger.warning(f"⚠️ Failed ensuring TrialConfig table: {e}")

    try:
        call_command("collectstatic", interactive=False, verbosity=0)
    except Exception as e:
        logger.warning(f"⚠️ collectstatic failed: {e}")

    try:
        import camera.management.commands.aimodel_script
        call_command("aimodel_script")
        logger.info("✅ AiModel instances created successfully.")
    except Exception as e:
        logger.warning(f"⚠️ aimodel_script failed or not present: {e}")

# --------------------------
# Background workers
# --------------------------
def run_periodic_tasks():
    """Run periodic tasks in background threads (no Celery)."""
    from camera.tasks import (
        save_in_out_stats_to_file,
        load_in_out_stats_from_file,
        generate_daily_in_out_reports,
    )

    tz = ZoneInfo("Asia/Kolkata")
    load_in_out_stats_from_file()

    def save_worker():
        while True:
            try:
                save_in_out_stats_to_file()
                logger.info("💾 Stats saved")
            except Exception as e:
                logger.error(f"❌ Error in save_worker: {e}")
            time.sleep(5)

    def email_worker(trigger_hour=20, trigger_minute=00):
        sent_this_minute = False
        while True:
            try:
                now_ist = datetime.now(tz)
                if now_ist.hour == trigger_hour and now_ist.minute == trigger_minute:
                    if not sent_this_minute:
                        generate_daily_in_out_reports()
                        logger.info("📧 Daily report sent")
                        sent_this_minute = True
                else:
                    sent_this_minute = False
            except Exception as e:
                logger.error(f"❌ Error in email_worker: {e}")
            time.sleep(30)

    threading.Thread(target=save_worker, daemon=True).start()
    threading.Thread(target=email_worker, daemon=True).start()

# --------------------------
# Uvicorn server
# --------------------------
def start_uvicorn():
    logger.warning("🚀 Starting Uvicorn server on 0.0.0.0:8000")
    uvicorn.run(
        "camera_streaming.asgi:application",
        host="0.0.0.0",
        port=8000,
        log_level="warning",  # ⬅️ keep uvicorn quiet
        reload=False,
    )

# --------------------------
# Entry point
# --------------------------
def main():
    run_startup()
    run_periodic_tasks()
    start_uvicorn()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
