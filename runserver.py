import os
import sys
from pathlib import Path
import uvicorn
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo



# Stop Ultralytics from trying pip installs at runtime
os.environ["YOLO_DISABLE_PIP"] = "1"

print("Setting DJANGO_SETTINGS_MODULE...")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "camera_streaming.settings")
os.environ.setdefault("SECRET_KEY", "your-secret-key")
os.environ.setdefault("DEBUG", "True")
os.environ.setdefault("ALLOWED_HOSTS", "localhost,127.0.0.1")
os.environ.setdefault("WS_BASE_URL", "ws://localhost:8000")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:5173")

# Resolve base dir (handles PyInstaller)
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent

# Prefer EXE path over any local checkout on PATH
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Model file envs
os.environ["YOLO_PPE_MODEL"] = str(BASE_DIR / "PPE_model.pt")
os.environ["YOLO_POSE_MODEL"] = str(BASE_DIR / "yolov8n-pose.pt")
os.environ["YOLO_FIRE_SMOKE_MODEL"] = str(BASE_DIR / "fire_smoke_model.pt")

# Persistent locations
PERSISTENT_DIR = os.path.expanduser("~/.camex")
os.makedirs(PERSISTENT_DIR, exist_ok=True)


def run_startup():
    """Apply fresh migrations and init DB, ensure TrialConfig table exists if missing."""
    from django.conf import settings
    import django
    from django.core.management import call_command
    from django.db import connection

    settings.DATABASES["default"]["NAME"] = os.path.join(PERSISTENT_DIR, "data.sqlite3")
    django.setup()

    try:
        print("📦 Applying fresh migrations...")
        call_command("makemigrations", "camera", interactive=False, verbosity=0)
        call_command("migrate", interactive=False, run_syncdb=True, verbosity=1, fake_initial=True)
        print("✅ Migrations applied.")
    except Exception as e:
        print(f"❌ migrate failed: {e}")

    # ✅ Ensure TrialConfig table exists (manual fallback)
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS camera_trialconfig (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expiry_date DATE NOT NULL,
                    created_at DATETIME NOT NULL
                )
            """)
        print("✅ Ensured camera_trialconfig table exists.")
    except Exception as e:
        print(f"⚠️ Failed ensuring TrialConfig table: {e}")

    # Collect static files
    try:
        print("🗂 Collecting static files...")
        call_command("collectstatic", interactive=False, verbosity=0)
    except Exception as e:
        print(f"⚠️ collectstatic failed: {e}")


    try:
        import camera.management.commands.aimodel_script  # ensure bundled
        print("Running aimodel_script command...")
        call_command("aimodel_script")
        print("✅ AiModel instances created successfully.")
    except Exception as e:
        print(f"⚠️ aimodel_script failed or not present: {e}")



def run_periodic_tasks():
    """Run periodic tasks in background thread (no Celery)."""
    from camera.tasks import save_in_out_stats_to_file, save_seat_stats_to_file,load_in_out_stats_from_file
    load_in_out_stats_from_file()
    def worker():
        while True:
            try:
                now = datetime.now(ZoneInfo("Asia/Riyadh"))
                print(f"🕒 Running periodic tasks at {now}")
                save_in_out_stats_to_file()
                save_seat_stats_to_file()
            except Exception as e:
                print(f"❌ Error in periodic tasks: {e}")
            time.sleep(5)  # run every 5 seconds

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def start_uvicorn():
    """Start Uvicorn server"""
    print("🚀 Starting Uvicorn server...")
    uvicorn.run(
        "camera_streaming.asgi:application",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )


def main():
    run_startup()

    # 🔄 Start periodic task runner
    run_periodic_tasks()

    # 🚀 Start Uvicorn (main thread)
    start_uvicorn()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
