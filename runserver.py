import os
import sys
from pathlib import Path
import uvicorn

print("Setting DJANGO_SETTINGS_MODULE...")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "camera_streaming.settings")
os.environ.setdefault("SECRET_KEY", "your-secret-key")
os.environ.setdefault("DEBUG", "True")
os.environ.setdefault("ALLOWED_HOSTS", "localhost,127.0.0.1")
os.environ.setdefault("WS_BASE_URL", "ws://localhost:8000")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:5173")

# Set BASE_DIR depending on execution context
if getattr(sys, 'frozen', False):  # Running from .exe
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent

sys.path.append(str(BASE_DIR))

# Set environment for model files
os.environ["YOLO_PPE_MODEL"] = str(BASE_DIR / "PPE_model.pt")
os.environ["YOLO_POSE_MODEL"] = str(BASE_DIR / "yolov8n-pose.pt")
os.environ["YOLO_FIRE_SMOKE_MODEL"] = str(BASE_DIR / "fire_smoke_model.pt")

# Create persistent DB directory (outside TEMP)
PERSISTENT_DIR = os.path.expanduser("~/.camex")
os.makedirs(PERSISTENT_DIR, exist_ok=True)

# Force Django to use persistent DB location
from django.conf import settings
DB_PATH = os.path.join(PERSISTENT_DIR, "data.sqlite3")
settings.DATABASES['default']['NAME'] = DB_PATH

# Setup Django
import django
from django.core.management import call_command

django.setup()

# Check for existing DB
def db_exists():
    return os.path.exists(DB_PATH)

if not db_exists():
    print("📦 First time setup: applying migrations...")
    call_command("migrate", interactive=False)
else:
    print("✅ Using existing DB, skipping re-migration.")

# 👇 Force-load the custom management command (important for PyInstaller)
import camera.management.commands.aimodel_script

# Run your custom management command
print("Running aimodel_script command...")
call_command("aimodel_script")

# Start the server
print(f"Base dir: {BASE_DIR}")
print("Starting Uvicorn server...")

uvicorn.run(
    "camera_streaming.asgi:application",
    host="0.0.0.0",
    port=8000,
    log_level="info",
    reload=False
)
