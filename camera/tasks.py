# camera/tasks.py
import cv2
import time
import json
import logging
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from shapely.geometry import Polygon, Point
from celery import shared_task
from datetime import datetime
import os
import time
import threading
from datetime import date
from django.conf import settings
from camera.models import UserAiModel, SeatStatsLog


@shared_task
def test_task():
    print(f"[{datetime.now()}] ✅ Celery test task ran successfully!")
    return "Done"


# -------------------------------
# Logging Setup
# -------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
import cv2
import json
import time
import torch
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict
from shapely.geometry import Polygon, Point
from django.conf import settings
from celery import shared_task
from camera.models import UserAiModel, SeatStatsLog
from ultralytics import YOLO
import logging

# Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------
# Seat Polygons
# -------------------------------
seats = {
    "seat_1": [(343.2, 368.9), (507.3, 275.4), (431.7, 157.4), (235.5, 222.8)],
    "seat_2": [(348.3, 374.1), (517.6, 290.8), (621.4, 438.2), (448.3, 533.1)],
    "seat_3": [(463.7, 563.8), (670.1, 501.0), (804.1, 719.0), (522.7, 719.0)],
    "seat_4": [(818.8, 575.4), (1025.3, 429.2), (1250.9, 594.6), (1137.2, 719.0), (843.2, 719.0), (770.1, 617.7)],
    "seat_5": [(665.0, 353.6), (838.1, 238.2), (1011.2, 427.9), (811.2, 574.1)],
}
poly_map = {name: Polygon(pts) for name, pts in seats.items()}

# -------------------------------
# Shared In-Memory Stats
# -------------------------------
stats_store = defaultdict(lambda: {
    name: {"dwell": 0.0, "empty": 0.0, "empty_total": 0.0}
    for name in seats
})
last_ts_map = {}  # Track last_ts per (user_id, camera_id)
lock = threading.Lock()

# -------------------------------
# Load YOLO Model
# -------------------------------
try:
    model = YOLO("yolo11n-pose.pt")
    logger.info("✅ YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading YOLO model: {e}")

# -------------------------------
# Seat Status Function
# -------------------------------
def seat_status(user_id, camera_id, img, results, stats):
    now = time.time()
    key = (user_id, camera_id)
    dt = now - last_ts_map.get(key, now)
    last_ts_map[key] = now

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]

    for name, poly in poly_map.items():
        occupied = any(poly.contains(Point(x, y)) for x, y in centers)
        s = stats[name]
        if occupied:
            s["dwell"] += dt
            s["empty"] = 0.0
        else:
            s["empty"] += dt
            s["empty_total"] += dt

    logger.info(f"📊 Seat stats updated for user={user_id}, cam={camera_id}, Δt={dt:.2f}s")

# -------------------------------
# Start Camera Stream Thread
# -------------------------------
def process_camera(user_id, camera_id, rtsp_url):
    logger.info(f"🎥 Starting stream for user={user_id}, camera={camera_id}")

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.warning(f"❌ Unable to open stream: {rtsp_url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"⚠️ Failed to read frame from camera {camera_id}")
            break

        try:
            results = model(frame)
            with lock:
                stats = stats_store[(user_id, camera_id)]
                seat_status(user_id, camera_id, frame, results, stats)
        except Exception as e:
            logger.error(f"❌ Frame processing error: {e}")

        time.sleep(0.5)

    cap.release()
    logger.info(f"⛔ Stream ended for user={user_id}, camera={camera_id}")

def start_camera_stream(user_id, camera_id, rtsp_url):
    logger.info(f"🧵 Launching camera thread for user={user_id}, cam={camera_id}")
    t = threading.Thread(target=process_camera, args=(user_id, camera_id, rtsp_url))
    t.daemon = True
    t.start()

# -------------------------------
# Celery Task
# -------------------------------
@shared_task
def save_seat_stats_to_file():
    now = datetime.now()
    if not (7 <= now.hour < 15):
        logger.info("⏰ Outside active hours (07:00 - 15:00), skipping save.")
        return

    logger.info("📥 Saving seat stats to files...")

    today = date.today()
    base_dir = os.path.join(settings.MEDIA_ROOT, "seat_stats")
    os.makedirs(base_dir, exist_ok=True)

    try:
        active_models = UserAiModel.objects.filter(
            aimodel__function_name="seat_status",
            is_active=True
        ).select_related("user", "camera")
    except Exception as e:
        logger.error(f"❌ Failed to fetch UserAiModels: {e}")
        return

    with lock:
        for model in active_models:
            user_id = model.user.id
            camera_id = model.camera.id
            rtsp_url = model.camera.rtsp_url
            start_camera_stream(user_id, camera_id, rtsp_url)

            key = (user_id, camera_id)
            stats = stats_store.get(key)
            if not stats:
                logger.info(f"🚫 No stats yet for user={user_id}, cam={camera_id}")
                continue

            file_name = f"seat_user{user_id}_cam{camera_id}_{today}.json"
            path = os.path.join(base_dir, file_name)

            snapshot = {
                "timestamp": time.time(),
                "stats": stats
            }

            data = []
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    logger.warning(f"⚠️ Couldn't load existing JSON: {e}")

            data.append(snapshot)

            try:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"💾 Stats saved to: {file_name}")
            except Exception as e:
                logger.error(f"❌ Error writing file {file_name}: {e}")

            try:
                SeatStatsLog.objects.get_or_create(
                    user_id=user_id,
                    camera_id=camera_id,
                    date=today,
                    defaults={"stats_file": f"seat_stats/{file_name}"}
                )
            except Exception as e:
                logger.warning(f"❌ DB log save failed: {e}")
