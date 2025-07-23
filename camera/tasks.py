# camera/tasks.py
import cv2
import time
import json
import logging
import numpy as np
from ultralytics import YOLO
from ultralytics import solutions
from collections import defaultdict
from camera.models import UserAiModel
from shapely.geometry import Polygon, Point
from celery import shared_task
from datetime import datetime

@shared_task
def test_task():
    print(f"[{datetime.now()}] ✅ Celery test task ran successfully!")
    return "Done"

import os
import json
import time
import threading
from datetime import date
from collections import defaultdict

import cv2
import numpy as np
from shapely.geometry import Polygon, Point

from celery import shared_task
from django.conf import settings
from ultralytics import YOLO

from camera.models import UserAiModel, SeatStatsLog

# -------------------------------
# Logging Setup
# -------------------------------
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------------------------
# Seat Polygons
# -------------------------------
seats = {
    "seat_1": [(343.2, 368.9), (507.3, 275.4), (431.7, 157.4), (235.5, 222.8)],
    "seat_2": [(348.3, 374.1), (517.6, 290.8), (621.4, 438.2), (448.3, 533.1)],
    "seat_3": [(463.7, 563.8), (670.1, 501.0), (804.1, 719.0), (522.7, 719.0)],
    "seat_4": [
        (818.8, 575.4),
        (1025.3, 429.2),
        (1250.9, 594.6),
        (1137.2, 719.0),
        (843.2, 719.0),
        (770.1, 617.7),
    ],
    "seat_5": [(665.0, 353.6), (838.1, 238.2), (1011.2, 427.9), (811.2, 574.1)],
}

poly_map = {name: Polygon(pts) for name, pts in seats.items()}
poly_int = {name: np.array(pts, np.int32) for name, pts in seats.items()}

# -------------------------------
# Shared In-Memory Stats
# -------------------------------
stats_store = defaultdict(lambda: {
    name: {"dwell": 0.0, "empty": 0.0, "empty_total": 0.0}
    for name in seats
})
lock = threading.Lock()
last_ts_map = {}

# -------------------------------
# Load YOLO Model
# -------------------------------
try:
    model = YOLO("yolo11n-pose.pt")
    logger.info("✅ YOLO model loaded successfully.")
except Exception as e:
    logger.info(f"❌ Error loading YOLO model: {e}")

# -------------------------------
# Seat Status Calculation
# -------------------------------
def seat_status(img, results, stats):
    now = time.time()
    dt = now - last_ts_map.get(id(img), now)
    last_ts_map[id(img)] = now

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

    logger.info(f"Seat status updated for frame. Time delta: {dt:.2f}s")

# -------------------------------
# Start Camera Stream Thread
# -------------------------------
def process_camera(user_id, camera_id, rtsp_url):
    logger.info(f"🎥 Starting stream for user={user_id}, camera={camera_id}")

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        logger.info(f"❌ Unable to open camera stream: {rtsp_url}")
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
                seat_status(frame, results, stats)
        except Exception as e:
            logger.info(f"❌ Error processing frame: {e}")

        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"⛔ Stream ended for user={user_id}, camera={camera_id}")

def start_camera_stream(user_id, camera_id, rtsp_url):
    logger.info(f"🧵 Launching camera thread for user={user_id}, cam={camera_id}")
    t = threading.Thread(target=process_camera, args=(user_id, camera_id, rtsp_url))
    t.daemon = True
    t.start()

# -------------------------------
# Celery Task: Save Seat Stats to File
# -------------------------------
@shared_task
def save_seat_stats_to_file():
    logger.info("📥 Saving seat stats to files...")

    today = date.today()
    base_dir = os.path.join(settings.MEDIA_ROOT, "seat_stats")
    os.makedirs(base_dir, exist_ok=True)

    try:
        active_models = UserAiModel.objects.filter(
            aimodel__function_name="seat_status",
            is_active=True
        ).select_related("user", "camera")
        logger.info(f'{active_models=}')

    except Exception as e:
        logger.info(f"❌ Failed to fetch UserAiModels: {e}")
        return

    with lock:
        logger.info(f'{lock=}')
        for model in active_models:
            user_id = model.user.id
            camera_id = model.camera.id
            rtsp_url = model.camera.rtsp_url  # ✅ fetch from Camera model
            start_camera_stream(user_id, camera_id, rtsp_url)
            key = (user_id, camera_id)
            stats = stats_store.get(key)
            logger.info(f'{stats=}')

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
                logger.info(f"❌ Error writing to file {file_name}: {e}")

            try:
                SeatStatsLog.objects.get_or_create(
                    user_id=user_id,
                    camera_id=camera_id,
                    date=today,
                    defaults={"stats_file": f"seat_stats/{file_name}"}
                )
            except Exception as e:
                logger.info(f"❌ DB save failed for stats log: {e}")
