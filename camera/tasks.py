# camera/tasks.py
import os
import cv2
import json
import time
import threading
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, date, time as dtime
from collections import defaultdict
from shapely.geometry import Polygon, Point
from zoneinfo import ZoneInfo

from django.conf import settings
from ultralytics import YOLO
from .models import UserAiModel, SeatStatsLog

# -------------------------------
# Logger
# -------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------
# Celery Test Task
# -------------------------------
# @shared_task
def test_task():
    print(f"[{datetime.now()}] ✅ Celery test task ran successfully!")
    return "Done"

# -------------------------------
# Seat Zones Loader
# -------------------------------
def load_seat_zones(user_id, camera_id, frame_width, frame_height):
    """
    Fetch seat zones from UserAiModel.zones in DB.
    Converts normalized coords (0-1) → pixels.
    Returns: poly_map (name→Polygon), stats dict (name→counters)
    """
    try:
        model = UserAiModel.objects.filter(
            user_id=user_id,
            camera_id=camera_id,
            aimodel__function_name="seat_status",
            is_active=True
        ).first()
        if not model or not model.zones:
            return {}, {}

        zones = json.loads(model.zones)
        poly_map = {}
        stats = {}

        for name, pts in zones.items():
            if 0 < pts[0][0] <= 1 and 0 < pts[0][1] <= 1:  # normalized
                pts_px = [(int(x * frame_width), int(y * frame_height)) for x, y in pts]
            else:
                pts_px = [(int(x), int(y)) for x, y in pts]

            poly_map[name] = Polygon(pts_px)
            stats[name] = {"dwell": 0.0, "empty": 0.0, "empty_total": 0.0}

        return poly_map, stats
    except Exception as e:
        logger.error(f"❌ Error loading seat zones: {e}")
        return {}, {}

# -------------------------------
# Shared In-Memory Stats
# -------------------------------
stats_store = defaultdict(dict)  # per (user,camera) → seat stats
last_ts_map = {}
last_processed_date = {}
lock = threading.Lock()

# -------------------------------
# Load YOLO Model
# -------------------------------
try:
    model = YOLO("yolo11n-pose.pt")
    logger.info("✅ YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading YOLO model: {e}")
    model = None

# -------------------------------
# Seat Status Function
# -------------------------------
def seat_status(user_id, camera_id, img, results):
    now = time.time()
    key = (user_id, camera_id)
    dt = now - last_ts_map.get(key, now)
    last_ts_map[key] = now

    H, W = img.shape[:2]
    poly_map, _ = load_seat_zones(user_id, camera_id, W, H)
    if not poly_map:
        logger.info(f"🚫 No seat zones for user={user_id}, cam={camera_id}")
        return

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]

    with lock:
        stats = stats_store.setdefault(key, {})
        for name, poly in poly_map.items():
            s = stats.setdefault(name, {"dwell": 0.0, "empty": 0.0, "empty_total": 0.0})
            occupied = any(poly.contains(Point(x, y)) for x, y in centers)
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

    sa_tz = ZoneInfo("Asia/Riyadh")
    shift_start = dtime(7, 0)
    shift_end = dtime(15, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"⚠️ Failed to read frame from camera {camera_id}")
            break

        now = datetime.now(sa_tz)
        today = now.date()
        current_time = now.time()
        key = (user_id, camera_id)

        # 🔄 Reset stats if new day
        if last_processed_date.get(key) != today:
            with lock:
                stats_store[key] = {}
                last_ts_map[key] = time.time()
                last_processed_date[key] = today
                logger.info(f"🔄 Stats reset for user={user_id}, cam={camera_id} on {today}")

        # ⏰ Skip processing if outside shift hours
        if not (shift_start <= current_time < shift_end):
            time.sleep(10)
            continue

        try:
            if model:
                results = model(frame)
                seat_status(user_id, camera_id, frame, results)
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
# Celery Task: Save Stats
# -------------------------------
# @shared_task
def save_seat_stats_to_file():
    now = datetime.now(ZoneInfo("Asia/Riyadh"))
    if not (7 <= now.hour < 15):
        logger.info("⏰ Outside active hours (07:00 - 15:00), skipping save.")
        return

    logger.info("📥 Saving seat stats to files...")
    today = now.date()
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

            snapshot = {"timestamp": time.time(), "stats": stats}

            data = []
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                except Exception as e:
                    logger.warning(f"⚠️ Couldn't load existing JSON: {e}")

            data.append(snapshot)

            try:
                with open(path, "w") as f:
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


# -------------------------------
# Shared In-Memory IN/OUT Stats
# -------------------------------
in_out_store = defaultdict(lambda: {"in": 0, "out": 0})
in_out_lock = threading.Lock()
last_in_out_date = {}

# -------------------------------
# Update from Frame Function
# -------------------------------
def update_in_out_stats(user_id, camera_id, counter):
    """
    Read live in/out counts from ObjectCounter, persist in memory store.
    """
    now = datetime.now(ZoneInfo("Asia/Riyadh"))
    key = (user_id, camera_id)
    logger.info(f"📥 [update_in_out_stats] Updating stats for user_id={user_id}, camera_id={camera_id}, key={key}")
    logger.info(f"🕒 Current Saudi time: {now.isoformat()}")

    with in_out_lock:
        logger.info("🔒 Acquired in_out_lock for updating in_out_store")

        # ✅ Just mirror the counter’s own running totals
        prev_in = in_out_store.get(key, {}).get("in", 0)
        prev_out = in_out_store.get(key, {}).get("out", 0)
        logger.info(f"📊 Previous memory values: IN={prev_in}, OUT={prev_out}")

        in_out_store[key]["in"] = counter.in_count
        in_out_store[key]["out"] = counter.out_count
        logger.info(f"🆕 Updated memory values: IN={counter.in_count}, OUT={counter.out_count}")

        snapshot = {
            "saudi_time": now.isoformat(),
            "in_total": counter.in_count,
            "out_total": counter.out_count,
        }
        logger.info(f"📦 Snapshot created: {snapshot}")

    logger.info(f"✅ [update_in_out_stats] Update complete for key={key}")
    return snapshot

# -------------------------------
# Celery Task: Save IN/OUT Stats
# -------------------------------
# @shared_task
def save_in_out_stats_to_file():
    now = datetime.now(ZoneInfo("Asia/Riyadh"))
    today = now.date()
    logger.info("📥 [save_in_out_stats_to_file] Start saving In/Out stats...")
    logger.info(f"📅 Current Saudi date: {today}, Time: {now.isoformat()}")

    base_dir = os.path.join(settings.MEDIA_ROOT, "in_out_stats")
    logger.info(f"📂 Base dir for saving stats: {base_dir}")
    os.makedirs(base_dir, exist_ok=True)

    try:
        active_models = UserAiModel.objects.filter(
            aimodel__function_name="in_out_count_people",
            is_active=True
        ).select_related("user", "camera")
        logger.info(f"🔍 Found {active_models.count()} active UserAiModels")
    except Exception as e:
        logger.error(f"❌ Failed to fetch InOut UserAiModels: {e}")
        return

    with in_out_lock:
        logger.info("🔒 Acquired in_out_lock for saving stats")

        for model in active_models:
            user_id = model.user.id
            camera_id = model.camera.id
            key = (user_id, camera_id)
            logger.info(f"➡️ Processing user_id={user_id}, camera_id={camera_id}, key={key}")

            stats = in_out_store.get(key, {"in": 0, "out": 0})
            logger.info(f"📊 Current in/out from memory: IN={stats['in']} OUT={stats['out']}")

            file_name = f"inout_user{user_id}_cam{camera_id}_{today}.json"
            path = os.path.join(base_dir, file_name)
            logger.info(f"📝 Target file: {path}")

            snapshot = {
                "saudi_time": now.isoformat(),
                "timestamp": time.time(),
                "stats": {
                    "in_count": stats["in"],
                    "out_count": stats["out"],
                },
            }
            logger.info(f"📦 Snapshot prepared: {snapshot}")

            # Load old file if exists
            data = []
            if os.path.exists(path):
                logger.info("📂 Existing file found, loading previous data...")
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    logger.info(f"📥 Loaded {len(data)} records from existing file")
                except Exception as e:
                    logger.warning(f"⚠️ Couldn't load existing JSON: {e}")

            data.append(snapshot)
            logger.info(f"🆕 Appended snapshot, total records now: {len(data)}")

            try:
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"💾 Successfully saved in/out stats → {file_name}")
            except Exception as e:
                logger.error(f"❌ Error writing in/out file {file_name}: {e}")

    logger.info("✅ [save_in_out_stats_to_file] Completed saving all stats")


def load_in_out_stats_from_file():
    """Load saved in/out stats into memory at startup (or restart)."""
    base_dir = os.path.join(settings.MEDIA_ROOT, "in_out_stats")
    os.makedirs(base_dir, exist_ok=True)

    today = datetime.now(ZoneInfo("Asia/Riyadh")).date()

    try:
        active_models = UserAiModel.objects.filter(
            aimodel__function_name="in_out_count_people",
            is_active=True
        ).select_related("user", "camera")
    except Exception as e:
        logger.error(f"❌ Failed to fetch InOut UserAiModels: {e}")
        return

    with in_out_lock:
        for model in active_models:
            user_id = model.user.id
            camera_id = model.camera.id
            key = (user_id, camera_id)

            file_name = f"inout_user{user_id}_cam{camera_id}_{today}.json"
            path = os.path.join(base_dir, file_name)

            if not os.path.exists(path):
                logger.info(f"ℹ️ No previous file for user={user_id}, cam={camera_id}")
                continue

            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    if data and isinstance(data, list):
                        last_snapshot = data[-1]["stats"]
                        in_out_store[key]["in"] = last_snapshot.get("in_count", 0)
                        in_out_store[key]["out"] = last_snapshot.get("out_count", 0)
                        logger.info(f"✅ Restored counts for user={user_id}, cam={camera_id}: "
                                    f"IN={in_out_store[key]['in']}, OUT={in_out_store[key]['out']}")
            except Exception as e:
                logger.error(f"❌ Failed to load {file_name}: {e}")
