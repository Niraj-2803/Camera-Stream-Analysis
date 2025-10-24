import os
import sys
import logging
from collections import defaultdict
from ultralytics import solutions
from camera.models import UserAiModel, InOutStats
from django.utils import timezone
from ultralytics import YOLO
import numpy as np
from shapely.geometry import Polygon, Point

logger = logging.getLogger(__name__)

# Cache for ObjectCounters
zone_counters = defaultdict(dict)

# Defaults
REGION = [(710, 216), (710, 204), (788, 298), (786, 307)]
MODEL_PATH = "yolo11n.pt"

def in_out_count_people(frame, counter, user_id=None, camera_id=None):
    """
    Process a frame to count IN/OUT people, update the counter,
    and persist counts in the database per user/camera/date.
    Ensures counts persist across reconnections and only reset at new day.
    """
    # Process frame using the counter

    if user_id is None or camera_id is None:
        logger.warning("⚠️ [in_out_count_people] user_id or camera_id missing, cannot persist counts")
        return None

    try:
        today = timezone.now().date()

        # Fetch today's record or create new one
        stats_obj, created = InOutStats.objects.get_or_create(
            user_id=user_id,
            camera_id=camera_id,
            date=today
        )

        # If it's not a new day (record exists), sync counter with DB
        if not created:
            counter.in_count = stats_obj.total_in
            counter.out_count = stats_obj.total_out
        else:
            # Created = True → New day, start fresh
            counter.in_count = 0
            counter.out_count = 0

        results = counter.process(frame)

        # Update DB with latest counter values
        stats_obj.total_in = counter.in_count
        stats_obj.total_out = counter.out_count
        stats_obj.save(update_fields=['total_in', 'total_out', 'updated_at'])

        snapshot = {
            "date": str(today),
            "total_in": stats_obj.total_in,
            "total_out": stats_obj.total_out
        }
        return snapshot

    except Exception as e:
        logger.error(f"❌ [in_out_count_people] DB update failed for user {user_id}, camera {camera_id}: {e}")
        return None

from django.utils import timezone
from camera.models import SeatStatusStats
from shapely.geometry import Point
from django.utils import timezone
import cv2
import numpy as np
import time

last_ts = None  

def seat_status(img, results, poly_map, poly_int, stats, user_id, camera_id):
    global last_ts
    now = time.time()
    dt = 0.0 if last_ts is None else now - last_ts
    last_ts = now

    # Extract detections
    result = results[0]
    boxes = (
        result.boxes.xyxy.cpu().numpy()
        if result.boxes is not None
        else np.empty((0, 4))
    )
    centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]

    # Process each seat polygon
    for i, (name, poly) in enumerate(poly_map.items()):
        occupied = any(poly.contains(Point(x, y)) for x, y in centers)

        # Save/update seat status in DB
        try:
            today = timezone.now().date()
            seat_obj, created = SeatStatusStats.objects.get_or_create(
                user_id=user_id,
                camera_id=camera_id,
                seat_name=name,
                date=today,
                defaults={
                    "is_occupied": occupied,
                    "dwell_time": 0.0,
                    "dwell_time_total": 0.0,
                    "empty": 0.0,
                    "empty_total": 0.0,
                    "posture": None,
                    "updated_at": timezone.now(),
                }
            )

            # Now update in-place instead of replacing
            if occupied:
                seat_obj.dwell_time += dt              # current occupied streak
                seat_obj.dwell_time_total += dt        # total occupied today
                seat_obj.empty = 0.0                   # reset empty streak
            else:
                seat_obj.empty += dt                   # current empty streak
                seat_obj.empty_total += dt             # total empty today
                seat_obj.dwell_time = 0.0              # reset dwell streak

            seat_obj.is_occupied = occupied
            seat_obj.posture = stats[name].get("posture")
            seat_obj.updated_at = timezone.now()
            seat_obj.save()

        except Exception as e:
            print(f"❌ Error saving seat status for {name}: {e}")


    return img

def _extract_region(zones, default_region, frame_width, frame_height):
    if not zones:
        return {"Default": default_region}
    if isinstance(zones, dict):
        out = {}
        for name, poly in zones.items():
            pixel_poly = []
            for pt in poly:
                if len(pt) == 2:  # percent coords
                    x_pixel = int((pt[0] / 100.0) * frame_width)
                    y_pixel = int((pt[1] / 100.0) * frame_height)
                    pixel_poly.append((x_pixel, y_pixel))
                else:
                    pixel_poly.append(tuple(map(int, pt)))
            out[name] = pixel_poly
        return out
    if isinstance(zones, list):
        try:
            pixel_poly = [(int((x / 100.0) * frame_width), int((y / 100.0) * frame_height)) for x, y in zones]
            return {"Zone1": pixel_poly}
        except Exception:
            return {"Default": default_region}
    return {"Default": default_region}

_POSE_MODEL = None

def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and PyInstaller.
    """
    try:
        base_path = sys._MEIPASS  # PyInstaller's temp dir
    except AttributeError:
        base_path = os.path.abspath(".")  # Normal run
    return os.path.join(base_path, relative_path)

def _get_pose_model():
    global _POSE_MODEL
    if _POSE_MODEL is None:
        _POSE_MODEL = YOLO(resource_path("yolo11n-pose.pt"))
    return _POSE_MODEL

def get_seat_polygons_from_model(user_id, camera_id, frame_width, frame_height):
    try:
        user_aimodel = UserAiModel.objects.filter(
            user_id=user_id,
            camera_id=camera_id,
            aimodel__function_name="seat_status",
            is_active=True,
        ).first()
        if not user_aimodel or not user_aimodel.zones:
            print("❌ No zone data found.")
            return {}, {}, {}

        # Normalize (convert % → pixels)
        zones = _extract_region(user_aimodel.zones, REGION, frame_width, frame_height)

        poly_map = {name: Polygon(pts) for name, pts in zones.items()}
        poly_int = {name: np.array(pts, np.int32) for name, pts in zones.items()}
        stats = {
            name: {"dwell": 0.0, "empty": 0.0}
            for name in zones
        }

        return poly_map, poly_int, stats

    except Exception as e:
        print(f"❌ Error retrieving seats: {e}")
        return {}, {}, {}


def execute_user_ai_models(user_id, camera_id, frame, rtsp_url=None, save_to_db=True):
    """
    Run enabled AI models and update DB counts
    """
    frame_height, frame_width = frame.shape[:2]
    logger.info(f"Frame dimensions: {frame_width}x{frame_height}")

    user_ai_models = UserAiModel.objects.filter(user_id=user_id, camera_id=camera_id, is_active=True)

    function_map = {
        "seat_status": seat_status,
        "in_out_count_people": in_out_count_people,
    }

    out_frame = frame.copy()
    pose_model = None
    pose_results = None

    def ensure_pose_results():
        nonlocal pose_model, pose_results
        if pose_results is None:
            if pose_model is None:
                pose_model = _get_pose_model()
            pose_results = pose_model(out_frame, verbose=False)
        return pose_results

    for user_ai_model in user_ai_models:
        ai_model = user_ai_model.aimodel
        function_name = ai_model.function_name
        if function_name not in function_map:
            print(f"No function found for AiModel {function_name}.")
            continue

        try:
            if function_name == "in_out_count_people":
                zones = _extract_region(user_ai_model.zones, REGION, frame_width, frame_height)
                for zone_name, region in zones.items():
                    zone_key = f"{camera_id}_{zone_name}"

                    # Reuse or create ObjectCounter
                    if zone_key not in zone_counters:
                        zone_counters[zone_key] = solutions.ObjectCounter(
                            model=MODEL_PATH,
                            region=region,
                            classes=[0],  # only persons
                            analytics_type="crossing",
                            show_in=False,
                            show_out=False,
                            verbose=False,
                        )
                        logger.info(f"Initialized ObjectCounter for zone {zone_key}")
                    
                    counter = zone_counters[zone_key]
                    frame_for_counter = frame.copy()
                    in_out_count_people(frame_for_counter, counter, user_id, camera_id)
                
            if function_name == "seat_status":
                frame_for_pose = frame.copy()
                results = ensure_pose_results()
                poly_map, poly_int, stats = get_seat_polygons_from_model(
                    user_id, camera_id, frame_width, frame_height
                )
                if not poly_map:
                    print("⚠️ No seat zones configured for seat_status.")
                    continue
                seat_status(frame_for_pose, results, poly_map, poly_int, stats, user_id, camera_id)

        except Exception as e:
            logger.error(f"❌ Error during {function_name} execution: {e}")

    return out_frame