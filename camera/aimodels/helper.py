import logging
from collections import defaultdict
from ultralytics import solutions
from camera.models import UserAiModel, InOutStats
from django.utils import timezone

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


def execute_user_ai_models(user_id, camera_id, frame, rtsp_url=None, save_to_db=True):
    """
    Run enabled AI models and update DB counts
    """
    frame_height, frame_width = frame.shape[:2]
    logger.info(f"Frame dimensions: {frame_width}x{frame_height}")

    user_ai_models = UserAiModel.objects.filter(user_id=user_id, camera_id=camera_id, is_active=True)
    out_frame = frame.copy()

    for user_ai_model in user_ai_models:
        ai_model = user_ai_model.aimodel
        if ai_model.function_name != "in_out_count_people":
            logger.info(f"Skipping unsupported model: {ai_model.function_name}")
            continue

        try:
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
                out_frame = in_out_count_people(out_frame, counter, user_id, camera_id)

        except Exception as e:
            logger.error(f"❌ Error during in_out_count_people execution: {e}")

    return out_frame