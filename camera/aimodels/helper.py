import cv2
import time
import json
import logging
import os
import sys
import numpy as np
from collections import defaultdict
from django.conf import settings
from shapely.geometry import Polygon
from ultralytics import solutions
from camera.models import UserAiModel, InOutCount
from camera.tasks import update_in_out_stats
from datetime import date

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cache for ObjectCounters
zone_counters = defaultdict(dict)

# Defaults
REGION = [(710, 216), (710, 204), (788, 298), (786, 307)]
MODEL_PATH = "yolo11n.pt"


# ---------------------------
# Utility: resource path
# ---------------------------
def resource_path(relative_path):
    """Get absolute path to resource (works for dev and PyInstaller)."""
    try:
        base_path = sys._MEIPASS  # PyInstaller temp dir
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ---------------------------
# Utility: draw label
# ---------------------------
def draw_label(
    img,
    text,
    org,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    txt_color=(255, 0, 0),
    bg_color=(0, 0, 0),
    thickness=2,
):
    (w, h), base = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 4
    cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad), bg_color, -1)
    cv2.putText(img, text, org, font, font_scale, txt_color, thickness)


# ---------------------------
# In/Out counting
# ---------------------------
def in_out_count_people(frame, counter, user_id=None, camera_id=None):
    results = counter.process(frame)
    img = results.plot_im.copy()

    logger.info(f"👉 Processed frame: IN={counter.in_count}, OUT={counter.out_count}")

    if user_id is not None and camera_id is not None:
        snapshot = update_in_out_stats(user_id, camera_id, counter)
        logger.info(f"📝 Updated global store → {snapshot}")
        try:
            from camera.tasks import save_in_out_stats_to_file
            save_in_out_stats_to_file()
            logger.info("💾 Forced immediate save to JSON")
        except Exception as e:
            logger.error(f"❌ Failed immediate save: {e}")
    else:
        logger.warning("⚠️ user_id or camera_id missing, cannot persist counts")

    return img



# ---------------------------
# Normalize zones
# ---------------------------
def _extract_region(zones, default_region, frame_width, frame_height):
    """
    Normalize zones into dict of {name: polygon} and convert percentages to pixel coordinates.
    """
    if not zones:
        return {"Default": default_region}

    if isinstance(zones, dict):
        out = {}
        for name, poly in zones.items():
            pixel_poly = []
            for pt in poly:
                if len(pt) == 2:
                    # Convert % to pixels
                    x_pixel = (pt[0] / 100.0) * frame_width
                    y_pixel = (pt[1] / 100.0) * frame_height
                    pixel_poly.append((int(x_pixel), int(y_pixel)))
                else:
                    pixel_poly.append(tuple(map(int, pt)))
            out[name] = pixel_poly
        return out

    if isinstance(zones, list):
        try:
            pixel_poly = [
                (int((x / 100.0) * frame_width), int((y / 100.0) * frame_height))
                for x, y in zones
            ]
            return {"Zone1": pixel_poly}
        except Exception:
            return {"Default": default_region}

    return {"Default": default_region}


# ---------------------------
# Main AI execution
# ---------------------------
# camera/consumers.py
zone_previous_counts = {}  # Track previous counts to calculate increments

def execute_user_ai_models(user_id, camera_id, frame, rtsp_url=None, save_to_json=False, save_to_db=True):
    """
    Enhanced function that updates both JSON files and database on detection
    """
    frame_height, frame_width = frame.shape[:2]
    logger.info(f"Frame dimensions: {frame_width}x{frame_height}")

    user_ai_models = UserAiModel.objects.filter(
        user_id=user_id, camera_id=camera_id, is_active=True
    )

    out_frame = frame.copy()

    for user_ai_model in user_ai_models:
        ai_model = user_ai_model.aimodel
        function_name = ai_model.function_name
        logger.info(f"Calling function: {function_name}")

        if function_name != "in_out_count_people":
            print(f"# SECURITY CLEANING → Skipping unsupported model: {function_name}")
            continue

        try:
            zones = _extract_region(user_ai_model.zones, REGION, frame_width, frame_height)
            print(f"[Zones] Final zones for camera {camera_id}: {zones}")

            for zone_name, region in zones.items():
                print(f"Running counter for {zone_name}: {region}")
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
                        show_conf=False,
                        show_labels=False,
                        line_width=1,
                        verbose=False,
                    )
                    print(f"✅ Initialized ObjectCounter for zone {zone_key}")
                    # ✅ Initialize previous counts when creating new counter
                    zone_previous_counts[zone_key] = {'in': 0, 'out': 0}

                counter = zone_counters[zone_key]
                out_frame = in_out_count_people(out_frame, counter, user_id, camera_id)

                # ✅ Get current total counts
                current_in = counter.in_count
                current_out = counter.out_count

                # ✅ Calculate actual increments since last frame
                prev_counts = zone_previous_counts.get(zone_key, {'in': 0, 'out': 0})
                in_increment = current_in - prev_counts['in']
                out_increment = current_out - prev_counts['out']

                # ✅ Update previous counts for next iteration
                zone_previous_counts[zone_key] = {'in': current_in, 'out': current_out}

                print(f"[DEBUG] Zone {zone_name}: Total IN={current_in}, OUT={current_out}")
                print(f"[DEBUG] Zone {zone_name}: Increments IN+{in_increment}, OUT+{out_increment}")

                # ✅ Update database only when there's actual increment
                if save_to_db and (in_increment > 0 or out_increment > 0):
                    try:
                        ive_crop=Truenout_record = InOutCount.update_counts(
                            user_id=user_id,
                            camera_id=camera_id,
                            in_increment=in_increment,  # ✅ NOW CORRECT - actual increment
                            out_increment=out_increment,  # ✅ NOW CORRECT - actual increment
                            target_date=date.today()
                        )
                        print(f"🗄️ DB Updated → Zone:{zone_name}, IN+{in_increment}, OUT+{out_increment} | Total: IN={inout_record.in_count}, OUT={inout_record.out_count}")
                    except Exception as e:
                        logger.error(f"❌ Failed to update database: {e}")

                logger.info(f"[AI Worker] Zone {zone_name} → Increments: IN+{in_increment}, OUT+{out_increment}")

                # Save to JSON only when there's actual increment
                if save_to_json and (in_increment > 0 or out_increment > 0):
                    try:
                        from camera.tasks import save_in_out_stats_to_file
                        save_in_out_stats_to_file()
                        logger.info("💾 Saved counts to JSON (save_to_json=True)")
                    except Exception as e:
                        logger.error(f"❌ Failed to save JSON: {e}")

        except Exception as e:
            logger.error(f"❌ Error during in_out_count_people execution: {e}")

    return out_frame



# ---------------------------
# Debugging helper
# ---------------------------
def save_sample_data_to_json(user_id, camera_id, frame):
    sample_data = {
        "user_id": user_id,
        "camera_id": camera_id,
        "data": "sample",  # placeholder
    }

    with open("sample_data.json", "w") as json_file:
        json.dump(sample_data, json_file, indent=4)
    logger.info(f"✅ Sample data saved to JSON: {sample_data}")


# =============================
# SECURITY CLEANING
# =============================
# - Removed blur_faces
# - Removed pixelate_people
# - Removed count_cars / count_people
# - Removed heatmap
# - Removed posture tracking
# - Removed seat_status
# - Removed PPE detection
# - Removed fire_smoke detection
# - Removed people_time_tracker
