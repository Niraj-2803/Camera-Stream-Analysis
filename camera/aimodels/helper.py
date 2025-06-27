import os
import cv2
import json
import time
import numpy as np
import logging
from collections import defaultdict
from shapely.geometry import Polygon, Point
from ultralytics import YOLO, solutions
from camera.models import UserAiModel

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Use Case 1: Blur Faces ---
def blur_faces(frame, results):
    logger.info("Starting face blurring process.")
    result = results[0]
    try:
        kpts = result.keypoints.xy.cpu().numpy()
        confs = result.keypoints.conf.cpu().numpy()
    except Exception as e:
        logger.error("Error accessing keypoints: %s", e)
        return frame

    h_img, w_img = frame.shape[:2]
    num_faces_blurred = 0

    for idx, (person_kpts, person_conf) in enumerate(zip(kpts, confs)):
        head_pts = person_kpts[[0, 1, 2, 3, 4], :]
        head_conf = person_conf[[0, 1, 2, 3, 4]]
        valid = head_conf > 0.7
        pts = head_pts[valid]

        if pts.shape[0] < 2:
            continue

        xs, ys = pts[:, 0], pts[:, 1]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        pw = int((x2 - x1) * 0.3)
        ph = int((y2 - y1) * 0.3)
        x1_p, y1_p = max(0, x1 - pw), max(0, y1 - ph)
        x2_p = min(w_img, x2 + pw)
        y2_p = min(h_img, y2 + ph)

        side = max(x2_p - x1_p, y2_p - y1_p)
        x2_s = min(w_img, x1_p + side)
        y2_s = min(h_img, y1_p + side)

        roi = frame[y1_p:y2_s, x1_p:x2_s]
        if roi.size > 0:
            k = max(1, int(side * 0.3) // 2 * 2 + 1)
            frame[y1_p:y2_s, x1_p:x2_s] = cv2.blur(roi, (k, k))
            num_faces_blurred += 1

    logger.info(f"Total faces blurred: {num_faces_blurred}")
    return frame

# --- Use Case 2: Pixelate People ---
def pixelate_people(frame, boxes, pixel_size=10):
    for box in boxes:
        if int(box.cls[0]) != 0:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        region = frame[y1:y2, x1:x2]
        if region.size > 0:
            small = cv2.resize(region, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = pixelated
    return frame

# --- Use Case 3: Count People ---
last_count_log_time = 0

def count_people(frame, results):
    global last_count_log_time
    if not results or len(results) == 0:
        logger.warning("No results returned by model.")
        return frame, 0

    result = results[0]
    try:
        boxes = result.boxes.xyxy.cpu().numpy()
    except Exception as e:
        logger.error("Error accessing boxes: %s", e)
        return frame, 0

    if boxes.size == 0:
        cv2.putText(frame, "Count: 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return frame, 0

    order = np.argsort(boxes[:, 0])
    count = len(order)

    for idx, i in enumerate(order, start=1):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, str(idx), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    current_time = time.time()
    if current_time - last_count_log_time >= 60:
        logger.info(f"People Count: {count}")
        last_count_log_time = current_time

    return frame, count

# --- Use Case 4: Generate Heatmap ---
def generate_people_heatmap(model_path, colormap=cv2.COLORMAP_PARULA):
    heatmap = solutions.Heatmap(model=model_path, colormap=colormap, classes=[0], show=False)
    return lambda frame: heatmap(frame).plot_im

# --- Seat Status ---
def draw_label(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6,
               txt_color=(255, 0, 0), bg_color=(0, 0, 0), thickness=2):
    (w, h), base = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 2
    cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad), bg_color, -1)
    cv2.putText(img, text, org, font, font_scale, txt_color, thickness)

def seat_status(img, results):
    seats = {
        'seat_1': [(343.2,368.9),(507.3,275.4),(431.7,157.4),(235.5,222.8)],
        'seat_2': [(348.3,374.1),(517.6,290.8),(621.4,438.2),(448.3,533.1)],
        'seat_3': [(463.7,563.8),(670.1,501.0),(804.1,719.0),(522.7,719.0)],
        'seat_4': [(818.8,575.4),(1025.3,429.2),(1250.9,594.6),(1137.2,719.0),(843.2,719.0),(770.1,617.7)],
        'seat_5': [(665.0,353.6),(838.1,238.2),(1011.2,427.9),(811.2,574.1)]
    }
    poly_map = {name: Polygon(pts) for name, pts in seats.items()}
    empty_since = {name: None for name in seats}
    empty_duration = {name: 0 for name in seats}
    stats = {name: defaultdict(float) for name in seats}
    fps = 25

    now = time.time()
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.empty((0,4))
    centers = [((x1 + x2)/2, (y1 + y2)/2) for x1,y1,x2,y2 in boxes]

    for name, poly in poly_map.items():
        occupied = any(poly.contains(Point(x, y)) for x, y in centers)

        if occupied:
            stats[name]['dwell'] += 1.0 / fps
            empty_since[name] = None
            empty_duration[name] = 0.0
        else:
            if empty_since[name] is None:
                empty_since[name] = now
            empty_duration[name] = now - empty_since[name]

        stats[name]['empty'] = round(empty_duration[name], 2)

        pts = np.array(poly.exterior.coords[:-1], np.int32)
        cv2.polylines(img, [pts], True, (255, 0, 0), 2)
        cx, cy = map(int, poly.centroid.coords[0])
        draw_label(img, f"{name} dwell: {stats[name]['dwell']:.1f}s", (cx - 40, cy + 6), txt_color=(0, 255, 0))
        draw_label(img, f"{name} empty: {empty_duration[name]:.1f}s", (cx - 40, cy - 20), txt_color=(0, 255, 0))

    return img

# --- Main execution ---
def execute_user_ai_models(user_id, camera_id, frame, rtsp_url=None, save_to_json=False):
    logger.info(f"Processing user_id={user_id}, camera_id={camera_id}")
    user_ai_models = UserAiModel.objects.filter(user_id=user_id, camera_id=camera_id, is_active=True)
    model = YOLO('yolo11n-pose.pt')
    results = model(frame)

    for user_ai_model in user_ai_models:
        function_name = user_ai_model.aimodel.function_name
        logger.info(f"Executing {function_name}")

        if function_name == "blur_faces":
            frame = blur_faces(frame, results)
        elif function_name == "pixelate_people":
            frame = pixelate_people(frame, results[0].boxes)
        elif function_name == "count_people":
            frame, _ = count_people(frame, results)
        elif function_name == "seat_status":
            frame = seat_status(frame, results)
        else:
            logger.warning(f"No function found for {function_name}")

    if save_to_json:
        save_sample_data_to_json(user_id, camera_id, frame)

    return frame

def save_sample_data_to_json(user_id, camera_id, frame):
    data = {
        "user_id": user_id,
        "camera_id": camera_id,
        "shape": frame.shape
    }
    with open("sample_data.json", "w") as f:
        json.dump(data, f, indent=4)
    logger.info("Sample data saved to sample_data.json")
