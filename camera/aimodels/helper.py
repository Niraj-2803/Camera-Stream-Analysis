import cv2
import time
import json
import logging
from django.conf import settings
import numpy as np
import pytz
from ultralytics import YOLO
from ultralytics import solutions
from collections import defaultdict
from camera.models import UserAiModel
from shapely.geometry import Polygon, Point

import time
from datetime import datetime
from zoneinfo import ZoneInfo

from camera.tasks import update_in_out_stats

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


import os
import sys
from collections import defaultdict
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults

zone_counters = defaultdict(dict)


def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and PyInstaller.
    """
    try:
        base_path = sys._MEIPASS  # PyInstaller's temp dir
    except AttributeError:
        base_path = os.path.abspath(".")  # Normal run
    return os.path.join(base_path, relative_path)


def pixelate_people(frame, boxes, pixel_size=10):
    for box in boxes:
        cls = int(box.cls[0])
        if cls != 0:
            continue  # Only target class 0 (person)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        region = frame[y1:y2, x1:x2]
        if region.size > 0:
            small = cv2.resize(
                region, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR
            )
            pixelated = cv2.resize(
                small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
            )
            frame[y1:y2, x1:x2] = pixelated

    return frame


# 🧠 Use Case 3a: Count cars only
def count_cars(model_path, width, height, region_points=None):
    if region_points is None:
        region_points = [
            (20, int(height * 0.75)),
            (width - 20, int(height * 0.75)),
            (width - 20, int(height * 0.7)),
            (20, int(height * 0.7)),
        ]

    counter = solutions.ObjectCounter(
        show=False,
        region=region_points,
        model=model_path,
        classes=[1, 2, 3, 5, 7],  # Bicycle, car, motorcycle, bus, truck
    )

    def counter_fn(frame):
        results = counter(frame)
        return results.plot_im

    return counter_fn


# 🧠 Use Case 4: Generate heatmap for people only (class 0)
def generate_people_heatmap(model_path, colormap=cv2.COLORMAP_PARULA):
    heatmap = solutions.Heatmap(
        model=model_path,
        colormap=colormap,
        classes=[0],  # Only detect and visualize people (class 0)
        show=False,
    )

    def heatmap_fn(frame):
        results = heatmap(frame)
        return results.plot_im

    return heatmap_fn


# 🔁 Main processing engine
def process_video_stream(
    source, output_path, model_path, usecase="blur_faces", region_points=None
):
    model = YOLO(model_path)

    # Ensure proper stream formatting
    if source.startswith("http") and not source.endswith("/video"):
        source = source.rstrip("/") + "/video"

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Failed to open video source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20

    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    print(f"📹 Processing started on: {source}")
    print("▶️ Press ESC to stop...")

    object_counter_fn = None
    heatmap_fn = None

    if usecase == "count_cars":
        object_counter_fn = count_cars(model_path, width, height, region_points)
    elif usecase == "count_people":
        object_counter_fn = count_people(model_path, width, height, region_points)
    elif usecase == "people_heatmap":
        heatmap = solutions.Heatmap(
            model=model_path,
            colormap=cv2.COLORMAP_PARULA,
            classes=[0],  # 🧍‍♂️ Only track people
            show=False,
        )
        heatmap_fn = lambda frame: heatmap(frame).plot_im

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Frame not received. Exiting...")
            break

        # 🔌 Use cases
        if usecase == "blur_faces":
            results = model(frame)[0]
            processed_frame = blur_faces(frame, results.boxes)

        elif usecase == "pixelate_people":
            results = model(frame)[0]
            processed_frame = pixelate_people(frame, results.boxes)

        elif usecase in ["count_cars", "count_people"] and object_counter_fn:
            processed_frame = object_counter_fn(frame)

        elif usecase == "people_heatmap" and heatmap_fn:
            processed_frame = heatmap_fn(frame)

        else:
            print("❌ Unknown usecase. Skipping processing.")
            processed_frame = frame

        # 🎥 Display and record
        cv2.imshow("YOLO Use Case Output", processed_frame)
        video_writer.write(processed_frame)

        if cv2.waitKey(1) == 27:  # ESC key
            print("⏹️ Stopped by user.")
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"✅ Video saved to: {output_path}")


# _________________________________________________________________________________________________________________________________________
# Track Posture

# pose_model = YOLO("yolov8n-pose.pt")  # Use yolov8n-pose.pt for speed
pose_model = YOLO(resource_path("yolov8n-pose.pt"))
SEAT_COORDINATES = {
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

seat_poly = {n: Polygon(pts) for n, pts in SEAT_COORDINATES.items()}
stats = {s: defaultdict(float) for s in SEAT_COORDINATES}
owner_tid = {s: None for s in SEAT_COORDINATES}
owner_miss = {s: 0 for s in SEAT_COORDINATES}


# Lightweight function to process posture on a single frame
def process_posture_and_occupancy_frame(
    model, frame, seat_poly, stats, owner_tid, owner_miss, fps
):
    results = model.predict(frame, stream=False, verbose=False)[0]
    if not hasattr(results, "keypoints") or results.keypoints is None:
        return frame

    kps = results.keypoints.xy.cpu().numpy()
    kconf = results.keypoints.conf.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()
    tids = np.arange(len(kps))

    centroid, posture = {}, {}
    h = frame.shape[0]

    for i, kp in enumerate(kps):
        tid = int(tids[i])
        x0, y0, x1, y1 = boxes[i]
        centroid[tid] = Point((x0 + x1) / 2, (y0 + y1) / 2)
        ang = {k: np.nan for k in ("l_knee", "r_knee", "l_hip", "r_hip")}

        if kconf[i][11] > 0.3 and kconf[i][13] > 0.3 and kconf[i][15] > 0.3:
            ang["l_knee"] = compute_angle(kp[11], kp[13], kp[15])
        if kconf[i][12] > 0.3 and kconf[i][14] > 0.3 and kconf[i][16] > 0.3:
            ang["r_knee"] = compute_angle(kp[12], kp[14], kp[16])
        if kconf[i][5] > 0.3 and kconf[i][11] > 0.3 and kconf[i][13] > 0.3:
            ang["l_hip"] = compute_angle(kp[5], kp[11], kp[13])
        if kconf[i][6] > 0.3 and kconf[i][12] > 0.3 and kconf[i][14] > 0.3:
            ang["r_hip"] = compute_angle(kp[6], kp[12], kp[14])

        posture[tid] = classify_posture(kp, kconf[i], ang, img_h=h)

    for seat, poly in seat_poly.items():
        owner = owner_tid[seat]
        ids_in = [tid for tid, pt in centroid.items() if poly.contains(pt)]

        if owner is None and ids_in:
            owner_tid[seat] = ids_in[0]
            owner_miss[seat] = 0
        elif owner is not None:
            if owner in ids_in:
                stats[seat]["dwell"] += 1 / fps
                stats[seat][posture[owner].lower()] += 1 / fps
                owner_miss[seat] = 0
            else:
                owner_miss[seat] += 1
                if owner_miss[seat] > 30:
                    owner_tid[seat] = None
                    owner_miss[seat] = 0

    draw_seats(frame, seat_poly, stats)
    return frame


# In your WebSocket `stream_posture_and_occupancy`, update process_frame_callback:
def track_posture_and_occupancy(frame, boxes):
    fps = 30  # ideally measured dynamically
    start_time = time.time()
    process_posture_and_occupancy_frame(
        pose_model, frame, seat_poly, stats, owner_tid, owner_miss, fps
    )
    elapsed_time = time.time() - start_time
    sleep_time = max(0, (1 / fps) - elapsed_time)
    if sleep_time > 0:
        time.sleep(sleep_time)


def compute_angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    ba, bc = a - b, c - b
    d = np.linalg.norm(ba) * np.linalg.norm(bc)
    if d < 1e-6:
        return np.nan
    cos = np.dot(ba, bc) / d
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def classify_posture(kp, conf, ang, img_h=None, min_visible=8):
    legs_ok = all(conf[i] > 0.3 for i in (11, 12, 13, 14, 15, 16))
    if legs_ok:
        knee = np.nanmean([ang["l_knee"], ang["r_knee"]])
        hip = np.nanmean([ang["l_hip"], ang["r_hip"]])
        if knee < 120 or hip < 120:
            return "Sitting"
        if knee > 150 and hip > 150:
            return "Standing"
        return "Uncertain"

    if img_h is not None and np.sum(conf > 0.3) >= min_visible:
        s_y = np.nanmean([kp[5][1], kp[6][1]])
        h_y = np.nanmean([kp[11][1], kp[12][1]])
        return "Sitting" if 0.5 * (s_y + h_y) > img_h * 0.55 else "Standing"

    return "Uncertain"


def draw_seats(img, poly_map, stats):
    """
    Draw each seat polygon and its current dwell time on the frame.
    """
    for name, poly in poly_map.items():
        pts = np.array(poly.exterior.coords[:-1], np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        cx, cy = map(int, poly.centroid.coords[0])
        label_text = f"{name}: {stats[name]['dwell']:.1f}s"
        label_org = (cx - 40, cy + 6)

        draw_label(
            img,
            label_text,
            label_org,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.6,
            txt_color=(255, 0, 0),
            bg_color=(0, 0, 0),
            thickness=2,
        )


def draw_label(
    img,
    text,
    org,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.4,
    txt_color=(255, 0, 0),
    bg_color=(0, 0, 0),
    thickness=1,
):
    """
    Draw text with a solid background.
    """
    (w, h), base = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 2

    cv2.rectangle(
        img, (x - pad, y - h - pad), (x + w + pad, y + base + pad), bg_color, -1
    )
    cv2.putText(img, text, org, font, font_scale, txt_color, thickness)


# _________________________________________________________________________________________________________________________________________


# _________________________________________________________________________________________________________________________________________
# Blur faces


def blur_faces(frame, results):
    logger.info("Starting face blurring process.")

    result = results[0]  # single-image inference

    try:
        kpts = result.keypoints.xy.cpu().numpy()
        confs = result.keypoints.conf.cpu().numpy()
    except Exception as e:
        logger.error("Error accessing keypoints or confidence scores: %s", e)
        return frame

    h_img, w_img = frame.shape[:2]
    logger.info(f"Image dimensions: width={w_img}, height={h_img}")

    num_faces_blurred = 0

    for idx, (person_kpts, person_conf) in enumerate(zip(kpts, confs)):
        head_pts = person_kpts[[0, 1, 2, 3, 4], :]
        head_conf = person_conf[[0, 1, 2, 3, 4]]
        valid = head_conf > 0.7
        pts = head_pts[valid]

        if pts.shape[0] < 2:
            logger.debug(
                f"Skipping person {idx}: insufficient keypoints with confidence > 0.7"
            )
            continue

        xs, ys = pts[:, 0], pts[:, 1]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        pw = int((x2 - x1) * 0.3)
        ph = int((y2 - y1) * 0.3)
        x1_p, y1_p = max(0, x1 - pw), max(0, y1 - ph)
        x2_p = min(w_img, x2 + pw)
        y2_p = min(h_img, y2 + ph)

        w_box, h_box = x2_p - x1_p, y2_p - y1_p
        side = max(w_box, h_box)
        x2_s = min(w_img, x1_p + side)
        y2_s = min(h_img, y1_p + side)

        roi = frame[y1_p:y2_s, x1_p:x2_s]
        if roi.size > 0:
            BLUR_FACTOR = 0.3
            raw_k = side * BLUR_FACTOR
            # ensure odd integer ≥1
            k = max(1, int(raw_k) // 2 * 2 + 1)

            frame[y1_p:y2_s, x1_p:x2_s] = cv2.blur(roi, (k, k))
            logger.info(
                f"Blurred face for person {idx} in region: ({x1_p},{y1_p}) to ({x2_s},{y2_s})"
            )
            num_faces_blurred += 1
        else:
            logger.warning(f"Empty ROI for person {idx}; skipping.")

    logger.info(f"Finished processing. Total faces blurred: {num_faces_blurred}")
    return frame


# _________________________________________________________________________________________________________________________________________


# _________________________________________________________________________________________________________________________________________
# Count People

# Keep track of the last time count was logged
last_count_log_time = 0  # global or pass into function if needed


def count_people(frame, results):
    """
    Draws bounding boxes around each detected person, labels them 1…N (left to right),
    and logs count every 60 seconds.
    Returns: (annotated_frame, count)
    """
    global last_count_log_time

    if not results or len(results) == 0:
        logger.warning("No results returned by model.")
        return frame, 0

    result = results[0]

    try:
        boxes = result.boxes.xyxy.cpu().numpy()
    except Exception as e:
        logger.error("❌ Error accessing boxes: %s", e)
        return frame, 0

    if boxes.size == 0:
        logger.info("No people detected in frame.")
        cv2.putText(
            frame, "Count: 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )
        return frame, 0

    # sort boxes left to right
    order = np.argsort(boxes[:, 0])
    count = len(order)
    logger.debug(f"{count} people detected.")

    for idx, i in enumerate(order, start=1):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            str(idx),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

    # Add total count to frame
    cv2.putText(
        frame,
        f"Count: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    # Print count every 60 seconds
    current_time = time.time()
    if current_time - last_count_log_time >= 60:
        logger.info(f"👥 People Count: {count}")
        last_count_log_time = current_time

    return frame


# _________________________________________________________________________________________________________________________________________

# _________________________________________________________________________________________________________________________________________

# SEAT Status

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

# Create Shapely polygons and integer arrays for drawing
poly_map = {name: Polygon(pts) for name, pts in seats.items()}
poly_int = {name: np.array(pts, np.int32) for name, pts in seats.items()}

# Stats: dwell, current empty, total empty
stats = {name: {"dwell": 0.0, "empty": 0.0, "empty_total": 0.0} for name in seats}

# Timing variable (initialized on first call)
last_ts = None

from shapely.geometry import Polygon
import numpy as np


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
            name: {"dwell": 0.0, "empty": 0.0, "empty_total": 0.0}
            for name in zones
        }

        return poly_map, poly_int, stats

    except Exception as e:
        print(f"❌ Error retrieving seats: {e}")
        return {}, {}, {}


def seat_status(img, results, poly_map, poly_int, stats):
    global last_ts
    now = time.time()
    dt = 0.0 if last_ts is None else now - last_ts
    last_ts = now

    # Draw panel
    panel_x, panel_y = 10, 40
    line_h = 20
    panel_w = 280
    panel_h = line_h * len(poly_map) + 10
    cv2.rectangle(
        img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1
    )

    result = results[0]
    boxes = (
        result.boxes.xyxy.cpu().numpy()
        if result.boxes is not None
        else np.empty((0, 4))
    )
    centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]

    for cx, cy in centers:
        cv2.circle(img, (int(cx), int(cy)), radius=5, color=(0, 0, 255), thickness=-1)

    for i, (name, poly) in enumerate(poly_map.items()):
        occupied = any(poly.contains(Point(x, y)) for x, y in centers)

        if occupied:
            stats[name]["dwell"] += dt
            stats[name]["empty"] = 0.0
        else:
            stats[name]["empty"] += dt
            stats[name]["empty_total"] += dt

        cv2.polylines(img, [poly_int[name]], True, (255, 0, 0), 1)
        cx, cy = map(int, poly.centroid.coords[0])
        draw_label_seat_status(
            img, f"{name} dwell: {stats[name]['dwell']:.1f}s", (cx - 40, cy + 6)
        )
        draw_label_seat_status(
            img, f"{name} empty: {stats[name]['empty']:.1f}s", (cx - 40, cy - 20)
        )

        y = panel_y + (i + 1) * line_h
        cv2.putText(
            img,
            f"{name}: total empty {stats[name]['empty_total']:.1f}s",
            (panel_x + 5, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return img


# def seat_status(img, results, poly_map, poly_int, stats):
#     """
#     Update and draw seat occupancy stats using real elapsed time between calls.
#     """
#     global last_ts
#     now = time.time()
#     # Compute delta-time since last frame
#     if last_ts is None:
#         dt = 0.0
#     else:
#         dt = now - last_ts
#     last_ts = now

#     # Panel metrics (for overlay)
#     panel_x, panel_y = 10, 40
#     line_h = 20
#     panel_w = 280
#     panel_h = line_h * len(seats) + 10

#     # Draw panel background once per frame
#     cv2.rectangle(
#         img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1
#     )

#     # Extract detection boxes and compute centers
#     result = results[0]
#     boxes = (
#         result.boxes.xyxy.cpu().numpy()
#         if result.boxes is not None
#         else np.empty((0, 4))
#     )
#     centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]
#     for cx, cy in centers:
#         cv2.circle(img, (int(cx), int(cy)), radius=5, color=(0, 0, 255), thickness=-1)

#     # Update stats per seat
#     for i, (name, poly) in enumerate(poly_map.items()):
#         occupied = any(poly.contains(Point(x, y)) for x, y in centers)

#         if occupied:
#             stats[name]["dwell"] += dt
#             stats[name]["empty"] = 0.0
#         else:
#             stats[name]["empty"] += dt
#             stats[name]["empty_total"] += dt

#         # Draw seat polygon
#         cv2.polylines(img, [poly_int[name]], True, (255, 0, 0), 2)

#         # Draw labels at centroid
#         cx, cy = map(int, poly.centroid.coords[0])
#         draw_label_seat_status(
#             img, f"{name} dwell: {stats[name]['dwell']:.1f}s", (cx - 40, cy + 6)
#         )
#         draw_label_seat_status(
#             img, f"{name} empty: {stats[name]['empty']:.1f}s", (cx - 40, cy - 20)
#         )

#         # Overlay total-empty stats on panel
#         y = panel_y + (i + 1) * line_h
#         cv2.putText(
#             img,
#             f"{name}: total empty {stats[name]['empty_total']:.1f}s",
#             (panel_x + 5, y),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 255, 0),
#             2,
#         )
#         print(stats)

#     return img


def draw_label_seat_status(
    img,
    text,
    org,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.4,
    txt_color=(255, 0, 0),
    bg_color=(0, 0, 0),
    thickness=1,
):
    (w, h), base = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 2
    cv2.rectangle(
        img, (x - pad, y - h - pad), (x + w + pad, y + base + pad), bg_color, -1
    )
    cv2.putText(img, text, org, font, font_scale, txt_color, thickness)


# _________________________________________________________________________________________________________________________________________

# In/Out count

def in_out_count_people(frame, counter, user_id=None, camera_id=None):
    results = counter.process(frame)
    img = results.plot_im.copy()
    
    # ✅ Bigger IN/OUT labels
    big_font = cv2.FONT_HERSHEY_SIMPLEX
    big_scale = 1.0
    big_thickness = 3

    draw_label(img, f"Total IN: {counter.in_count}", (50, 50),
               font=big_font, font_scale=big_scale, thickness=big_thickness)
    draw_label(img, f"Total OUT: {counter.out_count}", (50, 100),
               font=big_font, font_scale=big_scale, thickness=big_thickness)

    if user_id is not None and camera_id is not None:
        # ✅ Update memory store
        snapshot = update_in_out_stats(user_id, camera_id, counter)
        logger.info(f"📝 [in_out_count_people] Updated global store → {snapshot}")

        # ✅ Immediately save JSON (no need to wait 5s worker)
        try:
            from camera.tasks import save_in_out_stats_to_file

            logger.info("💾 [in_out_count_people] Forcing immediate save...")
            save_in_out_stats_to_file()   # <-- force write here
        except Exception as e:
            logger.error(f"❌ [in_out_count_people] Failed immediate save: {e}")

    else:
        logger.warning("⚠️ [in_out_count_people] user_id or camera_id missing, cannot persist counts")

    return img


# _________________________________________________________________________________________________________________________________________


time_tracker_state = {}

# People time tracker
def initialize_zones_dict(zones_dict):
    return {label: Polygon(coords) for label, coords in zones_dict.items()}


def get_color(track_id, color_map):
    if track_id not in color_map:
        np.random.seed(track_id)
        color_map[track_id] = tuple(int(x) for x in np.random.randint(0, 255, size=3))
    return color_map[track_id]


def process_frame(frame, tracker_obj, region_polygons, TIMEZONE,
                  entry_time_per_zone, wall_clock_entry_times,
                  dwell_time_per_zone, color_map, user_id=None, camera_id=None):
    logs = []
    try:
        tracker_obj.extract_tracks(frame)
    except Exception as e:
        print("Tracking failed:", e)
        return frame, []

    annotator = SolutionAnnotator(frame, line_width=2)
    current_time = time.monotonic()

    # 🔹 Draw zones
    for label, poly in region_polygons.items():
        pts = [(int(x), int(y)) for x, y in poly.exterior.coords]
        color = (0, 255, 0) if hash(label) % 2 == 0 else (255, 0, 0)
        annotator.draw_region(pts, color=color, thickness=2)

    # 🔹 Process tracks
    for box, track_id, cls in zip(tracker_obj.boxes, tracker_obj.track_ids, tracker_obj.clss):
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        center = Point(cx, cy)

        inside_zones = [label for label, poly in region_polygons.items() if poly.intersects(center)]

        if inside_zones:
            for label in inside_zones:
                if track_id not in entry_time_per_zone[label]:
                    entry_time_per_zone[label][track_id] = current_time
                    wall_clock_entry_times[label][track_id] = datetime.now(TIMEZONE)

                elapsed = current_time - entry_time_per_zone[label][track_id]
                dwell_time_per_zone[label][track_id] = elapsed

                label_text = f"{int(elapsed)}s"
                color = get_color(track_id, color_map)
                annotator.box_label(box, label=label_text, color=color)

        else:
            for label in list(region_polygons.keys()):
                if track_id in entry_time_per_zone[label]:
                    entry_wall = wall_clock_entry_times[label].pop(track_id, None)
                    exit_wall = datetime.now(TIMEZONE)

                    entry_time_per_zone[label].pop(track_id, None)
                    dwell_time_per_zone[label].pop(track_id, None)

                    if entry_wall:
                        duration = round((exit_wall - entry_wall).total_seconds(), 2)
                        visit_id = f"{label}_{entry_wall.strftime('%Y%m%dT%H%M%S')}_{track_id}"

                        logs.append({
                            "visit_id": visit_id,
                            "zone": label,
                            "entry_time": entry_wall.isoformat(),
                            "exit_time": exit_wall.isoformat(),
                            "duration_seconds": duration
                        })

                        # 🔹 Save duration into history
                        state = time_tracker_state.get((user_id, camera_id))
                        if state:
                            state["history"][track_id].append(duration)
                            logger.info(f"✅ Stored visit {visit_id} → {duration} sec (track_id={track_id})")
                            logger.info(f"📊 Current history for cam {camera_id}: {dict(state['history'])}")

    result_img = annotator.result()
    return result_img, logs


def people_time_tracker(frame, people_tracker, zone_polygons, TIMEZONE,
                        entry_time_per_zone, wall_clock_entry_times,
                        dwell_time_per_zone, track_colors, user_id=None, camera_id=None):

    img, logs = process_frame(
        frame,
        people_tracker,
        zone_polygons,
        TIMEZONE,
        entry_time_per_zone,
        wall_clock_entry_times,
        dwell_time_per_zone,
        track_colors,
        user_id,
        camera_id
    )

    logger.info(f"People time tracker Stats---: {logs}")

    # 🔹 Print avg waiting time
    avg_wait = get_average_waiting_time(user_id, camera_id)
    logger.info(f"⏱️ Average waiting time so far: {avg_wait} seconds")

    return img


def _zones_key(zones_dict):
    # make zones hashable for "zones changed" detection
    return tuple(sorted((name, tuple(map(tuple, pts))) for name, pts in zones_dict.items()))


def get_time_tracker_state(user_id, camera_id, zones):
    key = (user_id, camera_id)
    zkey = _zones_key(zones)

    state = time_tracker_state.get(key)
    if (state is None) or (state["zones_key"] != zkey):
        # (Re)initialize on first run or if zones changed
        state = {
            "people_tracker": BaseSolution(model="yolo11n.pt", classes=[0], conf=0.3, device="cpu"),
            "zone_polygons": {label: Polygon(coords) for label, coords in zones.items()},
            "entry": {label: {} for label in zones},
            "dwell": {label: {} for label in zones},
            "wall":  {label: {} for label in zones},
            "colors": {},
            "tz": pytz.timezone("Asia/Riyadh"),
            "zones_key": zkey,
            "history": defaultdict(list),   
        }
        time_tracker_state[key] = state

    return state



def get_average_waiting_time(user_id, camera_id):
    state = time_tracker_state.get((user_id, camera_id))
    if not state:
        logger.info("⚠️ No state found yet.")
        return 0.0

    all_durations = [d for durations in state["history"].values() for d in durations]
    if not all_durations:
        logger.info("⚠️ No completed visits yet.")
        return 0.0

    avg_wait = sum(all_durations) / len(all_durations)
    logger.info(f"📈 Average wait from {len(all_durations)} visits = {avg_wait:.2f}s")
    return round(avg_wait, 2)


# _________________________________________________________________________________________________________________________________________
# PPE KIT


# PPE_WEIGHTS_PATH = r"PPE_model.pt"

# PPE_model = YOLO(PPE_WEIGHTS_PATH)
PPE_model = YOLO(resource_path("PPE_model.pt"))


def ppe_detection(frame, boxes):
    model = PPE_model
    results = model.predict(
        frame,
    )
    annotated = results[0].plot()
    return annotated


# _________________________________________________________________________________________________________________________________________


# _________________________________________________________________________________________________________________________________________
# Fire Detection

# FIRE_SMOKE_WEIGHTS_PATH   = r"fire_smoke_model.pt"

# FIRE_SMOKE_model = YOLO(FIRE_SMOKE_WEIGHTS_PATH)
FIRE_SMOKE_model = YOLO(resource_path("fire_smoke_model.pt"))


def fire_smoke_detection(frame, boxes):
    model = FIRE_SMOKE_model
    results = model.predict(
        frame,
    )
    annotated = results[0].plot()
    return annotated


# _________________________________________________________________________________________________________________________________________


# _________________________________________________________________________________________________________________________________________
# main function



# Defaults for the counter region/model (used if DB zones missing)
REGION = [(710, 216), (710, 204), (788, 298), (786, 307)]
MODEL_PATH = "yolo11n.pt"

# Cache pose model so it's loaded once per process
_POSE_MODEL = None


def _get_pose_model():
    global _POSE_MODEL
    if _POSE_MODEL is None:
        _POSE_MODEL = YOLO(resource_path("yolo11n-pose.pt"))
    return _POSE_MODEL


import logging

logger = logging.getLogger("camera_ai")

def _extract_region(zones, default_region, frame_width, frame_height):
    """
    Normalize zones into dict of {name: polygon} and convert percentages to pixel coordinates.
    """
    if not zones:
        return {"Default": default_region}
    
    if isinstance(zones, dict):
        out = {}
        for name, poly in zones.items():
            try:
                # Convert percentage coordinates to pixel coordinates
                pixel_poly = []
                for pt in poly:
                    if len(pt) == 2:
                        # Convert from percentage (0-100) to pixel coordinates
                        x_percent, y_percent = pt
                        x_pixel = (x_percent / 100.0) * frame_width
                        y_pixel = (y_percent / 100.0) * frame_height
                        pixel_poly.append((int(x_pixel), int(y_pixel)))
                    else:
                        # Assume already in pixel format
                        pixel_poly.append(tuple(map(int, pt)))
                out[name] = pixel_poly
                print(f"[Zones] Converted {name}: {poly} -> {pixel_poly}")
            except Exception as e:
                print(f"[Zones] Failed parsing {name}: {e}")
        return out
    
    if isinstance(zones, list):
        try:
            pixel_poly = []
            for pt in zones:
                x_percent, y_percent = pt
                x_pixel = (x_percent / 100.0) * frame_width
                y_pixel = (y_percent / 100.0) * frame_height
                pixel_poly.append((int(x_pixel), int(y_pixel)))
            return {"Zone1": pixel_poly}
        except Exception as e:
            print(f"[Zones] Failed parsing list: {e}")
            return {"Default": default_region}
    
    return {"Default": default_region}

def execute_user_ai_models(
    user_id, camera_id, frame, rtsp_url=None, save_to_json=False
):
    print(f"{user_id=}")
    print(f"{camera_id=}")
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    print(f"Frame dimensions: {frame_width}x{frame_height}")
    
    user_ai_models = UserAiModel.objects.filter(
        user_id=user_id, camera_id=camera_id, is_active=True
    )
    
    function_map = {
        "blur_faces": blur_faces,
        "pixelate_people": pixelate_people,
        "count_people": count_people,
        "generate_people_heatmap": generate_people_heatmap,
        "track_posture_and_occupancy": track_posture_and_occupancy,
        "seat_status": seat_status,
        "in_out_count_people": in_out_count_people,
        "fire_smoke_detction": fire_smoke_detection,
        "ppe_detection": ppe_detection,
        "people_time_tracker": people_time_tracker,

    }
    
    out_frame = frame.copy()
    pose_model = None
    pose_results = None
    
    def ensure_pose_results():
        nonlocal pose_model, pose_results
        if pose_results is None:
            if pose_model is None:
                pose_model = _get_pose_model()
            pose_results = pose_model(out_frame)
        return pose_results
    
    for user_ai_model in user_ai_models:
        ai_model = user_ai_model.aimodel
        function_name = ai_model.function_name
        print(f"Calling function: {function_name}")
        
        if function_name not in function_map:
            print(f"❌ No function found for AiModel {function_name}.")
            continue
        
        try:
            if function_name == "people_time_tracker":  # <- ensure this is elif in your final code
                zones = _extract_region(user_ai_model.zones, REGION, frame_width, frame_height)
                logger.info(f"[Zones] Final zones for camera {user_ai_model.camera_id}: {zones}")

                state = get_time_tracker_state(user_id, camera_id, zones)

                out_frame = people_time_tracker(
                    out_frame,  # use the current working frame
                    state["people_tracker"],
                    state["zone_polygons"],
                    state["tz"],
                    state["entry"],
                    state["wall"],
                    state["dwell"],
                    state["colors"],
                )


            elif function_name == "in_out_count_people":
                # Pass frame dimensions to convert percentages to pixels
                zones = _extract_region(user_ai_model.zones, REGION, frame_width, frame_height)
                logger.info(f"[Zones] Final zones for camera {user_ai_model.camera_id}: {zones}")
                
                for zone_name, region in zones.items():
                    logger.info(f"Running counter for {zone_name}: {region}")
                    # Create a unique key for the counter
                    zone_key = f"{camera_id}_{zone_name}"

                    # Reuse or create ObjectCounter
                    if zone_key not in zone_counters:
                        zone_counters[zone_key] = solutions.ObjectCounter(
                            model="yolo11n.pt",
                            region=region,
                            classes=[0],
                            analytics_type="crossing",  
                            show_in=False,
                            show_out=False,
                            show_conf=False,
                            show_labels=False,
                            line_width=1,
                        )
                        logger.info(f"✅ Initialized ObjectCounter for zone {zone_key}")

                    counter = zone_counters[zone_key]
                    out_frame = in_out_count_people(out_frame, counter, user_id, camera_id)
            elif function_name == "seat_status":
                results = ensure_pose_results()
                poly_map, poly_int, stats = get_seat_polygons_from_model(
                    user_id, camera_id, frame_width, frame_height
                )
                if not poly_map:
                    print("⚠️ No seat zones configured for seat_status.")
                    continue
                out_frame = seat_status(out_frame, results, poly_map, poly_int, stats)

            else:
                results = ensure_pose_results()
                func = function_map[function_name]
                out_frame = func(out_frame, results)
            
            print(f"✅ Processed frame using {function_name}.")
        except Exception as e:
            print(f"❌ Error during {function_name} execution: {e}")
    
    return out_frame


def save_sample_data_to_json(user_id, camera_id, frame):
    sample_data = {
        "user_id": user_id,
        "camera_id": camera_id,
        "data": "hello rajesh",  # Example: Save the shape of the processed frame
    }

    # Save data to a JSON file
    with open("sample_data.json", "w") as json_file:
        json.dump(sample_data, json_file, indent=4)
    print(f"✅ Sample data saved to JSON: {sample_data}")


# _________________________________________________________________________________________________________________________________________
