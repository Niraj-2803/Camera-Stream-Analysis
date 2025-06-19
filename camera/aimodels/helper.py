from collections import defaultdict
import cv2
from ultralytics import YOLO
from ultralytics import solutions
import numpy as np
import os

from camera.models import UserAiModel


# 🧠 Use Case 1: Blur the top part of the 'person' bounding box
def blur_faces(frame, boxes, blur_size=35, face_ratio=0.2):
    print("blur face execute")
    for box in boxes:
        cls = int(box.cls[0])
        if cls != 0:
            continue  # Only target class 0 (person)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_height = y2 - y1
        face_height = int(box_height * face_ratio)
        y2_face = y1 + face_height

        face_region = frame[y1:y2_face, x1:x2]
        if face_region.size > 0:
            blurred_face = cv2.blur(face_region, (blur_size, blur_size))
            frame[y1:y2_face, x1:x2] = blurred_face

    return frame


# 🧠 Use Case 2: Pixelate full person area (for privacy or censorship)
def pixelate_people(frame, boxes, pixel_size=10):
    for box in boxes:
        cls = int(box.cls[0])
        if cls != 0:
            continue  # Only target class 0 (person)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        region = frame[y1:y2, x1:x2]
        if region.size > 0:
            small = cv2.resize(region, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = pixelated

    return frame


# 🧠 Use Case 3a: Count cars only
def count_cars(model_path, width, height, region_points=None):
    if region_points is None:
        region_points = [
            (20, int(height * 0.75)),
            (width - 20, int(height * 0.75)),
            (width - 20, int(height * 0.7)),
            (20, int(height * 0.7))
        ]

    counter = solutions.ObjectCounter(
        show=False,
        region=region_points,
        model=model_path,
        classes=[1, 2, 3, 5, 7]  # Bicycle, car, motorcycle, bus, truck
    )

    def counter_fn(frame):
        results = counter(frame)
        return results.plot_im

    return counter_fn


# 🧠 Use Case 3b: Count people only
def count_people(model_path, width, region_points=None):
    print("count people execute")
    height =1
    if region_points is None:
        region_points = [
            (20, int(height * 0.5)),
            (width - 20, int(height * 0.5)),
            (width - 20, int(height * 0.45)),
            (20, int(height * 0.45))
        ]

    counter = solutions.ObjectCounter(
        show=False,
        region=region_points,
        model=model_path,
        classes=[0]  # Only people (class 0)
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
        show=False
    )

    def heatmap_fn(frame):
        results = heatmap(frame)
        return results.plot_im

    return heatmap_fn

# 🔁 Main processing engine
def process_video_stream(source, output_path, model_path, usecase="blur_faces", region_points=None):
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
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
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
            show=False
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


# helpers.py

import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
import json
import time
from collections import defaultdict


# Function to process posture and occupancy tracking
def track_posture_and_occupancy(model, source, output_path, stats_file='stats.json', show=True):
    print(f"🔗 Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Unable to open video source {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Video source opened. FPS: {fps}, Resolution: {w}x{h}")

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Seat polygons and tracking (add seat coordinates here)
    seats = { ... }  # Define your seat polygons here
    seat_poly = {n: Polygon(pts) for n, pts in seats.items()}
    stats = {s: defaultdict(float) for s in seats}
    owner_tid = {s: None for s in seats}
    owner_miss = {s: 0 for s in seats}

    print(f"⚙️ YOLO Model loading: {model}")
    model = YOLO(model)  # Load the YOLO model
    print(f"✅ Model {model} loaded successfully.")

    frame_idx = 0
    last_dump = time.time()

    # Start processing frames
    print("🎥 Start processing frames...")
    for res in model.track(source=source, stream=True, verbose=False):
        frame_idx += 1
        frame = res.orig_img.copy()

        kps = res.keypoints.xy.cpu().numpy()
        kconf = res.keypoints.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        tids = res.boxes.id.cpu().numpy()

        print(f"⏱ Frame {frame_idx}: Processing frame...")

        # Track posture and classify (use your logic)
        centroid, posture = {}, {}
        for i, kp in enumerate(kps):
            tid = int(tids[i])
            x0, y0, x1, y1 = boxes[i]
            centroid[tid] = Point((x0 + x1) / 2, (y0 + y1) / 2)
            ang = {k: np.nan for k in ('l_knee', 'r_knee', 'l_hip', 'r_hip')}
            print(f"👤 Tracking person {tid}, keypoints: {kp}")

            # Compute angles if the keypoints for legs are available
            if kconf[i][11] > .3 and kconf[i][13] > .3 and kconf[i][15] > .3:
                ang['l_knee'] = compute_angle(kp[11], kp[13], kp[15])
                print(f"    Left knee angle: {ang['l_knee']}")
            if kconf[i][12] > .3 and kconf[i][14] > .3 and kconf[i][16] > .3:
                ang['r_knee'] = compute_angle(kp[12], kp[14], kp[16])
                print(f"    Right knee angle: {ang['r_knee']}")
            if kconf[i][5] > .3 and kconf[i][11] > .3 and kconf[i][13] > .3:
                ang['l_hip'] = compute_angle(kp[5], kp[11], kp[13])
            if kconf[i][6] > .3 and kconf[i][12] > .3 and kconf[i][14] > .3:
                ang['r_hip'] = compute_angle(kp[6], kp[12], kp[14])
            
            posture[tid] = classify_posture(kp, kconf[i], ang, img_h=h)
            print(f"    Person {tid} posture: {posture[tid]}")

        # Track seat occupancy based on posture and update stats
        for seat, poly in seat_poly.items():
            owner = owner_tid[seat]
            ids_in = [tid for tid, pt in centroid.items() if poly.contains(pt)]
            print(f"🪑 Checking seat {seat}, owner: {owner}, ids_in: {ids_in}")

            if owner is None and ids_in:
                owner = owner_tid[seat] = ids_in[0]
                owner_miss[seat] = 0
                print(f"    Seat {seat} occupied by person {owner}.")
            elif owner is not None:
                if owner in ids_in:
                    stats[seat]['dwell'] += 1 / fps
                    stats[seat][posture[owner].lower()] += 1 / fps
                    owner_miss[seat] = 0
                    print(f"    Seat {seat} still occupied by person {owner}, updating stats.")
                else:
                    owner_miss[seat] += 1
                    if owner_miss[seat] > 30:
                        owner_tid[seat] = None
                        owner_miss[seat] = 0
                        print(f"    Seat {seat} is now unoccupied.")

        # Draw the seat occupancy and posture
        print(f"✏️ Drawing seat occupancy and posture on frame {frame_idx}...")
        draw_seats(frame, seat_poly, stats)

        writer.write(frame)
        if show:
            cv2.imshow("Posture Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save stats periodically every 100 frames
        if frame_idx % 100 == 0:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"📊 Stats saved to {stats_file} at frame {frame_idx}.")

    writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"✅ Video processing complete. Stats saved in {stats_file}")
    return stats


def compute_angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    ba, bc = a - b, c - b
    d = np.linalg.norm(ba) * np.linalg.norm(bc)
    if d < 1e-6:
        return np.nan
    cos = np.dot(ba, bc) / d
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def classify_posture(kp, conf, ang, img_h=None, min_visible=8):
    # Logic to classify posture
    # Checks the knee and hip angles to classify as Sitting, Standing, or Uncertain
    legs_ok = all(conf[i] > .3 for i in (11, 12, 13, 14, 15, 16))
    if legs_ok:
        knee = np.nanmean([ang['l_knee'], ang['r_knee']])
        hip = np.nanmean([ang['l_hip'], ang['r_hip']])
        if knee < 120 or hip < 120:
            return 'Sitting'
        if knee > 150 and hip > 150:
            return 'Standing'
        return 'Uncertain'

    if img_h is not None and np.sum(conf > .3) >= min_visible:
        s_y = np.nanmean([kp[5][1], kp[6][1]])
        h_y = np.nanmean([kp[11][1], kp[12][1]])
        return 'Sitting' if 0.5 * (s_y + h_y) > img_h * 0.55 else 'Standing'

    return 'Uncertain'


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
            thickness=2
        )


def draw_label(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.6, txt_color=(255, 0, 0), bg_color=(0, 0, 0), thickness=2):
    """
    Draw text with a solid background.
    """
    (w, h), base = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 2

    cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad), bg_color, -1)
    cv2.putText(img, text, org, font, font_scale, txt_color, thickness)


import json

def execute_user_ai_models(user_id, camera_id, frame, save_to_json=False):
    print(f'{user_id=}')
    print(f'{camera_id=}')
    
    # Fetch all UserAiModel instances for the given user and camera
    user_ai_models = UserAiModel.objects.filter(user_id=user_id, camera_id=camera_id, is_active=True)

    # Loop through each related AiModel and execute the corresponding function
    for user_ai_model in user_ai_models:
        ai_model = user_ai_model.aimodel
        print(f'{ai_model=}')
        
        # Get the function name from the AiModel instance
        function_name = ai_model.function_name
        print(f'Calling function: {function_name}')
        
        # Dynamically map the function name to the actual function
        function_map = {
            "blur_faces": blur_faces,
            "pixelate_people": pixelate_people,
            "count_people": count_people,
            "generate_people_heatmap": generate_people_heatmap,
            "track_posture_and_occupancy": track_posture_and_occupancy  # New function for posture tracking
        }

        # Check if the function exists in the map
        if function_name in function_map:
            function_to_execute = function_map[function_name]
            print(f"Executing {function_name} for user {user_id} and camera {camera_id}.")
            
            # If it's the 'track_posture' function, pass the correct arguments
            if function_name == "track_posture_and_occupancy":
                try:
                    result = function_to_execute(
                        model="yolo11m-pose.pt",  # You can replace with the appropriate model path
                        source=f"rtsp://camera_{camera_id}_url",  # Replace with actual camera RTSP URL
                        output_path="output_video.mp4",
                        stats_file='stats.json',  # You can specify stats output
                        show=True  # Or set this to False based on your needs
                    )
                    if save_to_json:
                        # Save data to a JSON file after processing
                        save_sample_data_to_json(user_id, camera_id, result)
                except Exception as e:
                    print(f"❌ Error during posture tracking execution: {e}")

            else:
                # For other AI models like `blur_faces`, `count_people`, etc.
                try:
                    boxes = []  # Replace this with actual detected boxes from your model
                    processed_frame = function_to_execute(frame, boxes)
                    print(f"Processed frame using {function_name}.")
                except Exception as e:
                    print(f"❌ Error during {function_name} execution: {e}")
        else:
            print(f"❌ No function found for AiModel {ai_model.function_name}.")

def save_sample_data_to_json(user_id, camera_id, frame):
    sample_data = {
        "user_id": user_id,
        "camera_id": camera_id,
        "data": "hello rajesh"  # Example: Save the shape of the processed frame
    }

    # Save data to a JSON file
    with open("sample_data.json", "w") as json_file:
        json.dump(sample_data, json_file, indent=4)
    print(f"✅ Sample data saved to JSON: {sample_data}")

