import cv2
from ultralytics import YOLO
from ultralytics import solutions
import numpy as np
import os


# 🧠 Use Case 1: Blur the top part of the 'person' bounding box
def blur_faces(frame, boxes, blur_size=35, face_ratio=0.2):
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
def count_people(model_path, width, height, region_points=None):
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


if __name__ == "__main__":
    input_source = r"input_video\gewan_indoor_1_min.mp4"
    output_file = r"output_video.mp4"
    model_path = "yolo11n.pt"

    # custom_region = [(100, 500), (1100, 500), (1100, 450), (100, 450)]

    # process_video_stream(
    #     source=input_source,
    #     output_path=output_file,
    #     model_path=model_path,
    #     usecase="people_heatmap"
    # )

    # 🔒 Other use cases (examples):
    # process_video_stream(source=input_source, output_path="output.mp4", model_path=model_path, usecase="blur_faces")
    # process_video_stream(source=input_source, output_path=output_file, model_path=model_path, usecase="pixelate_people")
    # process_video_stream(source=input_source, output_path="output.mp4", model_path=model_path, usecase="count_cars")
    # process_video_stream(source=input_source, output_path="output.mp4", model_path=model_path, usecase="count_people")
    process_video_stream(source=input_source, output_path="output.mp4", model_path=model_path, usecase="people_heatmap")
