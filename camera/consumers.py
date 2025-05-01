# consumers.py
import cv2
import threading
import time
from channels.generic.websocket import WebsocketConsumer
from .models import Camera

import cv2
import threading
import time
import os
from channels.generic.websocket import WebsocketConsumer
from .models import Camera

class CameraStreamConsumer(WebsocketConsumer):
    def connect(self):
        self.camera_id = self.scope["url_route"]["kwargs"]["camera_id"]
        self.accept()
        self.streaming = True

        # Load fallback image once
        fallback_path = os.path.join(os.path.dirname(__file__), 'no_frame.jpg')
        fallback_img = cv2.imread(fallback_path)
        if fallback_img is not None:
            _, self.fallback_buffer = cv2.imencode('.jpg', fallback_img)
        else:
            self.fallback_buffer = None
            print("[WARN] no_frame.jpg not found. Fallback frame unavailable.")

        threading.Thread(target=self.stream_video, daemon=True).start()

    def disconnect(self, close_code):
        self.streaming = False

    def build_rtsp_url(self, cam):
        # Preserve existing credentials in URL if present
        if "@" in cam.rtsp_url:
            return cam.rtsp_url + "?rtsp_transport=tcp"

        if cam.username and cam.password:
            parts = cam.rtsp_url.split("://")
            if len(parts) == 2:
                protocol, rest = parts
                return f"{protocol}://{cam.username}:{cam.password}@{rest}?rtsp_transport=tcp"

        return cam.rtsp_url + "?rtsp_transport=tcp"

    def send_fallback_frame(self):
        if self.fallback_buffer is not None:
            try:
                self.send(bytes_data=self.fallback_buffer.tobytes())
            except Exception as e:
                print(f"[EXCEPTION] Sending fallback frame failed: {e}")


    def stream_video(self):
        try:
            cam = Camera.objects.get(id=self.camera_id)
        except Camera.DoesNotExist:
            self.close()
            return

        rtsp_url = self.build_rtsp_url(cam)
        print(f"[INFO] Attempting to stream from: {rtsp_url}")

        while self.streaming:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

            time.sleep(1)  # short wait for decoder to initialize

            if not cap.isOpened():
                print("[WARN] Cannot open stream. Sending fallback and retrying...")
                self.send_fallback_frame()
                time.sleep(3)
                continue

            print("[INFO] Stream opened successfully.")

            while self.streaming and cap.isOpened():
                for _ in range(2):  # discard old frames
                    cap.grab()

                success, frame = cap.read()
                if not success:
                    print("[ERROR] Frame read failed. Reconnecting...")
                    self.send_fallback_frame()
                    break

                try:
                    _, buffer = cv2.imencode('.jpg', frame)
                    self.send(bytes_data=buffer.tobytes())
                except Exception as e:
                    print(f"[EXCEPTION] Sending frame failed: {e}")
                    break

                time.sleep(0.01)  # ~100 fps limit (adjust as needed)

            cap.release()
            print("[INFO] Camera stream ended. Reconnecting...")
            time.sleep(2)


# class CameraStreamConsumer(WebsocketConsumer):
#     def connect(self):
#         self.camera_id = self.scope["url_route"]["kwargs"]["camera_id"]
#         self.accept()
#         self.streaming = True
#         self.thread = threading.Thread(target=self.stream_video)
#         self.thread.start()

#     def disconnect(self, close_code):
#         self.streaming = False

#     def stream_video(self):
#         from .models import Camera
#         cam = Camera.objects.get(id=self.camera_id)
#         cap = cv2.VideoCapture(cam.rtsp_url)

#         model = YOLO("yolov8n.pt")  # Or your custom model for person detection

#         while self.streaming:
#             success, frame = cap.read()
#             if not success:
#                 continue

#             # Detect
#             results = model(frame)[0].boxes  # Get detections

#             # Blur faces (upper body assumed to contain face)
#             frame = blur_faces(frame, results)

#             # Encode + Send
#             _, buffer = cv2.imencode('.jpg', frame)
#             self.send(bytes_data=buffer.tobytes())
#             time.sleep(0.05)  # ~20fps

#         cap.release()
