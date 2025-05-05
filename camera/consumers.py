import cv2
import threading
import time
import os
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
        threading.Thread(target=self.stream_video, daemon=True).start()

    def disconnect(self, close_code):
        self.streaming = False

    def build_rtsp_url(self, cam):
        # If username/password already in URL, don't alter it
        if "@" in cam.rtsp_url:
            return cam.rtsp_url

        if cam.username and cam.password:
            parts = cam.rtsp_url.split("://")
            if len(parts) == 2:
                protocol, rest = parts
                return f"{protocol}://{cam.username}:{cam.password}@{rest}"
        return cam.rtsp_url

    def stream_video(self):
        try:
            cam = Camera.objects.get(id=self.camera_id)
        except Camera.DoesNotExist:
            self.close()
            return

        rtsp_url = self.build_rtsp_url(cam)

        # Print the RTSP URL to the terminal
        print(f"[INFO] Connecting to camera stream: {rtsp_url}")

        while self.streaming:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set the buffer size to minimize memory consumption
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            cap.set(cv2.CAP_PROP_FPS, 30)  # Optionally set FPS if supported

            if not cap.isOpened():
                print("[WARN] Could not open RTSP stream. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            print("[INFO] Stream opened successfully.")

            while self.streaming and cap.isOpened():
                # Optional: discard buffered frames
                for _ in range(3):
                    cap.grab()

                success, frame = cap.read()
                if not success:
                    print("[ERROR] Failed to read frame. Sending default image.")
                    # Load and send the default image "no_frame.jpg"
                    fallback_path = os.path.join(os.path.dirname(__file__), 'no_frame.jpg')
                    fallback_img = cv2.imread(fallback_path)
                    if fallback_img is not None:
                        _, self.fallback_buffer = cv2.imencode('.jpg', fallback_img)
                    else:
                        self.fallback_buffer = None
                        print("[WARN] no_frame.jpg not found. Fallback frame unavailable.")

                    # Send fallback frame if available
                    if self.fallback_buffer is not None:
                        self.send(bytes_data=self.fallback_buffer.tobytes())

                    time.sleep(0.01)  # adjust FPS if needed
                    continue

                try:
                    _, buffer = cv2.imencode('.jpg', frame)
                    self.send(bytes_data=buffer.tobytes())
                except Exception as e:
                    print(f"[EXCEPTION] Failed to send frame: {e}")
                    break

                time.sleep(0.01)  # adjust FPS if needed

            cap.release()
            print("[INFO] Stream released. Reconnecting in 5 seconds...")
            time.sleep(5)

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
