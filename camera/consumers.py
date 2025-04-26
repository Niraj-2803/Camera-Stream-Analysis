import cv2
import threading
import time
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

    def stream_video(self):
        cam = Camera.objects.get(id=self.camera_id)
        rtsp_url = cam.rtsp_url
        print(f"[INFO] Connecting to camera stream: {rtsp_url}")

        while self.streaming:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

            if not cap.isOpened():
                print("[WARN] Could not open RTSP stream. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            print("[INFO] Stream opened successfully.")

            while self.streaming and cap.isOpened():
                # Optional: discard buffered frames to reduce lag
                for _ in range(3):
                    cap.grab()

                success, frame = cap.read()
                if not success:
                    print("[ERROR] Failed to read frame. Attempting to reconnect...")
                    break  # Exit to outer loop and reconnect

                try:
                    _, buffer = cv2.imencode('.jpg', frame)
                    self.send(bytes_data=buffer.tobytes())
                except Exception as e:
                    print(f"[EXCEPTION] Failed to send frame: {e}")
                    break

                time.sleep(0.01)  # ~60 FPS or adjust based on latency

            cap.release()
            print("[INFO] Stream released. Reconnecting in 5 seconds...")
            time.sleep(5)  # Avoid tight reconnect loop

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
