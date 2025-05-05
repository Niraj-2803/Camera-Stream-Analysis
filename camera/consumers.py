

import logging
import cv2
import threading
import time
import os
from channels.generic.websocket import WebsocketConsumer
from .models import Camera

# Configure logging to output to stdout, which is captured by Gunicorn
logging.basicConfig(level=logging.DEBUG)  # Change level to DEBUG, INFO, etc. based on your needs
logger = logging.getLogger(__name__)

class CameraStreamConsumer(WebsocketConsumer):
    def connect(self):
        self.camera_id = self.scope["url_route"]["kwargs"]["camera_id"]
        self.accept()
        self.streaming = True
        threading.Thread(target=self.stream_video, daemon=True).start()

    def disconnect(self, close_code):
        self.streaming = False

    def build_rtsp_url(self, cam):
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

        # Log the RTSP URL to stdout (which will be captured by Gunicorn)
        logger.info(f"Connecting to camera stream: {rtsp_url}")

        while self.streaming:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            cap.set(cv2.CAP_PROP_FPS, 30)

            if not cap.isOpened():
                logger.warning("Could not open RTSP stream. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            logger.info("Stream opened successfully.")

            while self.streaming and cap.isOpened():
                for _ in range(3):
                    cap.grab()

                success, frame = cap.read()
                if not success:
                    logger.error("Failed to read frame from stream. Sending default image.")
                    fallback_path = os.path.join(os.path.dirname(__file__), 'no_frame.jpg')
                    fallback_img = cv2.imread(fallback_path)

                    if fallback_img is not None:
                        _, self.fallback_buffer = cv2.imencode('.jpg', fallback_img)
                        self.send(bytes_data=self.fallback_buffer.tobytes())
                    else:
                        self.fallback_buffer = None
                        logger.error("no_frame.jpg not found. Sending empty frame.")
                        self.send(bytes_data=b'')

                    time.sleep(0.01)
                    continue

                try:
                    _, buffer = cv2.imencode('.jpg', frame)
                    self.send(bytes_data=buffer.tobytes())
                except Exception as e:
                    logger.exception("Failed to send frame:")
                    break

                time.sleep(0.01)

            cap.release()
            logger.info("Stream released. Reconnecting in 5 seconds...")
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
