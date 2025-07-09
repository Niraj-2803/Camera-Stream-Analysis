import logging
import cv2
import threading
import time
import os
from channels.generic.websocket import WebsocketConsumer

from camera.aimodels.helper import *
from .models import Camera

# Configure logging to output to stdout, which is captured by Gunicorn
logging.basicConfig(level=logging.DEBUG)  # Change level to DEBUG, INFO, etc. based on your needs
logger = logging.getLogger(__name__)

# class CameraStreamConsumer(WebsocketConsumer):
#     def connect(self):
#         self.camera_id = self.scope["url_route"]["kwargs"]["camera_id"]
#         self.accept()
#         self.streaming = True
#         threading.Thread(target=self.stream_video, daemon=True).start()

#     def disconnect(self, close_code):
#         self.streaming = False

#     def build_rtsp_url(self, cam):
#         if "@" in cam.rtsp_url:
#             return cam.rtsp_url

#         if cam.username and cam.password:
#             parts = cam.rtsp_url.split("://")
#             if len(parts) == 2:
#                 protocol, rest = parts
#                 return f"{protocol}://{cam.username}:{cam.password}@{rest}"
#         return cam.rtsp_url

#     def send_fallback_frame(self):
#         # Load fallback image only once
#         fallback_path = os.path.join(os.path.dirname(__file__), 'no_frame.jpg')
#         fallback_img = cv2.imread(fallback_path)

#         if fallback_img is not None:
#             _, self.fallback_buffer = cv2.imencode('.jpg', fallback_img)
#             self.send(bytes_data=self.fallback_buffer.tobytes())
#         else:
#             self.fallback_buffer = None
#             logger.error("no_frame.jpg not found. Sending empty frame.")
#             self.send(bytes_data=b'')  # Send empty frame if no_frame.jpg is missing

#     def stream_video(self):
#         try:
#             cam = Camera.objects.get(id=self.camera_id)
#         except Camera.DoesNotExist:
#             self.close()
#             return

#         rtsp_url = self.build_rtsp_url(cam)

#         # Log the RTSP URL to stdout (which will be captured by Gunicorn)
#         logger.info(f"Connecting to camera stream: {rtsp_url}")

#         while self.streaming:
#             cap = cv2.VideoCapture(rtsp_url)
#             cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
#             cap.set(cv2.CAP_PROP_FPS, 30)

#             if not cap.isOpened():
#                 logger.warning("Could not open RTSP stream. Retrying in 5 seconds...")
#                 self.send_fallback_frame()  # Send fallback image when stream is unavailable
#                 time.sleep(5)
#                 continue

#             logger.info("Stream opened successfully.")

#             while self.streaming and cap.isOpened():
#                 for _ in range(3):  # Discard a few frames to ensure we don't send old frames
#                     cap.grab()

#                 success, frame = cap.read()
#                 if not success:
#                     logger.error("Failed to read frame from stream. Sending fallback image.")
#                     self.send_fallback_frame()  # Send fallback image if no frame is read
#                     time.sleep(0.01)  # Avoid tight loop, small delay before retrying
#                     continue

#                 try:
#                     _, buffer = cv2.imencode('.jpg', frame)
#                     self.send(bytes_data=buffer.tobytes())
#                 except Exception as e:
#                     logger.exception("Failed to send frame:")
#                     break

#                 time.sleep(0.01)  # Control frame rate (100 fps limit here)

#             cap.release()
#             logger.info("Stream released. Reconnecting in 5 seconds...")
#             time.sleep(5)  # Wait before trying to reconnect


class CameraStreamConsumer(WebsocketConsumer):
    def connect(self):
        # Extract user_id, camera_id, and mode from the URL
        self.user_id = self.scope["url_route"]["kwargs"]["user_id"]
        self.camera_id = self.scope["url_route"]["kwargs"]["camera_id"]
        mode = self.scope["url_route"]["kwargs"].get("mode", "normal")  # Default to 'normal' if not provided
        
        self.accept()
        self.streaming = True
        print(f"📡 Camera stream started for user {self.user_id}, camera {self.camera_id}, mode {mode}.")
        
        if mode == "aimodel":
            print("🌐 Mode set to 'aimodel'. Initiating AI model stream with posture and occupancy tracking.")
            threading.Thread(target=self.stream_posture_and_occupancy, daemon=True).start()
        else:
            threading.Thread(target=self.stream_video, daemon=True).start()

    def disconnect(self, close_code):
        self.streaming = False
        print(f"🛑 Camera {self.camera_id} stream stopped for user {self.user_id}.")

    def build_rtsp_url(self, cam):
        if "@" in cam.rtsp_url:
            return cam.rtsp_url
        if cam.username and cam.password:
            parts = cam.rtsp_url.split("://")
            if len(parts) == 2:
                protocol, rest = parts
                return f"{protocol}://{cam.username}:{cam.password}@{rest}"
        return cam.rtsp_url

    def send_fallback_frame(self):
        fallback_path = os.path.join(os.path.dirname(__file__), 'no_frame.jpg')
        fallback_img = cv2.imread(fallback_path)
        if fallback_img is not None:
            _, self.fallback_buffer = cv2.imencode('.jpg', fallback_img)
            self.send(bytes_data=self.fallback_buffer.tobytes())
            print("⚠️ Sending fallback frame (no frame available).")
        else:
            self.fallback_buffer = None
            self.send(bytes_data=b'')  # Send empty frame if no_frame.jpg is missing
            print("❌ no_frame.jpg missing, sending empty frame.")

    def _stream_camera(self, rtsp_url, process_frame_callback=None):
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print(f"❌ Failed to open RTSP stream. Retrying...")
            self.send_fallback_frame()
            return

        print("✅ RTSP stream opened successfully.")
        frame_counter = 0

        while self.streaming:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame not received. Skipping...")
                self.send_fallback_frame()
                continue
            frame_counter += 1

            # Print "Hello Rajesh" every 10 frames
            if frame_counter % 10 == 0:
                print("Hello Rajesh")

            # If a callback function is provided, process the frame
            if process_frame_callback:
                try:
                    process_frame_callback(frame)
                except Exception as e:
                    print(f"⚠️ Error during frame processing: {e}. Continuing with the next frame.")

            # Send the processed frame
            _, buffer = cv2.imencode('.jpg', frame)
            self.send(bytes_data=buffer.tobytes())

            time.sleep(0.01)  # Control frame rate

        cap.release()

    def stream_video(self):
        try:
            cam = Camera.objects.get(id=self.camera_id)
            print(f"🔗 Camera found: {cam.id} - {cam.rtsp_url}")
        except Camera.DoesNotExist:
            print(f"❌ Camera with ID {self.camera_id} does not exist.")
            self.close()
            return

        rtsp_url = self.build_rtsp_url(cam)

        self._stream_camera(rtsp_url)

    def stream_posture_and_occupancy(self):
        try:
            cam = Camera.objects.get(id=self.camera_id)
            print(f"🔗 Camera found: {cam.id} - {cam.rtsp_url}")
        except Camera.DoesNotExist:
            print(f"❌ Camera with ID {self.camera_id} does not exist.")
            self.close()
            return

        rtsp_url = self.build_rtsp_url(cam)

        # Define the frame processing callback for posture and occupancy
        def process_frame_callback(frame):
            execute_user_ai_models(
                user_id=self.user_id,
                camera_id=self.camera_id,
                frame=frame,
                rtsp_url=rtsp_url,
                save_to_json=False
            )

        self._stream_camera(rtsp_url, process_frame_callback)

import json
import threading
import time
from pathlib import Path
from datetime import timedelta, datetime
from channels.generic.websocket import WebsocketConsumer


class AnalyticsStreamConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        self.streaming = True
        print("📊 Analytics WebSocket connected.")
        threading.Thread(target=self.stream_live_analytics, daemon=True).start()

    def disconnect(self, close_code):
        self.streaming = False
        print("🛑 Analytics WebSocket disconnected.")

    def seconds_to_hm(self, seconds):
        td = timedelta(seconds=round(seconds))
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m"

    def hm_to_seconds(self, hm_string):
        try:
            parts = hm_string.lower().split("h")
            hours = int(parts[0].strip())
            minutes = int(parts[1].replace("m", "").strip())
            return hours * 3600 + minutes * 60
        except:
            return 0

    def build_person_from_seat(self, seat_name, seat_data, seat_id):
        dwell = seat_data.get("dwell", 0.0)
        empty_total = seat_data.get("empty_total", 0.0)
        system_time = dwell + empty_total

        # Apply the correct productivity formula
        productivity = round((dwell / system_time) * 100, 1) if system_time > 0 else 0.0
        alert = "Long away time" if empty_total >= 3600 else None

        return {
            "id": seat_id,
            "person": f"Person{str(seat_id).zfill(3)}",
            "status": "Active",
            "productivity": productivity,
            "sittingTime": self.seconds_to_hm(dwell),
            "standingTime": self.seconds_to_hm(0),
            "awayTime": self.seconds_to_hm(empty_total),
            "systemTime": round(system_time, 1),
            "productiveHours": self.seconds_to_hm(dwell),
            "totalHours": self.seconds_to_hm(system_time),
            "alerts": alert,
            "hasAlert": alert is not None
        }

    def get_today_analytics_file(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        return Path(f"analytics/{today_str}.json")

    def stream_live_analytics(self):
        analytics_file = self.get_today_analytics_file()
        if not analytics_file.exists():
            self.send(text_data=json.dumps({"error": f"{analytics_file.name} not found"}))
            self.close()
            return

        last_sent_timestamp = None

        while self.streaming:
            try:
                with open(analytics_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                self.send(text_data=json.dumps({"error": f"Error reading analytics file: {str(e)}"}))
                time.sleep(10)
                continue

            if not data:
                time.sleep(10)
                continue

            frame = data[-1]
            if frame.get("timestamp") == last_sent_timestamp:
                time.sleep(10)
                continue

            last_sent_timestamp = frame.get("timestamp")

            seat_stats = frame.get("stats", {})
            people = []

            total_productivity = 0.0
            total_productive_seconds = 0.0
            total_persons = 0
            active_alerts = 0

            for i, seat in enumerate(["seat_1", "seat_2", "seat_3", "seat_4", "seat_5"], start=1):
                if seat in seat_stats:
                    person = self.build_person_from_seat(seat, seat_stats[seat], seat_id=i)
                    people.append(person)
                    total_productivity += person["productivity"]
                    total_productive_seconds += self.hm_to_seconds(person["productiveHours"])
                    total_persons += 1
                    if person["hasAlert"]:
                        active_alerts += 1

            avg_productivity = round(total_productivity / total_persons, 1) if total_persons else 0.0
            total_productive_hours = round(total_productive_seconds / 3600, 1)

            response = {
                "saudi_time": frame.get("saudi_time"),
                "stats": people,
                "average_productivity": avg_productivity,
                "total_productive_hours": total_productive_hours,
                "active_alerts_count": active_alerts
            }

            self.send(text_data=json.dumps(response))
            time.sleep(10)
