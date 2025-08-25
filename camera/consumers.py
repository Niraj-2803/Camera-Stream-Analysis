import logging
import cv2
import threading
import time
import os
from channels.generic.websocket import WebsocketConsumer
import json
from pathlib import Path
from datetime import timedelta, datetime
from channels.generic.websocket import WebsocketConsumer
from django.conf import settings
from urllib.parse import parse_qs
from camera.aimodels.helper import *
from .models import Camera

# Configure logging to output to stdout, which is captured by Gunicorn
logging.basicConfig(level=logging.DEBUG)  # Change level to DEBUG, INFO, etc. based on your needs
logger = logging.getLogger(__name__)


import os, time, cv2
from queue import Queue, Full, Empty
from threading import Thread, Lock
from channels.generic.websocket import WebsocketConsumer

class CameraStreamConsumer(WebsocketConsumer):
    def connect(self):
        self.user_id = self.scope["url_route"]["kwargs"]["user_id"]
        self.camera_id = self.scope["url_route"]["kwargs"]["camera_id"]
        mode = self.scope["url_route"]["kwargs"].get("mode", "normal")

        self.accept()
        self.streaming = True
        print(f"📡 Camera stream started for user {self.user_id}, camera {self.camera_id}, mode {mode}.")

        # Shared AI state
        self.ai_queue = None            # frames → AI worker
        self.ai_result_lock = Lock()    # protects ai_last_vis
        self.ai_last_vis = None         # last annotated image from AI worker (np.ndarray)

        if mode == "aimodel":
            print("🌐 Mode set to 'aimodel'. Initiating AI model stream with posture and occupancy tracking.")
            self.ai_queue = Queue(maxsize=1)  # keep only the freshest frame to avoid lag
            self.ai_worker_thread = Thread(target=self._ai_worker_loop, daemon=True)
            self.ai_worker_thread.start()
            Thread(target=self.stream_video_with_ai, daemon=True).start()
        else:
            Thread(target=self.stream_video, daemon=True).start()

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
            self.send(bytes_data=b'')
            print("❌ no_frame.jpg missing, sending empty frame.")

    def _stream_camera(self, rtsp_url, process_frame_callback=None, overlay_callback=None):
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print(f"❌ Failed to open RTSP stream.")
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
            if frame_counter % 10 == 0:
                print("Hello Rajesh")

            # Non-blocking AI enqueue
            if process_frame_callback:
                try:
                    process_frame_callback(frame, frame_counter)
                except Exception as e:
                    print(f"⚠️ Error during frame processing enqueue: {e}")

            # Overlay latest AI result if any (non-blocking)
            frame_to_send = frame
            if overlay_callback:
                try:
                    vis = overlay_callback(frame)
                    if vis is not None:
                        frame_to_send = vis
                except Exception as e:
                    print(f"⚠️ Error during overlay: {e}")

            _, buffer = cv2.imencode('.jpg', frame_to_send)
            self.send(bytes_data=buffer.tobytes())

            time.sleep(0.005)  # small pacing; keeps latency low

        cap.release()

    # ---------- NORMAL mode ----------
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

    # ---------- AIMODEL mode (decoupled) ----------
    def stream_video_with_ai(self):
        try:
            cam = Camera.objects.get(id=self.camera_id)
            print(f"🔗 Camera found: {cam.id} - {cam.rtsp_url}")
        except Camera.DoesNotExist:
            print(f"❌ Camera with ID {self.camera_id} does not exist.")
            self.close()
            return

        rtsp_url = self.build_rtsp_url(cam)

        def process_frame_callback(frame, frame_counter):
            # Optionally resize for faster AI:
            # ai_frame = cv2.resize(frame, (640, 360))
            ai_frame = frame

            # Send every Nth frame to AI to reduce compute
            N = 3
            if frame_counter % N != 0 or self.ai_queue is None:
                return

            try:
                if self.ai_queue.full():
                    try:
                        _ = self.ai_queue.get_nowait()  # drop oldest
                    except Empty:
                        pass
                self.ai_queue.put_nowait(ai_frame)
            except Full:
                pass

        def overlay_callback(_frame):
            with self.ai_result_lock:
                return None if self.ai_last_vis is None else self.ai_last_vis

        self._stream_camera(rtsp_url, process_frame_callback, overlay_callback)

    # ---------- Single persistent AI worker ----------
    def _ai_worker_loop(self):
        while self.streaming:
            try:
                frame = self.ai_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                vis = execute_user_ai_models(
                    user_id=self.user_id,
                    camera_id=self.camera_id,
                    frame=frame,
                    rtsp_url=None,
                    save_to_json=False,
                )
                with self.ai_result_lock:
                    self.ai_last_vis = vis if vis is not None else None
            except Exception as e:
                print(f"⚠️ AI worker error: {e}")


class AnalyticsStreamConsumer(WebsocketConsumer):
    def connect(self):
        # Extract ?cam=analytics_2 from query string (default to analytics_1)
        query_string = self.scope.get("query_string", b"").decode()
        params = parse_qs(query_string)
        self.folder_name = params.get("cam", ["analytics_1"])[0]  # Default folder is analytics_1

        self.accept()
        self.streaming = True
        print(f"📊 WebSocket connected. Streaming from folder: {self.folder_name}")

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

        productivity = round((dwell / system_time) * 100, 1) if system_time > 0 else 0.0
        alert = "Long away time" if empty_total >= 3600 else None

        return {
            "id": seat_id,
            "person": seat_name,
            "status": "Active" if dwell > 0 else "Inactive",
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
        return Path(f"{self.folder_name}/{today_str}.json")

    def stream_live_analytics(self):
        last_sent_timestamp = None

        while self.streaming:
            analytics_file = self.get_today_analytics_file()

            if not analytics_file.exists():
                self.send(text_data=json.dumps({"error": f"{analytics_file.name} not found"}))
                time.sleep(10)
                continue

            try:
                with open(analytics_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                self.send(text_data=json.dumps({"error": f"Error reading file: {str(e)}"}))
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

            for idx, (seat_name, seat_data) in enumerate(seat_stats.items(), start=1):
                if seat_name == "overall":
                    continue
                person = self.build_person_from_seat(seat_name, seat_data, seat_id=idx)
                people.append(person)
                total_productivity += person["productivity"]
                total_productive_seconds += self.hm_to_seconds(person["productiveHours"])
                total_persons += 1
                if person["hasAlert"]:
                    active_alerts += 1

            avg_productivity = round(total_productivity / total_persons, 1) if total_persons else 0.0
            total_productive_hours = round(total_productive_seconds / 3600, 1)
            total_hours = round(sum(person["systemTime"] for person in people) / 3600, 1)

            response = {
                "saudi_time": frame.get("saudi_time"),
                "stats": people,
                "average_productivity": avg_productivity,
                "total_productive_hours": total_productive_hours,
                "total_hours": total_hours, 
                "active_alerts_count": active_alerts
            }


            self.send(text_data=json.dumps(response))
            time.sleep(10)





class LiveSeatStatsConsumer(WebsocketConsumer):
    def connect(self):
        query_string = self.scope.get("query_string", b"").decode()
        params = parse_qs(query_string)

        self.user_id = params.get("user_id", [None])[0]
        self.camera_id = params.get("camera_id", [None])[0]
        self.date_str = params.get("date", [datetime.now().strftime("%Y-%m-%d")])[0]
        self.mode = params.get("mode", ["seat"])[0]  # default seat mode

        if not self.user_id or not self.camera_id:
            self.close()
            return

        self.accept()
        self.streaming = True
        print(
            f"📡 WebSocket connected: user={self.user_id}, "
            f"cam={self.camera_id}, date={self.date_str}, mode={self.mode}"
        )
        threading.Thread(target=self.stream_stats, daemon=True).start()

    def disconnect(self, close_code):
        self.streaming = False
        print("🔌 WebSocket disconnected")

    # -------------------------------
    # Helpers
    # -------------------------------
    # def get_file_path(self):
    #     """Pick correct file depending on mode"""
    #     if self.mode == "inout":
    #         filename = f"inout_user{self.user_id}_cam{self.camera_id}_{self.date_str}.json"
    #     else:  # seat_status default
    #         filename = f"seat_user{self.user_id}_cam{self.camera_id}_{self.date_str}.json"
    #     return Path(settings.MEDIA_ROOT) / "seat_stats" / filename

    def get_file_path(self):
        """Pick correct file depending on mode"""
        if self.mode == "inout":
            filename = f"inout_user{self.user_id}_cam{self.camera_id}_{self.date_str}.json"
            return Path(settings.MEDIA_ROOT) / "in_out_stats" / filename
        else:  # seat_status default
            filename = f"seat_user{self.user_id}_cam{self.camera_id}_{self.date_str}.json"
            return Path(settings.MEDIA_ROOT) / "seat_stats" / filename


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
        except Exception:
            return 0

    def build_person_from_seat(self, seat_name, seat_data, seat_id):
        dwell = seat_data.get("dwell", 0.0)
        empty_total = seat_data.get("empty_total", 0.0)
        system_time = dwell + empty_total

        productivity = round((dwell / system_time) * 100, 1) if system_time > 0 else 0.0
        alert = "Long away time" if empty_total >= 3600 else None

        return {
            "id": seat_id,
            "person": seat_name,
            "status": "Active" if dwell > 0 else "Inactive",
            "productivity": productivity,
            "sittingTime": self.seconds_to_hm(dwell),
            "standingTime": self.seconds_to_hm(0),
            "awayTime": self.seconds_to_hm(empty_total),
            "systemTime": round(system_time, 1),
            "productiveHours": self.seconds_to_hm(dwell),
            "totalHours": self.seconds_to_hm(system_time),
            "alerts": alert,
            "hasAlert": alert is not None,
        }

    def build_seat_response(self, seat_stats):
        people = []
        total_productivity = 0.0
        total_productive_seconds = 0.0
        total_system_seconds = 0.0
        total_persons = 0
        active_alerts = 0

        for idx, (seat_name, seat_data) in enumerate(seat_stats.items(), start=1):
            person = self.build_person_from_seat(seat_name, seat_data, seat_id=idx)
            people.append(person)
            total_productivity += person["productivity"]
            total_productive_seconds += self.hm_to_seconds(person["productiveHours"])
            total_system_seconds += person["systemTime"]
            total_persons += 1
            if person["hasAlert"]:
                active_alerts += 1

        avg_productivity = (
            round(total_productivity / total_persons, 1) if total_persons else 0.0
        )
        total_productive_hours = round(total_productive_seconds / 3600, 1)
        total_hours = round(total_system_seconds / 3600, 1)

        return {
            "saudi_time": datetime.now().isoformat(),
            "stats": people,
            "average_productivity": avg_productivity,
            "total_productive_hours": total_productive_hours,
            "total_hours": total_hours,
            "active_alerts_count": active_alerts,
        }

    # -------------------------------
    # Streaming Loop
    # -------------------------------
    def stream_stats(self):
        last_sent_timestamp = None

        while self.streaming:
            filepath = self.get_file_path()

            if not filepath.exists():
                self.send(
                    text_data=json.dumps({"error": f"{filepath.name} not found"})
                )
                time.sleep(10)
                continue

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except Exception as e:
                self.send(
                    text_data=json.dumps({"error": f"Error reading file: {str(e)}"})
                )
                time.sleep(10)
                continue

            if not data or not isinstance(data, list):
                time.sleep(10)
                continue

            latest = data[-1]
            if latest.get("timestamp") == last_sent_timestamp:
                time.sleep(10)
                continue
            last_sent_timestamp = latest["timestamp"]

            # 🔹 Mode specific processing
            if self.mode == "seat":
                seat_stats = latest.get("stats", {})
                response = self.build_seat_response(seat_stats)
            else:  # inout
                stats = latest.get("stats", {})
                in_count = stats.get("in_count", 0)
                out_count = stats.get("out_count", 0)
                response = {
                    "saudi_time": datetime.now().isoformat(),
                    "in_count": in_count,
                    "out_count": out_count,
                    "total": in_count + out_count,
                }

            self.send(text_data=json.dumps(response))
            time.sleep(5)
