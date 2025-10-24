import logging
import time
import json
import cv2
from channels.generic.websocket import WebsocketConsumer
from django.utils import timezone
from .models import Camera, InOutStats, SeatStatusStats
from threading import Thread
from camera.aimodels.helper import *
from queue import Queue, Full, Empty

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AiWorkerConsumer(WebsocketConsumer):
    def connect(self):
        self.user_id = self.scope["url_route"]["kwargs"]["user_id"]
        self.camera_id = self.scope["url_route"]["kwargs"]["camera_id"]

        self.accept()
        self.streaming = True
        print(f"🤖 AI Worker started for user {self.user_id}, camera {self.camera_id}")

        # Queue to send frames to AI
        self.ai_queue = Queue(maxsize=1)

        # Initialize threads
        Thread(target=self._ai_worker_loop, daemon=True).start()
        Thread(target=self._stream_video, daemon=True).start()

    def disconnect(self, close_code):
        self.streaming = False
        print(f"🛑 AI Worker stopped for user {self.user_id}, camera {self.camera_id}")

    def build_rtsp_url(self, cam):
        if "@" in cam.rtsp_url:
            return cam.rtsp_url
        if cam.username and cam.password:
            parts = cam.rtsp_url.split("://")
            if len(parts) == 2:
                protocol, rest = parts
                return f"{protocol}://{cam.username}:{cam.password}@{rest}"
        return cam.rtsp_url

    def _stream_video(self):
        try:
            cam = Camera.objects.get(id=self.camera_id)
        except Camera.DoesNotExist:
            print(f"❌ Camera {self.camera_id} does not exist")
            self.close()
            return

        rtsp_url = self.build_rtsp_url(cam)
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"❌ Failed to open RTSP stream")
            return

        frame_counter = 0
        while self.streaming:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            frame_counter += 1
            # Send every Nth frame to AI to reduce load
            N = 5
            if frame_counter % N == 0:
                try:
                    if self.ai_queue.full():
                        _ = self.ai_queue.get_nowait()
                    self.ai_queue.put_nowait(frame)
                except Full:
                    pass

            time.sleep(0.005)
        cap.release()

    def _ai_worker_loop(self):
        last_snapshot = None

        while self.streaming:
            try:
                frame = self.ai_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                # Run active models
                execute_user_ai_models(
                    user_id=self.user_id,
                    camera_id=self.camera_id,
                    frame=frame,
                    rtsp_url=None,
                    save_to_db=True  # store results
                )

                today = timezone.now().date()

                # --- InOut ---
                stats_obj = InOutStats.objects.filter(
                    user_id=self.user_id,
                    camera_id=self.camera_id,
                    date=today
                ).first()

                in_out_data = {
                    "total_in": stats_obj.total_in if stats_obj else 0,
                    "total_out": stats_obj.total_out if stats_obj else 0
                }

                # --- Seat Status ---
                seat_objs = SeatStatusStats.objects.filter(
                    user_id=self.user_id,
                    camera_id=self.camera_id,
                    date=today
                )

                seat_data = []
                total_dwell_time = 0.0
                total_productivity = 0.0
                total_alerts = 0

                for s in seat_objs:
                    total_working_hours = s.dwell_time_total + s.empty_total
                    productivity = round((s.dwell_time_total / total_working_hours) * 100, 1) if total_working_hours > 0 else 0
                    alert = "Long away time" if s.empty >= 600 else None
                    if alert:
                        total_alerts += 1

                    total_dwell_time += float(s.dwell_time_total)
                    total_productivity += productivity

                    seat_data.append({
                        "seat_name": s.seat_name,
                        "occupied": s.is_occupied,
                        "dwell_time": float(s.dwell_time),
                        "dwell_time_total": float(s.dwell_time_total),
                        "empty": float(s.empty),
                        "empty_total": float(s.empty_total),
                        "productivity": productivity,
                        "alert": alert,
                        "has_alert": alert is not None
                    })

                # --- Compute summary ---
                total_persons = len(seat_objs)
                avg_productivity = round(total_productivity / total_persons, 1) if total_persons > 0 else 0
                total_productive_hours = total_dwell_time

                # --- Common response ---
                snapshot = {
                    "type": "ai_snapshot",
                    "date": str(today),
                    "in_out": in_out_data,
                    "seat_status": seat_data,
                    "summary": {
                        "total_persons": total_persons,
                        "total_productive_hours": total_productive_hours,
                        "avg_productivity": avg_productivity,
                        "total_alerts": total_alerts
                    }
                }


                # Only send if new
                if snapshot != last_snapshot:
                    self.send(text_data=json.dumps(snapshot))
                    last_snapshot = snapshot

            except Exception as e:
                print(f"⚠️ AI worker error: {e}")

