import logging
import cv2
import threading
import time
import os
import json
from queue import Queue, Full, Empty
from threading import Thread, Lock
from pathlib import Path
from datetime import datetime
from channels.generic.websocket import WebsocketConsumer
from django.conf import settings
from urllib.parse import parse_qs
from .models import Camera, InOutCount, User
from camera.aimodels.helper import execute_user_ai_models
from django.db.models import Sum, Q
from channels.db import database_sync_to_async

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================
# Camera Streaming
# ==============================
class CameraStreamConsumer(WebsocketConsumer):
    def connect(self):
        self.user_id = self.scope["url_route"]["kwargs"]["user_id"]
        self.camera_id = self.scope["url_route"]["kwargs"]["camera_id"]
        mode = self.scope["url_route"]["kwargs"].get("mode", "normal")

        self.accept()
        self.streaming = True
        logger.info(f"📡 Camera stream started for user={self.user_id}, camera={self.camera_id}, mode={mode}")

        # AI support
        self.ai_queue = None
        self.ai_result_lock = Lock()
        self.ai_last_vis = None

        if mode == "aimodel":
            self.ai_queue = Queue(maxsize=1)
            self.ai_worker_thread = Thread(target=self._ai_worker_loop, daemon=True)
            self.ai_worker_thread.start()
            Thread(target=self.stream_video_with_ai, daemon=True).start()
        else:
            Thread(target=self.stream_video, daemon=True).start()

    def disconnect(self, close_code):
        self.streaming = False
        logger.info(f"🛑 Camera {self.camera_id} stream stopped for user {self.user_id}")

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
        fallback_path = os.path.join(getattr(settings, "IMAGE_FILES", ""), "no_frame.jpg")
        fallback_img = cv2.imread(fallback_path)
        if fallback_img is not None:
            _, buffer = cv2.imencode('.jpg', fallback_img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            self.send(bytes_data=buffer.tobytes())
        else:
            self.send(bytes_data=b'')

    def _stream_camera(self, rtsp_url, process_frame_callback=None, overlay_callback=None):
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            logger.warning("❌ Failed to open RTSP stream.")
            self.send_fallback_frame()
            return

        logger.info("✅ RTSP stream opened successfully.")
        frame_counter = 0

        while self.streaming:
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ Frame not received. Skipping...")
                self.send_fallback_frame()
                continue

            frame_counter += 1

            if process_frame_callback:
                try:
                    process_frame_callback(frame, frame_counter)
                except Exception as e:
                    logger.warning(f"⚠️ Error enqueueing frame for AI: {e}")

            frame_to_send = frame
            if overlay_callback:
                try:
                    vis = overlay_callback(frame)
                    if vis is not None:
                        frame_to_send = vis
                except Exception as e:
                    logger.warning(f"⚠️ Error applying AI overlay: {e}")

            _, buffer = cv2.imencode('.jpg', frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            self.send(bytes_data=buffer.tobytes())
            time.sleep(0.005)

        cap.release()

    def stream_video(self):
        try:
            cam = Camera.objects.get(id=self.camera_id)
        except Camera.DoesNotExist:
            logger.error(f"❌ Camera with ID {self.camera_id} does not exist.")
            self.close()
            return

        rtsp_url = self.build_rtsp_url(cam)
        self._stream_camera(rtsp_url)

    def stream_video_with_ai(self):
        try:
            cam = Camera.objects.get(id=self.camera_id)
        except Camera.DoesNotExist:
            logger.error(f"❌ Camera with ID {self.camera_id} does not exist.")
            self.close()
            return

        rtsp_url = self.build_rtsp_url(cam)

        def process_frame_callback(frame, frame_counter):
            if frame_counter % 5 != 0 or self.ai_queue is None:
                return
            try:
                if self.ai_queue.full():
                    _ = self.ai_queue.get_nowait()
                self.ai_queue.put_nowait(frame)
            except Full:
                pass

        def overlay_callback(_frame):
            with self.ai_result_lock:
                return self.ai_last_vis

        self._stream_camera(rtsp_url, process_frame_callback, overlay_callback)

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
                    self.ai_last_vis = vis
            except Exception as e:
                logger.error(f"⚠️ AI worker error: {e}")


# ==============================
# Live In/Out Stats
# ==============================
class LiveSeatStatsConsumer(WebsocketConsumer):
    def connect(self):
        params = parse_qs(self.scope.get("query_string", b"").decode())
        self.user_id = params.get("user_id", [None])[0]
        self.camera_id = params.get("camera_id", [None])[0]
        self.date_str = params.get("date", [datetime.now().strftime("%Y-%m-%d")])[0]
        self.mode = params.get("mode", ["inout"])[0]  # force to inout

        if not self.user_id or not self.camera_id:
            self.close()
            return

        self.accept()
        self.streaming = True
        logger.info(f"📡 InOut WebSocket connected: user={self.user_id}, cam={self.camera_id}, date={self.date_str}")
        threading.Thread(target=self.stream_stats, daemon=True).start()

    def disconnect(self, close_code):
        self.streaming = False
        logger.info("🔌 InOut WebSocket disconnected")

    def get_file_path(self):
        filename = f"inout_user{self.user_id}_cam{self.camera_id}_{self.date_str}.json"
        return Path(settings.MEDIA_ROOT) / "in_out_stats" / filename

    def stream_stats(self):
        last_sent_timestamp = None
        while self.streaming:
            filepath = self.get_file_path()
            if not filepath.exists():
                self.send(text_data=json.dumps({"error": f"{filepath.name} not found"}))
                time.sleep(5)
                continue

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ JSON incomplete for {filepath}, retrying: {e}")
                time.sleep(1)
                continue
            except Exception as e:
                self.send(text_data=json.dumps({"error": f"Error reading file: {str(e)}"}))
                time.sleep(5)
                continue

            if not data:
                time.sleep(5)
                continue

            latest = data[-1]
            if latest.get("timestamp") == last_sent_timestamp:
                time.sleep(5)
                continue
            last_sent_timestamp = latest["timestamp"]

            stats = latest.get("stats", {})
            in_count = stats.get("in", 0)
            out_count = stats.get("out", 0)

            response = {
                "saudi_time": datetime.now().isoformat(),
                "in_count": in_count,
                "out_count": out_count,
                "total": in_count + out_count,
            }

            self.send(text_data=json.dumps(response))
            time.sleep(5)

class LiveDatabaseStatsConsumer(WebsocketConsumer):
    """
    WebSocket consumer that reads ONLY from database
    """
    def connect(self):
        params = parse_qs(self.scope.get("query_string", b"").decode())
        self.user_id = params.get("user_id", [None])[0]
        self.camera_id = params.get("camera_id", [None])[0]
        self.date_str = params.get("date", [datetime.now().strftime("%Y-%m-%d")])[0]

        if not self.user_id:
            self.close(code=4000, reason="user_id is required")
            return

        # Validate user exists
        try:
            User.objects.get(id=self.user_id)
        except User.DoesNotExist:
            self.close(code=4001, reason="Invalid user_id")
            return

        # If camera_id provided, validate it exists
        if self.camera_id:
            try:
                Camera.objects.get(id=self.camera_id)
            except Camera.DoesNotExist:
                self.close(code=4002, reason="Invalid camera_id")
                return

        self.accept()
        self.streaming = True
        logger.info(f"🗄️ Database WebSocket connected: user={self.user_id}, cam={self.camera_id}, date={self.date_str}")
        threading.Thread(target=self.stream_db_stats, daemon=True).start()

    def disconnect(self, close_code):
        self.streaming = False
        logger.info("🔌 Database WebSocket disconnected")

    def get_db_stats(self):
        """Get stats from database only"""
        try:
            target_date = datetime.strptime(self.date_str, "%Y-%m-%d").date()
            
            if self.camera_id:
                # Single camera stats
                try:
                    inout_record = InOutCount.objects.get(
                        user_id=self.user_id,
                        camera_id=self.camera_id,
                        date=target_date
                    )
                    return {
                        "camera_id": self.camera_id,
                        "camera_name": inout_record.camera.name,
                        "date": target_date.isoformat(),
                        "in_count": inout_record.in_count,
                        "out_count": inout_record.out_count,
                        "total": inout_record.total_count,
                        "last_updated": inout_record.last_updated.isoformat(),
                        "source": "database"
                    }
                except InOutCount.DoesNotExist:
                    return {
                        "camera_id": self.camera_id,
                        "date": target_date.isoformat(),
                        "in_count": 0,
                        "out_count": 0,
                        "total": 0,
                        "source": "database",
                        "message": "No data found for this date"
                    }
            else:
                # All cameras stats for user
                records = InOutCount.objects.filter(
                    user_id=self.user_id,
                    date=target_date
                ).select_related('camera')
                
                cameras_data = []
                total_in = 0
                total_out = 0
                
                for record in records:
                    camera_data = {
                        "camera_id": record.camera.id,
                        "camera_name": record.camera.name,
                        "in_count": record.in_count,
                        "out_count": record.out_count,
                        "total": record.total_count,
                        "last_updated": record.last_updated.isoformat()
                    }
                    cameras_data.append(camera_data)
                    total_in += record.in_count
                    total_out += record.out_count
                
                return {
                    "date": target_date.isoformat(),
                    "total_in_count": total_in,
                    "total_out_count": total_out,
                    "total_count": total_in + total_out,
                    "cameras": cameras_data,
                    "cameras_count": len(cameras_data),
                    "source": "database"
                }
                
        except Exception as e:
            logger.error(f"❌ Error fetching DB stats: {e}")
            return {"error": f"Database error: {str(e)}"}

    def stream_db_stats(self):
        last_sent_data = None
        
        while self.streaming:
            current_data = self.get_db_stats()

            # Skip sending if data hasn't changed
            if current_data == last_sent_data:
                time.sleep(5)
                continue

            last_sent_data = current_data.copy()
            
            # Add real-time timestamp
            current_data["saudi_time"] = datetime.now().isoformat()
            
            self.send(text_data=json.dumps(current_data))
            time.sleep(5)


# # ==============================
# # SECURITY CLEANING
# # ==============================
# # - Removed AnalyticsStreamConsumer (seat/productivity stats)
# # - Removed seat-related mode inside LiveSeatStatsConsumer
# # - All productivity/dwell/empty tracking stripped

# import logging
# import cv2
# import threading
# import time
# import os
# import json
# import subprocess
# from queue import Queue, Full, Empty
# from threading import Thread, Lock
# from pathlib import Path
# from datetime import datetime
# from django.http import StreamingHttpResponse, HttpResponse
# from django.conf import settings
# from django.views.decorators import gzip
# from urllib.parse import parse_qs
# from .models import Camera
# from camera.aimodels.helper import execute_user_ai_models

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ==============================
# # Camera Streaming with FFmpeg
# # ==============================

# class CameraStreamHandler:
#     def __init__(self, user_id, camera_id, mode="normal"):
#         self.user_id = user_id
#         self.camera_id = camera_id
#         self.mode = mode
#         self.streaming = True
#         self.ai_queue = None
#         self.ai_result_lock = Lock()
#         self.ai_last_vis = None
        
#         # Initialize AI processing if needed
#         if mode == "aimodel":
#             self.ai_queue = Queue(maxsize=1)
#             self.ai_worker_thread = Thread(target=self._ai_worker_loop, daemon=True)
#             self.ai_worker_thread.start()

#     def build_rtsp_url(self, cam):
#         if "@" in cam.rtsp_url:
#             return cam.rtsp_url
#         if cam.username and cam.password:
#             parts = cam.rtsp_url.split("://")
#             if len(parts) == 2:
#                 protocol, rest = parts
#                 return f"{protocol}://{cam.username}:{cam.password}@{rest}"
#         return cam.rtsp_url

#     def get_camera_info(self):
#         try:
#             cam = Camera.objects.get(id=self.camera_id)
#             return cam, self.build_rtsp_url(cam)
#         except Camera.DoesNotExist:
#             logger.error(f"❌ Camera with ID {self.camera_id} does not exist.")
#             return None, None

#     def generate_frames(self):
#         cam, rtsp_url = self.get_camera_info()
#         if not cam or not rtsp_url:
#             # Send fallback frame
#             fallback_path = os.path.join(getattr(settings, "IMAGE_FILES", ""), "no_frame.jpg")
#             if os.path.exists(fallback_path):
#                 with open(fallback_path, 'rb') as f:
#                     while self.streaming:
#                         yield (b'--frame\r\n'
#                                b'Content-Type: image/jpeg\r\n\r\n' + f.read() + b'\r\n')
#                         time.sleep(1)
#             return

#         # Use FFmpeg to process the RTSP stream
#         ffmpeg_cmd = [
#             'ffmpeg',
#             '-i', rtsp_url,
#             '-r', '15',  # Reduce frame rate to 15 FPS
#             '-s', '640x360',  
#             '-f', 'mjpeg',
#             '-qscale', '5',  
#             '-'
#         ]

#         process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
#         frame_counter = 0
#         frame_buffer = b''
        
#         while self.streaming and process.poll() is None:
#             # Read FFmpeg output
#             data = process.stdout.read(1024)
#             if not data:
#                 continue
                
#             frame_buffer += data
#             # Look for JPEG frame boundaries
#             start = frame_buffer.find(b'\xff\xd8')
#             end = frame_buffer.find(b'\xff\xd9')
            
#             if start != -1 and end != -1:
#                 jpeg_data = frame_buffer[start:end+2]
#                 frame_buffer = frame_buffer[end+2:]
                
#                 # Convert to numpy array for AI processing if needed
#                 if self.mode == "aimodel" and self.ai_queue is not None:
#                     nparr = np.frombuffer(jpeg_data, np.uint8)
#                     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
#                     # Process frame with AI (every 5th frame)
#                     frame_counter += 1
#                     if frame_counter % 5 == 0:
#                         try:
#                             if self.ai_queue.full():
#                                 _ = self.ai_queue.get_nowait()
#                             self.ai_queue.put_nowait(frame)
#                         except Full:
#                             pass
                    
#                     # Apply AI overlay if available
#                     with self.ai_result_lock:
#                         if self.ai_last_vis is not None:
#                             # Re-encode the frame with AI overlay
#                             _, buffer = cv2.imencode('.jpg', self.ai_last_vis, 
#                                                    [int(cv2.IMWRITE_JPEG_QUALITY), 40])
#                             jpeg_data = buffer.tobytes()
                
#                 # Yield the frame in M-JPEG format
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg_data + b'\r\n')
        
#         process.terminate()
#         try:
#             process.wait(timeout=5)
#         except subprocess.TimeoutExpired:
#             process.kill()

#     def _ai_worker_loop(self):
#         while self.streaming:
#             try:
#                 frame = self.ai_queue.get(timeout=0.5)
#             except Empty:
#                 continue

#             try:
#                 vis = execute_user_ai_models(
#                     user_id=self.user_id,
#                     camera_id=self.camera_id,
#                     frame=frame,
#                     rtsp_url=None,
#                     save_to_json=False,
#                 )
#                 with self.ai_result_lock:
#                     self.ai_last_vis = vis
#             except Exception as e:
#                 logger.error(f"⚠️ AI worker error: {e}")

#     def stop(self):
#         self.streaming = False


# # Django view to handle video streaming
# @gzip.gzip_page
# def video_feed(request, user_id, camera_id):
#     mode = request.GET.get('mode', 'normal')
    
#     def generate():
#         handler = CameraStreamHandler(user_id, camera_id, mode)
#         try:
#             for frame in handler.generate_frames():
#                 yield frame
#         finally:
#             handler.stop()
    
#     return StreamingHttpResponse(generate(), 
#                                 content_type='multipart/x-mixed-replace; boundary=frame')


# # ==============================
# # Live In/Out Stats (HTTP Endpoint)
# # ==============================

# def live_seat_stats(request):
#     params = request.GET
#     user_id = params.get("user_id")
#     camera_id = params.get("camera_id")
#     date_str = params.get("date", datetime.now().strftime("%Y-%m-%d"))
    
#     if not user_id or not camera_id:
#         return HttpResponse(json.dumps({"error": "Missing user_id or camera_id"}), 
#                           status=400, content_type="application/json")
    
#     filename = f"inout_user{user_id}_cam{camera_id}_{date_str}.json"
#     filepath = Path(settings.MEDIA_ROOT) / "in_out_stats" / filename
    
#     if not filepath.exists():
#         return HttpResponse(json.dumps({"error": f"{filepath.name} not found"}), 
#                           status=404, content_type="application/json")
    
#     try:
#         with open(filepath, "r") as f:
#             data = json.load(f)
#     except Exception as e:
#         return HttpResponse(json.dumps({"error": f"Error reading file: {str(e)}"}), 
#                           status=500, content_type="application/json")
    
#     if not data:
#         return HttpResponse(json.dumps({"error": "No data available"}), 
#                           status=404, content_type="application/json")
    
#     latest = data[-1]
#     stats = latest.get("stats", {})
#     in_count = stats.get("in_count", 0)
#     out_count = stats.get("out_count", 0)
    
#     response = {
#         "saudi_time": datetime.now().isoformat(),
#         "in_count": in_count,
#         "out_count": out_count,
#         "total": in_count + out_count,
#     }
    
#     return HttpResponse(json.dumps(response), content_type="application/json")
