import cv2
import time
import threading
import logging
from camera.aimodels.helper import execute_user_ai_models

logger = logging.getLogger(__name__)

def start_ai_worker(rtsp_url, user_id, camera_id):
    print(f"Entered start_ai_worker(user={user_id}, cam={camera_id})")

    def worker():
        print(f"AI worker starting for user={user_id}, camera={camera_id}")
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logger.error(f"Failed to open RTSP for AI worker (camera {camera_id})")
            return

        logger.info(f"AI Worker capturing frames for cam {camera_id}")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"⚠ Lost frame for camera {camera_id}, retrying in 1s...")
                time.sleep(1)
                continue

            try:
                execute_user_ai_models(user_id, camera_id, frame, rtsp_url=rtsp_url, save_to_db=True)
                print(f"✅ Frame sent to AI pipeline for cam {camera_id}")
            except Exception as e:
                logger.error(f"AI processing failed for camera {camera_id}: {e}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    logger.info(f"🚀 AI worker thread launched for camera {camera_id}")