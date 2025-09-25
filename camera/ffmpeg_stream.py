# camera/ffmpeg_stream.py
import os
import subprocess
import logging
from django.conf import settings
from pathlib import Path
from camera.ai_worker import start_ai_worker

logger = logging.getLogger(__name__)

def start_ffmpeg_stream(rtsp_url, user_id, camera_id):
    try:
        logger.warning(f"⚡ Entered start_ffmpeg_stream for cam {camera_id}")
        media_root = getattr(settings, "MEDIA_ROOT", Path("./media"))
        out_dir = Path(media_root) / "hls" / f"camera_{camera_id}"
        os.makedirs(out_dir, exist_ok=True)

        hls_path = out_dir / "index.m3u8"

        ffmpeg_cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-an",
            "-f", "hls",
            "-hls_time", "2",
            "-hls_list_size", "10",
            "-hls_flags", "delete_segments+append_list+program_date_time",
            "-hls_allow_cache", "0",
            str(hls_path),
        ]

        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.warning(f"🚀 FFmpeg started for camera {camera_id}, PID={proc.pid}, writing {hls_path}")

        # 🔹 Start AI Worker
        logger.warning(f"📡 About to call start_ai_worker for cam {camera_id}")
        start_ai_worker(rtsp_url, user_id, camera_id)
        logger.warning(f"📡 Returned from start_ai_worker call for cam {camera_id}")

    except Exception as e:
        logger.error(f"❌ start_ffmpeg_stream crashed for cam {camera_id}: {e}", exc_info=True)
