from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from .models import Camera
from .serializers import CameraSerializer
from django.http import StreamingHttpResponse
import cv2

class CameraListCreateView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        cameras = Camera.objects.filter(is_deleted=False)
        serializer = CameraSerializer(cameras, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = CameraSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

from django.shortcuts import get_object_or_404, render
from django.http import StreamingHttpResponse, HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from .models import Camera
import cv2

# MJPEG stream generator
def generate_frames(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 🔴 Stream endpoint (returns MJPEG frames)
class CameraStreamFeedView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        camera = get_object_or_404(Camera, pk=pk, is_deleted=False)
        return StreamingHttpResponse(
            generate_frames(camera.rtsp_url),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )

# 🔵 HTML viewer that embeds the stream
class CameraStreamViewer(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        return HttpResponse(f"""
            <html>
            <head><title>Live Stream</title></head>
            <body>
                <h2>Camera Stream - ID {pk}</h2>
                <img src="/api/camera/{pk}/stream/feed/" width="640" height="480" />
            </body>
            </html>
        """)
