from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from .models import Camera, CameraGroup, UserAiModel, AiModel
from .serializers import AiModelSerializer, CameraGroupActionSerializer, CameraSerializer, UserAiModelSerializer, UserAiModelActionSerializer
from django.http import StreamingHttpResponse, HttpResponse
from django.shortcuts import get_object_or_404
import cv2
from .aimodels.helper import blur_faces, process_video_stream
from drf_yasg.utils import swagger_auto_schema
from rest_framework.generics import GenericAPIView
from rest_framework.mixins import ListModelMixin
from rest_framework.pagination import PageNumberPagination
import requests
from django.shortcuts import render



# @login_required
def camera_dashboard(request):
    token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ1NjQ3ODg3LCJpYXQiOjE3NDU1NjE0ODcsImp0aSI6Ijg0MDQxZjU0NzFmYTQ1M2VhOGM5NTI2YmFhNmNiNDUwIiwidXNlcl9pZCI6MX0.QfbEs1A9lSaT09wGDNveacbmBMvvLcTh0z3LjIDO9jo"
    headers = {"Authorization": f"Bearer {token}"}
    api_url = "http://127.0.0.1:8000/api/camera/"
    response = requests.get(api_url, headers=headers)

    cameras = response.json().get("results", [])  # paginated DRF response
    return render(request, "camera/camera_dashboard.html", {"cameras": cameras})


class CameraPagination(PageNumberPagination):
    page_size = 9

class CameraListCreateView(GenericAPIView, ListModelMixin):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = CameraSerializer
    pagination_class = CameraPagination
    queryset = Camera.objects.filter(is_deleted=False)

    @swagger_auto_schema(responses={200: CameraSerializer(many=True)})
    def get(self, request):
        return self.list(request)

    @swagger_auto_schema(request_body=CameraSerializer, responses={201: CameraSerializer})
    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            serializer.save(created_by=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# MJPEG stream generator
import cv2
import time

def generate_frames(rtsp_url, reconnect_delay=5):
    print(f"[INFO] Starting stream from: {rtsp_url}")
    
    while True:
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            print("[WARN] Failed to open stream. Retrying in 5 seconds...")
            time.sleep(reconnect_delay)
            continue
        
        print("[INFO] Stream opened successfully.")
        
        while True:
            success, frame = cap.read()

            if not success:
                print("[ERROR] Failed to read frame. Attempting to reconnect...")
                cap.release()
                time.sleep(reconnect_delay)
                break  # Exit inner loop to retry connection

            # Optional: Resize or process frame here if needed
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )


class CameraStreamFeedView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        camera = get_object_or_404(Camera, pk=pk, is_deleted=False)
        return StreamingHttpResponse(
            generate_frames(camera.rtsp_url),
            content_type="multipart/x-mixed-replace; boundary=frame",
        )


# class AiModelListView(APIView):
#     permission_classes = [permissions.IsAuthenticated]  
#     @swagger_auto_schema(responses={200: "Video played locally with face blur."})
#     def get(self, request):
#         stream_url = "rtsp://admin:admin@192.168.1.4:1935"
#         output_file = r"output_video.mp4"
#         model_path = "yolo11n.pt"
#         # blur_faces(stream_url)
#         result = process_video_stream(
#         source=stream_url,
#         output_path=output_file,
#         model_path=model_path,
#         usecase="blur_faces"
#     )
#         return StreamingHttpResponse(
#             result,
#             content_type="multipart/x-mixed-replace; boundary=frame",
#         )

#         return Response({"message": "Video played locally with face blur."}, status=status.HTTP_200_OK)
#         # Response({"message": "Stream processed (or attempted to)"}, status=status.HTTP_200_OK)

#         # models = AiModel.objects.all()
#         # serializer = AiModelSerializer(models, many=True)
#         # return Response(serializer.data, status=status.HTTP_200_OK)
    

class ActivateAiModelView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    @swagger_auto_schema(request_body=UserAiModelActionSerializer, responses={200: UserAiModelSerializer})
    def post(self, request):
        serializer = UserAiModelActionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user = request.user
        aimodel_id = serializer.validated_data['aimodel_id']
        is_active = serializer.validated_data['is_active']
        aimodel = AiModel.objects.get(id=aimodel_id)

        # Create or update the UserAiModel instance
        obj, created = UserAiModel.objects.update_or_create(
            user=user,
            aimodel=aimodel,
            defaults={'is_active': is_active}
        )

        return Response(UserAiModelSerializer(obj).data, status=status.HTTP_200_OK)


class AddCameraToGroupAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(
        request_body=CameraGroupActionSerializer,
        operation_summary="Add multiple cameras to a group by group name",
        responses={200: "Cameras added to group successfully"}
    )
    def post(self, request):
        serializer = CameraGroupActionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        group_name = serializer.validated_data['group_name']
        camera_ids = serializer.validated_data['camera_ids']

        group, created = CameraGroup.objects.get_or_create(
                    name=group_name,
                    created_by=request.user
                )
        updated = 0
        for cam_id in camera_ids:
            try:
                camera = Camera.objects.get(id=cam_id, created_by=request.user)
                camera.group = group
                camera.save()
                updated += 1
            except Camera.DoesNotExist:
                continue  # skip invalid camera ids

        return Response(
            {"message": f"{updated} camera(s) added to group '{group_name}'."},
            status=status.HTTP_200_OK
        )

class ListCamerasInGroupAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="List cameras in a specific group",
        responses={200: "List of cameras in the group"}
    )
    def get(self, request, group_id):
        group = get_object_or_404(CameraGroup, id=group_id, created_by=request.user)
        cameras = group.cameras.all()
        serializer = CameraSerializer(cameras, many=True)

        return Response({
            "group": group.name,
            "cameras": serializer.data
        }, status=status.HTTP_200_OK)


class CreateAiModelView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(
        request_body=AiModelSerializer,
        responses={201: AiModelSerializer, 400: "Bad Request"},
        operation_summary="Create AI Model",
        operation_description="Creates a new AI model with a name and function_name."
    )
    def post(self, request):
        serializer = AiModelSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)