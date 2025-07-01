from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from .models import Camera, CameraGroup, UserAiModel, AiModel
from .serializers import AiModelSerializer, AssignAiModelSerializer, CameraGroupActionSerializer, CameraSerializer, UserAiModelSerializer, UserAiModelActionSerializer, UserCameraQuerySerializer
from django.shortcuts import get_object_or_404
from drf_yasg.utils import swagger_auto_schema
from rest_framework.generics import GenericAPIView
from rest_framework.mixins import ListModelMixin
from rest_framework.pagination import PageNumberPagination
from collections import defaultdict
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import Camera
import requests
from django.shortcuts import render

class LiveWallAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        mode = request.GET.get('mode', 'normal')

        cameras = Camera.objects.filter(created_by=user, is_active=True).select_related('group')
        base_url = settings.WS_BASE_URL.rstrip("/")

        all_cameras = []
        grouped = defaultdict(list)

        for cam in cameras:
            camera_data = {
                "id": cam.id,
                "camera_name": cam.name,
                "location": cam.location or "",
                # Include user_id and camera_id in the ws_stream_url
                "ws_stream_url": f"{base_url}/ws/stream/{user.id}/{cam.id}/{mode}/"
            }

            # Add to full camera list
            all_cameras.append(camera_data)

            # Group by group name
            group_name = cam.group.name if cam.group else "Unassigned"
            grouped[group_name].append(camera_data)

        # Structure the group-wise camera data
        grouped_data = [
            {
                "group_name": group,
                "cameras": cam_list
            }
            for group, cam_list in grouped.items()
        ]

        return Response({
            "cameras": all_cameras,
            "data": grouped_data
        })



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


# @login_required
def camera_dashboard(request):
    token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ2NTI1NDk2LCJpYXQiOjE3NDY0MzkwOTYsImp0aSI6IjQzMmM5YzcwNGFhZDQ4NjViNGEzYzYzNTliYzJlZDA1IiwidXNlcl9pZCI6MX0.tG5EFouftZl9hqrN3AmcnJjF8vqKEUbUE1LXnx1mJC8"
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


# class ActivateAiModelView(APIView):
#     permission_classes = [permissions.IsAuthenticated]
#     @swagger_auto_schema(request_body=UserAiModelActionSerializer, responses={200: UserAiModelSerializer})
#     def post(self, request):
#         serializer = UserAiModelActionSerializer(data=request.data)
#         if not serializer.is_valid():
#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#         user = request.user
#         aimodel_id = serializer.validated_data['aimodel_id']
#         is_active = serializer.validated_data['is_active']
#         aimodel = AiModel.objects.get(id=aimodel_id)

#         # Create or update the UserAiModel instance
#         obj, created = UserAiModel.objects.update_or_create(
#             user=user,
#             aimodel=aimodel,
#             defaults={'is_active': is_active}
#         )

#         return Response(UserAiModelSerializer(obj).data, status=status.HTTP_200_OK)

class ActivateAiModelView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(request_body=UserAiModelActionSerializer, responses={200: UserAiModelSerializer})
    def post(self, request):
        serializer = UserAiModelActionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        aimodel_id = serializer.validated_data['aimodel_id']
        is_active = serializer.validated_data['is_active']

        try:
            aimodel = AiModel.objects.get(id=aimodel_id)
        except AiModel.DoesNotExist:
            return Response({"detail": "AiModel not found."}, status=status.HTTP_404_NOT_FOUND)

        # Update AiModel status only
        aimodel.status = "Active" if is_active else "Inactive"
        aimodel.save()

        # You can return AiModel data, or a simple success message
        return Response({
            "id": aimodel.id,
            "name": aimodel.name,
            "status": aimodel.status
        }, status=status.HTTP_200_OK)

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
        operation_description="Creates a new AI model with name, function_name, icon, version, and status."
    )
    def post(self, request):
        serializer = AiModelSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(
        responses={200: AiModelSerializer(many=True)},
        operation_summary="List AI Models",
        operation_description="Returns a list of all AI models."
    )
    def get(self, request):
        queryset = AiModel.objects.all()
        serializer = AiModelSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from camera.models import UserAiModel, AiModel, Camera
from users.models import User
from camera.serializers import UserAiModelSerializer

class UserAiModelView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    
class UserAiModelView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(
        request_body=AssignAiModelSerializer,
        responses={201: "Created", 400: "Bad Request"},
        operation_summary="Assign AI Models to User",
        operation_description="Assigns a list of AiModels to a user and camera."
    )
    def post(self, request):
        serializer = AssignAiModelSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        user_id = serializer.validated_data['user_id']
        camera_id = serializer.validated_data['camera_id']
        ai_model_ids = serializer.validated_data['ai_model_ids']

        try:
            user = User.objects.get(id=user_id)
            camera = Camera.objects.get(id=camera_id)
        except (User.DoesNotExist, Camera.DoesNotExist):
            return Response({"detail": "Invalid user or camera ID."}, status=404)

        created_items = []
        duplicate_items = []

        for model_id in ai_model_ids:
            try:
                aimodel = AiModel.objects.get(id=model_id)
            except AiModel.DoesNotExist:
                continue

            # Check if the combination already exists
            if UserAiModel.objects.filter(user=user, aimodel=aimodel, camera=camera).exists():
                duplicate_items.append({"user_id": user.id, "aimodel_id": aimodel.id, "camera_id": camera.id})
                continue

            # Create new association
            obj = UserAiModel.objects.create(
                user=user,
                aimodel=aimodel,
                camera=camera,
                is_active=True
            )
            created_items.append(obj.id)

        # Return error if duplicates found
        if duplicate_items:
            return Response({
                "detail": "Some entries already exist.",
                "duplicates": duplicate_items,
                "created_ids": created_items
            }, status=400 if not created_items else 207)  # 207: Multi-Status

        return Response({"created_ids": created_items}, status=201)

    @swagger_auto_schema(
        query_serializer=UserCameraQuerySerializer,
        responses={200: AiModelSerializer(many=True)},
        operation_summary="Get AI Models for User and Camera",
        operation_description="Returns AI models assigned to a specific user and camera."
    )
    def get(self, request):
        serializer = UserCameraQuerySerializer(data=request.query_params)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        user_id = serializer.validated_data['user_id']
        camera_id = serializer.validated_data['camera_id']

        user_models = UserAiModel.objects.filter(user_id=user_id, camera_id=camera_id).select_related('aimodel')
        ai_models = [um.aimodel for um in user_models]

        serialized_data = AiModelSerializer(ai_models, many=True)
        return Response(serialized_data.data, status=200)
