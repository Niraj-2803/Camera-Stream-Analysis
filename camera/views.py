import cv2
import time
import os
import requests
from collections import defaultdict
from django.conf import settings
from django.shortcuts import get_object_or_404, render
from rest_framework import status, permissions
from rest_framework.generics import GenericAPIView
from rest_framework.mixins import ListModelMixin
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from datetime import datetime, date
from drf_yasg.utils import swagger_auto_schema
import threading
from django.db.models import Sum
from django.http import JsonResponse
from .ffmpeg_stream import start_ffmpeg_stream
from drf_yasg import openapi
from .models import Camera, CameraGroup, UserAiModel, AiModel, TrialConfig, InOutCount
from .serializers import (
    AiModelSerializer,
    AssignAiModelSerializer,
    CameraGroupActionSerializer,
    CameraSerializer,
    UserAiModelSerializer,
    UserCameraQuerySerializer,
    ExpiryConfigSerializer,
    ExpiryStatusSerializer,
    UserAiModelZoneSerializer,
    StartStreamResponseSerializer,
    StartStreamRequestSerializer
)
from users.models import User


class LiveWallAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        mode = request.GET.get("mode", "normal")

        cameras = Camera.objects.filter(created_by=user, is_active=True).select_related("group")
        base_url = settings.WS_BASE_URL.rstrip("/")

        all_cameras = []
        grouped = defaultdict(list)

        for cam in cameras:
            camera_data = {
                "id": cam.id,
                "camera_name": cam.name,
                "location": cam.location or "",
                "ws_stream_url": f"{base_url}/ws/stream/{user.id}/{cam.id}/{mode}/",
            }
            all_cameras.append(camera_data)

            group_name = cam.group.name if cam.group else "Unassigned"
            grouped[group_name].append(camera_data)

        grouped_data = [{"group_name": group, "cameras": cam_list} for group, cam_list in grouped.items()]

        return Response({"cameras": all_cameras, "data": grouped_data})


def generate_frames(rtsp_url, reconnect_delay=5):
    while True:
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            time.sleep(reconnect_delay)
            continue

        while True:
            success, frame = cap.read()
            if not success:
                cap.release()
                time.sleep(reconnect_delay)
                break

            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"


def camera_dashboard(request):
    # TODO: move hardcoded JWT token to .env
    token = "HARDCODED_TOKEN_HERE"
    headers = {"Authorization": f"Bearer {token}"}
    api_url = "http://127.0.0.1:8000/api/camera/"
    response = requests.get(api_url, headers=headers)
    cameras = response.json().get("results", [])
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


class ActivateAiModelView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(request_body=UserAiModelSerializer, responses={200: UserAiModelSerializer})
    def post(self, request):
        serializer = UserAiModelSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        aimodel_id = serializer.validated_data["aimodel"]["id"]
        is_active = serializer.validated_data["is_active"]

        try:
            aimodel = AiModel.objects.get(id=aimodel_id)
        except AiModel.DoesNotExist:
            return Response({"detail": "AiModel not found."}, status=status.HTTP_404_NOT_FOUND)

        aimodel.status = "Active" if is_active else "Inactive"
        aimodel.save()

        return Response(
            {"id": aimodel.id, "name": aimodel.name, "status": aimodel.status},
            status=status.HTTP_200_OK,
        )


class AddCameraToGroupAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(
        request_body=CameraGroupActionSerializer,
        operation_summary="Add multiple cameras to a group by group name",
    )
    def post(self, request):
        serializer = CameraGroupActionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        group_name = serializer.validated_data["group_name"]
        camera_ids = serializer.validated_data["camera_ids"]

        group, _ = CameraGroup.objects.get_or_create(name=group_name, created_by=request.user)

        updated = 0
        for cam_id in camera_ids:
            try:
                camera = Camera.objects.get(id=cam_id, created_by=request.user)
                camera.group = group
                camera.save()
                updated += 1
            except Camera.DoesNotExist:
                continue

        return Response({"message": f"{updated} camera(s) added to group '{group_name}'."}, status=200)


class ListCamerasInGroupAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(operation_summary="List cameras in a specific group")
    def get(self, request, group_id):
        group = get_object_or_404(CameraGroup, id=group_id, created_by=request.user)
        cameras = group.cameras.all()
        serializer = CameraSerializer(cameras, many=True)
        return Response({"group": group.name, "cameras": serializer.data}, status=200)


class CreateAiModelView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(request_body=AiModelSerializer)
    def post(self, request):
        serializer = AiModelSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

    @swagger_auto_schema(responses={200: AiModelSerializer(many=True)})
    def get(self, request):
        queryset = AiModel.objects.all()
        serializer = AiModelSerializer(queryset, many=True)
        return Response(serializer.data, status=200)


class UserAiModelView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(request_body=AssignAiModelSerializer)
    def post(self, request):
        serializer = AssignAiModelSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        user_id = serializer.validated_data["user_id"]
        camera_id = serializer.validated_data["camera_id"]
        ai_model_ids = serializer.validated_data["ai_model_ids"]

        try:
            user = User.objects.get(id=user_id)
            camera = Camera.objects.get(id=camera_id)
        except (User.DoesNotExist, Camera.DoesNotExist):
            return Response({"detail": "Invalid user or camera ID."}, status=404)

        created_items, duplicate_items = [], []

        for model_id in ai_model_ids:
            try:
                aimodel = AiModel.objects.get(id=model_id)
            except AiModel.DoesNotExist:
                continue

            if UserAiModel.objects.filter(user=user, aimodel=aimodel, camera=camera).exists():
                duplicate_items.append(
                    {"user_id": user.id, "aimodel_id": aimodel.id, "camera_id": camera.id}
                )
                continue

            obj = UserAiModel.objects.create(user=user, aimodel=aimodel, camera=camera, is_active=True)
            created_items.append(obj.id)

        if duplicate_items:
            return Response(
                {"detail": "Some entries already exist.", "duplicates": duplicate_items, "created_ids": created_items},
                status=400 if not created_items else 207,
            )

        return Response({"created_ids": created_items}, status=201)

    @swagger_auto_schema(query_serializer=UserCameraQuerySerializer)
    def get(self, request):
        serializer = UserCameraQuerySerializer(data=request.query_params)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        user_id = serializer.validated_data["user_id"]
        camera_id = serializer.validated_data["camera_id"]

        user_models = UserAiModel.objects.filter(user_id=user_id, camera_id=camera_id).select_related("aimodel")
        ai_models = [um.aimodel for um in user_models]
        serialized_data = AiModelSerializer(ai_models, many=True)
        return Response(serialized_data.data, status=200)


class ExpiryConfigView(APIView):
    permission_classes = [permissions.AllowAny]

    @swagger_auto_schema(request_body=ExpiryConfigSerializer)
    def post(self, request):
        serializer = ExpiryConfigSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        password = serializer.validated_data["password"]
        expiry_date = serializer.validated_data["expiry_date"]

        # TODO: move ADMIN_PASSWORD to .env
        ADMIN_PASSWORD = "HARDCODED_PASSWORD"

        if password != ADMIN_PASSWORD:
            return Response({"detail": "Invalid password"}, status=401)

        TrialConfig.objects.all().delete()
        TrialConfig.objects.create(expiry_date=expiry_date)
        return Response({"message": "Expiry date set successfully"}, status=200)

    @swagger_auto_schema(responses={200: ExpiryStatusSerializer})
    def get(self, request):
        config = TrialConfig.objects.first()
        if not config:
            return Response({"status": "not_set"}, status=200)

        today = time.strftime("%Y-%m-%d")
        if today > str(config.expiry_date):
            return Response({"status": "expired", "expiry_date": config.expiry_date}, status=200)

        return Response(
            {"status": "active", "expiry_date": config.expiry_date},
            status=200,
        )


class UserAiModelZoneView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(
        request_body=UserAiModelZoneSerializer,
        responses={200: "Updated", 404: "Not Found", 400: "Bad Request"},
        operation_summary="Update Zones for UserAiModel",
        operation_description=(
            "Adds new zones or updates existing ones for a UserAiModel instance. "
            "If a zone key already exists, it will be overwritten. "
            "If it's new, it will be appended."
        )
    )
    def post(self, request):
        serializer = UserAiModelZoneSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user_id = serializer.validated_data['user_id']
        camera_id = serializer.validated_data['camera_id']
        aimodel_id = serializer.validated_data['aimodel_id']
        new_zones = serializer.validated_data['zones']

        try:
            user_aimodel = UserAiModel.objects.get(
                user_id=user_id,
                camera_id=camera_id,
                aimodel_id=aimodel_id
            )
        except UserAiModel.DoesNotExist:
            return Response({"detail": "UserAiModel entry not found."}, status=404)

        # Merge logic
        current_zones = user_aimodel.zones or {}
        if not isinstance(current_zones, dict):
            current_zones = {}

        for key, val in new_zones.items():
            current_zones[key] = val  # update if exists, else add

        user_aimodel.zones = current_zones
        user_aimodel.save(update_fields=["zones"])

        return Response({
            "detail": "Zones updated successfully.",
            "zones": current_zones
        }, status=200)

class StartStreamView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="Start a camera stream and return HLS URL + zones",
        request_body=StartStreamRequestSerializer,
        responses={
            200: StartStreamResponseSerializer,
            400: openapi.Response(
                description="Invalid request",
                examples={"application/json": {"error": "Camera not found or inactive"}},
            ),
        },
    )
    def post(self, request):
        serializer = StartStreamRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        camera_id = serializer.validated_data["camera_id"]

        # Fetch camera from DB
        try:
            camera = Camera.objects.get(id=camera_id, created_by=request.user, is_active=True)
        except Camera.DoesNotExist:
            return Response({"error": "Camera not found or inactive"}, status=400)

        rtsp_url = camera.rtsp_url

        # Run ffmpeg stream in background
        t = threading.Thread(target=start_ffmpeg_stream, args=(rtsp_url, request.user.id, camera_id))
        t.daemon = True
        t.start()

        # HLS URL
        hls_url = f"/media/hls/camera_{camera_id}/index.m3u8"

        # Fetch zones for this camera + user
        user_ai_models = UserAiModel.objects.filter(user=request.user, camera=camera, is_active=True).select_related("aimodel")

        zones_data = {}
        for uam in user_ai_models:
            zones_data[str(uam.aimodel.name)] = {
                "ai_model": uam.aimodel.function_name,
                "zones": uam.zones or {}
            }

        return Response({
            "hls_url": hls_url,
            "zones": zones_data
        }, status=200)
    
# Add these to your views.py

class InOutStatsAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="Get InOut statistics with various filters",
        manual_parameters=[
            openapi.Parameter('query_type', openapi.IN_QUERY, description="Type of query: 'single', 'all_cameras', 'range', or 'aggregate'", type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('user_id', openapi.IN_QUERY, description="User ID", type=openapi.TYPE_INTEGER, required=True),
            openapi.Parameter('camera_id', openapi.IN_QUERY, description="Camera ID (required for 'single' and 'range' with specific camera)", type=openapi.TYPE_INTEGER, required=False),
            openapi.Parameter('date', openapi.IN_QUERY, description="Date in YYYY-MM-DD format (for 'single' and 'all_cameras')", type=openapi.TYPE_STRING, required=False),
            openapi.Parameter('start_date', openapi.IN_QUERY, description="Start date in YYYY-MM-DD format (for 'range' and 'aggregate')", type=openapi.TYPE_STRING, required=False),
            openapi.Parameter('end_date', openapi.IN_QUERY, description="End date in YYYY-MM-DD format (for 'range' and 'aggregate')", type=openapi.TYPE_STRING, required=False),
        ],
        responses={
            200: openapi.Response(
                description="Statistics data",
                examples={
                    "application/json": {
                        "type": "single_camera_result",
                        "data": {
                            "camera_id": 19,
                            "camera_name": "Main Entrance",
                            "date": "2024-09-16",
                            "in_count": 25,
                            "out_count": 23,
                            "total": 48,
                            "last_updated": "2024-09-16T14:30:00",
                            "source": "database"
                        }
                    }
                }
            ),
            400: "Bad Request",
            404: "Not Found"
        }
    )
    def get(self, request):
        query_type = request.query_params.get('query_type')
        user_id = request.query_params.get('user_id')
        camera_id = request.query_params.get('camera_id')
        date_str = request.query_params.get('date', datetime.now().strftime("%Y-%m-%d"))
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')

        if not user_id:
            return Response({'error': 'user_id is required'}, status=400)

        if not query_type:
            return Response({'error': 'query_type is required'}, status=400)

        # Validate user exists
        try:
            User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({'error': 'Invalid user_id'}, status=404)

        try:
            if query_type == 'single':
                return self.handle_single_camera(user_id, camera_id, date_str)
            elif query_type == 'all_cameras':
                return self.handle_all_cameras(user_id, date_str)
            elif query_type == 'range':
                return self.handle_range_query(user_id, camera_id, start_date, end_date)
            elif query_type == 'aggregate':
                return self.handle_aggregate_query(user_id, camera_id, start_date, end_date)
            else:
                return Response({
                    'error': 'Invalid query_type. Use: single, all_cameras, range, or aggregate'
                }, status=400)

        except Exception as e:
            return Response({'error': str(e)}, status=500)

    def handle_single_camera(self, user_id, camera_id, date_str):
        """Get stats for a single camera on a specific date"""
        if not camera_id:
            return Response({'error': 'camera_id is required for single camera query'}, status=400)

        # Validate camera exists
        try:
            Camera.objects.get(id=camera_id)
        except Camera.DoesNotExist:
            return Response({'error': 'Invalid camera_id'}, status=404)

        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            try:
                inout_record = InOutCount.objects.get(
                    user_id=user_id,
                    camera_id=camera_id,
                    date=target_date
                )
                return Response({
                    'type': 'single_camera_result',
                    'data': {
                        "camera_id": int(camera_id),
                        "camera_name": inout_record.camera.name,
                        "date": target_date.isoformat(),
                        "in_count": inout_record.in_count,
                        "out_count": inout_record.out_count,
                        "total": inout_record.total_count,
                        "last_updated": inout_record.last_updated.isoformat(),
                        "source": "database"
                    }
                })
            except InOutCount.DoesNotExist:
                return Response({
                    'type': 'single_camera_result',
                    'data': {
                        "camera_id": int(camera_id),
                        "date": target_date.isoformat(),
                        "in_count": 0,
                        "out_count": 0,
                        "total": 0,
                        "source": "database",
                        "message": "No data found for this date"
                    }
                })
        except ValueError:
            return Response({'error': 'Invalid date format. Use YYYY-MM-DD'}, status=400)

    def handle_all_cameras(self, user_id, date_str):
        """Get stats for all cameras of a user on a specific date"""
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            records = InOutCount.objects.filter(
                user_id=user_id,
                date=target_date
            ).select_related('camera')
            
            cameras_data = []
            total_in = 0
            total_out = 0
            
            for record in records:
                camera_data = {
                    "camera_id": record.camera.id,
                    "camera_name": record.camera.name,
                    "location": record.camera.location or "",
                    "in_count": record.in_count,
                    "out_count": record.out_count,
                    "total": record.total_count,
                    "last_updated": record.last_updated.isoformat()
                }
                cameras_data.append(camera_data)
                total_in += record.in_count
                total_out += record.out_count
            
            return Response({
                'type': 'all_cameras_result',
                'data': {
                    "date": target_date.isoformat(),
                    "total_in_count": total_in,
                    "total_out_count": total_out,
                    "total_count": total_in + total_out,
                    "cameras": cameras_data,
                    "cameras_count": len(cameras_data),
                    "source": "database"
                }
            })
        except ValueError:
            return Response({'error': 'Invalid date format. Use YYYY-MM-DD'}, status=400)

    def handle_range_query(self, user_id, camera_id, start_date, end_date):
        """Get stats for date range"""
        try:
            queryset = InOutCount.objects.filter(user_id=user_id)
            
            if camera_id:
                # Validate camera exists
                try:
                    Camera.objects.get(id=camera_id)
                except Camera.DoesNotExist:
                    return Response({'error': 'Invalid camera_id'}, status=404)
                queryset = queryset.filter(camera_id=camera_id)
                
            if start_date:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                queryset = queryset.filter(date__gte=start_date)
            if end_date:
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                queryset = queryset.filter(date__lte=end_date)
                
            records = queryset.select_related('camera').order_by('-date')
            
            results = []
            for record in records:
                results.append({
                    'date': record.date.isoformat(),
                    'camera_id': record.camera.id,
                    'camera_name': record.camera.name,
                    'location': record.camera.location or "",
                    'in_count': record.in_count,
                    'out_count': record.out_count,
                    'total_count': record.total_count,
                    'last_updated': record.last_updated.isoformat()
                })
                
            return Response({
                'type': 'range_result',
                'data': results,
                'count': len(results),
                'filters': {
                    'user_id': int(user_id),
                    'camera_id': int(camera_id) if camera_id else None,
                    'start_date': start_date.isoformat() if start_date else None,
                    'end_date': end_date.isoformat() if end_date else None
                }
            })
        except ValueError:
            return Response({'error': 'Invalid date format. Use YYYY-MM-DD'}, status=400)

    def handle_aggregate_query(self, user_id, camera_id, start_date, end_date):
        """Get aggregated stats"""
        try:
            queryset = InOutCount.objects.filter(user_id=user_id)
            
            if camera_id:
                # Validate camera exists
                try:
                    Camera.objects.get(id=camera_id)
                except Camera.DoesNotExist:
                    return Response({'error': 'Invalid camera_id'}, status=404)
                queryset = queryset.filter(camera_id=camera_id)
                
            if start_date:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                queryset = queryset.filter(date__gte=start_date)
            if end_date:
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                queryset = queryset.filter(date__lte=end_date)
                
            aggregated = queryset.aggregate(
                total_in=Sum('in_count'),
                total_out=Sum('out_count'),
                total_overall=Sum('total_count')
            )
            
            return Response({
                'type': 'aggregate_result',
                'data': {
                    'total_in': aggregated['total_in'] or 0,
                    'total_out': aggregated['total_out'] or 0,
                    'total_overall': aggregated['total_overall'] or 0,
                    'date_range': {
                        'start': start_date.isoformat() if start_date else None,
                        'end': end_date.isoformat() if end_date else None
                    },
                    'camera_id': int(camera_id) if camera_id else None,
                    'user_id': int(user_id)
                }
            })
        except ValueError:
            return Response({'error': 'Invalid date format. Use YYYY-MM-DD'}, status=400)


class CameraDeleteAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="Delete a camera by camera_id and user_id",
        manual_parameters=[
            openapi.Parameter('camera_id', openapi.IN_QUERY, description="Camera ID to delete", type=openapi.TYPE_INTEGER, required=True),
            openapi.Parameter('user_id', openapi.IN_QUERY, description="User ID (owner of the camera)", type=openapi.TYPE_INTEGER, required=True),
        ],
        responses={
            200: openapi.Response(
                description="Camera deleted successfully",
                examples={
                    "application/json": {
                        "message": "Camera deleted successfully",
                        "deleted_camera": {
                            "id": 19,
                            "name": "Main Entrance",
                            "location": "Front Door"
                        }
                    }
                }
            ),
            400: "Bad Request - Missing parameters",
            404: "Camera not found",
            403: "Permission denied - Camera doesn't belong to user"
        }
    )
    def delete(self, request):
        camera_id = request.query_params.get('camera_id')
        user_id = request.query_params.get('user_id')

        if not camera_id:
            return Response({'error': 'camera_id is required'}, status=400)
        
        if not user_id:
            return Response({'error': 'user_id is required'}, status=400)

        try:
            # Validate user exists
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({'error': 'Invalid user_id'}, status=404)

        try:
            # Find the camera belonging to the specified user
            camera = Camera.objects.get(id=camera_id, created_by=user)
        except Camera.DoesNotExist:
            return Response({
                'error': 'Camera not found or does not belong to the specified user'
            }, status=404)

        # Store camera info before deletion
        camera_info = {
            "id": camera.id,
            "name": camera.name,
            "location": camera.location or "",
            "rtsp_url": camera.rtsp_url
        }

        # Soft delete by setting is_deleted=True (if you have this field)
        # Or hard delete
        try:
            # If you have soft delete field
            if hasattr(camera, 'is_deleted'):
                camera.is_deleted = True
                camera.is_active = False
                camera.save()
                action = "marked as deleted"
            else:
                # Hard delete
                camera.delete()
                action = "permanently deleted"

            return Response({
                'message': f'Camera {action} successfully',
                'deleted_camera': camera_info
            }, status=200)

        except Exception as e:
            return Response({
                'error': f'Failed to delete camera: {str(e)}'
            }, status=500)
