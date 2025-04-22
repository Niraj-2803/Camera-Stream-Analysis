from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.shortcuts import get_object_or_404
from .models import Event
from .serializers import EventSerializer
from drf_yasg.utils import swagger_auto_schema
from rest_framework.parsers import MultiPartParser, FormParser



class EventListCreateAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]  # 💡 needed for image upload
    @swagger_auto_schema(responses={200: EventSerializer(many=True)})
    def get(self, request):
        events = Event.objects.all().order_by('-timestamp')
        serializer = EventSerializer(events, many=True)
        return Response(serializer.data)
    

    @swagger_auto_schema(request_body=EventSerializer, responses={201: EventSerializer})

    def post(self, request):
        serializer = EventSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


class EventDetailAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self, pk):
        return get_object_or_404(Event, pk=pk)

    @swagger_auto_schema(responses={200: EventSerializer})
    def get(self, request, pk):
        event = self.get_object(pk)
        serializer = EventSerializer(event)
        return Response(serializer.data)

    @swagger_auto_schema(request_body=EventSerializer, responses={200: EventSerializer})
    def put(self, request, pk):
        event = self.get_object(pk)
        serializer = EventSerializer(event, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(responses={204: 'Event deleted'})
    def delete(self, request, pk):
        event = self.get_object(pk)
        event.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
