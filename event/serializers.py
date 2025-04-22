from rest_framework import serializers

from camera.models import Camera
from .models import Event


class EventSerializer(serializers.ModelSerializer):
    camera_id = serializers.PrimaryKeyRelatedField(
        queryset=Camera.objects.all(),
        source='camera'
    )

    class Meta:
        model = Event
        fields = ['id', 'camera_id', 'event_type', 'description', 'image', 'timestamp']
        read_only_fields = ['id', 'timestamp']
