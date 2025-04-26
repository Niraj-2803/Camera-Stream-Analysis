from rest_framework import serializers
from .models import Camera, UserAiModel, AiModel


class CameraSerializer(serializers.ModelSerializer):
    created_by = serializers.StringRelatedField(read_only=True) 

    class Meta:
        model = Camera
        fields = ["id", "name", "rtsp_url", "location", "created_at", "created_by"]
        read_only_fields = ["created_at", "created_by"]



class AiModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = AiModel
        fields = ['id', 'name', 'function_name']

class UserAiModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserAiModel
        fields = ['aimodel', 'is_active']


class UserAiModelActionSerializer(serializers.Serializer):
    aimodel_id = serializers.IntegerField(required=True)
    is_active = serializers.BooleanField(required=True)

    def validate_aimodel_id(self, value):
        if not AiModel.objects.filter(id=value).exists():
            raise serializers.ValidationError("AiModel with this ID does not exist.")
        return value
    

class CameraGroupActionSerializer(serializers.Serializer):
    group_name = serializers.CharField()
    camera_ids = serializers.ListField(
        child=serializers.IntegerField(), allow_empty=False
    )
