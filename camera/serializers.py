from rest_framework import serializers
from .models import Camera, UserAiModel, AiModel

class CameraSerializer(serializers.ModelSerializer):
    created_by = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = Camera
        fields = [
            "id",
            "name",
            "rtsp_url",
            "location",
            "username",
            "password",
            "created_at",
            "created_by",
        ]
        read_only_fields = ["created_at", "created_by"]

class AiModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = AiModel
        fields = ["id", "name", "function_name", "icon", "version", "status"]

class UserAiModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserAiModel
        fields = ["aimodel", "is_active"]

class UserAiModelActionSerializer(serializers.Serializer):
    aimodel_id = serializers.IntegerField(required=True)
    is_active = serializers.BooleanField(required=True)

    def validate_aimodel_id(self, value):
        if not AiModel.objects.filter(id=value).exists():
            raise serializers.ValidationError("AiModel with this ID does not exist.")
        return value

class CameraGroupActionSerializer(serializers.Serializer):
    group_name = serializers.CharField()
    camera_ids = serializers.ListField(child=serializers.IntegerField(), allow_empty=False)

class AssignAiModelSerializer(serializers.Serializer):
    user_id = serializers.IntegerField(required=True)
    camera_id = serializers.IntegerField(required=True)
    ai_model_ids = serializers.ListField(
        child=serializers.IntegerField(), required=True, allow_empty=False
    )

class UserCameraQuerySerializer(serializers.Serializer):
    user_id = serializers.IntegerField(required=True)
    camera_id = serializers.IntegerField(required=True)

class ExpiryConfigSerializer(serializers.Serializer):
    password = serializers.CharField()
    expiry_date = serializers.DateField()

class ExpiryStatusSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=["not_set", "expired", "active"])
    expiry_date = serializers.DateField(required=False)
    days_left = serializers.IntegerField(required=False)


class UserAiModelZoneSerializer(serializers.Serializer):
    user_id = serializers.IntegerField()
    camera_id = serializers.IntegerField()
    aimodel_id = serializers.IntegerField()
    zones = serializers.DictField(
        child=serializers.ListField(
            child=serializers.ListField(
                child=serializers.FloatField()
            )
        )
    )

class StartStreamRequestSerializer(serializers.Serializer):
    camera_id = serializers.IntegerField(required=True)

class ZoneDetailSerializer(serializers.Serializer):
    ai_model = serializers.CharField()
    zones = serializers.DictField(
        child=serializers.ListField(
            child=serializers.ListField(
                child=serializers.FloatField()
            )
        )
    )

class StartStreamResponseSerializer(serializers.Serializer):
    hls_url = serializers.CharField()
    zones = ZoneDetailSerializer(many=True, required=False)


