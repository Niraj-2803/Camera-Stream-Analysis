from django.db import models
from users.models import BaseModel, User

class CameraGroup(BaseModel):
    name = models.CharField(max_length=100)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='camera_groups')

    def __str__(self):
        return self.name

class Camera(BaseModel):
    name = models.CharField(max_length=100)
    rtsp_url = models.CharField(max_length=500)
    location = models.CharField(max_length=255, blank=True, null=True, default="")
    username = models.CharField(max_length=100, blank=True, null=True, default="")
    password = models.CharField(max_length=100, blank=True, null=True, default="")
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='cameras')
    group = models.ForeignKey(CameraGroup, on_delete=models.SET_NULL, null=True, blank=True, related_name='cameras')

    def __str__(self):
        return self.name

class AiModel(models.Model):
    name = models.CharField(max_length=100)
    function_name = models.CharField(max_length=100)
    icon = models.CharField(max_length=200, default="")  # Path to icon as string
    version = models.CharField(max_length=10, default="1.0")
    status = models.CharField(max_length=20, default="Inactive")

    def __str__(self):
        return self.name


class UserAiModel(BaseModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user')
    aimodel = models.ForeignKey(AiModel, on_delete=models.CASCADE, related_name='ai')
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='camera')
    is_active = models.BooleanField(default=False)

    class Meta:
        unique_together = ('user', 'aimodel', 'camera')