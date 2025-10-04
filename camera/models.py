from django.db import models
from users.models import BaseModel, User
from datetime import date
from django.utils import timezone

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
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='cameras',null=True, blank=True)
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


class UserAiModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user')
    aimodel = models.ForeignKey(AiModel, on_delete=models.CASCADE, related_name='ai')
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='camera')
    is_active = models.BooleanField(default=False)
    zones = models.JSONField(default=dict, blank=True, null=True)  # Add this line
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    is_deleted = models.BooleanField(default=False, null=True, blank=True)


    class Meta:
        unique_together = ('user', 'aimodel', 'camera')


# models.py
class SeatStatsLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    stats_file = models.FileField(upload_to="seat_stats/")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'camera', 'date')


class TrialConfig(models.Model):
    expiry_date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Expiry: {self.expiry_date}"
    
    
class InOutStats(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    total_in = models.IntegerField(default=0)
    total_out = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ai_inoutstats"
        unique_together = ('user', 'camera', 'date')

    def increment_in(self, value=1):
        self.total_in += value
        self.save(update_fields=['total_in', 'updated_at'])

    def increment_out(self, value=1):
        self.total_out += value
        self.save(update_fields=['total_out', 'updated_at'])

    @property
    def total_count(self):
        return self.total_in + self.total_out
    
class RestrictedZoneAlert(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    zone_name = models.CharField(max_length=100)
    date = models.DateField()
    image = models.ImageField(upload_to="restricted_alerts/")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "camera", "date"])
        ]
        ordering = ["-created_at"]

    def __str__(self):
        return f"Alert - {self.zone_name} ({self.date})"
