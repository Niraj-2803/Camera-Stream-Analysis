from django.db import models
from camera.models import Camera
from users.models import BaseModel

# Create your models here.

class Event(BaseModel):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    event_type = models.CharField(max_length=100)  
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='events/', null=True, blank=True)
    description = models.TextField(null=True, blank=True)
