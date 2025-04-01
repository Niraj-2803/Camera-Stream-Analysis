from django.db import models
from users.models import BaseModel

class Camera(BaseModel):
    name = models.CharField(max_length=100)
    rtsp_url = models.CharField(max_length=500)

    def __str__(self):
        return self.name
