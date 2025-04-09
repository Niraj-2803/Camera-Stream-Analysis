from django.db import models
from users.models import BaseModel, User


class Camera(BaseModel):
    name = models.CharField(max_length=100)
    rtsp_url = models.CharField(max_length=500)
    location = models.CharField(max_length=255, blank=True, null=True,default="") 
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='cameras')

    def __str__(self):
        return self.name


class AiModel(BaseModel):
    name = models.CharField(max_length=100)
    function_name = models.CharField(max_length=100)
    
    def __str__(self):
        return self.name
    
    
class UserAiModel(BaseModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user_models')
    aimodel = models.ForeignKey(AiModel, on_delete=models.CASCADE, related_name='user_models')
    is_active = models.BooleanField(default=False)

    class Meta:
        unique_together = ('user', 'aimodel')
