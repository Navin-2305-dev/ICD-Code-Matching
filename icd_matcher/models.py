# icd_matcher/models.py
from django.db import models

class ICDCategory(models.Model):
    code = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=255)
    definition = models.TextField(blank=True, null=True)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.code} - {self.title}"

    class Meta:
        indexes = [models.Index(fields=['code'])]