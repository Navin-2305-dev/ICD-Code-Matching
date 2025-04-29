# icd_matcher/models.py
from django.db import models
import json
from jsonfield import JSONField

class ICDCategory(models.Model):
    code = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=255)
    definition = models.TextField(blank=True, null=True)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE)
    inclusions = models.TextField(blank=True, null=True)  # Store as JSON string
    exclusions = models.TextField(blank=True, null=True)  # Store as JSON string

    def get_inclusions(self):
        """Parse inclusions JSON string to list."""
        return json.loads(self.inclusions) if self.inclusions else []

    def get_exclusions(self):
        """Parse inclusions JSON string to list."""
        return json.loads(self.exclusions) if self.exclusions else []

    def __str__(self):
        return f"{self.code} - {self.title}"

    class Meta:
        indexes = [models.Index(fields=['code'])]

class MedicalAdmissionDetails(models.Model):
    patient_data = models.TextField()
    admission_date = models.DateField()
    predicted_icd_code = models.CharField(max_length=20, blank=True, null=True)  # Top match
    prediction_accuracy = models.FloatField(blank=True, null=True)  # Top match confidence
    predicted_icd_codes = JSONField(default=list, blank=True)  # All matches as list of dicts
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'icd_matcher_medicaladmissiondetails'
        verbose_name = 'Medical Admission Detail'
        verbose_name_plural = 'Medical Admission Details'

    def __str__(self):
        return f"Admission on {self.admission_date}: {self.patient_data[:50]}..."