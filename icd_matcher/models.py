from django.db import models
from django.core.exceptions import ValidationError
from django.db.models import JSONField
from django.contrib.postgres.indexes import GinIndex
import json
import logging

logger = logging.getLogger(__name__)

class ICDCategory(models.Model):
    code = models.CharField(max_length=10, unique=True)
    title = models.CharField(max_length=255)
    definition = models.TextField(blank=True)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL)
    inclusions = models.TextField(blank=True)
    exclusions = models.TextField(blank=True)

    def clean(self):
        for field in ['inclusions', 'exclusions']:
            value = getattr(self, field)
            if value:
                try:
                    parsed = json.loads(value) if value else []
                    if not isinstance(parsed, list):
                        raise ValidationError(f"{field.capitalize()} must be a list.")
                except json.JSONDecodeError:
                    raise ValidationError(f"{field.capitalize()} must be valid JSON.")
            else:
                setattr(self, field, json.dumps([]))  # Ensure empty fields are stored as empty list JSON

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.code}: {self.title}"

    def get_inclusions(self):
        return json.loads(self.inclusions) if self.inclusions else []

    def get_exclusions(self):
        return json.loads(self.exclusions) if self.exclusions else []

class MedicalAdmissionDetails(models.Model):
    patient_data = models.TextField()
    admission_date = models.DateField()
    predicted_icd_code = models.CharField(max_length=10, null=True, blank=True)
    prediction_accuracy = models.FloatField(null=True, blank=True)
    predicted_icd_codes = JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['admission_date']),
            models.Index(fields=['predicted_icd_code']),
            GinIndex(fields=['predicted_icd_codes']),
        ]

    def clean(self):
        if not self.patient_data or not self.patient_data.strip():
            raise ValidationError("patient_data cannot be empty.")
        if self.prediction_accuracy is not None and not (0 <= self.prediction_accuracy <= 100):
            raise ValidationError("Prediction accuracy must be between 0 and 100.")
        codes = self.predicted_icd_codes
        if isinstance(codes, str):
            try:
                codes = json.loads(codes)
                self.predicted_icd_codes = codes
            except json.JSONDecodeError:
                raise ValidationError("predicted_icd_codes contains invalid JSON.")
        if codes:
            if not isinstance(codes, list):
                raise ValidationError("predicted_icd_codes must be a list.")
            for code in codes:
                if not isinstance(code, dict) or not all(k in code for k in ['code', 'confidence', 'title']):
                    raise ValidationError(
                        "Each predicted_icd_codes entry must be a dict with 'code', 'confidence', and 'title' keys."
                    )
                if not isinstance(code['confidence'], (int, float)) or code['confidence'] < 0 or code['confidence'] > 100:
                    raise ValidationError("Confidence score must be a number between 0 and 100.")
        if self.predicted_icd_code and not ICDCategory.objects.filter(code=self.predicted_icd_code).exists():
            raise ValidationError(f"Invalid ICD code: {self.predicted_icd_code}")

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Admission on {self.admission_date}: {self.patient_data[:50]}"