from django.db import models
from django.core.exceptions import ValidationError
from django.db.models import JSONField
from django.contrib.postgres.indexes import GinIndex

class ICDCategory(models.Model):
    code = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=255)
    definition = models.TextField(blank=True, null=True)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE)
    inclusions = JSONField(default=list, blank=True)
    exclusions = JSONField(default=list, blank=True)

    def get_inclusions(self):
        return self.inclusions or []

    def get_exclusions(self):
        return self.exclusions or []

    class Meta:
        indexes = [
            models.Index(fields=['code', 'title']),
            models.Index(fields=['title'], name='icd_title_idx'),
        ]

    def clean(self):
        if not self.code or not self.code.strip():
            raise ValidationError("Code cannot be empty.")
        if not self.title or not self.title.strip():
            raise ValidationError("Title cannot be empty.")

    def __str__(self):
        return f"{self.code} - {self.title[:50]}"

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
        if self.prediction_accuracy is not None and not (0 <= self.prediction_accuracy <= 100):
            raise ValidationError("Prediction accuracy must be between 0 and 100.")
        if self.predicted_icd_codes:
            for code in self.predicted_icd_codes:
                if not isinstance(code, dict) or 'code' not in code or 'confidence' not in code:
                    raise ValidationError("Invalid predicted_icd_codes format.")

    def __str__(self):
        return f"Admission on {self.admission_date}: {self.patient_data[:50]}"