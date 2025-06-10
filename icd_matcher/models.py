from django.db import models
from django.core.exceptions import ValidationError
from django.db.models import JSONField
from django.contrib.postgres.indexes import GinIndex
import json
import logging

logger = logging.getLogger(__name__)

class ICDCategory(models.Model):
    code = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=255)
    definition = models.TextField(blank=True, null=True)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE)
    inclusions = JSONField(default=list, blank=True)
    exclusions = JSONField(default=list, blank=True)

    def get_inclusions(self):
        """Return inclusions synchronously."""
        return self.inclusions or []

    def get_exclusions(self):
        """Return exclusions synchronously."""
        return self.exclusions or []

    class Meta:
        indexes = [
            models.Index(fields=['code', 'title']),
            models.Index(fields=['title'], name='icd_title_idx'),
        ]

    def clean(self):
        """Validate model data synchronously."""
        if not self.code or not self.code.strip():
            raise ValidationError("Code cannot be empty.")
        if not self.title or not self.title.strip():
            raise ValidationError("Title cannot be empty.")
        # Validate JSON fields
        try:
            if not isinstance(self.inclusions, list):
                raise ValidationError("Inclusions must be a list.")
            if not isinstance(self.exclusions, list):
                raise ValidationError("Exclusions must be a list.")
        except TypeError:
            raise ValidationError("Inclusions and exclusions must be valid JSON lists.")

    def save(self, *args, **kwargs):
        """Override save to ensure validation."""
        self.clean()
        super().save(*args, **kwargs)

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
        """Validate model data synchronously."""
        # Validate patient_data as non-empty text (no JSON requirement)
        if not self.patient_data or not self.patient_data.strip():
            raise ValidationError("patient_data cannot be empty.")

        # Validate prediction_accuracy
        if self.prediction_accuracy is not None and not (0 <= self.prediction_accuracy <= 100):
            raise ValidationError("Prediction accuracy must be between 0 and 100.")

        # Validate predicted_icd_codes
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

        # Validate predicted_icd_code if provided
        if self.predicted_icd_code and not ICDCategory.objects.filter(code=self.predicted_icd_code).exists():
            raise ValidationError(f"Invalid ICD code: {self.predicted_icd_code}")

    def save(self, *args, **kwargs):
        """Override save to ensure validation."""
        self.clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Admission on {self.admission_date}: {self.patient_data[:50]}"