from django.db.models.signals import post_save
from django.dispatch import receiver
from icd_matcher.models import MedicalAdmissionDetails
from icd_matcher.tasks import predict_icd_code
import logging

logger = logging.getLogger(__name__)

@receiver(post_save, sender=MedicalAdmissionDetails)
def trigger_icd_prediction(sender, instance, created, **kwargs):
    """Trigger Celery task to predict ICD code when a new MedicalAdmissionDetails record is created."""
    if created:
        logger.info(f"New MedicalAdmissionDetails created (ID: {instance.id}). Triggering ICD prediction.")
        predict_icd_code.delay(instance.id)