import logging
from django.db.models.signals import post_save
from django.dispatch import receiver
from icd_matcher.models import MedicalAdmissionDetails
from icd_matcher.tasks import predict_icd_code
from celery.exceptions import MaxRetriesExceededError

logger = logging.getLogger(__name__)

@receiver(post_save, sender=MedicalAdmissionDetails)
def trigger_icd_prediction(sender, instance, created, **kwargs):
    if created:
        try:
            logger.info(f"Triggering ICD prediction for admission {instance.id}")
            predict_icd_code.apply_async(
                args=[instance.id],
                queue='high_priority',
                retry=True,
                retry_policy={
                    'max_retries': 3,
                    'interval_start': 30,
                    'interval_step': 60,
                    'interval_max': 300
                }
            )
        except MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for admission {instance.id}")
        except Exception as e:
            logger.error(f"Failed to trigger ICD prediction for admission {instance.id}: {e}")