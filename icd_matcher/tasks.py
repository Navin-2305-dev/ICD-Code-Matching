import logging
from celery import shared_task
from django.core.cache import cache
from django.conf import settings
from icd_matcher.models import MedicalAdmissionDetails
from icd_matcher.utils.rag_kag_pipeline import RAGKAGPipeline
from icd_matcher.utils.exceptions import ICDMatcherError
from asgiref.sync import sync_to_async
import uuid

logger = logging.getLogger(__name__)

@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    queue='high_priority'
)
async def predict_icd_code(self, admission_id):
    """Predict ICD code for a medical admission."""
    logger.info(f"Starting ICD prediction task for admission {admission_id}")
    try:
        admission = await sync_to_async(MedicalAdmissionDetails.objects.get)(id=admission_id)
        cache_key = f"predict_{admission.id}_{uuid.uuid4().hex}"
        cached_result = await sync_to_async(cache.get)(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for prediction: {cache_key}")
            await sync_to_async(setattr)(admission, 'predicted_icd_code', cached_result.get('predicted_icd_code', ''))
            await sync_to_async(setattr)(admission, 'prediction_accuracy', cached_result.get('prediction_accuracy', 0.0))
            await sync_to_async(setattr)(admission, 'predicted_icd_codes', cached_result.get('predicted_icd_codes', []))
            await sync_to_async(admission.save)()
            return cached_result

        pipeline = await RAGKAGPipeline.create()
        icd_matches = pipeline.run(admission.patient_data)
        if not icd_matches:
            logger.warning(f"No ICD matches found for admission {admission_id}")
            return {}

        top_match = None
        top_score = 0.0
        predicted_codes = []
        for condition, matches in icd_matches.items():
            for code, title, score in matches:
                predicted_codes.append({"code": code, "title": title, "confidence": score})
                if score > top_score:
                    top_score = score
                    top_match = (code, title)

        result = {
            'predicted_icd_code': top_match[0] if top_match else '',
            'prediction_accuracy': top_score,
            'predicted_icd_codes': predicted_codes
        }

        await sync_to_async(setattr)(admission, 'predicted_icd_code', result['predicted_icd_code'])
        await sync_to_async(setattr)(admission, 'prediction_accuracy', result['prediction_accuracy'])
        await sync_to_async(setattr)(admission, 'predicted_icd_codes', result['predicted_icd_codes'])
        await sync_to_async(admission.save)()
        
        await sync_to_async(cache.set)(
            cache_key,
            result,
            timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600)
        )
        logger.info(f"ICD prediction completed for admission {admission_id}")
        return result
    except MedicalAdmissionDetails.DoesNotExist:
        logger.error(f"Admission {admission_id} not found")
        raise self.retry()
    except Exception as e:
        logger.error(f"Error in ICD prediction for admission {admission_id}: {e}")
        if self.request.retries >= self.max_retries:
            logger.error("Max retries reached, task failed")
            return {}
        raise self.retry(exc=e)