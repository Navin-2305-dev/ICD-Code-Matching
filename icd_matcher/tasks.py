from celery import shared_task
from django.db import transaction
from icd_matcher.models import MedicalAdmissionDetails
from icd_matcher.utils.text_processing import generate_patient_summary
from icd_matcher.utils.embeddings import find_best_icd_match
import logging
import re

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def predict_icd_code(self, admission_id):
    """Celery task to predict ICD codes for a MedicalAdmissionDetails record."""
    try:
        with transaction.atomic():
            admission = MedicalAdmissionDetails.objects.select_for_update().get(id=admission_id)
            logger.info(f"Predicting ICD codes for admission {admission_id}")
            
            # Generate summary and conditions
            summary, conditions = generate_patient_summary(admission.patient_data)
            logger.debug(f"Generated conditions: {conditions}")
            
            # Find best ICD matches
            icd_matches = find_best_icd_match(conditions, admission.patient_data)
            logger.debug(f"ICD matches: {icd_matches}")
            
            # Collect all matches with confidence >= 60%
            all_matches = []
            best_code = None
            best_score = None
            
            # Relevance heuristic: boost codes with "calculus" or "ureter" in title
            relevance_terms = ['calculus', 'ureter', 'kidney', 'renal', 'stone']
            
            for condition, matches in icd_matches.items():
                for code, title, confidence in matches:
                    if code and confidence >= 60:
                        # Calculate relevance score
                        relevance_boost = 1.0
                        title_lower = title.lower()
                        if any(term in title_lower for term in relevance_terms):
                            relevance_boost = 1.2  # Boost relevant codes
                        adjusted_score = confidence * relevance_boost
                        
                        all_matches.append({
                            'code': code,
                            'title': title,
                            'confidence': round(confidence, 1)
                        })
                        
                        # Track the highest-scoring match
                        if best_score is None or adjusted_score > best_score:
                            best_code = code
                            best_score = adjusted_score
                            best_confidence = confidence
            
            # Sort matches by confidence (descending)
            all_matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Update the record
            if all_matches:
                admission.predicted_icd_codes = all_matches
                admission.predicted_icd_code = best_code
                admission.prediction_accuracy = round(best_confidence, 1) if best_confidence else None
                admission.save()
                logger.info(f"Updated admission {admission_id} with {len(all_matches)} ICD codes, top code: {best_code} ({best_confidence}%)")
            else:
                logger.warning(f"No ICD codes found for admission {admission_id}")
                admission.predicted_icd_codes = []
                admission.predicted_icd_code = None
                admission.prediction_accuracy = None
                admission.save()
                
    except MedicalAdmissionDetails.DoesNotExist:
        logger.error(f"Admission {admission_id} not found")
        raise
    except Exception as e:
        logger.error(f"Error predicting ICD codes for admission {admission_id}: {e}")
        self.retry(countdown=60, exc=e)