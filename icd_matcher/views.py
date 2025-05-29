import hashlib
import logging
import json
import re
from typing import List, Dict, Union
from django.shortcuts import render, get_object_or_404
from django.core.cache import cache
from django.conf import settings
from django.http import JsonResponse
from asgiref.sync import sync_to_async
from django.db.models import Q
from icd_matcher.forms import PatientInputForm
from icd_matcher.models import MedicalAdmissionDetails, ICDCategory
from icd_matcher.utils.embeddings import find_best_icd_match
from icd_matcher.utils.text_processing import generate_patient_summary, is_not_negated, reset_preprocess_counter
from icd_matcher.utils.db_utils import get_icd_title
from icd_matcher.utils.rag_kag_pipeline import RAGKAGPipeline
import uuid

logger = logging.getLogger(__name__)

async def patient_input(request):
    """Handle patient input form submission."""
    reset_preprocess_counter()
    logger.debug("Processing patient input")
    if request.method == 'POST':
        form = PatientInputForm(request.POST)
        if form.is_valid():
            try:
                logger.debug(f"Form data: {form.cleaned_data}")
                patient_data = _format_patient_text(form.cleaned_data)
                if not patient_data.strip():
                    logger.warning("No valid patient data provided")
                    form.add_error(None, "Please provide valid patient data")
                    return await sync_to_async(render)(request, 'patient_input.html', {'form': form})

                logger.info(f"Patient data: {patient_data}")
                summary, conditions = await generate_patient_summary(patient_data)
                # Filter out empty or invalid conditions
                conditions = [cond for cond in conditions if cond.strip()]
                if not conditions:
                    logger.warning("No valid conditions identified from patient data")
                    form.add_error(None, "No specific conditions identified. Please provide detailed medical information.")
                    return await sync_to_async(render)(request, 'patient_input.html', {'form': form})

                non_negated_conditions = [
                    cond for cond in conditions
                    if await sync_to_async(is_not_negated)(cond, patient_data)
                ]
                logger.info(f"Non-negated conditions: {non_negated_conditions}")

                predefined_codes = form.cleaned_data.get('predefined_icd_codes', []) or form.cleaned_data.get('predefined_icd_code', [])

                pipeline_result = {}
                if settings.ICD_MATCHING_SETTINGS.get('USE_RAG_KAG', True):
                    pipeline = await RAGKAGPipeline.create()
                    pipeline_result = await pipeline.run(patient_data, predefined_icd_code=predefined_codes)
                else:
                    icd_matches = await _compute_icd_matches(non_negated_conditions, patient_data)
                    pipeline_result = {
                        'patient_data': patient_data,
                        'summary': summary,
                        'conditions': non_negated_conditions,
                        'icd_matches': icd_matches,
                        'all_kg_scores': {},  # Empty if not using RAG/KAG
                        'predefined_icd_titles': get_icd_title(predefined_codes),
                        'admission_id': None
                    }

                # Prepare admission data
                patient_data_json = {
                    'admission_type': form.cleaned_data.get('ADMISSION_TYPE'),
                    'admission_status': form.cleaned_data.get('ADMISSION_STATUS'),
                    'discharge_ward': form.cleaned_data.get('DISCHARGE_WARD'),
                    'icd_remarks_admission': form.cleaned_data.get('ICD_REMARKS_A'),
                    'icd_remarks_discharge': form.cleaned_data.get('ICD_REMARKS_D'),
                }
                
                # Determine predicted ICD code and accuracy
                predicted_icd_code = None
                prediction_accuracy = 0.0
                predicted_icd_codes = []
                
                if pipeline_result.get('icd_matches', {}).get('Bilateral Hearing Loss'):
                    top_match = pipeline_result['icd_matches']['Bilateral Hearing Loss'][0]
                    predicted_icd_code = top_match[0]  # e.g., 'H90.0'
                    prediction_accuracy = top_match[2]  # e.g., 100.0
                    predicted_icd_codes = [match[0] for match in pipeline_result['icd_matches']['Bilateral Hearing Loss']]
                

                # Create MedicalAdmissionDetails instance
                admission_data = {
                    'patient_data': json.dumps(patient_data_json),
                    'admission_date': form.cleaned_data.get('ADMISSION_DATE'),
                    'predicted_icd_code': predicted_icd_code,
                    'prediction_accuracy': prediction_accuracy,
                    'predicted_icd_codes': predicted_icd_codes,
                }
                logger.debug(f"Admission data: {admission_data}")
                admission = await sync_to_async(MedicalAdmissionDetails.objects.create)(**admission_data)
                admission_id = str(admission.id)
                pipeline_result['admission_id'] = admission_id

                cache_key = f"patient_result_{admission_id}_{uuid.uuid4().hex}"
                result_data = {
                    'patient_data': pipeline_result.get('patient_data', patient_data),
                    'summary': pipeline_result.get('summary', summary),
                    'conditions': pipeline_result.get('conditions', non_negated_conditions),
                    'icd_matches': pipeline_result.get('icd_matches', {}),
                    'all_kg_scores': pipeline_result.get('all_kg_scores', {}),
                    'predefined_icd_titles': pipeline_result.get('predefined_icd_titles', get_icd_title(predefined_codes)),
                    'admission_id': admission_id,
                }
                await sync_to_async(cache.set)(
                    cache_key, result_data, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600)
                )
                logger.info(f"Results cached with key: {cache_key}")

                reset_preprocess_counter()
                return await sync_to_async(render)(request, 'result.html', result_data)
            except Exception as e:
                logger.error(f"Error processing patient data: {e}", exc_info=True)
                form.add_error(None, f"Error processing data: {str(e)}")
                reset_preprocess_counter()
                return await sync_to_async(render)(request, 'patient_input.html', {'form': form})
        else:
            logger.debug(f"Form errors: {form.errors}")
            reset_preprocess_counter()
            return await sync_to_async(render)(request, 'patient_input.html', {'form': form})
    else:
        form = PatientInputForm()
        reset_preprocess_counter()
        return await sync_to_async(render)(request, 'patient_input.html', {'form': form})

async def result(request):
    """Display cached results for a given admission ID."""
    admission_id = request.GET.get('admission_id')
    if not admission_id:
        logger.warning("No admission_id provided for result view")
        return await sync_to_async(render)(request, 'patient_input.html', {
            'form': PatientInputForm(),
            'error': 'No admission ID provided'
        })

    cache_key_pattern = f"patient_result_{admission_id}_*"
    keys = await sync_to_async(cache.keys)(cache_key_pattern)
    if not keys:
        logger.warning(f"No cached results found for admission_id: {admission_id}")
        return await sync_to_async(render)(request, 'patient_input.html', {
            'form': PatientInputForm(),
            'error': 'No results found for the provided admission ID'
        })

    result_data = await sync_to_async(cache.get)(keys[0])
    if not result_data:
        logger.warning(f"No cached results found for admission_id: {admission_id}")
        return await sync_to_async(render)(request, 'patient_input.html', {
            'form': PatientInputForm(),
            'error': 'No results found for the provided admission ID'
        })

    logger.info(f"Retrieved cached results for admission_id: {admission_id}")
    return await sync_to_async(render)(request, 'result.html', result_data)

async def search_icd(request):
    """Search ICD codes by query string."""
    query = request.GET.get('query', '').strip()
    if not query:
        return JsonResponse({'results': []})

    try:
        results = await sync_to_async(
            lambda: list(
                ICDCategory.objects.filter(
                    Q(title__icontains=query) | Q(code__icontains=query)
                ).values('code', 'title')[:10]
            )
        )()
        response = [{'code': item['code'], 'title': item['title']} for item in results]
        logger.info(f"ICD search for query '{query}' returned {len(response)} results")
        return JsonResponse({'results': response})
    except Exception as e:
        logger.error(f"Error searching ICD codes for query '{query}': {e}")
        return JsonResponse({'error': str(e)}, status=500)

async def admission_details_view(request, admission_id):
    """Display details for a specific admission."""
    try:
        admission = await sync_to_async(get_object_or_404)(MedicalAdmissionDetails, id=admission_id)
        cache_key = f"patient_result_{admission_id}_{uuid.uuid4().hex}"
        result_data = await sync_to_async(cache.get)(cache_key)

        if not result_data:
            patient_data = _format_patient_text({
                'ICD_REMARKS_ADMISSION': json.loads(admission.patient_data).get('icd_remarks_admission', ''),
                'ICD_REMARKS_DISCHARGE': json.loads(admission.patient_data).get('icd_remarks_discharge', ''),
                'DISCHARGE_WARD': json.loads(admission.patient_data).get('discharge_ward', ''),
                'ADMISSION_TYPE': json.loads(admission.patient_data).get('admission_type', ''),
                'ADMISSION_DATE': admission.admission_date,
                'ADMISSION_STATUS': json.loads(admission.patient_data).get('admission_status', ''),
            })
            summary, conditions = await generate_patient_summary(patient_data)
            conditions = [cond for cond in conditions if cond.strip()]
            non_negated_conditions = [
                cond for cond in conditions
                if await sync_to_async(is_not_negated)(cond, patient_data)
            ]
            pipeline = await RAGKAGPipeline.create()
            pipeline_result = await pipeline.run(patient_data)
            predefined_icd_titles = [
                {
                    'code': code,
                    'title': await sync_to_async(get_icd_title)(code) or 'Unknown',
                    'is_relevant': False  # Simplified for this view; can be enhanced to check relevance
                }
                for code in admission.predicted_icd_codes or []
            ]
            result_data = {
                'patient_data': pipeline_result.get('patient_data', patient_data),
                'summary': pipeline_result.get('summary', summary),
                'conditions': pipeline_result.get('conditions', non_negated_conditions),
                'icd_matches': pipeline_result.get('icd_matches', {}),
                'all_kg_scores': pipeline_result.get('all_kg_scores', {}),
                'predefined_icd_titles': pipeline_result.get('predefined_icd_titles', predefined_icd_titles),
                'admission_id': str(admission_id),
            }
            await sync_to_async(cache.set)(
                cache_key, result_data, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600)
            )

        result_data['admission'] = admission
        logger.info(f"Retrieved admission details for admission_id: {admission_id}")
        return await sync_to_async(render)(request, 'admission_details.html', result_data)
    except Exception as e:
        logger.error(f"Error retrieving admission details for admission_id {admission_id}: {e}")
        return await sync_to_async(render)(request, 'patient_input.html', {
            'form': PatientInputForm(),
            'error': f'Error retrieving admission details: {e}'
        })

async def predict_icd_code_view(request):
    """Handle ICD code prediction view."""
    reset_preprocess_counter()
    logger.debug("Processing ICD prediction request")
    
    if request.method == 'POST':
        form = PatientInputForm(request.POST)
        if form.is_valid():
            try:
                logger.debug(f"Form data: {form.cleaned_data}")
                patient_data = _format_patient_text(form.cleaned_data)
                if not patient_data.strip():
                    logger.warning("No valid patient data provided")
                    form.add_error(None, "Please provide valid patient data")
                    return await sync_to_async(render)(request, 'patient_input.html', {'form': form})

                pipeline = await RAGKAGPipeline.create()
                predefined_codes = form.cleaned_data.get('predefined_icd_codes', []) or form.cleaned_data.get('predefined_icd_code', [])
                pipeline_result = await pipeline.run(patient_data, predefined_icd_code=predefined_codes)
                
                summary, conditions = await generate_patient_summary(patient_data)
                conditions = [cond for cond in conditions if cond.strip()]
                if not conditions:
                    logger.warning("No valid conditions identified from patient data")
                    form.add_error(None, "No specific conditions identified. Please provide detailed medical information.")
                    return await sync_to_async(render)(request, 'patient_input.html', {'form': form})

                non_negated_conditions = [
                    cond for cond in conditions
                    if await sync_to_async(is_not_negated)(cond, patient_data)
                ]
                
                result_data = {
                    'patient_data': pipeline_result.get('patient_data', patient_data),
                    'summary': pipeline_result.get('summary', summary),
                    'conditions': pipeline_result.get('conditions', non_negated_conditions),
                    'icd_matches': pipeline_result.get('icd_matches', {}),
                    'all_kg_scores': pipeline_result.get('all_kg_scores', {}),
                    'predefined_icd_titles': pipeline_result.get('predefined_icd_titles', get_icd_title(predefined_codes)),
                    'admission_id': pipeline_result.get('admission_id', None),
                }
                
                reset_preprocess_counter()
                return await sync_to_async(render)(request, 'result.html', result_data)
            except Exception as e:
                logger.error(f"Error predicting ICD code: {e}", exc_info=True)
                form.add_error(None, f"Error processing data: {e}")
                reset_preprocess_counter()
                return await sync_to_async(render)(request, 'patient_input.html', {'form': form})
        else:
            logger.debug(f"Form errors: {form.errors}")
            reset_preprocess_counter()
            return await sync_to_async(render)(request, 'patient_input.html', {'form': form})
    else:
        form = PatientInputForm()
        reset_preprocess_counter()
        return await sync_to_async(render)(request, 'patient_input.html', {'form': form})

def _format_patient_text(form_data: Dict) -> str:
    """Format patient data into a consistent string."""
    form_data = {k.lower(): v for k, v in form_data.items()}
    
    remarks_a = form_data.get('icd_remarks_a', form_data.get('icd_remarks_admission', '')).strip()
    remarks_d = form_data.get('icd_remarks_d', form_data.get('icd_remarks_discharge', '')).strip()
    admission_date = form_data.get('admission_date', '')
    admission_type = form_data.get('admission_type', '')
    admission_status = form_data.get('admission_status', '')
    discharge_ward = form_data.get('discharge_ward', '')
    
    patient_text = []
    if remarks_a:
        patient_text.append(f"A: {remarks_a}")
    if remarks_d:
        patient_text.append(f"D: {remarks_d}")
    if discharge_ward:
        patient_text.append(f"Discharge Ward: {discharge_ward}")
    if admission_type:
        patient_text.append(f"Admission Type: {admission_type}")
    if admission_date:
        patient_text.append(f"Admission Date: {admission_date}")
    if admission_status:
        patient_text.append(f"Admission Status: {admission_status}")
    
    return " ".join(patient_text)

async def _compute_icd_matches(conditions: List[str], patient_data: str) -> Dict[str, List[tuple]]:
    """Compute ICD matches without RAG/KAG pipeline."""
    icd_matches = {}
    for condition in conditions:
        try:
            matches = await sync_to_async(find_best_icd_match)([condition], patient_data)
            icd_matches[condition] = matches.get(condition, [])
        except Exception as e:
            logger.error(f"Error computing matches for {condition}: {e}")
            icd_matches[condition] = []
    return icd_matches