import hashlib
import logging
import json
from typing import List, Dict, Tuple
from django.shortcuts import render, get_object_or_404
from django.core.cache import cache
from django.conf import settings
from django.http import JsonResponse
from django.db.models import Q
from icd_matcher.forms import PatientInputForm
from icd_matcher.models import MedicalAdmissionDetails, ICDCategory
from icd_matcher.utils.embeddings import find_best_icd_match
from icd_matcher.utils.text_processing import generate_patient_summary, is_not_negated, reset_preprocess_counter
from icd_matcher.utils.db_utils import get_icd_title
from icd_matcher.utils.rag_kag_pipeline import RAGKAGPipeline
from asgiref.sync import async_to_sync

logger = logging.getLogger(__name__)

@async_to_sync
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
                    return render(request, 'patient_input.html', {'form': form})

                logger.info(f"Patient data: {patient_data[:200]}")
                summary, conditions = await generate_patient_summary(patient_data)
                conditions = [cond for cond in conditions if cond.strip() and cond != "No conditions identified" and cond != "Surgical Admission"]
                if not conditions:
                    logger.warning("No valid conditions identified from patient data")
                    form.add_error(None, "No specific conditions identified. Please provide detailed medical information.")
                    return render(request, 'patient_input.html', {'form': form})

                non_negated_conditions = [
                    cond for cond in conditions
                    if await is_not_negated(cond, patient_data)
                ]
                logger.info(f"Non-negated conditions: {non_negated_conditions}")

                predefined_codes = form.cleaned_data.get('predefined_icd_codes', []) or form.cleaned_data.get('predefined_icd_code', [])
                if isinstance(predefined_codes, str):
                    predefined_codes = [predefined_codes]

                pipeline_result = {}
                if settings.ICD_MATCHING_SETTINGS.get('USE_RAG_KAG', True):
                    pipeline = await RAGKAGPipeline.create()
                    pipeline_result = await pipeline.run(patient_data, predefined_codes[0] if predefined_codes else None)
                else:
                    icd_matches = await _compute_icd_matches(non_negated_conditions, patient_data)
                    pipeline_result = {
                        'patient_data': patient_data,
                        'summary': summary,
                        'conditions': non_negated_conditions,
                        'condition_details': {},
                        'icd_matches': icd_matches,
                        'all_kg_scores': {},
                        'predefined_icd_titles': [
                            {
                                'code': code,
                                'title': await get_icd_title(code) or 'Unknown',
                                'is_relevant': False
                            } for code in predefined_codes
                        ],
                        'admission_id': None
                    }

                # Format icd_matches and all_kg_scores for template
                formatted_icd_matches = {}
                for condition, matches in pipeline_result.get('icd_matches', {}).items():
                    formatted_icd_matches[condition] = [
                        {'code': match['code'], 'title': match['title'], 'similarity': round(float(match['similarity']), 2)}
                        for match in matches
                    ] if matches else [{'code': '', 'title': f'No matches found for {condition}', 'similarity': 0}]

                formatted_kg_scores = {}
                for condition, scores in pipeline_result.get('all_kg_scores', {}).items():
                    formatted_kg_scores[condition] = [
                        {'code': score['code'], 'title': score['title'], 'similarity': round(float(score['similarity']), 2)}
                        for score in scores
                    ] if scores else [{'code': '', 'title': f'No knowledge graph scores for {condition}', 'similarity': 0}]

                # Prepare admission data
                patient_data_json = {
                    'admission_type': form.cleaned_data.get('ADMISSION_TYPE', ''),
                    'admission_status': form.cleaned_data.get('ADMISSION_STATUS', ''),
                    'discharge_ward': form.cleaned_data.get('DISCHARGE_WARD', ''),
                    'icd_remarks_admission': form.cleaned_data.get('ICD_REMARKS_A', ''),
                    'icd_remarks_discharge': form.cleaned_data.get('ICD_REMARKS_D', ''),
                }

                # Determine predicted ICD code and accuracy
                predicted_icd_code = None
                prediction_accuracy = 0.0
                predicted_icd_codes = []
                icd_matches = pipeline_result.get('icd_matches', {})
                if icd_matches:
                    top_score = 0.0
                    for condition, matches in icd_matches.items():
                        for match in matches:
                            score = float(match['similarity'])
                            predicted_icd_codes.append({'code': match['code'], 'title': match['title'], 'confidence': score})
                            if score > top_score:
                                top_score = score
                                predicted_icd_code = match['code']
                                prediction_accuracy = score

                # Create MedicalAdmissionDetails instance
                admission_data = {
                    'patient_data': json.dumps(patient_data_json),
                    'admission_date': form.cleaned_data.get('ADMISSION_DATE'),
                    'predicted_icd_code': predicted_icd_code,
                    'prediction_accuracy': prediction_accuracy,
                    'predicted_icd_codes': predicted_icd_codes,
                }
                logger.debug(f"Admission data: {admission_data}")
                admission = await MedicalAdmissionDetails.objects.acreate(**admission_data)
                admission_id = str(admission.id)
                pipeline_result['admission_id'] = admission_id

                # Cache results
                data_hash = hashlib.sha256(json.dumps(pipeline_result, sort_keys=True).encode()).hexdigest()
                cache_key = f"patient_result_{admission_id}_{data_hash}_v1"
                result_data = {
                    'patient_data': pipeline_result.get('patient_data', patient_data),
                    'summary': pipeline_result.get('summary', summary),
                    'conditions': pipeline_result.get('conditions', non_negated_conditions),
                    'condition_details': pipeline_result.get('condition_details', {}),
                    'icd_matches': formatted_icd_matches,
                    'all_kg_scores': formatted_kg_scores,
                    'predefined_icd_titles': pipeline_result.get('predefined_icd_titles', []),
                    'admission_id': admission_id,
                }
                cache.set(
                    cache_key, result_data, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600)
                )
                logger.info(f"Results cached with key: {cache_key}")

                reset_preprocess_counter()
                return render(request, 'result.html', result_data)
            except Exception as e:
                logger.error(f"Error processing patient data: {str(e)}", exc_info=True)
                form.add_error(None, f"Error processing data: {str(e)}")
                reset_preprocess_counter()
                return render(request, 'patient_input.html', {'form': form})
        else:
            logger.debug(f"Form errors: {form.errors}")
            reset_preprocess_counter()
            return render(request, 'patient_input.html', {'form': form})
    else:
        form = PatientInputForm()
        reset_preprocess_counter()
        return render(request, 'patient_input.html', {'form': form})

@async_to_sync
async def result(request):
    """Display cached results for a given admission ID."""
    admission_id = request.GET.get('admission_id')
    if not admission_id:
        logger.warning("No admission_id provided for result view")
        return render(request, 'patient_input.html', {
            'form': PatientInputForm(),
            'error': 'No admission ID provided'
        })

    cache_key_pattern = f"patient_result_{admission_id}_*_v1"
    keys = cache.keys(cache_key_pattern)
    if not keys:
        logger.warning(f"No cached results found for admission_id: {admission_id}")
        return render(request, 'patient_input.html', {
            'form': PatientInputForm(),
            'error': 'No results found for the provided admission ID'
        })

    result_data = cache.get(keys[0])
    if not result_data:
        logger.warning(f"No cached results found for admission_id: {admission_id}")
        return render(request, 'patient_input.html', {
            'form': PatientInputForm(),
            'error': 'No results found for the provided admission ID'
        })

    logger.info(f"Retrieved cached results for admission_id: {admission_id}")
    return render(request, 'result.html', result_data)

@async_to_sync
async def search_icd(request):
    """Search ICD codes by query string."""
    query = request.GET.get('query', '').strip()
    if not query:
        return JsonResponse({'results': []})

    try:
        results = await ICDCategory.objects.filter(
            Q(title__icontains=query) | Q(code__icontains=query)
        ).values('code', 'title')
        response = [{'code': item['code'], 'title': item['title']} for item in results]
        logger.info(f"ICD search for query '{query}' returned {len(response)} results")
        return JsonResponse({'results': response})
    except Exception as e:
        logger.error(f"Error searching ICD codes for query '{query}': {str(e)}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)

@async_to_sync
async def admission_details_view(request, admission_id):
    """Display details for a specific admission."""
    try:
        admission = await get_object_or_404(MedicalAdmissionDetails, id=admission_id)
        data_hash = hashlib.sha256(admission.patient_data.encode()).hexdigest()
        cache_key = f"patient_result_{admission_id}_{data_hash}_v1"
        result_data = cache.get(cache_key)

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
            conditions = [cond for cond in conditions if cond.strip() and cond != "No conditions identified" and cond != "Surgical Admission"]
            non_negated_conditions = [
                cond for cond in conditions
                if await is_not_negated(cond, patient_data)
            ]
            pipeline = await RAGKAGPipeline.create()
            pipeline_result = await pipeline.run(patient_data)
            formatted_icd_matches = {}
            for condition, matches in pipeline_result.get('icd_matches', {}).items():
                formatted_icd_matches[condition] = [
                    {'code': match['code'], 'title': match['title'], 'similarity': round(float(match['similarity']), 2)}
                    for match in matches
                ] if matches else [{'code': '', 'title': f'No matches found for {condition}', 'similarity': 0}]

            formatted_kg_scores = {}
            for condition, scores in pipeline_result.get('all_kg_scores', {}).items():
                formatted_kg_scores[condition] = [
                    {'code': score['code'], 'title': score['title'], 'similarity': round(float(score['similarity']), 2)}
                    for score in scores
                ] if scores else [{'code': '', 'title': f'No knowledge graph scores for {condition}', 'similarity': 0}]

            predefined_icd_titles = [
                {
                    'code': code.get('code', ''),
                    'title': await get_icd_title(code.get('code', '')) or 'Unknown',
                    'is_relevant': any(
                        code.get('code', '') in [m['code'] for m in matches]
                        for matches in pipeline_result.get('icd_matches', {}).values()
                    )
                }
                for code in admission.predicted_icd_codes or []
            ]
            result_data = {
                'patient_data': pipeline_result.get('patient_data', patient_data),
                'summary': pipeline_result.get('summary', summary),
                'conditions': pipeline_result.get('conditions', non_negated_conditions),
                'condition_details': pipeline_result.get('condition_details', {}),
                'icd_matches': formatted_icd_matches,
                'all_kg_scores': formatted_kg_scores,
                'predefined_icd_titles': predefined_icd_titles,
                'admission_id': str(admission_id),
            }
            cache.set(
                cache_key, result_data, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600)
            )

        result_data['admission'] = admission
        logger.info(f"Retrieved admission details for admission_id: {admission_id}")
        return render(request, 'admission_details.html', result_data)
    except Exception as e:
        logger.error(f"Error retrieving admission details for admission_id {admission_id}: {str(e)}", exc_info=True)
        return render(request, 'patient_input.html', {
            'form': PatientInputForm(),
            'error': f'Error retrieving admission details: {str(e)}'
        })

@async_to_sync
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
                    return render(request, 'patient_input.html', {'form': form})

                pipeline = await RAGKAGPipeline.create()
                predefined_codes = form.cleaned_data.get('predefined_icd_codes', []) or form.cleaned_data.get('predefined_icd_code', [])
                if isinstance(predefined_codes, str):
                    predefined_codes = [predefined_codes]
                pipeline_result = await pipeline.run(patient_data, predefined_codes[0] if predefined_codes else None)
                
                summary, conditions = await generate_patient_summary(patient_data)
                conditions = [cond for cond in conditions if cond.strip() and cond != "No conditions identified" and cond != "Surgical Admission"]
                if not conditions:
                    logger.warning("No valid conditions identified from patient data")
                    form.add_error(None, "No specific conditions identified. Please provide detailed medical information.")
                    return render(request, 'patient_input.html', {'form': form})

                non_negated_conditions = [
                    cond for cond in conditions
                    if await is_not_negated(cond, patient_data)
                ]
                
                formatted_icd_matches = {}
                for condition, matches in pipeline_result.get('icd_matches', {}).items():
                    formatted_icd_matches[condition] = [
                        {'code': match['code'], 'title': match['title'], 'similarity': round(float(match['similarity']), 2)}
                        for match in matches
                    ] if matches else [{'code': '', 'title': f'No matches found for {condition}', 'similarity': 0}]

                formatted_kg_scores = {}
                for condition, scores in pipeline_result.get('all_kg_scores', {}).items():
                    formatted_kg_scores[condition] = [
                        {'code': score['code'], 'title': score['title'], 'similarity': round(float(score['similarity']), 2)}
                        for score in scores
                    ] if scores else [{'code': '', 'title': f'No knowledge graph scores for {condition}', 'similarity': 0}]
                
                result_data = {
                    'patient_data': pipeline_result.get('patient_data', patient_data),
                    'summary': pipeline_result.get('summary', summary),
                    'conditions': pipeline_result.get('conditions', non_negated_conditions),
                    'condition_details': pipeline_result.get('condition_details', {}),
                    'icd_matches': formatted_icd_matches,
                    'all_kg_scores': formatted_kg_scores,
                    'predefined_icd_titles': pipeline_result.get('predefined_icd_titles', []),
                    'admission_id': pipeline_result.get('admission_id', None),
                }
                
                reset_preprocess_counter()
                return render(request, 'result.html', result_data)
            except Exception as e:
                logger.error(f"Error predicting ICD code: {str(e)}", exc_info=True)
                form.add_error(None, f"Error processing data: {str(e)}")
                reset_preprocess_counter()
                return render(request, 'patient_input.html', {'form': form})
        else:
            logger.debug(f"Form errors: {form.errors}")
            reset_preprocess_counter()
            return render(request, 'patient_input.html', {'form': form})
    else:
        form = PatientInputForm()
        reset_preprocess_counter()
        return render(request, 'patient_input.html', {'form': form})

def _format_patient_text(form_data: Dict) -> str:
    """Format patient data into a consistent string."""
    form_data_lower = {k.lower(): v for k, v in form_data.items()}
    
    remarks_a = form_data_lower.get('icd_remarks_a', form_data_lower.get('icd_remarks_admission', '')).strip()
    remarks_d = form_data_lower.get('icd_remarks_d', form_data_lower.get('icd_remarks_discharge', '')).strip()
    admission_date = form_data_lower.get('admission_date', '')
    admission_type = form_data_lower.get('admission_type', '')
    admission_status = form_data_lower.get('admission_status', '')
    discharge_ward = form_data_lower.get('discharge_ward', '')
    
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
        patient_text.append(str(admission_date))
    if admission_status:
        patient_text.append(f"Admission Status: {admission_status}")
    
    return " ".join(patient_text)

async def _compute_icd_matches(conditions: List[str], patient_data: str) -> Dict[str, List[Tuple[str, str, float]]]:
    """Compute ICD matches without RAG/KAG pipeline."""
    icd_matches = {}
    for condition in conditions:
        try:
            matches = await find_best_icd_match([condition], patient_data)
            icd_matches[condition] = [
                {'code': code, 'title': title, 'similarity': float(similarity)}
                for code, title, similarity in matches.get(condition, [])
            ]
            logger.debug(f"Computed matches for {condition}: {icd_matches[condition]}")
        except Exception as e:
            logger.error(f"Error computing ICD matches for condition '{condition}': {str(e)}")
            icd_matches[condition] = []
    return icd_matches