from django.db import connection
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import logging
from .forms import PatientInputForm
from .utils import (
    generate_patient_summary,
    find_best_icd_match,
    preprocess_text,
    get_negation_cues,
    is_not_negated
)
from .models import ICDCategory

logger = logging.getLogger(__name__)

# Handles patient data
@require_http_methods(["GET", "POST"])
def patient_input(request):
    if request.method == 'POST':
        form = PatientInputForm(request.POST)
        if form.is_valid():
            try:
                patient_text = _format_patient_text(form.cleaned_data)
                corrected_text = preprocess_text(patient_text)

                # Generate summary and extract conditions
                summary, conditions = generate_patient_summary(corrected_text)
                logger.info(f"Extracted conditions: {conditions}")
                negation_cues = get_negation_cues()

                non_negated_conditions = sorted(
                    [cond for cond in conditions if is_not_negated(cond, corrected_text, negation_cues)],
                    key=len, reverse=True
                )

                # Handle predefined ICD codes
                predefined_icd, existing_codes = _process_predefined_codes(form.cleaned_data)

                # Compute best ICD matches
                computed_icd = _compute_icd_matches(non_negated_conditions, corrected_text, existing_codes)

                result = {
                    'summary': summary,
                    'predefined_icd': '\n'.join(predefined_icd),
                    'computed_icd': '\n'.join(
                        f"{item['code']}: {item['title']} ({item['percent']}%) - {', '.join(item['conditions'])}"
                        for item in computed_icd
                    )
                }

                return render(request, 'result.html', {'result': result})

            except Exception as e:
                logger.exception("Error processing patient data")
                return render(request, 'error.html', {'error': "An error occurred processing the patient data."})
        else:
            return render(request, 'input_form.html', {'form': form, 'errors': form.errors})

    return render(request, 'input_form.html', {'form': PatientInputForm()})

# Search for ICD codes using FTS
@require_http_methods(["GET"])
def search_icd(request):
    query = request.GET.get('q', '').strip()
    limit = int(request.GET.get('limit', 20))

    if not query:
        return JsonResponse({'error': 'Query parameter is required'}, status=400)

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT code, title FROM icd_fts WHERE title MATCH %s ORDER BY rank LIMIT %s",
                [f'"{query}"', limit]
            )
            fts_results = cursor.fetchall()

        results = _fetch_icd_entries_with_parent([row[0] for row in fts_results]) if fts_results else []

        return JsonResponse({'results': results})

    except Exception as e:
        logger.exception("Error searching ICD codes")
        return JsonResponse({'error': 'An error occurred during search'}, status=500)

# Result page
def result(request):
    
    return render(request, 'result.html')

# Fetch ICD entries along with their parent categories
def _fetch_icd_entries_with_parent(codes):
    entries = []
    for code in codes:
        entry = ICDCategory.objects.filter(code=code).first()
        if entry:
            parent = entry.parent_category.code if entry.parent_category else None
            entries.append({
                "code": entry.code,
                "title": entry.title,
                "parent_code": parent
            })
    return entries

# Format patient text
def _format_patient_text(data):
    return (
        f"A: {data['ICD_REMARKS_A']}\n"
        f"D: {data['ICD_REMARKS_D']} - {data['DISCHARGE_WARD']} - "
        f"Follow-up after {data['ADMISSION_TYPE']} admission on "
        f"{data['ADMISSION_DATE']} with status {data['ADMISSION_STATUS']}"
    )

def _process_predefined_codes(data):
    code = data.get('predefined_icd_code')
    codes = [code] if code else []
    icd_descriptions = []

    for code in codes:
        entry = ICDCategory.objects.filter(code=code).first()
        icd_descriptions.append(f"{code}: {entry.title if entry else 'Unknown title'}")

    return icd_descriptions or ["No predefined ICD codes"], codes

def _compute_icd_matches(conditions, text, existing_codes):
    matches = find_best_icd_match(conditions, text, existing_codes)
    icd_to_conditions = {}
    icd_to_score = {}
    icd_to_title = {}

    for cond, code_scores in matches.items():
        for code, title, score in code_scores:
            if code and score >= 60:
                icd_to_conditions.setdefault(code, []).append(cond)
                icd_to_score[code] = max(icd_to_score.get(code, 0), score)
                icd_to_title[code] = title

    computed_icd = []
    for code, conds in icd_to_conditions.items():
        computed_icd.append({
            "code": code,
            "title": icd_to_title.get(code, "Unknown title"),
            "percent": round(icd_to_score[code], 1),
            "conditions": conds
        })

    return computed_icd or [{"code": None, "title": "No matching ICD codes", "percent": 0.0, "conditions": []}]