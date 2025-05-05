from django.db import connection
from ninja import NinjaAPI, Schema
from typing import Dict, List, Tuple, Optional
from datetime import date
from pydantic import Field, field_validator
import logging
from .utils.text_processing import generate_patient_summary, get_negation_cues, is_not_negated
from .utils.embeddings import (
    find_best_icd_match,
    preprocess_text
)
from .models import ICDCategory

logger = logging.getLogger(__name__)

api = NinjaAPI(
    title="ICD Code Matcher API",
    description="API for matching patient descriptions to ICD-10 codes",
    version="1.0.0"
)

class PatientInput(Schema):
    admission_date: date
    admission_type: str
    admission_status: str
    discharge_ward: str
    icd_remarks_admission: str = Field(..., description="ICD remarks on admission")
    icd_remarks_discharge: str = Field(..., description="ICD remarks on discharge")
    predefined_icd_codes: List[str] = Field(default=[], description="Predefined ICD codes")

class ICDCodeQuery(Schema):
    medical_text: str = Field(..., description="Patient medical description")
    existing_codes: List[str] = Field(default=[], description="Existing ICD codes")

class ICDMatch(Schema):
    code: Optional[str] = None
    title: Optional[str] = None
    confidence: float = Field(..., description="Confidence score (0-100)")

    @field_validator('confidence')
    def round_confidence(cls, v):
        return round(v, 1)

class ConditionMatch(Schema):
    condition: str
    matches: List[ICDMatch]

class PatientSummaryResponse(Schema):
    summary: str
    conditions: List[str]
    icd_matches: List[ConditionMatch]

class ICDCodeSearchResponse(Schema):
    code: str
    title: str
    definition: Optional[str] = None
    parent_code: Optional[str] = None
    parent_title: Optional[str] = None

@api.post("/match-patient", response=PatientSummaryResponse, tags=["ICD Matching"])
def match_patient(request, data: PatientInput):
    try:
        patient_text = (
            f"A: {data.icd_remarks_admission}\n"
            f"D: {data.icd_remarks_discharge} - {data.discharge_ward} - "
            f"Follow-up after {data.admission_type} admission on {data.admission_date} "
            f"with status {data.admission_status}"
        )

        corrected_text = preprocess_text(patient_text)
        summary, conditions = generate_patient_summary(corrected_text)
        negation_cues = get_negation_cues()
        non_negated_conditions = sorted(
            [cond for cond in conditions if is_not_negated(cond, corrected_text, negation_cues)],
            key=len, reverse=True
        )

        patient_codes = find_best_icd_match(
            non_negated_conditions,
            patient_text,
            data.predefined_icd_codes
        )

        icd_matches = []
        for condition, matches in patient_codes.items():
            condition_matches = []
            for code, title, score in matches:
                if code and score >= 60:
                    condition_matches.append(ICDMatch(code=code, title=get_icd_title(code), confidence=score))
                else:
                    condition_matches.append(ICDMatch(code=None, title=None, confidence=0))

            if condition_matches:
                icd_matches.append(ConditionMatch(condition=condition, matches=condition_matches))

        return PatientSummaryResponse(
            summary=summary,
            conditions=non_negated_conditions,
            icd_matches=icd_matches
        )
    except Exception as e:
        logger.error(f"Error processing patient data: {str(e)}")
        raise

@api.post("/match-text", response=List[ConditionMatch], tags=["ICD Matching"])
def match_medical_text(request, data: ICDCodeQuery):
    try:
        processed_text = preprocess_text(data.medical_text)
        summary, conditions = generate_patient_summary(processed_text)
        logger.info(f"Extracted conditions: {conditions}")
        negation_cues = get_negation_cues()
        non_negated_conditions = [
            cond for cond in conditions
            if is_not_negated(cond, processed_text, negation_cues)
        ]

        matches = find_best_icd_match(
            non_negated_conditions,
            processed_text,
            data.existing_codes
        )

        result = []
        for condition, code_matches in matches.items():
            condition_matches = []
            for code, title, confidence in code_matches:
                if code and confidence >= 60:
                    condition_matches.append(ICDMatch(code=code, title=get_icd_title(code), confidence=confidence))
                else:
                    condition_matches.append(ICDMatch(code=None, title=None, confidence=0))

            result.append(ConditionMatch(condition=condition, matches=condition_matches))

        return result
    except Exception as e:
        logger.error(f"Error matching medical text: {str(e)}")
        raise

@api.get("/search-icd", response=List[ICDCodeSearchResponse], tags=["ICD Search"])
def search_icd_codes(request, query: str, limit: int = 20):
    try:
        query = preprocess_text(query)
        results = []

        with connection.cursor() as cursor:
            sql = f"SELECT code, title FROM icd_fts WHERE title MATCH '\"{query}\"' ORDER BY rank LIMIT {limit}"
            cursor.execute(sql)
            fts_results = cursor.fetchall()

        if fts_results:
            codes = [row[0] for row in fts_results]
            icd_entries = ICDCategory.objects.filter(code__in=codes).select_related('parent')

            for entry in icd_entries:
                parent_code = entry.parent.code if entry.parent else None
                parent_title = entry.parent.title if entry.parent else None

                results.append(ICDCodeSearchResponse(
                    code=entry.code,
                    title=entry.title,
                    definition=entry.definition,
                    parent_code=parent_code,
                    parent_title=parent_title
                ))

        return results
    except Exception as e:
        logger.error(f"Error searching ICD codes: {str(e)}")
        raise

@api.get("/icd/{code}", response=ICDCodeSearchResponse, tags=["ICD Search"])
def get_icd_code(request, code: str):
    try:
        icd_entry = ICDCategory.objects.filter(code=code).select_related('parent').first()
        if not icd_entry:
            return api.create_response(
                request,
                {"message": f"ICD code {code} not found"},
                status=404
                
            )

        parent_code = icd_entry.parent.code if icd_entry.parent else None
        parent_title = icd_entry.parent.title if icd_entry.parent else None

        return ICDCodeSearchResponse(
            code=icd_entry.code,
            title=icd_entry.title,
            definition=icd_entry.definition,
            parent_code=parent_code,
            parent_title=parent_title
        )
    except Exception as e:
        logger.error(f"Error fetching ICD code details: {str(e)}")
        raise

def get_icd_title(code: str) -> str:
    try:
        icd_entry = ICDCategory.objects.filter(code=code).only('title').first()
        print(f"Code: {code}, Title: {icd_entry.title if icd_entry else 'Unknown title'}")
        title = icd_entry.title if icd_entry else "Unknown title"
        return title
    except Exception as e:
        logger.error(f"Error fetching title for code {code}: {e}")
        return "Unknown title"