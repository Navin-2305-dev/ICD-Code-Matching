from functools import cache
from http.client import HTTPException
from django.db import connection
from ninja import NinjaAPI, Schema
from typing import Dict, List, Optional
from datetime import date
from pydantic import Field, field_validator
import logging
from asgiref.sync import sync_to_async
from icd_matcher.utils.text_processing import generate_patient_summary, get_negation_cues, is_not_negated
from icd_matcher.utils.embeddings import find_best_icd_match
from icd_matcher.utils.text_processing import preprocess_text
from icd_matcher.models import ICDCategory, MedicalAdmissionDetails
from icd_matcher.utils.rag_kag_pipeline import RAGKAGPipeline
from icd_matcher.utils.db_utils import get_icd_title
from django.conf import settings
import uuid

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
    conditions: List[ConditionMatch]
    admission_id: Optional[int] = None

class ICDCodeSearchResponse(Schema):
    code: str
    title: str
    definition: Optional[str] = None
    parent_code: Optional[str] = None
    parent_title: Optional[str] = None

@api.post("/match-patient", response=PatientSummaryResponse, tags=["ICD Matching"])
async def match_patient_data(request, payload: PatientInput):
    """Match patient data to ICD codes and save to database."""
    try:
        patient_data = (
            f"A: {payload.icd_remarks_admission or ''} "
            f"D: {payload.icd_remarks_discharge or ''}"
        ).strip()
        if not patient_data:
            raise ValueError("No valid patient data provided")

        summary, conditions = await generate_patient_summary(patient_data)
        if not conditions or conditions == ['No conditions identified']:
            raise ValueError("No specific conditions identified")

        non_negated_conditions = [
            cond for cond in conditions
            if await sync_to_async(is_not_negated)(cond, patient_data)
        ]
        pipeline = await RAGKAGPipeline.create()
        icd_matches = pipeline.run(patient_data)
        condition_matches = [
            ConditionMatch(condition=cond, matches=[
                ICDMatch(code=code, title=await get_icd_title(code), confidence=score)
                for code, title, score in matches
            ])
            for cond, matches in icd_matches.items()
        ]

        admission = await sync_to_async(MedicalAdmissionDetails.objects.create)(
            patient_data=patient_data,
            admission_date=payload.admission_date,
            admission_type=payload.admission_type,
            admission_status=payload.admission_status,
            discharge_ward=payload.discharge_ward,
            predicted_icd_code=condition_matches[0].matches[0].code if condition_matches and condition_matches[0].matches else "",
            prediction_accuracy=condition_matches[0].matches[0].confidence if condition_matches and condition_matches[0].matches else 0.0,
            predicted_icd_codes=[{"code": m.code, "title": m.title, "confidence": m.confidence} for cm in condition_matches for m in cm.matches]
        )
        cache_key = f"icd_prediction_{admission.id}_{uuid.uuid4().hex}"
        await sync_to_async(cache.set)(
            cache_key, icd_matches, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600)
        )
        return PatientSummaryResponse(
            summary=summary,
            conditions=condition_matches,
            admission_id=admission.id
        )
    except Exception as e:
        logger.error(f"Error matching patient data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/match-text", response=PatientSummaryResponse, tags=["ICD Matching"])
async def match_medical_text(request, payload: ICDCodeQuery):
    """Match medical text to ICD codes without saving."""
    try:
        patient_data = payload.medical_text.strip()
        if not patient_data:
            raise ValueError("No valid medical text provided")

        summary, conditions = await generate_patient_summary(patient_data)
        if not conditions or conditions == ['No conditions identified']:
            raise ValueError("No specific conditions identified")

        non_negated_conditions = [
            cond for cond in conditions
            if await sync_to_async(is_not_negated)(cond, patient_data)
        ]
        pipeline = await RAGKAGPipeline.create()
        icd_matches = pipeline.run(patient_data)
        condition_matches = [
            ConditionMatch(condition=cond, matches=[
                ICDMatch(code=code, title=await get_icd_title(code), confidence=score)
                for code, title, score in matches
            ])
            for cond, matches in icd_matches.items()
        ]
        return PatientSummaryResponse(
            summary=summary,
            conditions=condition_matches,
            admission_id=None
        )
    except Exception as e:
        logger.error(f"Error matching medical text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/search-icd", response=List[ICDCodeSearchResponse], tags=["ICD Search"])
async def search_icd_codes(request, query: str, limit: int = 20):
    try:
        query = preprocess_text(query.strip())
        if not query:
            return []

        results = []
        with connection.cursor() as cursor:
            sql = "SELECT code, title FROM icd_fts WHERE title MATCH %s ORDER BY rank LIMIT %s"
            cursor.execute(sql, [f'"{query}"', limit])
            fts_results = cursor.fetchall()

        if fts_results:
            codes = [row[0] for row in fts_results]
            icd_entries = await sync_to_async(ICDCategory.objects.filter(code__in=codes).select_related('parent').all)()

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
        logger.error(f"Error searching ICD codes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/icd/{code}", response=ICDCodeSearchResponse, tags=["ICD Search"])
async def get_icd_code(request, code: str):
    try:
        icd_entry = await sync_to_async(ICDCategory.objects.filter(code=code).select_related('parent').first)()
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
        logger.error(f"Error fetching ICD code details: {e}")
        raise HTTPException(status_code=500, detail=str(e))