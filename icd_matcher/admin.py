import logging
import csv
import json
from typing import AsyncIterable
from django.contrib import admin
from django.http import StreamingHttpResponse
from django.core.cache import cache
from django.conf import settings
from asgiref.sync import sync_to_async
from icd_matcher.models import MedicalAdmissionDetails, ICDCategory
from icd_matcher.tasks import predict_icd_code
from icd_matcher.utils.exceptions import ICDPipelineError
from celery import group
import hashlib

logger = logging.getLogger(__name__)

def stream_csv(data: AsyncIterable[list]) -> AsyncIterable[bytes]:
    """Stream CSV rows as bytes for StreamingHttpResponse."""
    async def _stream():
        writer = csv.writer(_LineBuffer())
        async for row in data:
            writer.writerow(row)
            yield writer.stream.getvalue().encode('utf-8')
            writer.stream.reset()
    return _stream()

class _LineBuffer:
    """Helper class to mimic file-like object for csv.writer."""
    def __init__(self):
        self.data = []
    
    def write(self, value):
        self.data.append(value)
    
    def getvalue(self):
        return ''.join(self.data)
    
    def reset(self):
        self.data.clear()

@admin.register(ICDCategory)
class ICDCategoryAdmin(admin.ModelAdmin):
    list_display = ('code', 'title', 'parent')
    search_fields = ('code', 'title')
    list_filter = ('parent',)
    ordering = ('code',)
    actions = ['export_icd_categories']

    async def export_icd_categories(self, request, queryset):
        """Export selected ICD categories to CSV asynchronously."""
        try:
            cache_key = f"icd_categories_export_{hashlib.sha256(str([obj.id for obj in queryset]).encode()).hexdigest()}_v1"
            cached_csv = await sync_to_async(cache.get)(cache_key)
            if cached_csv:
                logger.debug(f"Cache hit for ICD categories export: {cache_key}")
                async def cached_stream():
                    yield cached_csv.encode('utf-8')
                response = StreamingHttpResponse(
                    cached_stream(),
                    content_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="icd_categories.csv"'}
                )
                return response

            async def generate_rows():
                yield ['Code', 'Title', 'Parent Code']
                async for obj in sync_to_async(queryset.iterator)():
                    parent_code = obj.parent.code if obj.parent else ''
                    yield [obj.code, obj.title, parent_code]

            csv_content = b''.join([line async for line in stream_csv(generate_rows())])
            await sync_to_async(cache.set)(
                cache_key, csv_content.decode('utf-8'),
                timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 86400)
            )
            logger.info(f"Exported {await sync_to_async(queryset.count)()} ICD categories")

            response = StreamingHttpResponse(
                stream_csv(generate_rows()),
                content_type='text/csv',
                headers={'Content-Disposition': 'attachment; filename="icd_categories.csv"'}
            )
            return response
        except ICDPipelineError as e:
            logger.error(f"ICDPipelineError exporting ICD categories: {e}", exc_info=True)
            self.message_user(request, f"Failed to export categories: {str(e)}", level='ERROR')
            raise
        except Exception as e:
            logger.error(f"Unexpected error exporting ICD categories: {e}", exc_info=True)
            self.message_user(request, f"Failed to export categories: {str(e)}", level='ERROR')
            raise ICDPipelineError(f"Error exporting ICD categories: {e}")

    export_icd_categories.short_description = "Export selected ICD categories to CSV"

@admin.register(MedicalAdmissionDetails)
class MedicalAdmissionDetailsAdmin(admin.ModelAdmin):
    list_display = ('admission_date', 'patient_data_short', 'top_icd_code', 'top_icd_accuracy', 'all_icd_codes')
    search_fields = ('patient_data', 'predicted_icd_code')
    list_filter = ('admission_date',)
    actions = ['reprocess_admissions', 'export_admissions']
    list_per_page = 50
    readonly_fields = ('predicted_icd_code', 'prediction_accuracy', 'predicted_icd_codes')

    def patient_data_short(self, obj):
        """Display truncated patient data."""
        return obj.patient_data[:50] + '...' if len(obj.patient_data) > 50 else obj.patient_data
    patient_data_short.short_description = 'Patient Data'

    def top_icd_code(self, obj):
        """Display top predicted ICD code."""
        return obj.predicted_icd_code or 'N/A'
    top_icd_code.short_description = 'Top ICD Code'

    def top_icd_accuracy(self, obj):
        """Display prediction accuracy."""
        return f"{obj.prediction_accuracy:.1f}%" if obj.prediction_accuracy is not None else 'N/A'
    top_icd_accuracy.short_description = 'Accuracy'

    def all_icd_codes(self, obj):
        """Display all predicted ICD codes with titles and confidence."""
        try:
            codes = obj.predicted_icd_codes
            if not codes:
                return "No codes predicted"
            if not isinstance(codes, list):
                logger.warning(f"predicted_icd_codes is not a list for admission {obj.id}: {type(codes)}")
                return "Invalid code format"
            return '; '.join(
                f"{code['code']}: {code['title']} ({code['confidence']:.1f}%)"
                for code in codes
                if isinstance(code, dict) and all(k in code for k in ['code', 'title', 'confidence'])
            ) or "No valid codes"
        except Exception as e:
            logger.error(f"Error formatting all_icd_codes for admission {obj.id}: {e}", exc_info=True)
            return "Error retrieving codes"

    all_icd_codes.short_description = 'All Predicted ICD Codes'

    async def reprocess_admissions(self, request, queryset):
        """Reprocess selected admissions asynchronously in batches."""
        try:
            batch_size = settings.ICD_MATCHING_SETTINGS.get('BATCH_SIZE', 32)
            admission_ids = [obj.id async for obj in sync_to_async(queryset.values('id').iterator)()]
            count = len(admission_ids)
            logger.info(f"Triggering reprocessing for {count} admissions")

            # Batch Celery tasks
            for i in range(0, count, batch_size):
                batch_ids = admission_ids[i:i + batch_size]
                tasks = [
                    predict_icd_code.signature(
                        (ad_id,),
                        queue='high_priority',
                        retry=True,
                        retry_policy={
                            'max_retries': 3,
                            'interval_start': 30,
                            'interval_step': 60,
                            'interval_max': 300
                        }
                    )
                    for ad_id in batch_ids
                ]
                await sync_to_async(group(*tasks).apply_async)()

            self.message_user(request, f"Triggered reprocessing for {count} admissions.")
            logger.info(f"Scheduled {count} admissions for reprocessing")
        except celery.exceptions.MaxRetriesExceededError as e:
            logger.error(f"Max retries exceeded during reprocessing: {e}", exc_info=True)
            self.message_user(request, f"Some tasks failed due to max retries: {str(e)}", level='ERROR')
        except Exception as e:
            logger.error(f"Error reprocessing admissions: {e}", exc_info=True)
            self.message_user(request, f"Failed to reprocess: {str(e)}", level='ERROR')
            raise ICDPipelineError(f"Error reprocessing admissions: {e}")

    reprocess_admissions.short_description = "Reprocess selected admissions"

    async def export_admissions(self, request, queryset):
        """Export selected admissions to CSV asynchronously."""
        try:
            cache_key = f"admissions_export_{hashlib.sha256(str([obj.id for obj in queryset]).encode()).hexdigest()}_v1"
            cached_csv = await sync_to_async(cache.get)(cache_key)
            if cached_csv:
                logger.debug(f"Cache hit for admissions export: {cache_key}")
                async def cached_stream():
                    yield cached_csv.encode('utf-8')
                response = StreamingHttpResponse(
                    cached_stream(),
                    content_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="admissions.csv"'}
                )
                return response

            async def generate_rows():
                yield ['Admission Date', 'Patient Data', 'Top ICD Code', 'Accuracy', 'All ICD Codes']
                async for obj in sync_to_async(queryset.iterator)():
                    codes = obj.predicted_icd_codes or []
                    if not isinstance(codes, list):
                        logger.warning(f"Invalid predicted_icd_codes for admission {obj.id}: {codes}")
                        codes = []
                    all_codes = '; '.join(
                        f"{code['code']} ({code['confidence']:.1f}%)"
                        for code in codes
                        if isinstance(code, dict) and all(k in code for k in ['code', 'confidence'])
                    ) if codes else 'None'
                    yield [
                        obj.admission_date,
                        obj.patient_data,
                        obj.predicted_icd_code or 'N/A',
                        f"{obj.prediction_accuracy:.1f}%" if obj.prediction_accuracy is not None else 'N/A',
                        all_codes
                    ]

            csv_content = b''.join([line async for line in stream_csv(generate_rows())])
            await sync_to_async(cache.set)(
                cache_key, csv_content.decode('utf-8'),
                timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 86400)
            )
            logger.info(f"Exported {await sync_to_async(queryset.count)()} admissions")

            response = StreamingHttpResponse(
                stream_csv(generate_rows()),
                content_type='text/csv',
                headers={'Content-Disposition': 'attachment; filename="admissions.csv"'}
            )
            return response
        except ICDPipelineError as e:
            logger.error(f"ICDPipelineError exporting admissions: {e}", exc_info=True)
            self.message_user(request, f"Failed to export: {str(e)}", level='ERROR')
            raise
        except Exception as e:
            logger.error(f"Unexpected error exporting admissions: {e}", exc_info=True)
            self.message_user(request, f"Failed to export: {str(e)}", level='ERROR')
            raise ICDPipelineError(f"Error exporting admissions: {e}")

    export_admissions.short_description = "Export selected admissions to CSV"