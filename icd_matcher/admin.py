import logging
import csv
from django.contrib import admin
from django.http import HttpResponse
from icd_matcher.models import MedicalAdmissionDetails, ICDCategory
from icd_matcher.tasks import predict_icd_code
from icd_matcher.utils.exceptions import ICDPipelineError

logger = logging.getLogger(__name__)

@admin.register(ICDCategory)
class ICDCategoryAdmin(admin.ModelAdmin):
    """
    Admin interface for managing ICDCategory records.
    """
    list_display = ('code', 'title', 'parent')
    search_fields = ('code', 'title')
    list_filter = ('parent',)
    ordering = ('code',)
    actions = ['export_icd_categories']

    def export_icd_categories(self, request, queryset):
        """Export selected ICD categories to CSV."""
        try:
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="icd_categories.csv"'
            writer = csv.writer(response)
            writer.writerow(['Code', 'Title', 'Parent Code'])
            for obj in queryset:
                parent_code = obj.parent.code if obj.parent else ''
                writer.writerow([obj.code, obj.title, parent_code])
            logger.info(f"Exported {queryset.count()} ICD categories")
            return response
        except Exception as e:
            logger.error(f"Error exporting ICD categories: {e}")
            self.message_user(request, f"Error exporting: {e}", level='ERROR')
            raise ICDPipelineError(f"Error exporting ICD categories: {e}")
    export_icd_categories.short_description = "Export selected ICD categories to CSV"

@admin.register(MedicalAdmissionDetails)
class MedicalAdmissionDetailsAdmin(admin.ModelAdmin):
    """
    Admin interface for managing MedicalAdmissionDetails records.
    """
    list_display = ('admission_date', 'patient_data_short', 'top_icd_code', 'top_icd_accuracy', 'all_icd_codes')
    search_fields = ('patient_data', 'predicted_icd_code')
    list_filter = ('admission_date',)
    actions = ['reprocess_admissions', 'export_admissions']
    list_per_page = 50

    def patient_data_short(self, obj):
        """Display a shortened version of patient data."""
        return obj.patient_data[:50] + '...' if len(obj.patient_data) > 50 else obj.patient_data
    patient_data_short.short_description = 'Patient Data'

    def top_icd_code(self, obj):
        """Display the top predicted ICD code."""
        return obj.predicted_icd_code or 'N/A'
    top_icd_code.short_description = 'Top ICD Code'

    def top_icd_accuracy(self, obj):
        """Display the prediction accuracy."""
        return f"{obj.prediction_accuracy:.1f}%" if obj.prediction_accuracy else 'N/A'
    top_icd_accuracy.short_description = 'Accuracy'

    def all_icd_codes(self, obj):
        """Display all predicted ICD codes with confidence scores."""
        return '; '.join(
            f"{code['code']}: {code['title']} ({code['confidence']:.1f}%)"
            for code in obj.predicted_icd_codes
        ) or 'None'
    all_icd_codes.short_description = 'All ICD Codes'

    def reprocess_admissions(self, request, queryset):
        """Reprocess selected admissions for ICD prediction."""
        try:
            for admission in queryset:
                predict_icd_code.delay(admission.id)
            self.message_user(request, f"Reprocessing {queryset.count()} admissions.")
            logger.info(f"Triggered reprocessing for {queryset.count()} admissions")
        except Exception as e:
            logger.error(f"Error reprocessing admissions: {e}")
            self.message_user(request, f"Error reprocessing: {e}", level='ERROR')
            raise ICDPipelineError(f"Error reprocessing admissions: {e}")
    reprocess_admissions.short_description = "Reprocess selected admissions"

    def export_admissions(self, request, queryset):
        """Export selected admissions to CSV."""
        try:
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="admissions.csv"'
            writer = csv.writer(response)
            writer.writerow([
                'Admission Date', 'Patient Data', 'Top ICD Code', 'Accuracy', 'All ICD Codes'
            ])
            for obj in queryset:
                all_codes = '; '.join(
                    f"{code['code']} ({code['confidence']:.1f}%)" for code in obj.predicted_icd_codes
                )
                writer.writerow([
                    obj.admission_date,
                    obj.patient_data,
                    obj.predicted_icd_code or 'N/A',
                    f"{obj.prediction_accuracy:.1f}%" if obj.prediction_accuracy else 'N/A',
                    all_codes or 'None'
                ])
            logger.info(f"Exported {queryset.count()} admissions")
            return response
        except Exception as e:
            logger.error(f"Error exporting admissions: {e}")
            self.message_user(request, f"Error exporting: {e}", level='ERROR')
            raise ICDPipelineError(f"Error exporting admissions: {e}")
    export_admissions.short_description = "Export selected admissions to CSV"