from django.contrib import admin
from icd_matcher.models import ICDCategory, MedicalAdmissionDetails
from django.utils.html import format_html

@admin.register(ICDCategory)
class ICDCategoryAdmin(admin.ModelAdmin):
    list_display = ('code', 'title', 'parent')
    search_fields = ('code', 'title', 'inclusions', 'exclusions')
    list_filter = ('parent',)

@admin.register(MedicalAdmissionDetails)
class MedicalAdmissionDetailsAdmin(admin.ModelAdmin):
    list_display = ('admission_date', 'patient_data_short', 'top_icd_code', 'top_icd_accuracy', 'all_icd_codes')
    search_fields = ('patient_data', 'predicted_icd_code')
    list_filter = ('admission_date',)
    readonly_fields = ('predicted_icd_code', 'prediction_accuracy', 'predicted_icd_codes', 'created_at', 'updated_at')
    
    def patient_data_short(self, obj):
        return obj.patient_data[:50] + '...' if len(obj.patient_data) > 50 else obj.patient_data
    patient_data_short.short_description = 'Patient Data'
    
    def top_icd_code(self, obj):
        return obj.predicted_icd_code or 'N/A'
    top_icd_code.short_description = 'Top ICD Code'
    
    def top_icd_accuracy(self, obj):
        return f"{obj.prediction_accuracy:.1f}%" if obj.prediction_accuracy else 'N/A'
    top_icd_accuracy.short_description = 'Top Confidence'
    
    def all_icd_codes(self, obj):
        if not obj.predicted_icd_codes:
            return 'No matches'
        formatted_codes = [
            f"{match['code']}: {match['title']} ({match['confidence']}%)"
            for match in obj.predicted_icd_codes
        ]
        # Truncate to 3 codes for display, with a "more" indicator
        if len(formatted_codes) > 3:
            return format_html('{}; <i>...{} more</i>', '; '.join(formatted_codes[:3]), len(formatted_codes) - 3)
        return '; '.join(formatted_codes)
    all_icd_codes.short_description = 'All ICD Codes'