from django.contrib import admin
from icd_matcher.models import MedicalAdmissionDetails, ICDCategory
from icd_matcher.tasks import predict_icd_code

@admin.register(ICDCategory)
class ICDCategoryAdmin(admin.ModelAdmin):
    list_display = ('code', 'title', 'parent')
    search_fields = ('code', 'title')
    list_filter = ('parent',)
    ordering = ('code',)

@admin.register(MedicalAdmissionDetails)
class MedicalAdmissionDetailsAdmin(admin.ModelAdmin):
    list_display = ('admission_date', 'patient_data_short', 'top_icd_code', 'top_icd_accuracy', 'all_icd_codes')
    search_fields = ('patient_data', 'predicted_icd_code')
    list_filter = ('admission_date',)
    actions = ['reprocess_admissions']
    list_per_page = 50

    def patient_data_short(self, obj):
        return obj.patient_data[:50] + '...' if len(obj.patient_data) > 50 else obj.patient_data
    patient_data_short.short_description = 'Patient Data'

    def top_icd_code(self, obj):
        return obj.predicted_icd_code or 'N/A'
    top_icd_code.short_description = 'Top ICD Code'

    def top_icd_accuracy(self, obj):
        return f"{obj.prediction_accuracy:.1f}%" if obj.prediction_accuracy else 'N/A'
    top_icd_accuracy.short_description = 'Accuracy'

    def all_icd_codes(self, obj):
        return '; '.join(
            f"{code['code']}: {code['title']} ({code['confidence']:.1f}%)"
            for code in obj.predicted_icd_codes
        ) or 'None'
    all_icd_codes.short_description = 'All ICD Codes'

    def reprocess_admissions(self, request, queryset):
        for admission in queryset:
            predict_icd_code.delay(admission.id)
        self.message_user(request, f"Reprocessing {queryset.count()} admissions.")
    reprocess_admissions.short_description = "Reprocess selected admissions"