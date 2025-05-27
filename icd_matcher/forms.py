from django import forms

class PatientInputForm(forms.Form):
    ADMISSION_DATE = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    ADMISSION_TYPE = forms.CharField(max_length=100)
    ADMISSION_STATUS = forms.CharField(max_length=100)
    DISCHARGE_WARD = forms.CharField(max_length=100)
    ICD_REMARKS_A = forms.CharField(widget=forms.Textarea, label="ICD Remarks on Admission")
    ICD_REMARKS_D = forms.CharField(widget=forms.Textarea, label="ICD Remarks on Discharge")
    predefined_icd_code = forms.CharField(max_length=10, required=False, label="Enter ICD code (predefined)")