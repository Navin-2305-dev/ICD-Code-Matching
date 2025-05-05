# icd_matcher/tests/test_models.py
import os
import sys
import json
import pytest
import django

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'icd_matcher_project.settings')
django.setup()

from icd_matcher.models import ICDCategory, MedicalAdmissionDetails

@pytest.mark.django_db
class TestICDCategoryModel:
    def test_create_icd_category(self):
        category = ICDCategory.objects.create(
            code="N20.1",
            title="Calculus of ureter",
            inclusions=json.dumps(["ureteric calculus", "kidney stones"]),
            exclusions=json.dumps(["malposition"])
        )
        assert category.code == "N20.1"
        assert category.title == "Calculus of ureter"
        assert category.get_inclusions() == ["ureteric calculus", "kidney stones"]
        assert category.get_exclusions() == ["malposition"]
        assert str(category) == "N20.1 - Calculus of ureter"

    def test_empty_inclusions_exclusions(self):
        category = ICDCategory.objects.create(code="N20.0", title="Calculus of kidney")
        assert category.get_inclusions() == []
        assert category.get_exclusions() == []

@pytest.mark.django_db
class TestMedicalAdmissionDetailsModel:
    def test_create_medical_admission(self):
        admission = MedicalAdmissionDetails.objects.create(
            patient_data="History of bilateral ureteric calculus",
            admission_date="2025-04-28",
            predicted_icd_code="N20.1",
            prediction_accuracy=92.7,
            predicted_icd_codes=[
                {"code": "N20.1", "title": "Calculus of ureter", "confidence": 92.7},
                {"code": "N20.0", "title": "Calculus of kidney", "confidence": 86.3}
            ]
        )
        assert admission.patient_data == "History of bilateral ureteric calculus"
        assert admission.predicted_icd_code == "N20.1"
        assert admission.prediction_accuracy == 92.7
        assert len(admission.predicted_icd_codes) == 2
        assert admission.predicted_icd_codes[0]["code"] == "N20.1"
        assert str(admission).startswith("Admission on 2025-04-28: History of bilateral ureteric calculus")