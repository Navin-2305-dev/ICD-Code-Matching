# icd_matcher/tests/test_tasks.py
import os
import sys
import pytest
import django

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'icd_matcher_project.settings')
django.setup()

from django.test import TestCase
from icd_matcher.models import MedicalAdmissionDetails
from icd_matcher.tasks import predict_icd_code
from unittest.mock import patch

@pytest.mark.django_db
class TestPredictICDCodeTask(TestCase):
    def setUp(self):
        self.admission = MedicalAdmissionDetails.objects.create(
            patient_data="History of bilateral ureteric calculus",
            admission_date="2025-04-28"
        )

    @patch("icd_matcher.tasks.generate_patient_summary")
    @patch("icd_matcher.tasks.find_best_icd_match")
    def test_predict_icd_code_task(self, mock_find_best_icd_match, mock_generate_patient_summary):
        # Mock dependencies
        mock_generate_patient_summary.return_value = (
            "Ureteric calculus detected",
            ["ureteric calculus"]
        )
        mock_find_best_icd_match.return_value = {
            "ureteric calculus": [
                ("N20.1", "Calculus of ureter", 92.7),
                ("N20.0", "Calculus of kidney", 86.3)
            ]
        }

        # Run task synchronously
        predict_icd_code(self.admission.id)

        # Refresh admission
        self.admission.refresh_from_db()

        # Assertions
        assert self.admission.predicted_icd_code == "N20.1"
        assert self.admission.prediction_accuracy == 92.7
        assert len(self.admission.predicted_icd_codes) == 2
        assert self.admission.predicted_icd_codes[0] == {
            "code": "N20.1",
            "title": "Calculus of ureter",
            "confidence": 92.7
        }

    @patch("icd_matcher.tasks.generate_patient_summary")
    @patch("icd_matcher.tasks.find_best_icd_match")
    def test_no_icd_matches(self, mock_find_best_icd_match, mock_generate_patient_summary):
        # Mock empty results
        mock_generate_patient_summary.return_value = ("No conditions", [])
        mock_find_best_icd_match.return_value = {}

        predict_icd_code(self.admission.id)

        self.admission.refresh_from_db()
        assert self.admission.predicted_icd_code is None
        assert self.admission.prediction_accuracy is None
        assert self.admission.predicted_icd_codes == []