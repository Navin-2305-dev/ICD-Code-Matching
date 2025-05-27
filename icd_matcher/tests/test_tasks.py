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
from icd_matcher.utils.db_utils import setup_fts_table
from unittest.mock import patch

@pytest.mark.django_db
class TestPredictICDCodeTask(TestCase):
    def setUp(self):
        # Setup FTS5 table
        setup_fts_table()
        self.admission = MedicalAdmissionDetails.objects.create(
            patient_data="History of bilateral ureteric calculus",
            admission_date="2025-04-28"
        )

    @patch("icd_matcher.tasks.generate_patient_summary")
    @patch("icd_matcher.tasks.find_best_icd_match")
    def test_predict_icd_code_task(self, mock_find_best_icd_match, mock_generate_patient_summary):
        # Mock dependencies
        mock_generate_patient_summary.return_value = (
            "The patient has a history of bilateral ureteric calculus (kidney stones in both ureters).",
            ["Bilateral Ureteric Calculus"]
        )
        mock_find_best_icd_match.return_value = {
            "Bilateral Ureteric Calculus": [
                ("N20.1", "Calculus of ureter", 95.0),
                ("N20.0", "Calculus of kidney", 86.3)
            ]
        }

        # Run task synchronously
        predict_icd_code(self.admission.id)

        # Refresh admission
        self.admission.refresh_from_db()

        # Assertions
        assert self.admission.predicted_icd_code == "N20.1"
        assert abs(self.admission.prediction_accuracy - 95.0) < 0.1
        assert len(self.admission.predicted_icd_codes) == 2
        assert self.admission.predicted_icd_codes[0] == {
            "code": "N20.1",
            "title": "calculus of ureter",
            "confidence": 95.0
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