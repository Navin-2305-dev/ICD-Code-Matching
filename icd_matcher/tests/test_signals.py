# icd_matcher/tests/test_signals.py
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
from unittest.mock import patch

@pytest.mark.django_db
class TestSignals(TestCase):
    @patch("icd_matcher.signals.predict_icd_code.delay")
    def test_post_save_signal(self, mock_predict_icd_code):
        admission = MedicalAdmissionDetails.objects.create(
            patient_data="History of bilateral ureteric calculus",
            admission_date="2025-04-28"
        )
        mock_predict_icd_code.assert_called_once_with(admission.id)