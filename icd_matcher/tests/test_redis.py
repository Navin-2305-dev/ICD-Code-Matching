# icd_matcher/tests/test_redis.py
import os
import sys
import pytest
import json
import django
import fakeredis
from celery import Celery

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
class TestRedisIntegration(TestCase):
    def setUp(self):
        self.app = Celery('test')
        self.app.conf.update(
            task_always_eager=True,  # Ensures it runs inline
            task_store_eager_result=True,
        )
        self.admission = MedicalAdmissionDetails.objects.create(
            patient_data="History of bilateral ureteric calculus",
            admission_date="2025-04-28"
        )

    @patch("icd_matcher.tasks.generate_patient_summary")
    @patch("icd_matcher.tasks.find_best_icd_match")
    def test_task_stores_metadata_in_redis(self, mock_find_best_icd_match, mock_generate_patient_summary):
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

        result = predict_icd_code.apply(args=[self.admission.id])

        assert result.status == "SUCCESS"
        assert result.result is None  # if your task returns None
