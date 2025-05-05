# icd_matcher/tests/test_admin.py
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

from django.test import TestCase, Client
from django.contrib.auth.models import User
from icd_matcher.models import MedicalAdmissionDetails
from icd_matcher.admin import MedicalAdmissionDetailsAdmin
from django.contrib.admin.sites import AdminSite

@pytest.mark.django_db
class TestMedicalAdmissionDetailsAdmin(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_superuser(
            username="admin", password="admin123", email="admin@example.com"
        )
        self.client.login(username="admin", password="admin123")
        self.admission = MedicalAdmissionDetails.objects.create(
            patient_data="History of bilateral ureteric calculus",
            admission_date="2025-04-28",
            predicted_icd_code="N20.1",
            prediction_accuracy=92.7,
            predicted_icd_codes=[
                {"code": "N20.1", "title": "Calculus of ureter", "confidence": 92.7},
                {"code": "N20.0", "title": "Calculus of kidney", "confidence": 86.3}
            ]
        )
        self.site = AdminSite()
        self.admin = MedicalAdmissionDetailsAdmin(MedicalAdmissionDetails, self.site)

    def test_admin_list_display(self):
        response = self.client.get("/admin/icd_matcher/medicaladmissiondetails/")
        assert response.status_code == 200
        assert "N20.1" in response.content.decode()
        assert "92.7%" in response.content.decode()
        assert "Calculus of ureter (92.7%)" in response.content.decode()

    def test_admin_methods(self):
        assert self.admin.top_icd_code(self.admission) == "N20.1"
        assert self.admin.top_icd_accuracy(self.admission) == "92.7%"
        assert "N20.1: Calculus of ureter (92.7%); N20.0: Calculus of kidney (86.3%)" in self.admin.all_icd_codes(self.admission)
        assert self.admin.patient_data_short(self.admission) == "History of bilateral ureteric calculus"