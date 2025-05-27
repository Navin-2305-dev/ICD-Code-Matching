from django.urls import path, include
from . import views
from .api import api

urlpatterns = [
    # Frontend views
    path('', views.patient_input, name='patient_input'),
    path('result/', views.result, name='result'),
    path('search/', views.search_icd, name='search_icd'),
    
    # Async API views
    path('predict/', views.predict_icd_code_view, name='predict_icd_code'),
    path('admission/<int:admission_id>/', views.admission_details_view, name='admission_details'),
    
    # Django-Ninja API
    path('api/', api.urls),
]