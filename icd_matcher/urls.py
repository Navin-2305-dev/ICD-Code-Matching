# icd_matcher/urls.py
from django.urls import path
from . import views
from .api import api

urlpatterns = [
    # Traditional Django views
    path('', views.patient_input, name='patient_input'),
    path('result/', views.result, name='result'),
    path('search/', views.search_icd, name='search_icd'),
    
    # Django-ninja API
    path('api/', api.urls),
]
