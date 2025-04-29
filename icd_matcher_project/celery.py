import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'icd_matcher_project.settings')

app = Celery('icd_matcher_project')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()