# icd_matcher_project/urls.py
from django.contrib import admin
from django.urls import include, path
from icd_matcher import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('icd_matcher.urls')),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
