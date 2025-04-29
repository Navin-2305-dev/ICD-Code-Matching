from django.apps import AppConfig

class IcdMatcherConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'icd_matcher'

    def ready(self):
        import icd_matcher.signals