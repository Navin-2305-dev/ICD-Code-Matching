from django.apps import AppConfig

class IcdMatcherConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'icd_matcher'

    def ready(self):
        """Run initialization tasks after app registry is ready."""
        pass  # Graph building moved to management command