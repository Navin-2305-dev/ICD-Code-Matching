import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_KEY = 'django-insecure-46w1zzvshb#9=z+0csj6em+y)$(g#bj+qz^334*t92&cdaub+2'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_extensions',
    'icd_matcher.apps.IcdMatcherConfig',
]

ICD_MATCHING_SETTINGS = {
    'MISTRAL_LOCAL_URL': 'http://localhost:11434/api/generate',
    'MISTRAL_MODEL': 'mistral',
    'CACHE_TTL': 86400,  # 24 hours
    'BATCH_SIZE': 32,
    'FTS_QUERY_LIMIT': 10,
    'MAX_CANDIDATES': 15,
    'MIN_SIMILARITY_SCORE': 60,
    'MAX_SIMILARITY_SCORE': 95,
    'ALLOW_CATEGORY_CODES': True,
    'NEGATION_WINDOW': 10,
    'MAX_PIPELINE_ITERATIONS': 2,  # Allow one refinement iteration
    'PREPROCESS_CALL_LIMIT': 1000,
    'GRAPH_BUILD_BATCH_SIZE': 1000,
    'USE_RAG_KAG': True,
    'NEGATION_CUES': [
        "no", "not", "denies", "negative", "without", "absent", "ruled out",
        "non", "never", "lacks", "excludes", "rules out", "negative for",
        "free of", "deny", "denying", "unremarkable for"
    ]
}

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'icd_matcher_cache',
        'KEY_FUNCTION': lambda key, key_prefix, version: key.replace(':', '_').replace(' ', '_')[:250],
    }
}

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'icd_matcher_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'icd_matcher_project.wsgi.application'
ASGI_APPLICATION = "icd_matcher_project.asgi.application"

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'icd_matcher.log',
        },
    },
    'loggers': {
        'icd_matcher': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60