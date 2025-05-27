# 💡 ICD Code Matching - Intelligent Medical Mapping with Django & NLP

A powerful Django-based web application that automates the mapping of medical remarks to ICD-10 codes using advanced natural language processing (NLP), semantic search, and knowledge graph augmentation. Built with Django, Django-Ninja, sentence-transformers, and optional Mistral LLM integration, this project streamlines medical coding for healthcare professionals.

---

## 📦 Project Highlights

- **Automated ICD-10 Mapping**: Matches patient medical remarks to standardized ICD-10 codes with high accuracy.
- **NLP-Powered**: Uses sentence embeddings for semantic similarity and SQLite FTS5 for efficient full-text search.
- **Knowledge Graph**: Enhances matching by capturing relationships between ICD codes.
- **Negation Detection**: Identifies negated conditions (e.g., "no fever") to avoid incorrect coding.
- **Fast API Access**: Provides lightweight REST API endpoints via Django-Ninja.
- **Scalable Backend**: Processes tasks asynchronously with Celery and Redis.
- **Web Interface**: Includes a demo interface for testing and visualization.
- **Extensible**: Supports optional Mistral LLM for enhanced confidence scoring.

---

## 🧱 Tech Stack

- **Framework**: Django 5.2, Django-Ninja 1.3.0
- **NLP**: Sentence Transformers (`paraphrase-MiniLM-L6-v2`)
- **Knowledge Graph**: NetworkX for ICD code relationships
- **Database**: SQLite (with FTS5 for full-text search)
- **Task Queue**: Celery 5.3.6 with Redis 5.0.1 as the broker and result backend
- **LLM (Optional)**: Mistral via local API (`http://localhost:11434/api/generate`)
- **Dependencies**: NumPy, Requests, Tenacity, Pydantic 2.11.0, django-extensions, jsonfield, networkx, eventlet
- **Package Manager**: `uv` (recommended) or `pip`
- **Testing**: pytest 7.4.4, pytest-django 4.7.0, pytest-mock 3.12.0, fakeredis 2.20.0
- **Environment**: Tested on Windows (`platform win32`), compatible with Mac/Linux

---

## 🚀 Getting Started

Follow these steps to set up and run the ICD Code Matching project on your local machine.

### Prerequisites

- **Python**: 3.12.9 or higher ([python.org](https://www.python.org/downloads/))
- **Redis**: Install locally or use Docker.
  - Windows: Download from [Redis Windows releases](https://github.com/microsoftarchive/redis/releases).
  - Mac/Linux: `brew install redis` or `sudo apt-get install redis-server`.
  - Docker: Requires Docker Desktop ([docker.com](https://www.docker.com/products/docker-desktop/)).
- **Git**: For cloning the repository.
- **uv**: Recommended Python package manager (optional).
  - Install: `pip install uv` ([uv documentation](https://github.com/astral-sh/uv)).
- **WSL2** (optional, for Docker on Windows): Install via `wsl --install`.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://gitlab.com/program_aum/computation/icd_cleaning_tools.git
   cd icd_cleaning_tools
   ```

2. **Create a Virtual Environment**:
   Using `uv` (recommended):
   ```bash
   uv venv
   ```
   Or with `venv`:
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies**:
   Using `uv`:
   ```bash
   uv pip install -r requirements.txt
   ```
   Or with `pip`:
   ```bash
   pip install django==5.2 django-ninja==1.3.0 celery==5.3.6 redis==5.0.1 sentence-transformers==3.2.0 networkx==3.4.0 eventlet==0.36.0 pytest==7.4.4 pytest-django==4.7.0 pytest-mock==3.12.0 fakeredis==2.20.0 numpy requests tenacity pydantic==2.11.0 django-extensions jsonfield
   ```

5. **Set Up the Database**:
   Apply migrations to create the SQLite database (`db.sqlite3`):
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Load ICD Data**:
   Populate the database with ICD-10 codes:
   ```bash
   python manage.py load_icd_data icd_version_2019.json
   ```
   Ensure `icd_version_2019.json` is in the project root or provide the correct path.

7. **Set Up FTS5 Table**:
   Create the full-text search virtual table:
   ```bash
   python manage.py setup_fts
   ```
   Or manually:
   ```bash
   python manage.py shell
   ```
   ```python
   from icd_matcher.utils.db_utils import setup_fts_table
   setup_fts_table()
   ```

8. **Configure Redis**:
   Choose one of the following methods:

   **Option A: Local Redis Installation**:
   - Install Redis for Windows from [GitHub](https://github.com/microsoftarchive/redis/releases).
   - Start Redis:
     ```bash
     redis-server
     ```
   - Verify:
     ```bash
     redis-cli ping
     ```
     - Expected: `PONG`

   **Option B: Redis via Docker**:
   - Install and start Docker Desktop.
   - Run Redis container:
     ```bash
     docker run -d -p 6379:6379 --name redis redis
     ```
   - Verify:
     ```bash
     docker ps
     redis-cli ping
     ```
     - Expected: `PONG`

9. **Verify Django Settings**:
   Ensure `icd_matcher_project/settings.py` includes:
   ```python
   INSTALLED_APPS = [
       'django.contrib.admin',
       'django.contrib.auth',
       'django.contrib.contenttypes',
       'django.contrib.sessions',
       'django.contrib.messages',
       'django.contrib.staticfiles',
       'icd_matcher.apps.IcdMatcherConfig',
       'django_ninja',
       'django_extensions',
   ]

   DATABASES = {
       'default': {
           'ENGINE': 'django.db.backends.sqlite3',
           'NAME': BASE_DIR / 'db.sqlite3',
       }
   }

   CELERY_BROKER_URL = 'redis://localhost:6379/0'
   CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

   ICD_MATCHING_SETTINGS = {
       'BATCH_SIZE': 32,
       'MAX_CANDIDATES': 25,
       'MIN_SIMILARITY_SCORE': 50.0,
       'MAX_SIMILARITY_SCORE': 95.0,
       'INCLUSION_BOOST': 1.2,
       'EXCLUSION_PENALTY': 0.5,
       'MAX_WORKERS': 4,
       'CACHE_TTL': 3600,
   }

   LOGGING = {
       'version': 1,
       'disable_existing_loggers': False,
       'handlers': {
           'console': {'class': 'logging.StreamHandler'},
           'file': {'class': 'logging.FileHandler', 'filename': 'icd_matcher.log'},
       },
       'loggers': {
           'icd_matcher': {
               'handlers': ['console', 'file'],
               'level': 'DEBUG',
               'propagate': True,
           },
       },
   }
   ```

---

## 📂 Input Fields

The project processes patient records with the following fields:

- **DISCHARGE_REMARKS**: General discharge notes or summaries.
- **ICD_REMARKS_A**: Primary medical remarks for ICD coding.
- **ICD_REMARKS_D**: Additional diagnostic remarks.
- **Predefined ICD_code**: Manually assigned ICD-10 code (if available).
- **Generated Matching ICD Code(s)**: Automatically generated ICD-10 codes with confidence scores, stored in `MedicalAdmissionDetails.predicted_icd_codes` (JSONField).

Remarks are processed using NLP, embedded with sentence-transformers, matched via semantic similarity, and augmented with a knowledge graph.

---

## 🧠 Features

- **Condition Summary Generation**: Summarizes patient remarks (e.g., "History of bilateral ureteric calculus" → "The patient has a history of bilateral ureteric calculus (kidney stones in both ureters).").
- **Negation Detection**: Excludes negated conditions (e.g., "no fever").
- **Semantic Matching**: Uses sentence embeddings for high-accuracy ICD-10 code matching.
- **Knowledge Graph**: Enhances matching by leveraging ICD code relationships (e.g., `N20.1` → `N20`).
- **Full-Text Search**: Utilizes SQLite FTS5 for fast ICD code lookups.
- **LLM Integration**: Optional Mistral LLM for refined confidence scoring (local API at `http://localhost:11434/api/generate`).
- **Asynchronous Processing**: Handles large datasets with Celery and Redis.
- **API Access**: Exposes endpoints like `/api/match-text` via Django-Ninja.
- **Caching**: Stores embeddings with Django’s cache framework.
- **Parallel Processing**: Matches conditions concurrently for performance.

---

## 🔧 Running the Application

### 1. Start Redis Server

Ensure Redis is running:

- Local:
  ```bash
  redis-server
  ```
- Docker:
  ```bash
  docker start redis
  ```
- Verify:
  ```bash
  redis-cli ping
  ```
  - Expected: `PONG`

### 2. Start Celery Worker

Run Celery for asynchronous task processing (Windows requires `--pool=eventlet`):

```bash
celery -A icd_matcher_project worker --pool=eventlet --loglevel=info
```

- Expected:
  ```
  [2025-05-13 20:18:00,123: INFO/MainProcess] Connected to redis://localhost:6379//
  [2025-05-13 20:18:00,124: INFO/MainProcess] celery@<machine> ready.
  ```

### 3. Start Django Development Server

Run the Django server:

```bash
python manage.py runserver
```

- Access:
  - Web interface: `http://127.0.0.1:8000/`
  - API docs: `http://127.0.0.1:8000/api/docs` (Ninja Swagger UI)
- Expected: Django welcome page or 404 (if no root URL defined).

### 4. Test API Endpoint

Test the `/api/match-text` endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/match-text \
-H "Content-Type: application/json" \
-d '{"medical_text": "History of bilateral ureteric calculus", "existing_codes": []}'
```

- Expected Response:
  ```json
  {
      "summary": "The patient has a history of bilateral ureteric calculus (kidney stones in both ureters).",
      "conditions": ["Bilateral Ureteric Calculus"],
      "matches": {
          "Bilateral Ureteric Calculus": [
              {"code": "N20.1", "title": "Calculus of ureter", "confidence": 95.0},
              {"code": "N20.0", "title": "Calculus of kidney", "confidence": 86.3},
              {"code": "N20", "title": "Calculus of kidney and ureter", "confidence": 70.0}
          ]
      }
  }
  ```

### 5. Test Celery Task

Trigger the `predict_icd_code` task:

```bash
python manage.py shell
```

```python
from icd_matcher.models import MedicalAdmissionDetails
from icd_matcher.tasks import predict_icd_code
from django.utils import timezone
admission = MedicalAdmissionDetails.objects.create(
    patient_data="History of bilateral ureteric calculus",
    admission_date=timezone.now()
)
predict_icd_code.delay(admission.id)
```

- Check Celery logs for task execution.
- Verify database update:
  ```python
  admission.refresh_from_db()
  print(admission.predicted_icd_codes)
  ```
  - Expected:
    ```python
    [
        {'code': 'N20.1', 'title': 'Calculus of ureter', 'confidence': 95.0},
        {'code': 'N20.0', 'title': 'Calculus of kidney', 'confidence': 86.3}
    ]
    ```

### 6. Run Test Suite

Execute the test suite:

```bash
pytest icd_matcher/tests/ -v
```

- Expected:
  ```
  collected 11 items
  icd_matcher/tests/test_admin.py::TestMedicalAdmissionDetailsAdmin::test_admin_list_display PASSED
  ...
  icd_matcher/tests/test_tasks.py::TestPredictICDCodeTask::test_predict_icd_code_task PASSED
  ============================== 11 passed in 15.00s ==============================
  ```

- Suppress warnings (Django 6.0, Pydantic) in `pytest.ini`:
  ```ini
  [pytest]
  filterwarnings =
      ignore::DeprecationWarning:pydantic.*
      ignore::RemovedInDjango60Warning
  ```

---

## 🛠️ Useful Commands

### Database Management
- **Apply Migrations**:
  ```bash
  python manage.py makemigrations
  python manage.py migrate
  ```

- **Load ICD Data**:
  ```bash
  python manage.py load_icd_data icd_version_2019.json
  ```

- **Set Up FTS Table**:
  ```bash
  python manage.py setup_fts
  ```

- **Clear Embedding Cache**:
  ```bash
  python manage.py shell
  ```
  ```python
  from icd_matcher.utils.embeddings import clear_embedding_cache
  clear_embedding_cache()
  ```

### Task Execution
- **Run a Task**:
  ```bash
  python manage.py shell
  ```
  ```python
  from icd_matcher.tasks import predict_icd_code
  from icd_matcher.models import MedicalAdmissionDetails
  admission = MedicalAdmissionDetails.objects.create(
      patient_data="History of bilateral ureteric calculus",
      admission_date="2025-05-13"
  )
  result = predict_icd_code.delay(admission.id)
  print(result.id)
  print(MedicalAdmissionDetails.objects.get(id=admission.id).predicted_icd_codes)
  ```

- **Inspect Redis**:
  ```bash
  python icd_matcher/inspect_redis.py
  ```

### Testing
- **Run Redis Test**:
  ```bash
  pytest icd_matcher/tests/test_redis.py -v
  ```

- **Run All Tests**:
  ```bash
  pytest icd_matcher/tests/ -v
  ```

### Celery and Redis
- **Start Celery Worker**:
  ```bash
  celery -A icd_matcher_project worker --pool=eventlet --loglevel=info
  ```

- **Clear Redis**:
  ```bash
  redis-cli
  FLUSHDB
  ```

---

## 🔧 Developer Notes

- **Core Logic**:
  - `icd_matcher/utils/embeddings.py`: Handles sentence embeddings, synonym mapping (e.g., "Bilateral Ureteric Calculus" → "ureteric calculus"), and knowledge graph fallback.
  - `icd_matcher/utils/db_utils.py`: Manages FTS5 setup and queries.
  - `icd_matcher/tasks.py`: Contains `predict_icd_code` for asynchronous ICD matching.
  - `icd_matcher/utils/knowledge_graph.py`: Implements NetworkX-based ICD code relationships.

- **Database**: Uses SQLite (`db.sqlite3`) for production and `:memory:` for tests.
- **Models**:
  - `MedicalAdmissionDetails`: Stores patient data and `predicted_icd_codes` (JSONField).
  - `ICDCategory`: Stores ICD-10 codes, titles, inclusions, and exclusions.
- **Celery Tasks**: `predict_icd_code` processes admissions, storing results in `predicted_icd_codes`.
- **Redis**: Broker and result backend (`redis://localhost:6379/0`).
- **Testing**: Uses `pytest` with `fakeredis` to mock Redis.
- **FTS5**: Virtual table (`icd_fts`) is created via `setup_fts` command.
- **Recent Fixes**:
  - Resolved `test_predict_icd_code_task` failure by fixing case sensitivity (`"calculus of ureter"` vs. `"Calculus of ureter"`) and condition normalization in `embeddings.py`.
  - Added `synonym_map` to handle "Bilateral Ureteric Calculus".

### Key Files
- `icd_matcher/models.py`: Defines `MedicalAdmissionDetails` and `ICDCategory`.
- `icd_matcher/tasks.py`: Implements `predict_icd_code`.
- `icd_matcher/utils/embeddings.py`: Manages NLP and knowledge graph logic.
- `icd_matcher/utils/db_utils.py`: Handles FTS5 setup and queries.
- `icd_matcher/inspect_redis.py`: Inspects Celery task results.
- `icd_matcher/tests/`: Contains tests for admin, models, tasks, Redis, and RAG/KAG.

---

## 🐞 Troubleshooting

### Database Issues
- **Error**: `OperationalError: no such table: icd_matcher_medicaladmissiondetails`
  - **Fix**:
    ```bash
    python manage.py migrate
    ```
  - Verify `settings.py`:
    ```python
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }
    ```

- **Verify Database**:
  ```bash
  python manage.py shell
  ```
  ```python
  from icd_matcher.models import MedicalAdmissionDetails
  print(MedicalAdmissionDetails.objects.all().values('id', 'predicted_icd_codes'))
  ```

### Redis Issues
- **Error**: `bind: No such file or directory`
  - **Fix**: Check port `6379`:
    ```bash
    netstat -aon | findstr :6379
    taskkill /PID <pid> /F
    ```
  - Use a config file:
    ```bash
    echo "port 6379\nbind 127.0.0.1\nmaxmemory 256mb" > redis.conf
    redis-server redis.conf
    ```

- **Error**: `Redis connection failed`
  - **Fix**:
    ```bash
    redis-server
    redis-cli ping
    ```

- **Inspect Redis**:
  ```bash
  redis-cli
  KEYS celery-task-meta-*
  ```

### Docker Issues
- **Error**: `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`
  - **Fix**:
    - Start Docker Desktop.
    - Check service:
      ```bash
      Get-Service -Name com.docker.service
      Start-Service -Name com.docker.service
      ```
    - Update WSL:
      ```bash
      wsl --update
      ```

### Celery Issues
- **Tasks Not Executing**:
  - Ensure Redis is running.
  - Verify `icd_matcher_project/celery.py`:
    ```python
    from celery import Celery
    import os
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'icd_matcher_project.settings')
    app = Celery('icd_matcher_project')
    app.config_from_object('django.conf:settings', namespace='CELERY')
    app.autodiscover_tasks()
    ```
  - Use `--pool=eventlet` on Windows.

### Test Failures
- **Error**: `test_predict_icd_code_task` fails due to case sensitivity.
  - **Fix**: Ensure `embeddings.py` and `test_tasks.py` are updated (see recent fixes).
- **No FTS Results**:
  - Re-run:
    ```bash
    python manage.py setup_fts
    ```
  - Populate `ICDCategory`:
    ```bash
    python manage.py shell
    ```
    ```python
    from icd_matcher.models import ICDCategory
    ICDCategory.objects.create(
        code="N20.1",
        title="Calculus of ureter",
        inclusions='["ureteric calculus"]',
        exclusions='["malposition"]'
    )
    ```

### API Errors
- **404**: Verify `icd_matcher_project/urls.py` and `icd_matcher/api.py`.
- **500**: Set `DEBUG=True` in `settings.py` and check `icd_matcher.log`.

---

## 📚 Additional Resources

- **Django-Ninja**: [Documentation](https://django-ninja.rest-framework.com/)
- **Sentence Transformers**: [Hugging Face](https://huggingface.co/sentence-transformers)
- **Celery**: [Documentation](https://docs.celeryproject.org/)
- **Redis**: [Getting Started](https://redis.io/docs/getting-started/)
- **Mistral LLM**: [Ollama](https://ollama.ai/) for local API setup
- **NetworkX**: [Documentation](https://networkx.org/documentation/stable/)

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 📬 Contact

For issues or questions, open an issue on the [GitLab repository](https://gitlab.com/program_aum/computation/icd_cleaning_tools).