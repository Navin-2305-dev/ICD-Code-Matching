# 💡 ICD Code Matching - Intelligent Medical Mapping with Django & NLP

A powerful Django-based web application that automates the mapping of medical remarks to ICD-10 codes using advanced natural language processing (NLP) and semantic search. Built with Django, Django Ninja, sentence-transformers, and optional Mistral LLM integration, this project streamlines medical coding for healthcare professionals.

---

## 📦 Project Highlights

- **Automated ICD-10 Mapping**: Matches patient medical remarks to standardized ICD-10 codes with high accuracy.
- **NLP-Powered**: Uses sentence embeddings and full-text search for precise condition matching.
- **Negation Detection**: Intelligently identifies negated conditions (e.g., "no fever").
- **Fast API Access**: Provides a lightweight REST API via Django-Ninja.
- **Scalable Backend**: Processes tasks asynchronously with Celery and Redis.
- **Web Interface**: Includes a demo interface for testing and visualization.
- **Extensible**: Supports optional Mistral LLM for enhanced confidence scoring.

---

## 🧱 Tech Stack

- **Framework**: Django 4.2, Django-Ninja 1.3.0
- **NLP**: Sentence Transformers (`paraphrase-MiniLM-L6-v2`)
- **Database**: SQLite (with FTS5 for full-text search)
- **Task Queue**: Celery 5.3.6 with Redis 5.0.1 as the broker and result backend
- **LLM (Optional)**: Mistral via local API (`http://localhost:11434/api/generate`)
- **Dependencies**: NumPy, Requests, Tenacity, Pydantic 2.11.0, django-extensions, jsonfield
- **Package Manager**: `uv` (recommended for virtual environment and dependency management)
- **Testing**: pytest 7.4.4, pytest-django 4.7.0, pytest-mock 3.12.0, fakeredis 2.20.0

---

## 🚀 Getting Started

Follow these steps to set up and run the ICD Code Matching project on your local machine.

### Prerequisites

- **Python**: 3.12.9 or higher
- **Redis**: Install and run Redis server (`redis-server`) for Celery task queue and result backend.
  - Windows: Download from [Redis Windows releases](https://github.com/microsoftarchive/redis/releases) or use WSL.
  - Mac/Linux: `brew install redis` or `sudo apt-get install redis-server`.
- **Git**: For cloning the repository.
- **uv**: Recommended Python package manager (optional but preferred).
  - Install: `pip install uv` or follow [uv documentation](https://github.com/astral-sh/uv).

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

7. **Set Up FTS5 Table** (Optional):
   Create the full-text search virtual table:
   ```bash
   python manage.py shell
   ```
   ```python
   from icd_matcher.utils import setup_fts_table
   setup_fts_table()
   ```

8. **Start Redis Server**:
   Ensure Redis is running:
   ```bash
   redis-server
   redis-cli ping  # Should return PONG
   ```

9. **Run the Celery Worker**:
   Start the Celery worker for asynchronous task processing:
   ```bash
   celery -A icd_matcher_project worker --pool=solo --loglevel=info
   ```

10. **Run the Development Server**:
    Start the Django server:
    ```bash
    python manage.py runserver
    ```
    Access the web interface at `http://127.0.0.1:8000`.

11. **Test the API**:
    Open the Django-Ninja API documentation:
    ```
    http://127.0.0.1:8000/api/docs
    ```

12. **Run Sample ICD Matching** (Optional):
    Execute sample ICD matching logic:
    ```bash
    python manage.py runscript process_icd_data
    ```

---

## 📂 Input Fields

The project processes patient records with the following fields:

- **DISCHARGE_REMARKS**: General discharge notes or summaries.
- **ICD_REMARKS_A**: Primary medical remarks for ICD coding.
- **ICD_REMARKS_D**: Additional diagnostic remarks.
- **Predefined ICD_code**: Manually assigned ICD-10 code (if available).
- **Generated Matching ICD Code(s)**: Automatically generated ICD-10 codes with confidence scores, stored in `MedicalAdmissionDetails.predicted_icd_codes`.

Remarks are processed using NLP, embedded with sentence-transformers, and matched to ICD-10 codes via semantic similarity.

---

## 🧠 Features

- **Condition Summary Generation**: Automatically summarizes patient remarks for clarity.
- **Negation Detection**: Identifies negated conditions (e.g., "no fever") to avoid incorrect coding.
- **Semantic Matching**: Uses sentence embeddings for high-accuracy ICD-10 code matching.
- **Full-Text Search**: Leverages SQLite FTS5 for fast ICD code lookups.
- **LLM Integration**: Optional Mistral LLM for enhanced confidence scoring (local API at `http://localhost:11434/api/generate`).
- **Asynchronous Processing**: Handles large datasets efficiently with Celery and Redis.
- **API Access**: Provides fast, lightweight endpoints via Django-Ninja.
- **Caching**: Stores embeddings to optimize performance.
- **Parallel Processing**: Matches conditions concurrently for speed.

---

## 🔧 Developer Notes

- **Core Logic**: Located in `icd_matcher/utils.py` (FTS setup, embedding generation) and `icd_matcher/tasks.py` (ICD matching tasks).
- **Database**: Uses SQLite (`db.sqlite3`) for production and `:memory:` for tests.
- **Models**: `MedicalAdmissionDetails` stores patient data and `predicted_icd_codes` (JSONField).
- **Celery Tasks**: `predict_icd_code` processes admissions asynchronously, storing results in `predicted_icd_codes`.
- **Redis**: Acts as Celery’s broker and result backend (`redis://localhost:6379/0`).
- **Testing**: Uses `pytest` with `fakeredis` to mock Redis in tests.
- **FTS5**: Virtual table (`icd_fts`) is auto-created during migration or manually via `setup_fts_table()`.

### Key Files
- `icd_matcher/models.py`: Defines `MedicalAdmissionDetails` and `ICDCategory` models.
- `icd_matcher/tasks.py`: Contains `predict_icd_code` task for ICD matching.
- `icd_matcher/utils`: Handles FTS setup, embedding caching, and NLP utilities.
- `icd_matcher/inspect_redis.py`: Inspects Celery task results and database records.
- `icd_matcher/tests/test_redis.py`: Tests Redis integration for task metadata.

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
  python manage.py shell
  ```
  ```python
  from icd_matcher.utils import setup_fts_table
  setup_fts_table()
  ```

- **Clear Embedding Cache**:
  ```bash
  python manage.py shell
  ```
  ```python
  from icd_matcher.utils import clear_embedding_cache
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
      patient_data="Test ureteric calculus",
      admission_date="2025-04-28"
  )
  result = predict_icd_code.delay(admission.id)
  print(result.id)
  print(result.get(timeout=30))  # Returns None
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
  celery -A icd_matcher_project worker --pool=solo --loglevel=info
  ```

- **Clear Redis**:
  ```bash
  redis-cli
  FLUSHDB
  ```

---

## 🐞 Troubleshooting

### Database Issues
- **Error**: `OperationalError: no such table: icd_matcher_medicaladmissiondetails`
  - **Cause**: The worker or shell is using an un-migrated database.
  - **Fix**:
    ```bash
    python manage.py migrate
    ```
    Ensure `settings.py` uses `db.sqlite3` for the worker:
    ```python
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
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
- **Error**: `Redis connection failed`
  - **Fix**:
    ```bash
    redis-server
    redis-cli ping  # Should return PONG
    ```

- **Inspect Redis**:
  ```bash
  redis-cli
  KEYS celery-task-meta-*
  ```
---

## 📚 Additional Resources

- **Django-Ninja**: [Documentation](https://django-ninja.rest-framework.com/)
- **Sentence Transformers**: [Hugging Face](https://huggingface.co/sentence-transformers)
- **Celery**: [Documentation](https://docs.celeryproject.org/)
- **Redis**: [Getting Started](https://redis.io/docs/getting-started/)
- **Mistral LLM**: [Ollama](https://ollama.ai/) for local API setup

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---
