# 💡 ICD Code Matching - Intelligent Medical Mapping with Django & NLP 🧠

A powerful Django-based web application that automates the process of mapping medical remarks to ICD-10 codes using state-of-the-art NLP and semantic search. Built with Django, Django Ninja, sentence-transformers, and optional Mistral LLM integration.

────────────────────────────────────────────────────────────────────────────

## 📦 Project Highlights

✔️ Match patient medical remarks to standardized ICD-10 codes  
✔️ Leverage full-text search (FTS5) and sentence embeddings  
✔️ Detect negated conditions intelligently  
✔️ Fast API access using Django-Ninja  
✔️ Ready-to-run with a web interface for demo/testing  

────────────────────────────────────────────────────────────────────────────

## 🧱 Tech Stack

- Django + Django-Ninja
- Sentence Transformers (`paraphrase-MiniLM-L6-v2`)
- SQLite (FTS5-enabled)
- Mistral LLM (optional, for enhanced confidence scoring)
- NumPy, Requests, Tenacity
- `uv` (ultra fast Python package manager)

────────────────────────────────────────────────────────────────────────────

## 🚀 Getting Started

# Step 1: Clone the repository
$ git clone https://github.com/your-username/icd_code_matching.git

$ cd icd_code_matching

# Step 2: Create virtual environment using uv
$ uv venv

# Step 3: Activate the virtual environment
$ source .venv/bin/activate   # (or .venv\Scripts\activate on Windows)

# Step 4: Install dependencies
$ uv pip install --requirements requirements.txt

# Step 5: Apply migrations and load ICD data
$ python manage.py makemigrations

$ python manage.py migrate

$ python manage.py load_icd_data "icd_10\icd_version_2019.json"

# Step 6: Run the development server
$ python manage.py runserver

# Step 7: (Optional) Run sample ICD matching logic
$ python manage.py runscript process_icd_data 

# Step 8: (API Test) Open Django-Ninja API docs
http://127.0.0.1:8000/api/docs

────────────────────────────────────────────────────────────────────────────

## 📂 Input Fields

This project expects the following fields for each patient record:

- DISCHARGE_REMARKS
- ICD_REMARKS_A
- ICD_REMARKS_D
- Predefined ICD_code
- Generated Matching ICD Code(s)

All remarks are processed, embedded, and semantically matched to the nearest ICD codes.

────────────────────────────────────────────────────────────────────────────

## 🧠 Features

- 📝 Automatic condition summary generation
- ❌ Medical negation detection (e.g. "no fever")
- ⚡ Embedding-based ICD title similarity scoring
- 🔍 FTS5 Full-text ICD lookup
- 🧪 LLM-based reasoning via Mistral (local API)
- 🧬 Parallelized condition matching for speed
- 📡 Lightweight and fast API access (Django-Ninja)

────────────────────────────────────────────────────────────────────────────

## 🧪 Example Output

🔸 Summary:
"The patient was suffering from acute gastroenteritis and dehydration."

🔸 Suggested ICD Codes:
- A09: Infectious gastroenteritis (92.1%)
- E86.0: Dehydration (89.3%)

────────────────────────────────────────────────────────────────────────────

## 🔧 Developer Notes

- Core logic resides in `utils.py`
- Supports Mistral API at `http://localhost:11434/api/generate`
- Embedding caching and ORM fallback included
- FTS5-backed `icd_fts` virtual table is auto-created

────────────────────────────────────────────────────────────────────────────

## 🛠️ Useful Commands

# Run FTS table setup manually (optional)
$ python manage.py shell
>>> from icd_matcher.utils import setup_fts_table

>>> setup_fts_table()

# Clear cached embeddings (dev mode)
>>> from icd_matcher.utils import clear_embedding_cache

>>> clear_embedding_cache()
