import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

async def load_synonyms():
    base_dir = Path(__file__).resolve().parent.parent.parent
    synonym_file = base_dir / 'data' / 'helper_files' /'synonyms.json'
    default_synonyms = {
        'hearing loss': ['deafness', 'auditory impairment'],
        'calculus': ['stone', 'concretion'],
        'bilateral': ['both', 'dual-sided'],
        'pain': ['ache', 'discomfort', 'soreness'],
        'fever': ['pyrexia', 'hyperthermia'],
        'diabetes': ['hyperglycemia', 'diabetes mellitus'],
        'hypertension': ['high blood pressure', 'htn'],
        'myocardial infarction': ['heart attack', 'mi'],
        'covid-19': ['coronavirus', 'sars-cov-2'],
        'vaccination': ['immunization', 'shot'],
    }
    try:
        if synonym_file.exists():
            with open(synonym_file, 'r', encoding='utf-8') as f:
                loaded_synonyms = json.load(f)
            default_synonyms.update(loaded_synonyms)
            logger.info(f"Loaded synonyms from {synonym_file}")
        else:
            logger.warning(f"Synonym file {synonym_file} not found, using defaults")
        return default_synonyms
    except Exception as e:
        logger.error(f"Failed to load synonyms: {e}")
        return default_synonyms