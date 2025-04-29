import logging
import re
import requests
from typing import List, Tuple, Optional, Set
from django.core.cache import cache
from django.conf import settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .exceptions import TextPreprocessingError, MistralQueryError, ConditionExtractionError

logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    """Preprocess text by normalizing case, handling medical terms, and collapsing whitespace."""
    if not isinstance(text, str):
        logger.error("Input text must be a string")
        raise TextPreprocessingError("Input text must be a string")
    
    try:
        text = text.lower().strip()
        return text.strip()
    except Exception as e:
        logger.error(f"Text preprocessing failed: {e}")
        raise TextPreprocessingError(f"Text preprocessing failed: {e}")

def get_negation_cues() -> List[str]:
    """Return a list of negation cues from settings or default."""
    return getattr(settings, 'ICD_MATCHING_SETTINGS', {}).get('NEGATION_CUES', [
        "no", "not", "denies", "negative", "without", "absent", "ruled out", 
        "non", "never", "lacks", "excludes", "rules out", "negative for", 
        "free of", "deny", "denying", "unremarkable for"
    ])

def is_not_negated(condition: str, text: str, negation_cues: Optional[List[str]] = None, window: int = None) -> bool:
    """Check if a condition is not negated in the text with caching."""
    if not condition or not text:
        return True
    
    cache_key = f"negation_check_{hash(condition)}_{hash(text[:1000])}"
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"Cache hit for negation check: {condition}")
        return cached_result
    
    try:
        text_lower = text.lower() if isinstance(text, str) else ""
        condition_lower = condition.lower() if isinstance(condition, str) else ""
        
        if not text_lower or not condition_lower:
            return True
        
        if not negation_cues:
            negation_cues = get_negation_cues()
        
        if window is None:
            window = getattr(settings, 'ICD_MATCHING_SETTINGS', {}).get('NEGATION_WINDOW', 5)  # Tighter window
        
        negation_pattern = '|'.join(negation_cues)
        pattern = rf'\b(?:{negation_pattern})\b(?:\s+\w+){{0,2}}\s+{re.escape(condition_lower)}\b'
        if re.search(pattern, text_lower):
            cache.set(cache_key, False, timeout=settings.ICD_MATCHING_SETTINGS['CACHE_TTL'])
            return False
        
        matches = list(re.finditer(rf'\b{re.escape(condition_lower)}\b', text_lower))
        if not matches:
            cache.set(cache_key, True, timeout=settings.ICD_MATCHING_SETTINGS['CACHE_TTL'])
            return True
        
        for match in matches:
            start_idx = match.start()
            context_start = max(0, start_idx - window * 5)
            preceding_context = text_lower[context_start:start_idx]
            if any(cue in preceding_context.split() for cue in negation_cues):
                cache.set(cache_key, False, timeout=settings.ICD_MATCHING_SETTINGS['CACHE_TTL'])
                return False
        
        cache.set(cache_key, True, timeout=settings.ICD_MATCHING_SETTINGS['CACHE_TTL'])
        return True
    except Exception as e:
        logger.error(f"Negation check failed for condition '{condition}': {e}")
        raise TextPreprocessingError(f"Negation check failed: {e}")

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.RequestException, requests.Timeout, ConnectionError))
)
def query_mistral(prompt: str) -> str:
    """Query the local Mistral model with retries and caching."""
    cache_key = f"mistral_response_{hash(prompt)}"
    cached_response = cache.get(cache_key)
    
    if cached_response is not None:
        logger.debug(f"Cache hit for Mistral response: {prompt[:100]}...")
        return cached_response
    
    logger.debug(f"Cache miss for Mistral response: {prompt[:100]}...")
    
    url = getattr(settings, 'ICD_MATCHING_SETTINGS', {}).get('MISTRAL_API_URL', "http://localhost:11434/api/generate")
    payload = {
        "model": "mistral", 
        "prompt": prompt, 
        "stream": False, 
        "temperature": 0.1,
    }
    
    try:
        logger.debug(f"Querying Mistral with prompt: {prompt[:100]}...")
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json().get('response', '').strip()
        cache.set(cache_key, result, timeout=settings.ICD_MATCHING_SETTINGS['CACHE_TTL'])
        logger.debug(f"Cached Mistral response for prompt: {prompt[:100]}...")
        return result
    except (requests.RequestException, requests.Timeout, ConnectionError) as e:
        logger.error(f"Mistral request failed: {e}")
        raise MistralQueryError(f"Mistral query failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in Mistral query: {e}")
        raise MistralQueryError(f"Unexpected error in Mistral query: {e}")

def generate_patient_summary(patient_text: str) -> Tuple[str, List[str]]:
    """Generate a summary and extract conditions from patient text with status handling."""
    if not patient_text or not isinstance(patient_text, str):
        logger.warning("No valid patient data provided")
        return "No patient data provided.", ["No conditions identified"]
    
    try:
        patient_text = preprocess_text(patient_text)
        logger.debug(f"Preprocessed patient text: {patient_text}")
    except TextPreprocessingError as e:
        logger.error(f"Text preprocessing failed: {e}")
        raise
    
    if not patient_text:
        return "The patient was suffering from no conditions identified.", ["No conditions identified"]
    
    prompt = (
        f"From the following patient data, generate a concise summary of the patient's condition only if those details are explicitly mentioned."
        f" Extract all the applicable diseases, conditions, and significant symptoms "
        f"from the text. Provide the summary in a single sentence, and list the conditions "
        f"as a list of bullet points starting with a hyphen. Ignore negated conditions (e.g., 'no pain') but include "
        f"non-negated symptoms like 'pain' or 'swelling'. Return summary followed by 'Conditions:' and the list.\n\n"
        f"{patient_text}"
    )
    
    try:
        mistral_response = query_mistral(prompt)
        logger.debug(f"Mistral response: {mistral_response[:200]}...")
        
        if "Conditions:" in mistral_response:
            summary_part, condition_part = mistral_response.split("Conditions:", 1)
            conditions = [
                re.sub(r'\s*\(.*?\)', '', line).strip('- ').strip()
                for line in condition_part.split('\n')
                if line.strip().startswith('-') and line.strip('- ').strip()
            ]
            
            if not conditions:
                conditions = [line.strip() for line in condition_part.split('\n') if line.strip().startswith('*')]
                conditions = [re.sub(r'^\*\s*', '', cond).strip() for cond in conditions if cond.strip()]
            
            if not conditions:
                summary_conditions = re.findall(r'suffering from\s+(.+?)(?:,|\.|$)', summary_part)
                if summary_conditions:
                    possible_conditions = summary_conditions[0].split(' and ')
                    conditions = [cond.strip() for cond in possible_conditions if cond.strip()]
        else:
            summary_part = mistral_response
            conditions = [line.strip() for line in mistral_response.split('\n') if line.strip().startswith('-')]
            conditions = [re.sub(r'^-+\s*', '', cond).strip() for cond in conditions if cond.strip()]
        
        summary = summary_part.strip()
        
        conditions = [cond for cond in conditions if is_not_negated(cond.split(' - ')[0], patient_text)]
        
        if not conditions or all(c.lower() == "no conditions identified" for c in conditions):
            conditions = ["No conditions identified"]
            if not summary.strip():
                summary = "The patient was suffering from no conditions identified."
    except Exception as e:
        logger.warning(f"Mistral failed: {e}. Using fallback extraction.")
        try:
            words = patient_text.split()
            conditions = []
            for i in range(len(words)):
                for j in range(1, min(5, len(words) - i)):
                    phrase = ' '.join(words[i:i+j])
                    if 2 <= len(phrase.split()) <= 4 and len(phrase) > 5 and is_not_negated(phrase, patient_text):
                        status = 'recovered' if 'approved status' in patient_text or 'recovered' in patient_text else 'unknown'
                        conditions.append(f"{phrase} - {status}")
            
            conditions = list(set(conditions))[:5]
            
            if not conditions:
                conditions = ["No conditions identified"]
            
            summary = f"The patient was suffering from {' and '.join(c.split(' - ')[0] for c in conditions) if conditions and conditions[0] != 'No conditions identified' else 'no conditions identified'}"
        except Exception as e:
            logger.error(f"Fallback condition extraction failed: {e}")
            raise ConditionExtractionError(f"Condition extraction failed: {e}")
    
    logger.info(f"Summary generated: {summary[:100]}...")
    print(f"Summary: {summary}")
    print()
    print(f"Conditions: {conditions}")
    logger.info(f"Conditions extracted: {conditions}")
    return summary, conditions