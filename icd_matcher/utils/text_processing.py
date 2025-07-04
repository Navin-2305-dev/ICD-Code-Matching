import logging
import re
import hashlib
from typing import List, Tuple, Optional
from django.core.cache import cache
from django.conf import settings
from asgiref.sync import sync_to_async
import requests
import json
from threading import local
from icd_matcher.utils.synonyms import load_synonyms
from icd_matcher.utils.exceptions import MistralQueryError
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)
_thread_local = local()

def _get_preprocess_counter():
    if not hasattr(_thread_local, 'preprocess_counter'):
        _thread_local.preprocess_counter = 0
    return _thread_local.preprocess_counter

def _increment_preprocess_counter():
    if not hasattr(_thread_local, 'preprocess_counter'):
        _thread_local.preprocess_counter = 0
    _thread_local.preprocess_counter += 1
    return _thread_local.preprocess_counter

def reset_preprocess_counter():
    _thread_local.preprocess_counter = 0

async def preprocess_text(text: str, bypass_counter_limit: bool = False) -> str:
    if not isinstance(text, str):
        logger.error("Input text must be a string")
        raise ValueError("Input text must be a string")
    if not text.strip():
        logger.debug("Empty text provided, returning empty string")
        return ""

    if not bypass_counter_limit:
        count = _increment_preprocess_counter()
        max_calls = settings.ICD_MATCHING_SETTINGS.get('PREPROCESS_CALL_LIMIT', 1000)
        if count > max_calls:
            logger.error("Excessive preprocess_text calls detected, possible loop")
            raise RuntimeError("Excessive preprocess_text calls")

    text_hash = hashlib.sha256(text[:1000].encode()).hexdigest()
    cache_key = f"preprocessed_text_{text_hash}"
    cached_text = await sync_to_async(cache.get)(cache_key)
    if cached_text is not None:
        logger.debug(f"Cache hit for preprocess_text: {text[:200]}")
        return cached_text

    try:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        synonyms = await load_synonyms()
        for term, syn_list in synonyms.items():
            for syn in syn_list:
                text = re.sub(rf'\b{re.escape(syn)}\b', term, text, flags=re.IGNORECASE)
        await sync_to_async(cache.set)(cache_key, text, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
        logger.debug(f"Preprocessed text: {text[:200]}")
        return text
    except Exception as e:
        logger.error(f"Text preprocessing failed: {e}")
        raise ValueError(f"Text preprocessing failed: {e}")

def get_negation_cues() -> List[str]:
    return settings.ICD_MATCHING_SETTINGS.get('NEGATION_CUES', [
        "no", "not", "denies", "negative", "without", "absent", "ruled out",
        "non", "never", "lacks", "excludes", "rules out", "negative for",
        "free of", "deny", "denying", "unremarkable for"
    ])

async def is_not_negated(condition: str, text: str, negation_cues: Optional[List[str]] = None, window: int = None) -> bool:
    if not condition or not text:
        logger.debug("No condition or text, returning True")
        return True

    condition_hash = hashlib.sha256(condition[:50].encode()).hexdigest()
    text_hash = hashlib.sha256(text[:1000].encode()).hexdigest()
    cache_key = f"negation_check_{condition_hash}_{text_hash}"
    cached_result = await sync_to_async(cache.get)(cache_key)
    if cached_result is not None:
        logger.debug(f"Cache hit for negation check: {condition} -> {cached_result}")
        return cached_result

    try:
        text_lower = await preprocess_text(text.lower())
        condition_lower = await preprocess_text(condition.lower())
        negation_cues = negation_cues or get_negation_cues()
        window = window or settings.ICD_MATCHING_SETTINGS.get('NEGATION_WINDOW', 10)

        negation_pattern = '|'.join(
            rf'\b{re.escape(cue)}\b(?:\s+(?:\w+\s+){{0,5}})?{re.escape(condition_lower)}\b'
            for cue in negation_cues
        )
        if re.search(negation_pattern, text_lower, re.IGNORECASE):
            logger.debug(f"Negation detected for condition: {condition}")
            await sync_to_async(cache.set)(cache_key, False, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
            return False

        matches = list(re.finditer(rf'\b{re.escape(condition_lower)}\b', text_lower, re.IGNORECASE))
        if not matches:
            logger.debug(f"No matches for condition: {condition}")
            await sync_to_async(cache.set)(cache_key, True, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
            return True

        for match in matches:
            start_idx = match.start()
            context_start = max(0, start_idx - window * 5)
            preceding_context = text_lower[context_start:start_idx]
            words = preceding_context.split()
            if any(cue in words[-window:] for cue in negation_cues):
                logger.debug(f"Negation cue found in context for condition: {condition}")
                await sync_to_async(cache.set)(cache_key, False, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
                return False

        logger.debug(f"No negation for condition: {condition}")
        await sync_to_async(cache.set)(cache_key, True, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
        return True
    except Exception as e:
        logger.error(f"Negation check failed for condition '{condition}': {e}")
        raise ValueError(f"Negation check failed: {e}")

async def check_mistral_health() -> bool:
    try:
        url = settings.ICD_MATCHING_SETTINGS.get('MISTRAL_LOCAL_URL', 'http://localhost:11434/api/generate')
        response = requests.get(url.replace('/generate', '/'), timeout=5)
        response.raise_for_status()
        logger.info("Mistral model is available")
        return True
    except Exception as e:
        logger.warning(f"Mistral model health check failed: {e}")
        return False

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def query_local_mistral(prompt: str) -> str:
    if not prompt.strip():
        logger.warning("Empty prompt provided")
        raise MistralQueryError("Empty prompt provided")

    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    cache_key = f"mistral_response_{prompt_hash}"
    cached = await sync_to_async(cache.get)(cache_key)
    if cached:
        logger.debug(f"Cache hit for Mistral response: {prompt[:100]}...")
        return cached

    try:
        url = settings.ICD_MATCHING_SETTINGS.get('MISTRAL_LOCAL_URL', 'http://localhost:11434/api/generate')
        payload = {
            "model": settings.ICD_MATCHING_SETTINGS.get('MISTRAL_MODEL', 'mistral'),
            "prompt": prompt,
            "options": {"temperature": 0.1, "num_predict": 500}
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                if 'response' not in data:
                    logger.warning("Invalid response format from Mistral")
                    raise MistralQueryError("Invalid response format")
                result += data['response']
        result = result.strip()
        if not result:
            logger.warning("Empty response from Mistral")
            raise MistralQueryError("Empty response from Mistral")
        await sync_to_async(cache.set)(cache_key, result, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
        logger.debug(f"Cached LLM response for prompt: {prompt[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Local Mistral query failed: {e}")
        raise MistralQueryError(f"Local Mistral query failed: {e}")

async def generate_patient_summary(patient_text: str) -> Tuple[str, List[str]]:
    if not patient_text or not isinstance(patient_text, str):
        logger.warning("No valid patient data provided")
        return "No patient data provided.", ["No conditions identified"]

    try:
        patient_text = await preprocess_text(patient_text)
        logger.debug(f"Preprocessed patient text: {patient_text}")
    except Exception as e:
        logger.error(f"Text preprocessing failed: {e}")
        raise ValueError(f"Text preprocessing failed: {e}")

    if not patient_text:
        return "The patient was suffering from no conditions identified.", ["No conditions identified"]

    prompt = (
        f"Analyze the following patient medical data and extract all diseases, conditions, and significant symptoms explicitly mentioned. "
        f"Generate a concise summary in one sentence describing the patient's condition. "
        f"List only specific, non-negated diseases or symptoms as bullet points starting with a hyphen. "
        f"Exclude vague terms like 'conditions other than' or 'follow-up examination' unless they specify a disease. "
        f"Ignore negated terms (e.g., 'no pain', 'denies fever') but include non-negated symptoms like 'pain' or 'swelling'. "
        f"Recognize medical abbreviations (e.g., 'B/L' as 'bilateral', 'INV' as 'investigation') and infer relevant medical terms from context. "
        f"Use synonyms from the medical knowledge base to enhance condition extraction. "
        f"Return format: Summary: <sentence>\nConditions:\n- <condition1>\n- <condition2>\n\n"
        f"Patient data: {patient_text}"
    )

    try:
        mistral_response = await query_local_mistral(prompt)
        logger.debug(f"LLM response: {mistral_response[:200]}...")

        if "Conditions:" in mistral_response:
            summary_part, condition_part = mistral_response.split("Conditions:", 1)
            conditions = [
                re.sub(r'\s*\(.*?\)', '', line).strip('- ').strip()
                for line in condition_part.split('\n')
                if line.strip().startswith('-') and line.strip('- ').strip()
            ]
            summary = summary_part.replace("Summary:", "").strip()
        else:
            summary = mistral_response.strip()
            conditions = [
                re.sub(r'^-+\s*', '', line).strip()
                for line in mistral_response.split('\n')
                if line.strip().startswith('-') and line.strip('- ').strip()
            ]

        conditions = [
            cond for cond in conditions
            if await is_not_negated(cond.split(' - ')[0], patient_text)
        ]

        if not conditions:
            conditions = ["No conditions identified"]
            summary = summary or "The patient was suffering from no conditions identified."

        logger.info(f"Final summary: {summary[:100]}...")
        logger.info(f"Final conditions extracted: {conditions}")
        return summary, conditions
    except MistralQueryError as e:
        logger.warning(f"LLM failed: {e}. Using regex fallback.")
        medical_terms = r'\b([a-zA-Z\s-]+(?:\s*(?:disease|disorder|syndrome|condition|pain|symptom|infection|loss))\b)'
        conditions = [
            match.group(0) for match in re.finditer(medical_terms, patient_text, re.IGNORECASE)
            if await is_not_negated(match.group(0), patient_text)
        ]
        conditions = list(set(conditions))
        if not conditions:
            conditions = ["No conditions identified"]
        summary = f"The patient was suffering from {' and '.join(conditions)}."
        logger.info(f"Fallback summary: {summary[:100]}...")
        logger.info(f"Fallback conditions: {conditions}")
        return summary, conditions
