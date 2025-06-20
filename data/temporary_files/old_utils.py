# For Refernce
import re
import logging
import functools
from typing import List, Tuple, Dict, Optional, Union, Any, Set
import numpy as np
import requests
from django.db import connection, transaction
from django.db.models import Q
from django.db.models.functions import Length
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import concurrent.futures
from functools import lru_cache
from icd_matcher.models import ICDCategory

logger = logging.getLogger(__name__)

EMBEDDING_CACHE = {}
TITLE_EMBEDDING_CACHE = {}
ICD_TITLE_CACHE = {}

@functools.lru_cache(maxsize=1)
def get_embedder():
    """Lazily load the sentence transformer model only when needed"""
    try:
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Failed to load sentence transformer model: {e}")
        return None

def get_sentence_embedder():
    """Get the sentence embedder with error handling"""
    try:
        embedder = get_embedder()
        if embedder is None:
            raise ValueError("Embedder not available")
        return embedder
    except Exception as e:
        logger.error(f"Error accessing sentence transformer model: {e}")
        class DummyEmbedder:
            def encode(self, sentences, **kwargs):
                if isinstance(sentences, str):
                    return np.zeros(384)
                return np.zeros((len(sentences), 384))
        return DummyEmbedder()

# Setup FTS5 table for ICD lookup
def setup_fts_table():
    logger.info("Setting up FTS5 table for ICD matching")
    with connection.cursor() as cursor:
        # Drop existing FTS table if it exists
        cursor.execute("DROP TABLE IF EXISTS icd_fts")
        
        # Create new FTS5 table
        cursor.execute("""
            CREATE VIRTUAL TABLE icd_fts USING fts5(
                code UNINDEXED,
                title,
                tokenize='porter unicode61'
            )
        """)
        
        # Populate FTS table with all ICD codes
        cursor.execute("""
            INSERT INTO icd_fts(code, title)
            SELECT code, title FROM icd_matcher_icdcategory
        """)
        
        # Create triggers to keep FTS table in sync
        cursor.execute("""
            CREATE TRIGGER icd_ai AFTER INSERT ON icd_matcher_icdcategory
            BEGIN
                INSERT INTO icd_fts(code, title) VALUES (NEW.code, NEW.title);
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER icd_ad AFTER DELETE ON icd_matcher_icdcategory
            BEGIN
                DELETE FROM icd_fts WHERE code = OLD.code;
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER icd_au AFTER UPDATE ON icd_matcher_icdcategory
            BEGIN
                UPDATE icd_fts SET title = NEW.title WHERE code = NEW.code;
            END
        """)
        
    logger.info("FTS5 table setup complete")

def clear_embedding_cache():
    global EMBEDDING_CACHE, TITLE_EMBEDDING_CACHE, ICD_TITLE_CACHE
    EMBEDDING_CACHE.clear()
    TITLE_EMBEDDING_CACHE.clear()
    ICD_TITLE_CACHE.clear()
    logger.info("Embedding cache cleared")

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[,.;:!?](?!\d)', ' ', text)
    return text

# Getting negated cues 
@lru_cache(maxsize=1)
def get_negation_cues() -> List[str]:
    return [
        "no", "not", "denies", "negative", "without", "absent", "ruled out", 
        "non", "never", "lacks", "excludes", "rules out", "negative for", 
        "free of", "deny", "denying", "unremarkable for"
    ]

# Matching negated terms
def is_not_negated(condition: str, text: str, negation_cues: Optional[List[str]] = None, window: int = 8) -> bool:
    if not condition or not text:
        return True
    
    text_lower = text.lower() if isinstance(text, str) else ""
    condition_lower = condition.lower() if isinstance(condition, str) else ""
    
    if not text_lower or not condition_lower:
        return True
    
    if not negation_cues:
        negation_cues = get_negation_cues()
    
    # Combined pattern for efficiency
    negation_pattern = '|'.join(negation_cues)
    
    # Check direct negation patterns
    for cue in negation_cues:
        pattern = rf'\b{re.escape(cue)}\b(?:\s+\w+){{0,3}}\s+{re.escape(condition_lower)}\b'
        if re.search(pattern, text_lower):
            return False
    
    # Check for window-based negation
    matches = list(re.finditer(rf'\b{re.escape(condition_lower)}\b', text_lower))
    if not matches:
        return True
    
    for match in matches:
        start_idx = match.start()
        context_start = max(0, start_idx - 50)
        preceding_context = text_lower[context_start:start_idx]
        
        if not any(cue in preceding_context.split() for cue in negation_cues):
            return True
    
    return False

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.RequestException, requests.Timeout, ConnectionError))
)
def query_mistral(prompt: str, max_retries: int = 3) -> str:
    """
    Query the local Mistral model with improved error handling and retries
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral", 
        "prompt": prompt, 
        "stream": False, 
        "temperature": 0.1,
    }
    
    try:
        logger.debug(f"Querying Mistral with prompt: {prompt[:100]}...")
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except requests.RequestException as e:
        logger.error(f"Mistral request failed: {e}")
        raise
    except Exception as е:
        logger.error(f"Unexpected error in Mistral query: {e}")
        return ""

# Generate patient summary and extract conditions
def generate_patient_summary(patient_text: str) -> Tuple[str, List[str]]:
    if not patient_text or not isinstance(patient_text, str):
        return "No patient data provided.", ["No conditions identified"]
    
    patient_text = preprocess_text(patient_text)
    if not patient_text:
        return "The patient was suffering from no conditions identified.", ["No conditions identified"]
    
    prompt = (
        f"From the following patient data, generate a concise summary of the patient's condition starting with "
        f"'The patient was suffering from [disease]', optionally followed by ', received [medication/treatment]' and/or ', and recovered in [days] days' "
        f"only if those details are explicitly mentioned. Extract all diseases, conditions, and significant symptoms "
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
            
            # Fallback to asterisk list if no hyphen list found
            if not conditions:
                conditions = [line.strip() for line in condition_part.split('\n') if line.strip().startswith('*')]
                conditions = [re.sub(r'^\*\s*', '', cond).strip() for cond in conditions if cond.strip()]
            
            # Further fallback to extracting from summary
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
        
        if not conditions or all(c.lower() == "no conditions identified" for c in conditions):
            conditions = ["No conditions identified"]
            if not summary.strip():
                summary = "The patient was suffering from no conditions identified."
    except Exception as e:
        logger.warning(f"Mistral failed: {e}. Using fallback extraction.")
        words = patient_text.split()
        conditions = []
        for i in range(len(words)):
            for j in range(1, min(7, len(words) - i)):
                phrase = ' '.join(words[i:i+j])
                if 2 <= len(phrase.split()) <= 6 and len(phrase) > 5:
                    conditions.append(phrase)
        
        conditions = list(set(conditions))[:10]
        
        if not conditions:
            conditions = ["No conditions identified"]
        
        summary = f"The patient was suffering from {' and '.join(conditions[:3]) if conditions and conditions[0] != 'No conditions identified' else 'no conditions identified'}"

    logger.info(f"Summary generated: {summary[:100]}...")
    logger.info(f"Conditions extracted: {conditions}")
    return summary, conditions

# Getting title embedding
def get_title_embedding(title: str) -> np.ndarray:
    global TITLE_EMBEDDING_CACHE
    
    if title in TITLE_EMBEDDING_CACHE:
        return TITLE_EMBEDDING_CACHE[title]
    
    embedder = get_sentence_embedder()
    embedding = embedder.encode(title.lower(), convert_to_numpy=True, normalize_embeddings=True)
    
    # Only cache if we have space (limit cache size)
    if len(TITLE_EMBEDDING_CACHE) < 10000:
        TITLE_EMBEDDING_CACHE[title] = embedding
    
    return embedding

# Batch encoding texts 
def batch_encode_texts(texts: List[str], batch_size: int = 50) -> np.ndarray:
    embedder = get_sentence_embedder()
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedder.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        results.append(batch_embeddings)
    
    return np.vstack(results) if results else np.array([])

def execute_fts_query(query_terms: str, limit: int = 25) -> List[Tuple[str, str]]:
    results = []
    
    query_terms = query_terms.replace("'", "''")
    
    try:
        with connection.cursor() as cursor:
            sql = f"SELECT code, title FROM icd_fts WHERE title MATCH ? ORDER BY rank LIMIT ?"
            logger.debug(f"Executing FTS query: {sql} with terms: {query_terms}")
            cursor.execute(sql, (query_terms, limit))
            results = cursor.fetchall()
    except Exception as e:
        logger.error(f"FTS query failed: {e}")
    
    return results

def get_icd_title(code: str) -> str:
    """Get the title for an ICD code with caching"""
    global ICD_TITLE_CACHE
    if code in ICD_TITLE_CACHE:
        return ICD_TITLE_CACHE[code]
    
    try:
        icd_entry = ICDCategory.objects.filter(code=code).only('title').first()
        title = icd_entry.title if icd_entry else "Unknown title"
        if len(ICD_TITLE_CACHE) < 10000:
            ICD_TITLE_CACHE[code] = title
        return title
    except Exception as e:
        logger.error(f"Error fetching title for code {code}: {e}")
        return "Unknown title"

# Find Matching ICD Codes
def find_best_icd_match(
    conditions: List[str], 
    patient_medical_data: str, 
    existing_codes: Optional[List[str]] = None,
    use_parallel: bool = True
) -> Dict[str, List[Tuple[str, str, float]]]:
    if not conditions:
        logger.warning("No conditions provided.")
        return {}
    
    results = {}
    patient_medical_data = preprocess_text(patient_medical_data or "")
    existing_codes = existing_codes or []
    
    preprocessed_conditions = [preprocess_text(cond) for cond in conditions if preprocess_text(cond)]
    if not preprocessed_conditions:
        return {}
    
    valid_conditions = [(idx, cond) for idx, cond in enumerate(preprocessed_conditions) 
                       if cond.lower() != "no conditions identified"]
    
    if not valid_conditions:
        for condition in conditions:
            results[condition] = []
        return results
    
    status_words = {"resolved", "active", "chronic", "recovered", "past", "history", "of"}
    
    try:
        condition_indices = [idx for idx, _ in valid_conditions]
        condition_texts = [cond for _, cond in valid_conditions]
        
        if condition_texts:
            condition_embeddings = batch_encode_texts(condition_texts)
        else:
            condition_embeddings = np.array([])
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {e}")
        condition_embeddings = np.zeros((len(valid_conditions), 384))
    
    # Process each condition (parallel or sequential)
    if use_parallel and len(valid_conditions) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(valid_conditions))) as executor:
            future_to_condition = {
                executor.submit(
                    _process_single_condition, 
                    conditions[idx], 
                    cond_text,
                    idx,
                    condition_embeddings[i] if len(condition_embeddings) > i else None,
                    patient_medical_data,
                    status_words
                ): conditions[idx]
                for i, (idx, cond_text) in enumerate(valid_conditions)
            }
            
            for future in concurrent.futures.as_completed(future_to_condition):
                condition = future_to_condition[future]
                try:
                    condition_results = future.result()
                    results[condition] = condition_results
                except Exception as e:
                    logger.error(f"Error processing condition '{condition}': {e}")
                    results[condition] = []
    else:
        for i, (idx, cond_text) in enumerate(valid_conditions):
            try:
                embedding = condition_embeddings[i] if len(condition_embeddings) > i else None
                condition_results = _process_single_condition(
                    conditions[idx], 
                    cond_text,
                    idx,
                    embedding,
                    patient_medical_data,
                    status_words
                )
                results[conditions[idx]] = condition_results
            except Exception as e:
                logger.error(f"Error processing condition '{conditions[idx]}': {e}")
                results[conditions[idx]] = []
    
    for condition in conditions:
        if condition not in results:
            results[condition] = []
    
    return results

# Process a single condition to find all matching ICD codes
def _process_single_condition(
    original_condition: str,
    condition_text: str, 
    condition_idx: int,
    condition_embedding: Optional[np.ndarray],
    patient_data: str,
    status_words: Set[str]
) -> List[Tuple[str, str, float]]:
    
    logger.info(f"Processing condition: {original_condition}")
    query_term = re.sub('|'.join(rf'\b{word}\b' for word in status_words), '', condition_text).strip() or condition_text
    
    fts_results = execute_fts_query(f'"{query_term}"')
    
    # Fall back to ORM
    if len(fts_results) < 3:
        logger.info(f"FTS returned {len(fts_results)} results for '{original_condition}'. Supplementing with ORM.")
        
        # Create query for title matching
        query = Q(title__icontains=query_term)
        
        # Add individual word matching for terms over 3 chars
        for word in query_term.split():
            if len(word) > 3:
                query |= Q(title__icontains=word)
        
        with transaction.atomic():
            matches = (
                ICDCategory.objects.filter(query)
                .filter(icdcategory__isnull=True)
                .select_related('parent')
                .only('id', 'code', 'title', 'parent__code')
            )
            
            existing_codes = {result[0] for result in fts_results}
            for match in matches:
                if match.code not in existing_codes:
                    fts_results.append((match.code, match.title))
                    existing_codes.add(match.code)
    
    if not fts_results:
        logger.warning(f"No matches found for '{original_condition}'.")
        return []
    
    # Get ICD codes from FTS results
    icd_codes = [row[0] for row in fts_results]
    
    # Fetching leaf node ICD Values
    with transaction.atomic():
        try:
            matches = (
                ICDCategory.objects.filter(code__in=icd_codes)
                .select_related('parent')
                .only('id', 'code', 'title', 'parent__code')
            )
            
            matches = [m for m in matches if not hasattr(m, 'icdcategory_set') or not m.icdcategory_set.exists()]
            
            if not matches:
                logger.warning(f"No leaf node matches for '{original_condition}'.")
                return [(fts_results[0][0], get_icd_title(fts_results[0][0]), 60.0)] if fts_results else []
            
            match_titles = [m.title.lower() for m in matches]
            
            try:
                if condition_embedding is None:
                    embedder = get_sentence_embedder()
                    condition_embedding = embedder.encode(
                        condition_text,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                
                # Batch encode all titles at once
                match_embeddings = batch_encode_texts(match_titles)
                
                # Calculate similarities
                condition_embedding_reshaped = condition_embedding.reshape(1, -1)
                similarities = np.dot(match_embeddings, condition_embedding_reshaped.T).flatten()
                
                # Calculate similarity scores for all matches above threshold
                similarity_scores = [
                    (matches[i].code, matches[i].title, max(50, min(95, (sim + 1) * 50)))
                    for i, sim in enumerate(similarities)
                    if (sim + 1) * 50 >= 50
                ]
                
                # Sort by similarity score
                similarity_scores.sort(key=lambda x: x[2], reverse=True)
                
            except Exception as e:
                logger.error(f"Similarity calculation failed: {e}")
                similarity_scores = [(m.code, m.title, 60.0) for m in matches]
            
            if not similarity_scores:
                logger.warning(f"No similarity matches for '{original_condition}'.")
                return [(fts_results[0][0], get_icd_title(fts_results[0][0]), 60.0)] if fts_results else []
            
            candidate_list = similarity_scores
            candidate_text = "\n".join([f"{code}: {title} ({score:.1f}%)" for code, title, score in candidate_list])
            
            try:
                prompt = (
                    f"Analyze the patient's medical data and the clinical condition: '{original_condition}'.\n\n"
                    f"Your task is to select *all clinically relevant ICD-10 codes* from the candidate list below that match the given patient condition and data, "
                    f"based on clinical relevance and context. Ensure to:\n"
                    f"- Include all codes that are appropriate for the condition and supported by the medical data.\n"
                    f"- Exclude any negated conditions.\n"
                    f"- Prioritize codes from the candidate list.\n"
                    f"- You may suggest additional ICD-10 codes only if they are clairement supported by the patient's medical data.\n"
                    f"Return the results in the following format, one per line:\n"
                    f"[icd_code]: icd_title (X%)\n"
                    f"Where 'X' is your confidence level (0 to 100) based on how well the code matches the condition and context.\n"
                    f"If no ICD-10 codes are appropriate, return '[]'.\n\n"
                    f"Patient Medical Data:\n{patient_data}\n\n"
                    f"Candidate ICD-10 Codes with Similarity Scores:\n{candidate_text}"
                )
                
                mistral_response = query_mistral(prompt)
                print(f"Prompt sent to Mistral: {prompt}...")
                print(mistral_response)
                logger.debug(f"Mistral response for '{original_condition}': {mistral_response[:100]}...")
                
                matched_codes = []
                if mistral_response.strip() and mistral_response.strip() != '[]':
                    for line in mistral_response.split('\n'):
                        match = re.search(r'(?:\[)?([A-Za-z]\d+(?:\.\d+)?)(?:\])?:?\s*(.+?)\s*\((\d+(?:\.\d+)?)(?:%|\))', line.strip())
                        if match:
                            code, title, percent = match.groups()
                            code = code.strip()
                            percent = float(percent)
                            
                            # Validate code exists in database
                            db_title = get_icd_title(code)
                            if db_title != "Unknown title":
                                matched_codes.append((code, db_title, percent))
                                logger.debug(f"Accepted match: {code} ({percent}%)")
                
                # Remove duplicates
                matched_codes_dict = {}
                for code, title, score in matched_codes:
                    if code not in matched_codes_dict or score > matched_codes_dict[code][2]:
                        matched_codes_dict[code] = (code, title, score)
                
                final_matches = list(matched_codes_dict.values())
                
                # Return Mistral's results if available
                if final_matches:
                    return sorted(final_matches, key=lambda x: x[2], reverse=True)
                
            except Exception as e:
                logger.error(f"Mistral refinement failed for '{original_condition}': {e}")
            
            # Return all similarity scores as fallback
            return [(code, title, score) 
                    for code, title, score in sorted(similarity_scores, key=lambda x: x[2], reverse=True)]
                    
        except Exception as e:
            logger.error(f"Error processing matches for '{original_condition}': {e}")
            return [(fts_results[0][0], get_icd_title(fts_results[0][0]), 60.0)] if fts_results else []