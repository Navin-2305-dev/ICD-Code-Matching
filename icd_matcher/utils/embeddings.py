import logging
import functools
import re
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from sentence_transformers import SentenceTransformer
from django.core.cache import cache
from django.conf import settings
from icd_matcher.models import ICDCategory
from icd_matcher.utils.db_utils import get_icd_title, execute_fts_query
import concurrent.futures
from django.db.models import Q
from django.db import connection, transaction
from icd_matcher.utils.text_processing import preprocess_text, query_mistral, is_not_negated
from .exceptions import (
    EmbedderLoadError, EmbeddingGenerationError, FTSQueryError, SimilarityCalculationError,
    MistralQueryError, TextPreprocessingError
)

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=1)
def get_embedder():
    """Lazily load the sentence transformer model."""
    try:
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Failed to load sentence transformer model: {e}")
        raise EmbedderLoadError(f"Failed to load sentence transformer model: {e}")

def get_sentence_embedder():
    """Get the sentence embedder with error handling."""
    try:
        embedder = get_embedder()
        if embedder is None:
            raise ValueError("Embedder not available")
        return embedder
    except Exception as e:
        logger.error(f"Error accessing sentence transformer model: {e}")
        raise EmbedderLoadError(f"Error accessing sentence transformer model: {e}")

def clear_embedding_cache():
    """Clear the embedding cache."""
    logger.info("Embedding cache clearing is handled by Django's cache framework")

def get_title_embedding(title: str) -> np.ndarray:
    """Get the embedding for a title with caching."""
    cache_key = f"title_embedding_{hash(title)}"
    cached_embedding = cache.get(cache_key)
    
    if cached_embedding is not None:
        logger.debug(f"Cache hit for title embedding: {title}")
        return cached_embedding
    
    logger.debug(f"Cache miss for title embedding: {title}")
    try:
        embedder = get_sentence_embedder()
        embedding = embedder.encode(title.lower(), convert_to_numpy=True, normalize_embeddings=True)
        cache.set(cache_key, embedding, timeout=settings.ICD_MATCHING_SETTINGS['CACHE_TTL'])
        logger.debug(f"Cached title embedding for: {title}")
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding for title '{title}': {e}")
        raise EmbeddingGenerationError(f"Embedding generation failed for title '{title}': {e}")

def batch_encode_texts(texts: List[str], batch_size: int = None) -> np.ndarray:
    """Batch encode a list of texts into embeddings."""
    if batch_size is None:
        batch_size = settings.ICD_MATCHING_SETTINGS['BATCH_SIZE']
    
    embedder = get_sentence_embedder()
    results = []
    
    logger.debug(f"Batch encoding {len(texts)} texts with batch size {batch_size}")
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = embedder.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            results.append(batch_embeddings)
        
        encoded = np.vstack(results) if results else np.array([])
        logger.debug(f"Completed batch encoding, resulting shape: {encoded.shape}")
        return encoded
    except Exception as e:
        logger.error(f"Batch encoding failed: {e}")
        raise EmbeddingGenerationError(f"Batch encoding failed: {e}")

def find_best_icd_match(
    conditions: List[str], 
    patient_medical_data: str, 
    existing_codes: Optional[List[str]] = None,
    use_parallel: bool = True
) -> Dict[str, List[Tuple[str, str, float]]]:
    """Find the best ICD matches for a list of conditions."""
    if not conditions:
        logger.warning("No conditions provided.")
        return {}
    
    results = {}
    try:
        patient_medical_data = preprocess_text(patient_medical_data or "")
    except Exception as e:
        logger.error(f"Text preprocessing failed: {e}")
        raise TextPreprocessingError(f"Text preprocessing failed: {e}")
    
    existing_codes = existing_codes or []
    
    preprocessed_conditions = [
        cond for cond in [preprocess_text(cond) for cond in conditions if cond]
        if is_not_negated(cond.split(' - ')[0], patient_medical_data)
    ]
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
        condition_texts = [cond for _, cond in valid_conditions]
        if condition_texts:
            condition_embeddings = batch_encode_texts(condition_texts)
        else:
            condition_embeddings = np.array([])
    except EmbeddingGenerationError as e:
        logger.error(f"Batch embedding generation failed: {e}")
        condition_embeddings = np.zeros((len(valid_conditions), 384))
    
    max_workers = getattr(settings, 'ICD_MATCHING_SETTINGS', {}).get('MAX_WORKERS', 4)
    if use_parallel and len(valid_conditions) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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

def _process_single_condition(
    original_condition: str,
    condition_text: str, 
    condition_idx: int,
    condition_embedding: Optional[np.ndarray],
    patient_data: str,
    status_words: Set[str]
) -> List[Tuple[str, str, float]]:
    """Process a single condition to find matching ICD codes."""
    logger.info(f"Processing condition: {original_condition}")
    query_term = re.sub('|'.join(rf'\b{word}\b' for word in status_words), '', condition_text).strip() or condition_text
    
    # Define synonyms for common medical terms
    synonym_map = {
        'bilateral ureteric calculus': ['kidney stones', 'ureteral calculi', 'renal calculi'],
        'kidney stones': ['ureteric calculus', 'renal calculi', 'urolithiasis'],
        'hearing loss': ['deafness', 'auditory impairment'],
    }
    
    try:
        # Sanitize query term for FTS: remove problematic characters
        sanitized_query = re.sub(r'[^\w\s-]', '', query_term).strip()
        fts_results = []
        if sanitized_query:
            # Try exact phrase match first
            exact_query = f'"{sanitized_query}"'
            fts_results = execute_fts_query(exact_query, limit=10)
            if len(fts_results) < 3:
                logger.info(f"Exact FTS returned {len(fts_results)} results for '{original_condition}'. Trying loose match.")
                fts_results.extend(execute_fts_query(sanitized_query, limit=10))
                fts_results = list(set(fts_results))[:10]  # Remove duplicates
        
        # Try synonyms if results are insufficient
        if len(fts_results) < 3 and original_condition.lower() in synonym_map:
            logger.info(f"Insufficient FTS results for '{original_condition}'. Trying synonyms: {synonym_map[original_condition.lower()]}")
            for synonym in synonym_map[original_condition.lower()]:
                sanitized_synonym = re.sub(r'[^\w\s-]', '', synonym).strip()
                if sanitized_synonym:
                    fts_results.extend(execute_fts_query(sanitized_synonym, limit=10))
            fts_results = list(set(fts_results))[:10]  # Remove duplicates
    except FTSQueryError as e:
        logger.error(f"FTS query failed for condition '{original_condition}': {e}")
        fts_results = []
    
    if len(fts_results) < 3:
        logger.info(f"FTS returned {len(fts_results)} results for '{original_condition}'. Supplementing with ORM.")
        try:
            # Build a more specific ORM query
            query = Q(title__icontains=query_term)
            for word in query_term.split():
                if len(word) > 3:
                    query &= (Q(title__icontains=word) | Q(inclusions__icontains=word))
            
            # Include synonyms in ORM query
            if original_condition.lower() in synonym_map:
                for synonym in synonym_map[original_condition.lower()]:
                    query |= Q(title__icontains=synonym) | Q(inclusions__icontains=synonym)
            
            with transaction.atomic():
                matches = (
                    ICDCategory.objects.filter(query)
                    .filter(icdcategory__isnull=True)
                    .select_related('parent')
                    .only('id', 'code', 'title', 'parent__code', 'inclusions', 'exclusions')
                    .order_by('title')[:10]  # Limit to most relevant
                )
                
                existing_codes = {result[0] for result in fts_results}
                for match in matches:
                    if match.code not in existing_codes:
                        fts_results.append((match.code, match.title))
                        existing_codes.add(match.code)
        except Exception as e:
            logger.error(f"ORM query failed for condition '{original_condition}': {e}")
    
    if not fts_results:
        logger.warning(f"No matches found for '{original_condition}'.")
        return []
    
    icd_codes = [row[0] for row in fts_results]
    max_candidates = getattr(settings, 'ICD_MATCHING_SETTINGS', {}).get('MAX_CANDIDATES', 25)
    if len(icd_codes) > max_candidates:
        icd_codes = icd_codes[:max_candidates]
    
    try:
        with transaction.atomic():
            matches = (
                ICDCategory.objects.filter(code__in=icd_codes)
                .select_related('parent')
                .only('id', 'code', 'title', 'parent__code', 'inclusions', 'exclusions')
            )
            
            matches = [m for m in matches if not hasattr(m, 'icdcategory_set') or not m.icdcategory_set.exists()]
            
            if not matches:
                logger.warning(f"No leaf node matches for '{original_condition}'.")
                return [(fts_results[0][0], get_icd_title(fts_results[0][0]), 70.0)] if fts_results else []
            
            match_titles = [m.title.lower() for m in matches]
            
            try:
                if condition_embedding is None:
                    embedder = get_sentence_embedder()
                    condition_embedding = embedder.encode(
                        condition_text,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                
                match_embeddings = batch_encode_texts(match_titles)
                condition_embedding_reshaped = condition_embedding.reshape(1, -1)
                similarities = np.dot(match_embeddings, condition_embedding_reshaped.T).flatten()
                
                min_score = settings.ICD_MATCHING_SETTINGS['MIN_SIMILARITY_SCORE']
                max_score = settings.ICD_MATCHING_SETTINGS['MAX_SIMILARITY_SCORE']
                inclusion_boost = getattr(settings, 'ICD_MATCHING_SETTINGS', {}).get('INCLUSION_BOOST', 1.1)
                exclusion_penalty = getattr(settings, 'ICD_MATCHING_SETTINGS', {}).get('EXCLUSION_PENALTY', 0.5)
                
                similarity_scores = []
                for i, sim in enumerate(similarities):
                    score = max(min_score, min(max_score, (sim + 1) * 50))
                    match = matches[i]
                    
                    inclusion_match = False
                    exclusion_match = False
                    
                    if match.inclusions:
                        inclusions = preprocess_text(match.inclusions).lower()
                        for term in inclusions.split():
                            if term in patient_data.lower():
                                inclusion_match = True
                                score *= inclusion_boost
                                break
                    
                    if match.exclusions:
                        exclusions = preprocess_text(match.exclusions).lower()
                        for term in exclusions.split():
                            if term in patient_data.lower():
                                exclusion_match = True
                                score *= exclusion_penalty
                                break
                    
                    if exclusion_match and not inclusion_match:
                        continue
                    
                    if not is_not_negated(match.title, patient_data):
                        continue
                    
                    similarity_scores.append((match.code, match.title, min(max_score, score)))
                
                similarity_scores.sort(key=lambda x: x[2], reverse=True)
                
            except Exception as e:
                logger.error(f"Similarity calculation failed: {e}")
                raise SimilarityCalculationError(f"Similarity calculation failed for condition '{original_condition}': {e}")
            
            if not similarity_scores:
                logger.warning(f"No similarity matches for '{original_condition}'.")
                return [(fts_results[0][0], get_icd_title(fts_results[0][0]), 70.0)] if fts_results else []
            
            candidate_list = similarity_scores
            candidate_text = "\n".join([
                f"{code}: {title} ({score:.1f}%)" +
                (f"\n  Inclusions: {next((m.inclusions for m in matches if m.code == code), 'None')}" if next((m.inclusions for m in matches if m.code == code), None) else "") +
                (f"\n  Exclusions: {next((m.exclusions for m in matches if m.code == code), 'None')}" if next((m.exclusions for m in matches if m.code == code), None) else "")
                for code, title, score in candidate_list
            ])
            
            try:
                prompt = (
                    f"Analyze the patient's medical data and the clinical condition: '{original_condition}'.\n\n"
                    f"Your task is to select *all clinically relevant ICD-10 codes* from the candidate list below that match the given patient condition and data, "
                    f"based on clinical relevance and context. Ensure to:\n"
                    f"- Include all codes that are appropriate for the condition and supported by the medical data.\n"
                    f"- Exclude any negated conditions.\n"
                    f"- Prioritize codes from the candidate list.\n"
                    f"Return the results in the following format, one per line:\n"
                    f"[icd_code]: icd_title (X%)\n"
                    f"Where 'X' is your confidence level (0 to 100) based on how well the code matches the condition and context.\n"
                    f"If no ICD-10 codes are appropriate, return '[]'.\n\n"
                    f"Patient Medical Data:\n{patient_data}\n\n"
                    f"Candidate ICD-10 Codes with Similarity Scores:\n{candidate_text}"
                )
                
                mistral_response = query_mistral(prompt)
                logger.debug(f"Mistral response for '{original_condition}': {mistral_response[:100]}...")
                
                matched_codes = []
                if mistral_response.strip() and mistral_response.strip() != '[]':
                    for line in mistral_response.split('\n'):
                        match = re.search(r'(?:\[)?([A-Za-z]\d+(?:\.\d+)?)(?:\])?:?\s*(.+?)\s*\((\d+(?:\.\d+)?)(?:%|\))', line.strip())
                        if match:
                            code, title, percent = match.groups()
                            code = code.strip()
                            percent = float(percent)
                            
                            db_title = get_icd_title(code)
                            if db_title != "Unknown title":
                                matched_codes.append((code, db_title, percent))
                                logger.debug(f"Accepted match: {code} ({percent}%)")
                
                matched_codes_dict = {}
                for code, title, score in matched_codes:
                    if code not in matched_codes_dict or score > matched_codes_dict[code][2]:
                        matched_codes_dict[code] = (code, title, score)
                
                final_matches = list(matched_codes_dict.values())
                
                if final_matches:
                    return sorted(final_matches, key=lambda x: x[2], reverse=True)
                
            except MistralQueryError as e:
                logger.error(f"Mistral refinement failed for '{original_condition}': {e}")
                # Return all similarity scores as fallback
                return [(code, title, score) 
                        for code, title, score in sorted(similarity_scores, key=lambda x: x[2], reverse=True)
                        if is_not_negated(title, patient_data)]
                    
    except Exception as e:
        logger.error(f"Error processing matches for '{original_condition}': {e}")
        return [(fts_results[0][0], get_icd_title(fts_results[0][0]), 70.0)] if fts_results else []