import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
from django.core.cache import cache
from django.conf import settings
from icd_matcher.models import ICDCategory
from icd_matcher.utils.text_processing import preprocess_text, is_not_negated
from icd_matcher.utils.knowledge_graph import KnowledgeGraph
from icd_matcher.utils.db_utils import execute_fts_query, get_icd_title
import uuid

logger = logging.getLogger(__name__)

class EmbedderSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            try:
                cls._instance = SentenceTransformer('dmis-lab/biobert-v1.1')
            except Exception as e:
                logger.warning(f"BioBERT failed: {e}. Using fallback.")
                cls._instance = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        return cls._instance

def get_sentence_embedder():
    try:
        return EmbedderSingleton.get_instance()
    except Exception as e:
        logger.error(f"Error accessing embedder: {e}")
        raise ValueError(f"Error accessing embedder: {e}")

def get_title_embedding(title: str) -> np.ndarray:
    cache_key = f"title_embedding_{title[:50]}_{uuid.uuid4().hex}"
    cached_embedding = cache.get(cache_key)
    if cached_embedding is not None:
        logger.debug(f"Cache hit for title embedding: {title}")
        return cached_embedding

    try:
        embedder = get_sentence_embedder()
        embedding = embedder.encode(title.lower(), convert_to_numpy=True, normalize_embeddings=True)
        cache.set(cache_key, embedding, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
        logger.debug(f"Cached title embedding for: {title}")
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding for title '{title}': {e}")
        raise ValueError(f"Embedding generation failed for title '{title}': {e}")

def batch_encode_texts(texts: List[str], batch_size: int = None) -> np.ndarray:
    if not texts:
        logger.warning("Empty text list provided for batch encoding")
        return np.array([])

    batch_size = batch_size or settings.ICD_MATCHING_SETTINGS.get('BATCH_SIZE', 32)
    embedder = get_sentence_embedder()
    results = []

    logger.debug(f"Batch encoding {len(texts)} texts with batch size {batch_size}")
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch = [text if isinstance(text, str) else "" for text in batch]
            if not batch:
                continue
            batch_embeddings = embedder.encode(
                batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            results.append(batch_embeddings)
        encoded = np.vstack(results) if results else np.array([])
        logger.debug(f"Completed batch encoding, resulting shape: {encoded.shape}")
        return encoded
    except Exception as e:
        logger.error(f"Batch encoding failed: {e}")
        raise ValueError(f"Batch encoding failed: {e}")

def find_best_icd_match(conditions: List[str], patient_text: str) -> Dict[str, List[Tuple[str, str, float]]]:
    """Find the best ICD code matches for given conditions."""
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        icd_matches = {}

        for condition in conditions:
            norm_cond = condition.lower().strip()
            logger.debug(f"Processing condition: {norm_cond}")

            # Fetch candidate ICD codes
            fts_results = execute_fts_query(norm_cond, limit=30)
            logger.debug(f"FTS candidates for {norm_cond}: {fts_results}")

            if not fts_results:
                logger.warning(f"No FTS candidates found for {norm_cond}")
                icd_matches[condition] = []
                continue

            # Prepare texts for embedding
            candidate_texts = [item['title'] for item in fts_results]
            candidate_codes = [item['code'] for item in fts_results]

            try:
                # Compute embeddings
                condition_embedding = model.encode([norm_cond], convert_to_tensor=True)
                candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True)

                # Compute cosine similarities
                similarities = np.dot(condition_embedding, candidate_embeddings.T) / (
                    np.linalg.norm(condition_embedding) * np.linalg.norm(candidate_embeddings, axis=1)
                )
                similarities = similarities.flatten()
                logger.debug(f"Similarity scores for {norm_cond}: {similarities.tolist()}")

                # Boost scores for more specific codes
                boosted_similarities = similarities.copy()
                for i, code in enumerate(candidate_codes):
                    if len(code) > 3:  # Subcodes (e.g., N20.1)
                        boosted_similarities[i] += 0.1

                # Normalize scores to 0-100 range
                max_score = max(boosted_similarities)
                if max_score > 1.0:
                    boosted_similarities = boosted_similarities / max_score
                boosted_similarities = boosted_similarities * 100

                # Compute term overlap as a fallback
                condition_terms = norm_cond.split()
                term_overlaps = []
                for title in candidate_texts:
                    title_lower = title.lower()
                    overlap = sum(1 for term in condition_terms if term in title_lower)
                    term_overlaps.append(overlap)

                # Combine embedding scores with term overlap
                combined_scores = []
                for i in range(len(candidate_codes)):
                    embedding_score = boosted_similarities[i]
                    overlap_score = term_overlaps[i] * 25  # Increase weight of overlap
                    # If embedding score is low, boost with overlap
                    final_score = embedding_score if embedding_score > 75 else (embedding_score + overlap_score)
                    combined_scores.append((final_score, candidate_codes[i], candidate_texts[i]))

                # Get top matches
                combined_scores.sort(reverse=True)
                matches = [
                    (code, title, float(score))
                    for score, code, title in combined_scores
                    if float(score) > 75.0  # Higher threshold
                ]

                # Filter out category codes if specific subcodes are present
                filtered_matches = []
                has_specific = any(len(match[0]) > 3 for match in matches)
                for match in matches:
                    code = match[0]
                    if has_specific and len(code) <= 3:  # Skip category codes
                        continue
                    filtered_matches.append(match)

                icd_matches[condition] = filtered_matches
                logger.debug(f"Matches for {norm_cond}: {filtered_matches}")
            except Exception as e:
                logger.error(f"Embedding computation failed for {norm_cond}: {e}")
                icd_matches[condition] = []

        return icd_matches
    except Exception as e:
        logger.error(f"Error in find_best_icd_match: {e}")
        return {condition: [] for condition in conditions}