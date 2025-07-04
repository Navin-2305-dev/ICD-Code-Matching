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
import hashlib
from asgiref.sync import sync_to_async
import re

logger = logging.getLogger(__name__)

class EmbedderSingleton:
    _instance = None
    _abbreviation_dict = {
        'b/l': 'bilateral',
        'inv': 'investigation',
        'htn': 'hypertension',
        'dm': 'diabetes mellitus',
        'cad': 'coronary artery disease',
        'copd': 'chronic obstructive pulmonary disease',
        'mi': 'myocardial infarction',
    }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            try:
                cls._instance = SentenceTransformer('dmis-lab/biobert-v1.1')
            except Exception as e:
                logger.warning(f"BioBERT failed: {e}. Using fallback.")
                cls._instance = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        return cls._instance

    @classmethod
    def expand_abbreviations(cls, text: str) -> str:
        for abbr, full in cls._abbreviation_dict.items():
            text = re.sub(rf'\b{re.escape(abbr)}\b', full, text, flags=re.IGNORECASE)
        return text

async def get_sentence_embedder():
    try:
        return EmbedderSingleton.get_instance()
    except Exception as e:
        logger.error(f"Error accessing embedder: {e}")
        raise ValueError(f"Error accessing embedder: {e}")

async def get_title_embedding(title: str, context: str = None) -> np.ndarray:
    title_hash = hashlib.sha256((title + (context or ''))[:1000].encode()).hexdigest()
    cache_key = f"title_embedding_{title_hash}_v1"
    cached_embedding = await sync_to_async(cache.get)(cache_key)
    if cached_embedding is not None:
        logger.debug(f"Cache hit for title embedding: {title}")
        return cached_embedding

    try:
        embedder = await get_sentence_embedder()
        text_to_embed = await preprocess_text(EmbedderSingleton.expand_abbreviations(title.lower()))
        if context:
            context = await preprocess_text(EmbedderSingleton.expand_abbreviations(context.lower()))
            text_to_embed = f"{text_to_embed} [CONTEXT] {context}"
        embedding = embedder.encode(text_to_embed, convert_to_numpy=True, normalize_embeddings=True)
        await sync_to_async(cache.set)(cache_key, embedding, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
        logger.debug(f"Cached title embedding for: {title}")
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding for title '{title}': {e}")
        raise ValueError(f"Embedding generation failed for title '{title}': {e}")

async def batch_encode_texts(texts: List[str], context: str = None, batch_size: int = None) -> np.ndarray:
    if not texts:
        logger.warning("Empty text list provided for batch encoding")
        return np.array([])

    batch_size = batch_size or settings.ICD_MATCHING_SETTINGS.get('BATCH_SIZE', 32)
    embedder = await get_sentence_embedder()
    results = []

    logger.debug(f"Batch encoding {len(texts)} texts with batch size {batch_size}")
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [await preprocess_text(EmbedderSingleton.expand_abbreviations(text)) if isinstance(text, str) else "" for text in batch]
            if context:
                context = await preprocess_text(EmbedderSingleton.expand_abbreviations(context))
                batch = [f"{text} [CONTEXT] {context}" for text in batch]
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

async def find_best_icd_match(conditions: List[str], patient_text: str) -> Dict[str, List[Tuple[str, str, float]]]:
    try:
        model = await get_sentence_embedder()
        icd_matches = {}

        for condition in conditions:
            norm_cond = await preprocess_text(condition.lower().strip())
            logger.debug(f"Processing condition: {norm_cond}")

            fts_results = await execute_fts_query(norm_cond, limit=30)
            logger.debug(f"FTS candidates for {norm_cond}: {fts_results}")

            if not fts_results:
                logger.warning(f"No FTS candidates found for {norm_cond}")
                icd_matches[condition] = []
                continue

            candidate_texts = [item['title'] for item in fts_results]
            candidate_codes = [item['code'] for item in fts_results]

            try:
                condition_embedding = await batch_encode_texts([norm_cond], context=patient_text)
                candidate_embeddings = await batch_encode_texts(candidate_texts, context=patient_text)

                similarities = np.dot(condition_embedding, candidate_embeddings.T).flatten()
                similarities = similarities / (
                    np.linalg.norm(condition_embedding) * np.linalg.norm(candidate_embeddings, axis=1)
                )
                logger.debug(f"Similarity scores for {norm_cond}: {similarities.tolist()}")

                boosted_similarities = similarities.copy()
                for i, code in enumerate(candidate_codes):
                    if len(code) > 3:
                        boosted_similarities[i] += 0.1

                max_score = max(boosted_similarities)
                if max_score > 1.0:
                    boosted_similarities = boosted_similarities / max_score
                boosted_similarities = boosted_similarities * 100

                condition_terms = norm_cond.split()
                term_overlaps = [
                    sum(1 for term in condition_terms if term in title.lower())
                    for title in candidate_texts
                ]

                combined_scores = []
                for i in range(len(candidate_codes)):
                    embedding_score = boosted_similarities[i]
                    overlap_score = term_overlaps[i] * 25
                    final_score = embedding_score if embedding_score > 75 else (embedding_score + overlap_score)
                    combined_scores.append((final_score, candidate_codes[i], candidate_texts[i]))

                combined_scores.sort(reverse=True)
                matches = [
                    (code, title, float(score))
                    for score, code, title in combined_scores
                    if float(score) > 75.0
                ]

                filtered_matches = []
                has_specific = any(len(match[0]) > 3 for match in matches)
                for match in matches:
                    code = match[0]
                    if has_specific and len(code) <= 3:
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
