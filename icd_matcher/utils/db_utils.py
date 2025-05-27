import logging
import re
from typing import List, Tuple, Optional
from django.core.cache import cache
from django.db import connection
from django.db.models import Q
from django.conf import settings
from icd_matcher.models import ICDCategory
from icd_matcher.utils.exceptions import FTSQueryError, ICDTitleRetrievalError

logger = logging.getLogger(__name__)

def setup_fts_table():
    """Set up the FTS5 table for ICD matching with triggers."""
    logger.info("Setting up FTS5 table for ICD matching")
    try:
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS icd_fts")
            cursor.execute("""
                CREATE VIRTUAL TABLE icd_fts USING fts5(
                    code UNINDEXED,
                    title,
                    inclusions,
                    exclusions,
                    tokenize='porter unicode61'
                )
            """)
            cursor.execute("""
                INSERT INTO icd_fts(code, title, inclusions, exclusions)
                SELECT code, title, inclusions, exclusions FROM icd_matcher_icdcategory
            """)
            cursor.execute("""
                CREATE TRIGGER icd_ai AFTER INSERT ON icd_matcher_icdcategory
                BEGIN
                    INSERT INTO icd_fts(code, title, inclusions, exclusions)
                    VALUES (NEW.code, NEW.title, NEW.inclusions, NEW.exclusions);
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
                    UPDATE icd_fts SET title = NEW.title, inclusions = NEW.inclusions, exclusions = NEW.exclusions
                    WHERE code = NEW.code;
                END
            """)
        logger.info("FTS5 table setup complete")
    except Exception as e:
        logger.error(f"Failed to set up FTS5 table: {e}")
        raise ValueError(f"FTS5 table setup failed: {e}")

def get_icd_title(code: str) -> str:
    """Retrieve the title for an ICD code."""
    if not code:
        return None

    cache_key = f"icd_title_{code.upper()}"
    cached_title = cache.get(cache_key)
    if cached_title is not None:
        logger.debug(f"Cache hit for ICD title: {code}")
        return cached_title

    try:
        icd_entry = ICDCategory.objects.filter(code__iexact=code).only('title').first()
        title = icd_entry.title if icd_entry else None
        cache.set(cache_key, title, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
        logger.debug(f"Cached ICD title for {code}: {title}")
        return title
    except Exception as e:
        logger.error(f"Error fetching title for code {code}: {e}")
        return None

def execute_fts_query(query: str, limit: int = 20) -> list:
    """Execute a full-text search query on ICD categories using SQLite."""
    try:
        query = query.lower().strip()
        terms = query.split()
        synonyms = []

        # Require all terms to be present
        query_conditions = Q()
        for term in terms:
            query_conditions &= (Q(title__icontains=term) | Q(inclusions__icontains=term))

        # Add synonyms as optional
        synonym_conditions = Q()
        for term in synonyms:
            synonym_conditions |= Q(title__icontains=term) | Q(inclusions__icontains=term)

        # Combine conditions
        final_conditions = query_conditions & synonym_conditions if synonyms else query_conditions

        results = ICDCategory.objects.filter(final_conditions).values('code', 'title')[:limit]
        results_list = list(results)

        if len(results_list) < limit:
            # Fallback to broader search
            broad_conditions = Q()
            for term in terms + synonyms:
                broad_conditions |= Q(title__icontains=term) | Q(inclusions__icontains=term)
            additional_results = ICDCategory.objects.filter(broad_conditions).exclude(
                code__in=[r['code'] for r in results_list]
            ).values('code', 'title')[:limit - len(results_list)]
            results_list.extend(additional_results)

        logger.debug(f"SQLite FTS query results for '{query}': {results_list}")
        return results_list
    except Exception as e:
        logger.error(f"FTS query failed for {query}: {e}")
        return []