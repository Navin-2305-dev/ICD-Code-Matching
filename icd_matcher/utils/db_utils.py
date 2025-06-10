import logging
import re
from typing import List, Tuple, Optional
from django.core.cache import cache
from django.db import connection
from django.db.models import Q
from django.conf import settings
from icd_matcher.models import ICDCategory
from icd_matcher.utils.exceptions import FTSQueryError, ICDTitleRetrievalError
from asgiref.sync import sync_to_async

from icd_matcher.utils.text_processing import preprocess_text

logger = logging.getLogger(__name__)

async def setup_fts_table():
    """Set up the FTS5 table for ICD matching with triggers."""
    logger.info("Setting up FTS5 table for ICD matching")
    try:
        async with connection.cursor() as cursor:
            await cursor.execute("DROP TABLE IF EXISTS icd_fts")
            await cursor.execute("""
                CREATE VIRTUAL TABLE icd_fts USING fts5(
                    code UNINDEXED,
                    title,
                    inclusions,
                    exclusions,
                    tokenize='porter unicode61'
                )
            """)
            await cursor.execute("""
                INSERT INTO icd_fts(code, title, inclusions, exclusions)
                SELECT code, title, inclusions, exclusions FROM icd_matcher_icdcategory
            """)
            await cursor.execute("""
                CREATE TRIGGER icd_ai AFTER INSERT ON icd_matcher_icdcategory
                BEGIN
                    INSERT INTO icd_fts(code, title, inclusions, exclusions)
                    VALUES (NEW.code, NEW.title, NEW.inclusions, NEW.exclusions);
                END
            """)
            await cursor.execute("""
                CREATE TRIGGER icd_ad AFTER DELETE ON icd_matcher_icdcategory
                BEGIN
                    DELETE FROM icd_fts WHERE code = OLD.code;
                END
            """)
            await cursor.execute("""
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

async def get_icd_title(code: str) -> str:
    """Retrieve the title for an ICD code."""
    if not code:
        return None

    cache_key = f"icd_title_{code.upper()}_v1"
    cached_title = await sync_to_async(cache.get)(cache_key)
    if cached_title is not None:
        logger.debug(f"Cache hit for ICD title: {code}")
        return cached_title

    try:
        icd_entry = await sync_to_async(
            lambda: ICDCategory.objects.filter(code__iexact=code).only('title').first()
        )()
        title = icd_entry.title if icd_entry else None
        await sync_to_async(cache.set)(cache_key, title, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
        logger.debug(f"Cached ICD title for {code}: {title}")
        return title
    except Exception as e:
        logger.error(f"Error fetching title for code {code}: {e}")
        return None

async def execute_fts_query(query: str, limit: int = 20) -> list:
    """Execute a full-text search query on ICD categories using SQLite."""
    try:
        query = (await sync_to_async(preprocess_text)(query)).lower().strip()
        if not query:
            return []

        terms = query.split()
        query_conditions = Q()
        for term in terms:
            query_conditions &= (Q(title__icontains=term) | Q(inclusions__icontains=term))

        results = await sync_to_async(
            lambda: list(ICDCategory.objects.filter(query_conditions).values('code', 'title')[:limit])
        )()
        
        if len(results) < limit:
            broad_conditions = Q()
            for term in terms:
                broad_conditions |= Q(title__icontains=term) | Q(inclusions__icontains=term)
            additional_results = await sync_to_async(
                lambda: list(
                    ICDCategory.objects.filter(broad_conditions)
                    .exclude(code__in=[r['code'] for r in results])
                    .values('code', 'title')[:limit - len(results)]
                )
            )()
            results.extend(additional_results)

        logger.debug(f"FTS query results for '{query}': {results}")
        return results
    except Exception as e:
        logger.error(f"FTS query failed for {query}: {e}")
        return []