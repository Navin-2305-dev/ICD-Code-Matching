import logging
import re
from typing import List, Tuple, Optional
from django.core.cache import cache
from django.db import connection, transaction
from django.db.models import Q
from django.conf import settings
from icd_matcher.models import ICDCategory
from .exceptions import DatabaseSetupError, FTSQueryError, ICDTitleRetrievalError

logger = logging.getLogger(__name__)

def setup_fts_table():
    """Set up the FTS5 table for ICD matching with triggers for synchronization."""
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
        raise DatabaseSetupError(f"FTS5 table setup failed: {e}")

def execute_fts_query(query_terms: str, limit: int = None) -> List[Tuple[str, str]]:
    """Execute an FTS query to retrieve ICD codes and titles."""
    if limit is None:
        limit = getattr(settings, 'ICD_MATCHING_SETTINGS', {}).get('FTS_QUERY_LIMIT', 25)
    
    results = []
    if not query_terms or not isinstance(query_terms, str):
        logger.warning("Empty or invalid query terms provided")
        return results
    
    # Sanitize query terms: remove problematic characters and escape quotes
    query_terms = query_terms.strip()
    query_terms = re.sub(r'\s+', ' ', query_terms)
    query_terms = query_terms.replace("'", "''")
    
    # Split query into words for partial matching
    words = query_terms.split()
    if len(words) > 1:
        # Construct FTS query with OR for individual words and exact phrase
        fts_query = f'"{query_terms}" OR ' + ' OR '.join(words)
    else:
        fts_query = query_terms
    
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT code, title FROM icd_fts 
                WHERE icd_fts MATCH %s 
                ORDER BY rank LIMIT %s
            """
            logger.debug(f"Executing FTS query: {sql} with terms: {fts_query}, limit: {limit}")
            cursor.execute(sql, (fts_query, limit))
            results = cursor.fetchall()
        if not results:
            logger.info(f"No FTS results for query: {fts_query}")
        logger.debug(f"FTS query results: {results}")
        return results
    except Exception as e:
        logger.error(f"FTS query failed for terms '{fts_query}': {e}")
        raise FTSQueryError(f"FTS query execution failed: {e}")

def get_icd_title(code: str) -> str:
    """Get the title for an ICD code with caching."""
    cache_key = f"icd_title_{code}"
    cached_title = cache.get(cache_key)
    
    if cached_title is not None:
        logger.debug(f"Cache hit for ICD title: {code}")
        return cached_title
    
    logger.debug(f"Cache miss for ICD title: {code}")
    try:
        icd_entry = ICDCategory.objects.filter(code=code).only('title').first()
        title = icd_entry.title if icd_entry else "Unknown title"
        cache_ttl = getattr(settings, 'ICD_MATCHING_SETTINGS', {}).get('CACHE_TTL', 3600)
        cache.set(cache_key, title, timeout=cache_ttl)
        logger.debug(f"Cached ICD title for {code}: {title}")
        return title
    except Exception as e:
        logger.error(f"Error fetching title for code {code}: {e}")
        raise ICDTitleRetrievalError(f"Failed to retrieve title for ICD code {code}: {e}")