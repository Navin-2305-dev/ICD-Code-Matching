import logging
import networkx as nx
from typing import List, Tuple, Set
from django.core.cache import cache
from django.conf import settings
from icd_matcher.models import ICDCategory
from icd_matcher.utils.text_processing import preprocess_text, reset_preprocess_counter
from icd_matcher.utils.db_utils import get_icd_title
from community import community_louvain
from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)

async def rebuild_knowledge_graph():
    """Build and cache the ICD knowledge graph efficiently."""
    reset_preprocess_counter()
    logger.info("Starting knowledge graph rebuild")
    graph = nx.DiGraph()

    # Process ICD entries in batches
    batch_size = settings.ICD_MATCHING_SETTINGS.get('GRAPH_BUILD_BATCH_SIZE', 1000)
    icd_entries = ICDCategory.objects.select_related('parent').all()
    try:
        total_entries = await sync_to_async(icd_entries.count)()
    except Exception as e:
        logger.error(f"Failed to count ICD entries: {e}")
        raise
    logger.debug(f"Processing {total_entries} ICD entries in batches of {batch_size}")

    preprocessed_titles = {}
    for start in range(0, total_entries, batch_size):
        try:
            batch = await sync_to_async(lambda: list(icd_entries[start:start + batch_size]))()
            batch_titles = {
                entry.code: await sync_to_async(preprocess_text)(entry.title, bypass_counter_limit=True)
                for entry in batch
            }
            preprocessed_titles.update(batch_titles)
            logger.debug(f"Preprocessed {len(batch_titles)} titles in batch {start//batch_size + 1}")
        except Exception as e:
            logger.error(f"Failed to process batch {start//batch_size + 1}: {e}")
            raise

    # Add nodes and parent-child edges
    for entry in await sync_to_async(lambda: list(icd_entries))():
        graph.add_node(
            entry.code,
            title=entry.title,
            inclusions=entry.get_inclusions(),
            exclusions=entry.get_exclusions()
        )
        if entry.parent:
            graph.add_edge(entry.parent.code, entry.code, type="parent_child", weight=1.0)
            graph.add_edge(entry.code, entry.parent.code, type="child_parent", weight=0.5)

    # Process inclusions and exclusions efficiently
    term_to_codes = {}
    for entry in preprocessed_titles:
        try:
            inclusions = (await sync_to_async(lambda: ICDCategory.objects.get(code=entry).get_inclusions)())[:2]
            exclusions = (await sync_to_async(lambda: ICDCategory.objects.get(code=entry).get_exclusions)())[:2]
            for term in inclusions:
                term = await sync_to_async(preprocess_text)(term, bypass_counter_limit=True)
                term_to_codes.setdefault(term.lower(), set()).add(entry)
            for term in exclusions:
                term = await sync_to_async(preprocess_text)(term, bypass_counter_limit=True)
                term_to_codes.setdefault(term.lower(), set()).add(entry)
        except Exception as e:
            logger.error(f"Failed to process inclusions/exclusions for {entry}: {e}")
            continue

    for entry, entry_title in preprocessed_titles.items():
        for term, codes in term_to_codes.items():
            if term in entry_title.lower() and entry not in codes:
                for code in codes:
                    if code != entry:
                        graph.add_edge(entry, code, type="semantic_inclusion", weight=0.8)
                        graph.add_edge(code, entry, type="semantic_exclusion", weight=0.2)

    undirected_graph = graph.to_undirected()
    communities = community_louvain.best_partition(undirected_graph, weight='weight')
    try:
        await sync_to_async(cache.set)("icd_knowledge_graph", graph, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
        await sync_to_async(cache.set)("icd_communities", communities, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
        logger.info("Knowledge graph rebuilt and cached")
    except Exception as e:
        logger.error(f"Failed to cache knowledge graph: {e}")
        raise
    reset_preprocess_counter()
    return graph, communities

class KnowledgeGraph:
    def __init__(self):
        """Initialize the knowledge graph."""
        self.graph = {}
        self.is_initialized = False
        # Define a synonym dictionary for medical terms
        self.synonyms = {
            'hearing loss': ['deafness'],
            'calculus': ['stone'],
            'bilateral': ['both'],
            'pain': ['ache', 'discomfort'],
            'fever': ['pyrexia'],
            'diabetes': ['hyperglycemia'],
        }

    async def initialize(self):
        """Asynchronously initialize the knowledge graph with ICD data."""
        logger.debug("Initializing knowledge graph")
        cache_key = "knowledge_graph_icd"
        cached_graph = await sync_to_async(cache.get)(cache_key)

        if cached_graph:
            self.graph = cached_graph
            self.is_initialized = True
            logger.info("Loaded knowledge graph from cache")
            return

        try:
            icd_entries = await sync_to_async(
                lambda: list(ICDCategory.objects.all().values('code', 'title', 'inclusions', 'exclusions'))
            )()
            logger.debug(f"Fetched {len(icd_entries)} ICD entries")
            for entry in icd_entries:
                code = entry['code']
                try:
                    inclusions = entry['inclusions'] or ''
                    exclusions = entry['exclusions'] or ''
                    if isinstance(inclusions, str):
                        inclusions = [term.strip() for term in inclusions.split(',') if term.strip()]
                    if isinstance(exclusions, str):
                        exclusions = [term.strip() for term in exclusions.split(',') if term.strip()]

                    self.graph[code] = {
                        'title': entry['title'],
                        'inclusions': inclusions,
                        'exclusions': exclusions,
                        'related_codes': []
                    }
                except Exception as e:
                    logger.error(f"Failed to process inclusions/exclusions for {code}: {e}")
                    self.graph[code] = {
                        'title': entry['title'],
                        'inclusions': [],
                        'exclusions': [],
                        'related_codes': []
                    }

            # Link related codes
            for code in self.graph:
                for other_code in self.graph:
                    if code != other_code:
                        title_match = any(
                            term in self.graph[other_code]['title'].lower()
                            for term in self.graph[code]['inclusions']
                        )
                        inclusion_match = any(
                            term in ' '.join(self.graph[other_code]['inclusions']).lower()
                            for term in self.graph[code]['inclusions']
                        )
                        same_category = code[:3] == other_code[:3]
                        if title_match or inclusion_match or same_category:
                            self.graph[code]['related_codes'].append(other_code)

            await sync_to_async(cache.set)(
                cache_key,
                self.graph,
                timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600)
            )
            self.is_initialized = True
            logger.info("Knowledge graph rebuilt and cached")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph: {e}")
            raise

    async def query(self, condition: str) -> List[str]:
        """Query the graph for related ICD codes."""
        if not self.is_initialized:
            await self.initialize()

        condition = condition.lower().strip()
        if not condition:
            logger.warning("Empty condition provided for Knowledge Graph query")
            return []

        condition_terms = condition.split()
        # Expand terms with synonyms
        expanded_terms = []
        for term in condition_terms:
            expanded_terms.append(term)
            for key, values in self.synonyms.items():
                if term == key:
                    expanded_terms.extend(values)
                elif term in values:
                    expanded_terms.append(key)

        scored_codes = []

        for code, data in self.graph.items():
            title = data['title'].lower() if data['title'] else ''
            inclusions = [inc.lower() for inc in data['inclusions']]
            exclusions = [exc.lower() for exc in data['exclusions']]

            # Calculate term overlap score, including synonyms
            term_overlap = sum(
                1 for term in expanded_terms
                if any(term in title or term in inc for inc in inclusions)
            )
            if term_overlap == 0:
                continue

            # Exclude codes where condition terms match exclusions
            excluded = any(
                any(term in exc for term in condition_terms) for exc in exclusions
            )
            if excluded:
                continue

            # Expand range codes (e.g., H90-H95)
            expanded_codes = [code]
            if '-' in code:
                start, end = code.split('-')
                start_letter = start[0]
                start_num = int(start[1:]) if start[1:].isdigit() else 0
                end_num = int(end[1:]) if end[1:].isdigit() else 0
                for i in range(start_num, end_num + 1):
                    expanded_code = f"{start_letter}{i}"
                    if expanded_code in self.graph:
                        expanded_codes.append(expanded_code)

            # Include related codes
            all_codes = list(set(expanded_codes + data['related_codes']))
            specificity_score = max(len(c) for c in all_codes)  # Higher for more specific codes
            scored_codes.append((term_overlap, specificity_score, all_codes))

        # Sort by term overlap, then specificity
        scored_codes.sort(key=lambda x: (x[0], x[1]), reverse=True)
        related_codes = []
        for _, _, codes in scored_codes:
            codes.sort(key=len, reverse=True)
            related_codes.extend(codes)
            if len(related_codes) >= 15:
                break

        related_codes = list(set(related_codes))[:15]
        logger.debug(f"Knowledge graph query for '{condition}' returned: {related_codes}")
        return related_codes