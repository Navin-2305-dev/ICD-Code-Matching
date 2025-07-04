import logging
import networkx as nx
from typing import List, Dict
from django.core.cache import cache
from django.conf import settings
from icd_matcher.models import ICDCategory
from icd_matcher.utils.text_processing import preprocess_text, reset_preprocess_counter
from icd_matcher.utils.db_utils import get_icd_title
from icd_matcher.utils.synonyms import load_synonyms
from community import community_louvain
from asgiref.sync import sync_to_async
from sentence_transformers import SentenceTransformer, util
import json
import os
import re
import torch

logger = logging.getLogger(__name__)

# Define excluded codes and prefixes as a constant
EXCLUDED_CODES = {
    'specific_codes': [
        'A28.8', 'A33', 'A38', 'A41', 'A44', 'A48', 'A49', 'L98.2', 'W19',
        'D50.0', 'F44.6', 'F50.8', 'F52.0', 'I65.3', 'I66.4', 'K08.1',
        'K14.0', 'K14.1', 'K14.2', 'K14.4', 'K14.6', 'K14.9', 'K40.0',
        'K40.1', 'K40.2', 'K41.0', 'K41.1', 'K41.2', 'L65.8', 'L65.9',
        'L89.1', 'L98.7', 'N27.1', 'P02.1', 'P50.0', 'P50.1', 'P50.2',
        'P50.5', 'P50.8', 'P50.9', 'P61.3', 'Q36.0', 'Q37.0', 'Q37.2',
        'Q37.4', 'Q37.8', 'Q38.1', 'Q38.2', 'Q38.3', 'Q53.0', 'Q53.2',
        'Q60.1', 'Q60.4', 'Q65.1', 'Q65.4', 'Q89.2', 'R49.1', 'R63.0',
        'R63.4', 'S04.8', 'S05.2', 'S05.3', 'Z56.2', 'Z61.0', 'Z61.3',
        'Z89.3'
    ],
    'prefixes': ['T', 'E', 'G'],
    'z_prefix_exceptions': ['Z82.2']
}

async def rebuild_knowledge_graph():
    reset_preprocess_counter()
    logger.info("Starting knowledge graph rebuild")
    graph = nx.DiGraph()

    batch_size = settings.ICD_MATCHING_SETTINGS.get('GRAPH_BUILD_BATCH_SIZE', 1000)
    icd_entries = ICDCategory.objects.select_related('parent').all()
    total_entries = await sync_to_async(icd_entries.count)()

    preprocessed_titles = {}
    symptom_nodes = {}
    async for entry in icd_entries.iterator(chunk_size=batch_size):
        try:
            title = await preprocess_text(entry.title, bypass_counter_limit=True)
            preprocessed_titles[entry.code] = title
            graph.add_node(
                entry.code,
                type='icd_code',
                title=entry.title,
                inclusions=entry.get_inclusions() or [],
                exclusions=entry.get_exclusions() or [],
                definition=entry.definition or ''
            )
            if entry.parent:
                graph.add_edge(entry.parent.code, entry.code, type="parent_child", weight=1.0)
                graph.add_edge(entry.code, entry.parent.code, type="child_parent", weight=0.5)
            
            for symptom in entry.get_inclusions():
                symptom = symptom.lower().strip()
                if symptom:
                    symptom_nodes[symptom] = symptom_nodes.get(symptom, set()) | {entry.code}
                    graph.add_node(
                        f"symptom_{symptom}",
                        type='symptom',
                        description=symptom
                    )
                    graph.add_edge(entry.code, f"symptom_{symptom}", type="has_symptom", weight=0.7)
                    graph.add_edge(f"symptom_{symptom}", entry.code, type="related_to", weight=0.3)
        except Exception as e:
            logger.error(f"Failed to process entry {entry.code}: {e}")
            continue

    term_to_codes = {}
    async for entry in icd_entries.iterator(chunk_size=batch_size):
        try:
            inclusions = entry.get_inclusions()[:2]
            exclusions = entry.get_exclusions()[:2]
            for term in inclusions + exclusions:
                term = await preprocess_text(term, bypass_counter_limit=True)
                term_to_codes.setdefault(term.lower(), set()).add(entry.code)
        except Exception as e:
            logger.error(f"Failed to process inclusions/exclusions for {entry.code}: {e}")
            continue

    for entry, entry_title in preprocessed_titles.items():
        for term, codes in term_to_codes.items():
            if term in entry_title.lower() and entry not in codes:
                for code in codes:
                    if code != entry:
                        graph.add_edge(entry, code, type="semantic_inclusion", weight=0.8)
                        graph.add_edge(code, entry, type="semantic_exclusion", weight=0.2)

    undirected_graph = graph.to_undirected()
    communities = community_louvain.best_partition(undirected_graph, weight='weight', resolution=1.2)
    try:
        await sync_to_async(cache.set)("icd_knowledge_graph_v1", graph, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
        await sync_to_async(cache.set)("icd_communities_v1", communities, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600))
        logger.info("Knowledge graph rebuilt and cached")
    except Exception as e:
        logger.error(f"Failed to cache knowledge graph: {e}")
        raise
    reset_preprocess_counter()
    return graph, communities

class KnowledgeGraph:
    def __init__(self):
        self.graph = {}
        self.is_initialized = False
        self.synonyms = {}
        self.model = None

    async def initialize(self):
        logger.debug("Initializing knowledge graph")
        cache_key = "knowledge_graph_icd_v1"
        cached_graph = await sync_to_async(cache.get)(cache_key)

        if cached_graph:
            self.graph = cached_graph
            self.synonyms = await load_synonyms()
            self.model = await sync_to_async(SentenceTransformer)(
                'sentence-transformers/all-MiniLM-L6-v2',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.is_initialized = True
            logger.info("Loaded knowledge graph from cache")
            return

        try:
            self.synonyms = await load_synonyms()
            self.model = await sync_to_async(SentenceTransformer)(
                'sentence-transformers/all-MiniLM-L6-v2',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            icd_entries = await sync_to_async(
                lambda: list(ICDCategory.objects.all().values('code', 'title', 'inclusions', 'exclusions', 'definition'))
            )()
            logger.debug(f"Fetched {len(icd_entries)} ICD entries")
            for entry in icd_entries:
                code = entry['code']
                try:
                    inclusions = entry['inclusions'] or []
                    exclusions = entry['exclusions'] or []
                    if isinstance(inclusions, str):
                        inclusions = [term.strip() for term in inclusions.split(',') if term.strip()]
                    if isinstance(exclusions, str):
                        exclusions = [term.strip() for term in exclusions.split(',') if term.strip()]
                    self.graph[code] = {
                        'title': entry['title'],
                        'inclusions': inclusions,
                        'exclusions': exclusions,
                        'definition': entry['definition'] or '',
                        'related_codes': []
                    }
                except Exception as e:
                    logger.error(f"Failed to process data for {code}: {e}")
                    self.graph[code] = {
                        'title': entry['title'],
                        'inclusions': [],
                        'exclusions': [],
                        'definition': '',
                        'related_codes': []
                    }

            for code in self.graph:
                for other_code in self.graph:
                    if code != other_code and code[:1] == other_code[:1]:
                        title_match = any(
                            term in self.graph[other_code]['title'].lower()
                            for term in self.graph[code]['inclusions']
                        )
                        inclusion_match = any(
                            term in ' '.join(self.graph[other_code]['inclusions']).lower()
                            for term in self.graph[code]['inclusions']
                        )
                        definition_match = any(
                            term in self.graph[other_code]['definition'].lower()
                            for term in self.graph[code]['inclusions']
                        )
                        if title_match or inclusion_match or definition_match:
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

    async def query(self, condition: str) -> List[Dict]:
        if not self.is_initialized:
            await self.initialize()

        condition = condition.lower().strip()
        condition = re.sub(r'\bseizureure\b', 'seizure', condition)
        condition = re.sub(r'\s*\([^)]+\)', '', condition)
        if condition in ['b/l hearing loss', 'hearing loss', 'unspecified hearing loss', 'family history of hearing loss']:
            condition = 'bilateral hearing loss'
        if not condition:
            logger.warning("Empty condition provided for Knowledge Graph query")
            return []

        condition = condition.replace('b/l', 'bilateral').replace('hbp', 'hypertension').replace('dm', 'diabetes mellitus').replace('fs', 'febrile seizure').replace('se', 'status epilepticus')
        condition_terms = condition.split()
        expanded_terms = condition_terms.copy()
        for term in condition_terms:
            for key, values in self.synonyms.items():
                if term == key or term in [v.lower() for v in values]:
                    expanded_terms.extend([key] + values)

        scored_codes = []
        specific_codes = set(EXCLUDED_CODES['specific_codes'])
        prefixes = set(EXCLUDED_CODES['prefixes'])
        z_exceptions = set(EXCLUDED_CODES['z_prefix_exceptions'])
        condition_embedding = self.model.encode(condition, convert_to_tensor=True)
        for code, data in self.graph.items():
            if code in specific_codes or any(code.startswith(prefix) for prefix in prefixes) or \
               (code.startswith('Z') and code not in z_exceptions and re.match(r'^Z[0-8][0-9]\..*', code)):
                continue
            title = data['title'].lower() if data['title'] else ''
            inclusions = [inc.lower() for inc in data['inclusions']]
            definition = data['definition'].lower() if data['definition'] else ''

            exact_match = condition in title or condition in inclusions or condition in definition
            term_overlap = sum(
                1 for term in expanded_terms
                if term in title or term in ' '.join(inclusions) or term in definition
            )
            if term_overlap < 0.5 and not exact_match:
                continue

            excluded = any(
                any(term in exc for term in condition_terms) for exc in data['exclusions']
            )
            if excluded:
                continue

            candidate_text = f"{title} {' '.join(inclusions)} {definition}"
            candidate_embedding = self.model.encode(candidate_text, convert_to_tensor=True)
            score = util.cos_sim(condition_embedding, candidate_embedding)[0][0].item()
            if score > 0.5:
                scored_codes.append({'code': code, 'title': data['title'], 'similarity': score * 100})

        scored_codes.sort(key=lambda x: x['similarity'], reverse=True)
        related_codes = scored_codes
        logger.debug(f"Knowledge graph query for '{condition}' returned: {related_codes}")
        return related_codes
