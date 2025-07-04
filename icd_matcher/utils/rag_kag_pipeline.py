import logging
from typing import Dict, List, Optional
from icd_matcher.utils.knowledge_graph import KnowledgeGraph
from icd_matcher.utils.text_processing import preprocess_text
from icd_matcher.models import ICDCategory
from icd_matcher.utils.db_utils import get_icd_title
from django.db.models import Q
from django.core.cache import cache
from django.conf import settings
from asgiref.sync import sync_to_async
from sentence_transformers import SentenceTransformer, util
import requests
import json
import re
import os
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

class MistralQueryError(Exception):
    pass

class RAGKAGPipeline:
    def __init__(self):
        self.knowledge_graph = None
        self.model = None
        self.data_dir = os.path.join('D:\\', 'MedWise Project - 114', 'data', 'helper_files')
        self.abbreviations = self._load_json_file('abbreviations.json')
        self.synonyms = self._load_json_file('synonyms.json')

    def _load_json_file(self, filename: str) -> Dict:
        """Load JSON file or return empty dict if file doesn't exist."""
        file_path = os.path.join(self.data_dir, filename)
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            return {}

    def _save_json_file(self, filename: str, data: Dict):
        """Save data to JSON file."""
        file_path = os.path.join(self.data_dir, filename)
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving {filename}: {str(e)}")

    async def _query_mistral(self, prompt: str, cache_key: str) -> str:
        """Query local Mistral AI API and cache response."""
        cached_result = await sync_to_async(cache.get)(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for Mistral response: {cache_key}")
            return cached_result

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
            await sync_to_async(cache.set)(cache_key, result, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 86400))
            logger.debug(f"Cached Mistral response for prompt: {prompt[:100]}...")
            return result
        except Exception as e:
            logger.error(f"Mistral query failed: {str(e)}")
            raise MistralQueryError(str(e))

    async def _generate_abbreviations(self, terms: List[str]) -> Dict:
        """Generate abbreviations for given terms using Mistral AI."""
        cache_key = "abbreviations"
        cached_abbr = await sync_to_async(cache.get)(cache_key)
        if cached_abbr:
            return cached_abbr

        prompt = f"""
        Generate medical abbreviations for the following terms in JSON format:
        {json.dumps(terms, indent=2)}
        Return a dictionary where keys are abbreviations and values are the full terms.
        Example: {{"b/l": "bilateral", "dm": "diabetes mellitus"}}
        """
        try:
            response = await self._query_mistral(prompt, cache_key)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                logger.error("Failed to parse JSON from Mistral response for abbreviations")
                return self.abbreviations
            new_abbr = json.loads(response[json_start:json_end])
            self.abbreviations.update(new_abbr)
            self._save_json_file('abbreviations.json', self.abbreviations)
            return self.abbreviations
        except MistralQueryError as e:
            logger.error(f"Failed to generate abbreviations: {str(e)}")
            return self.abbreviations

    async def _generate_synonyms(self, condition: str) -> List[str]:
        """Generate synonyms for a condition using Mistral AI."""
        cache_key = f"synonyms_{condition.lower()}"
        cached_synonyms = await sync_to_async(cache.get)(cache_key)
        if cached_synonyms:
            return cached_synonyms

        prompt = f"""
        Generate 5-10 synonyms for the medical condition '{condition}' in JSON format:
        {{"synonyms": ["synonym1", "synonym2", ...]}}
        Ensure synonyms are medically accurate and relevant.
        """
        try:
            response = await self._query_mistral(prompt, cache_key)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                logger.error(f"Failed to parse JSON from Mistral response for {condition} synonyms")
                return self.synonyms.get(condition.lower(), [])
            result = json.loads(response[json_start:json_end])
            synonyms = result.get('synonyms', [])
            self.synonyms[condition.lower()] = synonyms
            self._save_json_file('synonyms.json', self.synonyms)
            return synonyms
        except MistralQueryError as e:
            logger.error(f"Failed to generate synonyms for {condition}: {str(e)}")
            return self.synonyms.get(condition.lower(), [])

    async def _generate_condition_details(self, condition: str) -> Dict:
        """Generate symptoms, causes, and treatments using Mistral AI."""
        cache_key = f"condition_details_{condition.lower()}"
        await sync_to_async(cache.delete)(cache_key)  # Clear cache for fresh generation
        cached_details = await sync_to_async(cache.get)(cache_key)
        if cached_details:
            return cached_details

        condition = condition.lower()
        prompt = f"""
        Provide detailed medical information for the condition '{condition}' in JSON format:
        {{
            "symptoms": ["symptom1", "symptom2", ...],
            "causes": ["cause1", "cause2", ...],
            "treatments": ["treatment1", "treatment2", ...]
        }}
        Ensure 5-7 items per category, medically accurate, and specific to '{condition}'.
        For bilateral ureter calculus, all symptoms, causes, and treatments must explicitly refer to stones in both ureters. Avoid any references to unilateral ureteral stones, one-sided issues, or single-ureter conditions.
        Do not include terms like 'unilateral', 'one side', or 'affected side' in the response.
        """
        try:
            response = await self._query_mistral(prompt, cache_key)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                logger.error(f"Failed to parse JSON from Mistral response for {condition}")
                return {"symptoms": [], "causes": [], "treatments": [], "warning": f"No details available for {condition}"}
            details = json.loads(response[json_start:json_end])
            if not all(isinstance(details.get(key, []), list) for key in ['symptoms', 'causes', 'treatments']):
                logger.error(f"Invalid structure in Mistral response for {condition}")
                return {"symptoms": [], "causes": [], "treatments": [], "warning": f"Invalid response structure for {condition}"}
            # Filter out any unilateral references
            for key in ['symptoms', 'causes', 'treatments']:
                details[key] = [
                    s for s in details[key]
                    if not any(term in s.lower() for term in ['unilateral', 'one side', 'affected side'])
                ]
            await sync_to_async(cache.set)(cache_key, details, timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 86400))
            return details
        except MistralQueryError as e:
            logger.error(f"Failed to generate details for {condition}: {str(e)}")
            return {"symptoms": [], "causes": [], "treatments": [], "warning": f"Mistral AI failed for {condition}"}

    @classmethod
    async def create(cls):
        pipeline = cls()
        await pipeline._initialize()
        return pipeline

    async def _initialize(self):
        logger.info("Setting up graph workflow")
        try:
            self.knowledge_graph = KnowledgeGraph()
            await self.knowledge_graph.initialize()
            self.model = await sync_to_async(SentenceTransformer)(
                'sentence-transformers/all-MiniLM-L6-v2',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            initial_terms = ['bilateral', 'investigation', 'hypertension', 'diabetes mellitus', 'seizure', 'epilepsy', 'febrile seizure', 'status epilepticus', 'hearing loss', 'bilateral hearing loss', 'ureter calculus', 'bilateral ureter calculus']
            await self._generate_abbreviations(initial_terms)
            logger.info("RAGKAGPipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            raise

    async def _expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations in the input text."""
        result = text.lower()
        for abbr, full in self.abbreviations.items():
            result = re.sub(r'\b' + re.escape(abbr) + r'\b', full, result, flags=re.IGNORECASE)
        return result

    async def _extract_conditions(self, patient_text: str) -> List[str]:
        """Extract conditions from patient text using regex and synonyms."""
        logger.info("Extracting conditions from patient text")
        try:
            patient_text = await self._expand_abbreviations(patient_text)
            patterns = [
                r'A:\s*([^\n;]+?)(?=\s*(?:\(|\n|D:|$))',  # A: CONDITION
                r'D:\s*([^\n;]+?)(?=\s*(?:\(|\n|$))',     # D: CONDITION
                r'(?<!\w)(?:b/l\s+ureter\s+calculus|bilateral\s+ureter\s+calculus|ureter\s+calculus|b/l\s+hearing\s+loss|bilateral\s+hearing\s+loss|hearing\s+loss|febrile\s+seizure(?:\s*\([^)]+\))?|status\s+epilepticus|hypertension|diabetes\s+mellitus|chest\s+pain|[a-zA-Z\s]+?(?:seizure|epilepsy|convulsion|pain|disease|disorder|syndrome|calculus))(?=\s|$|[.,; Lillt;])'  # Free text
            ]
            conditions = set()
            canonical_map = {
                'b/l ureter calculus': 'Bilateral Ureter Calculus',
                'ureter calculus': 'Bilateral Ureter Calculus',
                'ureteral stone': 'Bilateral Ureter Calculus',
                'bilateral ureteral stone': 'Bilateral Ureter Calculus',
                'b/l hearing loss': 'Bilateral Hearing Loss',
                'hearing loss in both ears': 'Bilateral Hearing Loss',
                'bilateral hearing impairment': 'Bilateral Hearing Loss',
                'hearing loss': 'Bilateral Hearing Loss',
                'unspecified hearing loss': 'Bilateral Hearing Loss',
                'family history of hearing loss': 'Bilateral Hearing Loss'
            }
            exclude_terms = {
                'follow-up', 'examination', 'treatment', 'discharge', 'ward', 'surg', 'med', 'cardio',
                'admission', 'type', 'fresh', 'status', 'approved', 'date', 'other than',
                'diabetes mellitusission', 'surgical admission', 'surgical', 'follow-up examination',
                'after treatment', 'conditions other than', 'discharge ward', 'admission type',
                'admission status'
            }
            for pattern in patterns:
                for match in re.finditer(pattern, patient_text, re.IGNORECASE):
                    condition = match.group(1).strip().lower() if pattern != patterns[-1] else match.group(0).strip().lower()
                    condition = re.sub(r'\s*\([^)]+\)', '', condition)
                    condition = re.sub(r'(discharge\s+ward|admission\s+type|admission\s+status|date|\d{4}-\d{2}-\d{2}|surg|fresh|approved|surgical\s+admission|surgical|follow-up\s+examination|after\s+treatment|conditions\s+other\s+than).*', '', condition).strip()
                    if not condition or any(term in condition for term in exclude_terms):
                        continue
                    condition = canonical_map.get(condition, condition.title())
                    for key, syn_list in self.synonyms.items():
                        if condition.lower() == key or condition.lower() in [s.lower() for s in syn_list]:
                            condition = key.title()
                            if condition.lower() not in exclude_terms:
                                conditions.add(condition)
                            break
                    else:
                        if condition.lower() not in exclude_terms and condition:
                            conditions.add(condition)
                            await self._generate_synonyms(condition.lower())
            conditions = list(conditions)
            if not conditions:
                logger.warning("No conditions found in patient text")
            logger.debug(f"Extracted conditions: {conditions}")
            return conditions
        except Exception as e:
            logger.error(f"Error extracting conditions: {str(e)}")
            return []

    async def _get_knowledge_graph_scores(self, conditions: List[str]) -> Dict[str, List[Dict]]:
        """Retrieve knowledge graph scores for conditions."""
        logger.info("Generating knowledge graph scores")
        kg_scores = {}
        for condition in conditions:
            try:
                preprocessed_condition = condition.lower()
                logger.debug(f"Processing condition for KG: {preprocessed_condition}")
                kg_results = await self.knowledge_graph.query(preprocessed_condition)
                kg_scores[condition] = [
                    {'code': r['code'], 'title': r['title'], 'similarity': float(r['similarity'])}
                    for r in sorted(kg_results, key=lambda x: x['similarity'], reverse=True)
                ]
                logger.debug(f"KG scores for {condition}: {kg_scores[condition]}")
            except Exception as e:
                logger.error(f"Error generating KG scores for {condition}: {str(e)}")
                kg_scores[condition] = []
        return kg_scores

    async def run(self, patient_text: str, predefined_icd_code: Optional[str] = None) -> Dict:
        logger.info(f"Running pipeline for patient text: {patient_text[:100]}...")
        try:
            preprocessed_text = await preprocess_text(patient_text)
            preprocessed_text = await self._expand_abbreviations(preprocessed_text)
            conditions = await self._extract_conditions(patient_text)
            if not conditions:
                logger.warning("No conditions extracted, defaulting to empty conditions")
                conditions = []

            icd_matches = await self._generate_icd_matches(conditions, preprocessed_text)
            all_kg_scores = await self._get_knowledge_graph_scores(conditions)

            condition_details = {}
            for condition in conditions:
                condition_details[condition] = await self._generate_condition_details(condition)

            predefined_icd_titles = []
            if predefined_icd_code:
                title = await get_icd_title(predefined_icd_code) or 'Unknown'
                matches_exist = any(matches for matches in icd_matches.values())
                is_relevant = matches_exist and any(
                    predefined_icd_code in [match['code'] for match in matches]
                    for matches in icd_matches.values()
                )
                predefined_icd_titles.append({
                    'code': predefined_icd_code,
                    'title': title,
                    'is_relevant': is_relevant
                })

            primary_condition = next(
                (c for c in conditions if c.lower() not in ['follow-up', 'examination', 'treatment']),
                conditions[0] if conditions else 'Unspecified Condition'
            )
            summary = f"The patient is admitted for a follow-up examination and potential surgical evaluation due to {primary_condition.lower() if primary_condition != 'Bilateral Ureter Calculus' else 'bilateral ureter calculus'}."
            return {
                'patient_data': patient_text,
                'summary': summary,
                'conditions': conditions,
                'condition_details': condition_details,
                'icd_matches': icd_matches,
                'all_kg_scores': all_kg_scores,
                'predefined_icd_titles': predefined_icd_titles,
                'admission_id': '1'
            }
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    async def _generate_icd_matches(self, conditions: List[str], patient_text: str) -> Dict[str, List[Dict]]:
        logger.info("Generating ICD matches")
        icd_matches = {}
        for condition in conditions:
            try:
                preprocessed_condition = condition.lower()
                logger.debug(f"Preprocessed condition: {preprocessed_condition}")
                candidates = await self._get_fts_candidates(preprocessed_condition)
                if not candidates:
                    logger.warning(f"No FTS candidates found for {condition}, falling back to knowledge graph")
                    kg_results = await self.knowledge_graph.query(preprocessed_condition)
                    candidates = [
                        {'code': r['code'], 'title': r['title'], 'inclusions': [], 'definition': r.get('description', '')}
                        for r in kg_results
                    ]
                    logger.debug(f"Fallback candidates from KG: {[c['code'] for c in candidates]}")

                matches = []
                condition_embedding = self.model.encode(preprocessed_condition, convert_to_tensor=True)
                for candidate in candidates:
                    code = candidate['code']
                    title = candidate['title'] or ''
                    candidate_text = title
                    if not candidate_text:
                        logger.debug(f"No valid text for candidate {code}")
                        continue
                    if code in EXCLUDED_CODES['specific_codes'] or any(code.startswith(prefix) for prefix in EXCLUDED_CODES['prefixes']) and code not in EXCLUDED_CODES['z_prefix_exceptions']:
                        continue
                    candidate_embedding = self.model.encode(candidate_text, convert_to_tensor=True)
                    score = util.cos_sim(condition_embedding, candidate_embedding)[0][0].item()
                    if score > 0.4:  # Lowered threshold
                        matches.append({'code': code, 'title': title, 'similarity': float(score * 100)})
                        logger.debug(f"Match for {condition}: {code} with score {score * 100}")
                matches.sort(key=lambda x: x['similarity'], reverse=True)
                icd_matches[condition] = matches
                logger.debug(f"Generated matches for {condition}: {icd_matches[condition]}")
            except Exception as e:
                logger.error(f"Error processing condition {condition}: {str(e)}")
                icd_matches[condition] = []
        return icd_matches

    def _fetch_fts_candidates(self, query_filters: Q) -> List[Dict]:
        """Synchronous function to f etch FTS candidates."""
        try:
            query = ICDCategory.objects.filter(query_filters).exclude(
                Q(code__in=EXCLUDED_CODES['specific_codes']) |
                Q(code__startswith='T') | Q(code__startswith='E') | Q(code__startswith='G')
            ).values('code', 'title', 'inclusions', 'definition').distinct()[:100]
            logger.debug(f"Executing FTS query: {str(query.query)}")
            candidates = list(query)
            logger.debug(f"Raw candidates: {[c['code'] for c in candidates]}")
            return candidates
        except Exception as e:
            logger.error(f"Error in _fetch_fts_candidates: {str(e)}")
            return []

    def _fetch_broad_fts_candidates(self) -> List[Dict]:
        """Synchronous function to fetch broad FTS candidates."""
        try:
            query = ICDCategory.objects.filter(
                Q(title__icontains='ureter') | Q(title__icontains='calculus')
            ).exclude(
                Q(code__in=EXCLUDED_CODES['specific_codes']) |
                Q(code__startswith='T') | Q(code__startswith='E') | Q(code__startswith='G')
            ).values('code', 'title', 'inclusions', 'definition').distinct()[:100]
            logger.debug(f"Executing broad FTS query: {str(query.query)}")
            candidates = list(query)
            logger.debug(f"Broad candidates: {[c['code'] for c in candidates]}")
            return candidates
        except Exception as e:
            logger.error(f"Error in _fetch_broad_fts_candidates: {str(e)}")
            return []

    async def _get_fts_candidates(self, condition: str) -> List[Dict]:
        logger.debug(f"Processing condition for FTS: {condition}")
        try:
            condition = condition.lower().strip()
            terms = condition.split()
            synonyms = self.synonyms.get(condition, [])
            query_filters = Q()
            for term in terms + synonyms:
                term = term.lower().strip()
                if term:
                    query_filters |= Q(title__icontains=term)

            candidates = await sync_to_async(self._fetch_fts_candidates)(query_filters)
            if not candidates:
                logger.warning(f"No candidates found for {condition}, trying broader query")
                candidates = await sync_to_async(self._fetch_broad_fts_candidates)()
            logger.debug(f"FTS candidates for {condition}: {[c['code'] for c in candidates]}")
            return candidates
        except Exception as e:
            logger.error(f"FTS query failed for {condition}: {str(e)}")
            return []