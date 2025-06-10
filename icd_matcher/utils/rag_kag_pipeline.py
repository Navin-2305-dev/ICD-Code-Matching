import logging
import asyncio
from typing import Dict, List, Tuple, TypedDict
from langgraph.graph import StateGraph, END
from icd_matcher.utils.text_processing import preprocess_text, generate_patient_summary
from icd_matcher.utils.knowledge_graph import KnowledgeGraph
from icd_matcher.utils.db_utils import get_icd_title
from django.conf import settings
from icd_matcher.models import ICDCategory
from asgiref.sync import sync_to_async
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    patient_text: str
    preprocessed_text: str
    summary: str
    conditions: List[str]
    kg_results: List[Tuple[str, str]]
    icd_matches: Dict[str, List[Tuple[str, str, float]]]
    all_kg_scores: Dict[str, List[Tuple[str, str, float]]]

class RAGKAGPipeline:
    def __init__(self):
        """Synchronous initialization of the RAGKAGPipeline."""
        self.graph = None
        self.knowledge_graph = None
        self.is_initialized = False

    @staticmethod
    async def create():
        """Async factory method to create and initialize the pipeline."""
        pipeline = RAGKAGPipeline()
        await pipeline._initialize()
        return pipeline

    async def _initialize(self):
        """Perform async initialization tasks."""
        if self.is_initialized:
            logger.debug("Pipeline already initialized")
            return

        try:
            logger.debug("Initializing knowledge graph")
            self.knowledge_graph = KnowledgeGraph()
            await self.knowledge_graph.initialize()

            logger.debug("Setting up graph workflow")
            self.graph = StateGraph(GraphState)
            self.graph.add_node("preprocess", self._preprocess_node)
            self.graph.add_node("retrieve", self._retrieve_node)
            self.graph.add_node("generate", self._generate_node)
            self.graph.add_edge("preprocess", "retrieve")
            self.graph.add_edge("retrieve", "generate")
            self.graph.add_edge("generate", END)
            self.graph.set_entry_point("preprocess")
            self.graph = self.graph.compile()

            self.is_initialized = True
            logger.info("RAGKAGPipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise

    async def _preprocess_node(self, state: GraphState) -> GraphState:
        """Preprocess the input text."""
        logger.debug("Preprocessing node")
        patient_text = state.get("patient_text", "").strip()
        if not patient_text:
            logger.warning("No patient text provided")
            state["preprocessed_text"] = ""
            return state

        try:
            preprocessed_text = await sync_to_async(preprocess_text)(patient_text)
            state["preprocessed_text"] = preprocessed_text
            logger.info(f"Patient text preprocessed: {preprocessed_text[:100]}...")
        except Exception as e:
            logger.error(f"Failed to preprocess text: {e}")
            state["preprocessed_text"] = patient_text  # Fallback to original text
        return state

    async def _retrieve_node(self, state: GraphState) -> GraphState:
        """Retrieve ICD codes from knowledge graph."""
        logger.debug("Retrieving from knowledge graph")
        preprocessed_text = state.get('preprocessed_text', '')
        conditions = [c.strip() for c in state.get('conditions', []) if c.strip()]

        if not preprocessed_text or not conditions:
            logger.warning("No text or conditions for retrieval")
            state['kg_results'] = []
            return state

        try:
            kg_codes = set()
            for condition in conditions:
                related_codes = await self.knowledge_graph.query(condition)
                kg_codes.update(related_codes)

            expanded_codes = []
            for code in kg_codes:
                if '-' in code:
                    start, end = code.split('-')
                    start_letter = start[0]
                    try:
                        start_num = int(start[1:]) if start[1:].isdigit() else 0
                        end_num = int(end[1:]) if end[1:].isdigit() else 0
                        for i in range(start_num, end_num + 1):
                            expanded_code = f"{start_letter}{i}"
                            if expanded_code not in expanded_codes and expanded_code in self.knowledge_graph.graph:
                                expanded_codes.append(expanded_code)
                    except ValueError:
                        logger.warning(f"Invalid code range: {code}")
                        continue
                else:
                    if code not in expanded_codes:
                        expanded_codes.append(code)

            kg_results = await sync_to_async(
                lambda: list(
                    ICDCategory.objects.filter(code__in=expanded_codes).values('code', 'title')
                )
            )()
            state["kg_results"] = [(item['code'], item['title']) for item in kg_results if item['title']]
            logger.info(f"Retrieved {len(kg_results)} knowledge graph results")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state['kg_results'] = []
        return state

    async def _generate_node(self, state: GraphState) -> GraphState:
        """Generate ICD matches with similarity scores."""
        logger.debug("Generating ICD matches")
        conditions = [c.strip() for c in state.get('conditions', []) if c.strip()]
        kg_results = state.get('kg_results', [])

        if not conditions or not kg_results:
            logger.warning("No conditions or KG results for generation")
            state['icd_matches'] = {c: [] for c in conditions} if conditions else {}
            state['all_kg_scores'] = {c: [] for c in conditions} if conditions else {}
            return state

        icd_matches = {}
        all_kg_scores = {}
        for condition in conditions:
            try:
                norm_cond = condition.lower().strip()
                condition_terms = norm_cond.split()
                expanded_terms = condition_terms.copy()
                for term in condition_terms:
                    for key, values in self.knowledge_graph.synonyms.items():
                        if term == key:
                            expanded_terms.extend(values)
                        elif term in values:
                            expanded_terms.append(key)

                all_scores = []
                for code, title in kg_results:
                    if not title:
                        continue
                    title_lower = title.lower()
                    term_overlap = sum(1 for term in expanded_terms if term in title_lower)
                    if term_overlap == 0:
                        continue
                    score = 80.0 + (term_overlap * 5)
                    if len(code) > 3:
                        score += 5.0
                    all_scores.append((code, title, score))

                all_scores.sort(key=lambda x: (x[2], len(x[0])), reverse=True)
                all_kg_scores[condition] = all_scores

                filtered_results = []
                has_specific = any(len(code) > 3 for code, _, _ in all_scores)
                for code, title, score in all_scores:
                    if has_specific and len(code) <= 3:
                        continue
                    filtered_results.append((code, title, score))

                icd_matches[condition] = filtered_results
                logger.debug(f"Matches for {condition}: {filtered_results}")
            except Exception as e:
                logger.error(f"Generation failed for {condition}: {e}")
                icd_matches[condition] = []
                all_kg_scores[condition] = []

        state['icd_matches'] = icd_matches
        state['all_kg_scores'] = all_kg_scores
        logger.info(f"Generated ICD matches: {icd_matches}")
        return state

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    async def run(self, patient_text: str, predefined_icd_code: str = None) -> Dict:
        """Run the pipeline asynchronously."""
        logger.debug(f"Running pipeline for patient text: {patient_text[:100]}...")
        try:
            if not patient_text.strip():
                logger.warning("Empty patient text provided")
                return {}

            summary, conditions = await generate_patient_summary(patient_text)
            state = {
                'patient_text': patient_text,
                'summary': summary,
                'conditions': [c for c in conditions if c.strip() and c != "No conditions identified"],
                'kg_results': [],
                'icd_matches': {},
                'all_kg_scores': {}
            }

            if not state['conditions']:
                logger.warning("No valid conditions identified")
                return state

            if not self.is_initialized:
                await self._initialize()

            result = await self.graph.ainvoke(state)
            icd_matches = result.get('icd_matches', {})
            all_kg_scores = result.get('all_kg_scores', {})

            predefined_icd_titles = []
            if predefined_icd_code:
                try:
                    icd_title = await get_icd_title(predefined_icd_code)  # Already async
                    is_relevant = any(
                        predefined_icd_code in [m[0] for m in matches]
                        for matches in icd_matches.values()
                    )
                    predefined_icd_titles.append({
                        'code': predefined_icd_code,
                        'title': icd_title or 'Unknown',
                        'is_relevant': is_relevant
                    })
                except Exception as e:
                    logger.error(f"Error processing predefined code {predefined_icd_code}: {e}")
                    predefined_icd_titles.append({
                        'code': predefined_icd_code,
                        'title': 'Unknown',
                        'is_relevant': False
                    })

            result = {
                'patient_data': patient_text,
                'summary': summary,
                'conditions': state['conditions'],
                'icd_matches': icd_matches,
                'all_kg_scores': all_kg_scores,
                'predefined_icd_titles': predefined_icd_titles,
                'admission_id': '61'  # Placeholder, updated dynamically
            }
            logger.info(f"Pipeline completed with matches: {icd_matches}")
            return result
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise