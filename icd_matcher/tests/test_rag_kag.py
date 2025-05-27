import asyncio
import logging
from icd_matcher.utils.rag_kag_pipeline import RAGKAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rag_kag():
    try:
        # Sample patient input
        patient_text = (
            "A: B/L HEARING LOSS(INV) D: FOLLOW-UP EXAMINATION AFTER TREATMENT FOR CONDITIONS OTHER THAN, "
            "Discharge Ward: SURG Admission Type: FRESH Admission Date: 2024-03-01 Admission Status: APPROVED"
        )
        predefined_icd_code = "L100"

        # Initialize the pipeline
        pipeline = await RAGKAGPipeline.create()
        logger.info("Testing RAG + KAG pipeline...")

        # Run the pipeline
        result = await pipeline.run(patient_text, predefined_icd_code)
        
        # Verify the output
        assert 'icd_matches' in result, "ICD matches not found in result"
        assert 'all_kg_scores' in result, "Knowledge Graph scores not found in result"
        assert 'predefined_icd_titles' in result, "Predefined ICD titles not found in result"
        
        icd_matches = result['icd_matches'].get('Bilateral Hearing Loss', [])
        assert len(icd_matches) > 0, "No ICD matches found for Bilateral Hearing Loss"
        assert icd_matches[0][0].startswith('H90'), "Expected ICD code starting with H90"
        assert icd_matches[0][2] >= 90.0, "Expected similarity score >= 90.0"

        logger.info("RAG + KAG pipeline test passed successfully!")
        logger.info(f"ICD Matches: {icd_matches}")
    except Exception as e:
        logger.error(f"RAG + KAG pipeline test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_rag_kag())