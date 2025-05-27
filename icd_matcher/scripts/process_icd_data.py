import logging
import pandas as pd
from django.conf import settings
from django.core.cache import cache
from icd_matcher.models import ICDCategory
from icd_matcher.utils.rag_kag_pipeline import RAGKAGPipeline
from icd_matcher.utils.text_processing import generate_patient_summary, is_not_negated
from icd_matcher.utils.exceptions import TrainingDataLoadError, TrainingDataProcessingError, ResultSavingError
from pathlib import Path
from celery import group
from asgiref.sync import sync_to_async
import asyncio

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

base_dir = Path(__file__).resolve().parent.parent
logs_dir = base_dir / 'data' / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)

log_file = logs_dir / 'process_icd_data.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)

async def get_icd_title(code):
    logger.debug(f"Retrieving title for ICD code: {code}")
    try:
        icd_entry = await sync_to_async(ICDCategory.objects.get)(code=code)
        logger.debug(f"Found title for {code}: {icd_entry.title}")
        return icd_entry.title
    except ICDCategory.DoesNotExist:
        logger.warning(f"ICD code {code} not found in database")
        return f"Title for {code}"

async def process_row(row, semaphore):
    """Process a single row of the training dataset."""
    async with semaphore:
        try:
            medical_text = (
                f"{row.get('DISCHARGE_REMARKS', '')} "
                f"A: {row.get('ICD_REMARKS_A', '')} "
                f"D: {row.get('ICD_REMARKS_D', '')}"
            ).strip()
            logger.debug(f"Processing row with medical text: {medical_text[:200]}")
            summary, conditions = await generate_patient_summary(medical_text)
            non_negated_conditions = [
                cond for cond in conditions
                if await sync_to_async(is_not_negated)(cond, medical_text)
            ]

            pipeline = RAGKAGPipeline()  # Instantiate the pipeline directly
            icd_matches = pipeline.run(medical_text)
            generated_codes = []
            for condition, matches in icd_matches.items():
                for code, title, score in matches:
                    if score >= settings.ICD_MATCHING_SETTINGS.get('MIN_SIMILARITY_SCORE', 60):
                        generated_codes.append(f"{code}: {title} ({score:.1f}%)")

            for code in generated_codes:
                code_part = code.split(':')[0]
                try:
                    icd_entry = await sync_to_async(ICDCategory.objects.get)(code=code_part)
                    icd_entry.title = code.split(': ')[1].split(' (')[0]
                    await sync_to_async(icd_entry.save)()
                except ICDCategory.DoesNotExist:
                    await sync_to_async(ICDCategory.objects.create)(
                        code=code_part,
                        title=code.split(': ')[1].split(' (')[0]
                    )

            cache_key = f"training_result_{hash(medical_text[:1000])}"
            await sync_to_async(cache.set)(
                cache_key,
                {'summary': summary, 'conditions': non_negated_conditions, 'icd_matches': icd_matches},
                timeout=settings.ICD_MATCHING_SETTINGS.get('CACHE_TTL', 3600)
            )

            return {
                'DISCHARGE_REMARKS': row.get('DISCHARGE_REMARKS', ''),
                'ICD_REMARKS_A': row.get('ICD_REMARKS_A', ''),
                'ICD_REMARKS_D': row.get('ICD_REMARKS_D', ''),
                'Predefined ICD_code': row.get('Predefined ICD_code', ''),
                'Generated Matching ICD Code(s)': '; '.join(generated_codes)
            }
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            raise TrainingDataProcessingError(f"Error processing row: {e}")

async def run():
    logger.info("Starting ICD code matching process")

    data_dir = base_dir / 'data' / 'training_data'
    data_dir.mkdir(parents=True, exist_ok=True)

    default_data_path = data_dir / 'Training_dataset.csv'
    data_path = Path(getattr(settings, 'TRAINING_DATA_PATH', default_data_path))
    logger.debug(f"Training data path: {data_path}")

    logger.info(f"Attempting to load training data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded training data with {len(df)} records")
        logger.debug(f"Columns in training data: {list(df.columns)}")
    except FileNotFoundError as e:
        logger.error(f"Training data file not found at {data_path}: {e}")
        raise TrainingDataLoadError(f"Training data file not found: {e}")
    except pd.errors.EmptyDataError as e:
        logger.error(f"Training data file at {data_path} is empty: {e}")
        raise TrainingDataLoadError(f"Training data file is empty: {e}")
    except Exception as e:
        logger.error(f"Error reading training data: {e}")
        raise TrainingDataLoadError(f"Error reading training data: {e}")

    expected_columns = ['DISCHARGE_REMARKS', 'ICD_REMARKS_A', 'ICD_REMARKS_D', 
                       'Predefined ICD_code', 'Generated Matching ICD Code(s)']
    if not all(col in df.columns for col in expected_columns):
        missing_cols = [col for col in expected_columns if col not in df.columns]
        logger.error(f"Missing columns in training data: {missing_cols}")
        raise TrainingDataLoadError(f"Missing columns in training data: {missing_cols}")

    results = []
    logger.info("Starting processing of training data rows")

    batch_size = settings.ICD_MATCHING_SETTINGS.get('BATCH_SIZE', 32)
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        tasks = [process_row(row, index) for index, row in batch.iterrows()]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {result}")
                continue
            results.append(result)

    output_dir = base_dir / 'data' / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'results.csv'

    logger.info(f"Attempting to save results to {output_path}")
    try:
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved results to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise ResultSavingError(f"Error saving results: {e}")

    logger.info("ICD code matching process completed successfully")

if __name__ == "__main__":
    asyncio.run(run())