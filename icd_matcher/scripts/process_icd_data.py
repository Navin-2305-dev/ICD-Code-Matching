import logging
import pandas as pd
from django.conf import settings
from icd_matcher.models import ICDCategory
from icd_matcher.utils.embeddings import find_best_icd_match
from icd_matcher.utils.text_processing import generate_patient_summary
from pathlib import Path
from icd_matcher.utils.exceptions import TrainingDataLoadError, TrainingDataProcessingError, ResultSavingError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ensure the logs directory exists
base_dir = Path(__file__).resolve().parent.parent.parent
logs_dir = base_dir / 'data' / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)

# File handler
log_file = logs_dir / 'process_icd_data.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers
logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get_icd_title(code):
    """Retrieve ICD title from ICDCategory or return a placeholder."""
    logger.debug(f"Retrieving title for ICD code: {code}")
    try:
        icd_entry = ICDCategory.objects.get(code=code)
        logger.debug(f"Found title for {code}: {icd_entry.title}")
        return icd_entry.title
    except ICDCategory.DoesNotExist:
        logger.warning(f"ICD code {code} not found in database")
        return f"Title for {code}"

def run():
    """Process training data to match ICD codes and save results."""
    logger.info("Starting ICD code matching process")

    # Define the data directory
    data_dir = base_dir / 'data' / 'training_data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get the training data path
    default_data_path = data_dir / 'Training_dataset.csv'
    data_path = Path(getattr(settings, 'TRAINING_DATA_PATH', default_data_path))
    logger.debug(f"Training data path: {data_path}")

    # Load the training data
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

    # Validate columns
    expected_columns = ['DISCHARGE_REMARKS', 'ICD_REMARKS_A', 'ICD_REMARKS_D', 
                       'Predefined ICD_code', 'Generated Matching ICD Code(s)']
    logger.debug(f"Expected columns: {expected_columns}")
    if not all(col in df.columns for col in expected_columns):
        missing_cols = [col for col in expected_columns if col not in df.columns]
        logger.error(f"Missing columns in training data: {missing_cols}")
        raise TrainingDataLoadError(f"Missing columns in training data: {missing_cols}")

    results = []
    logger.info("Starting processing of training data rows")

    for index, row in df.iterrows():
        logger.debug(f"Processing row {index}")
        try:
            discharge_remarks = row['DISCHARGE_REMARKS']
            icd_remarks_a = row['ICD_REMARKS_A']
            icd_remarks_d = row['ICD_REMARKS_D']
            predefined_icd = row['Predefined ICD_code']
            expected_matches = row['Generated Matching ICD Code(s)']

            logger.debug(f"Row {index} - Discharge Remarks: {discharge_remarks}")
            logger.debug(f"Row {index} - ICD Remarks A: {icd_remarks_a}")
            logger.debug(f"Row {index} - ICD Remarks D: {icd_remarks_d}")
            logger.debug(f"Row {index} - Predefined ICD: {predefined_icd}")
            logger.debug(f"Row {index} - Expected Matches: {expected_matches}")

            medical_text = f"{discharge_remarks} {icd_remarks_a} {icd_remarks_d}"
            logger.debug(f"Row {index} - Combined medical text: {medical_text}")

            summary, conditions = generate_patient_summary(medical_text)
            logger.info(f"Row {index}: Generated summary: {summary}")
            logger.debug(f"Row {index}: Identified conditions: {conditions}")

            matched_codes = []

            if not conditions or all(c.lower() == "no conditions identified" for c in conditions):
                logger.info(f"Row {index}: No valid conditions found")
                matched_codes.append("None (0.0%)")
            else:
                icd_matches = find_best_icd_match(conditions, medical_text)
                logger.debug(f"Row {index}: ICD matches: {icd_matches}")

                for condition, matches in icd_matches.items():
                    if not matches:
                        logger.info(f"Row {index}: No ICD matches for condition: {condition}")
                        continue

                    for code, title, score in matches:
                        match_str = f"{code}: {title} ({score:.1f}%)"
                        matched_codes.append(match_str)

                        logger.debug(f"Row {index}: Updating/creating ICDCategory for code {code}")
                        ICDCategory.objects.update_or_create(
                            code=code,
                            defaults={
                                'definition': medical_text
                            }
                        )
                        logger.info(
                            f"Row {index}: Condition: {condition}, "
                            f"Matched ICD: {code}, Score: {score:.1f}%, "
                            f"Predefined ICD: {predefined_icd}, "
                            f"Expected Matches: {expected_matches}"
                        )

            matched_codes_str = ", ".join(matched_codes) if matched_codes else "None (0.0%)"
            logger.debug(f"Row {index}: Matched codes string: {matched_codes_str}")

            results.append({
                'DISCHARGE_REMARKS': discharge_remarks,
                'ICD_REMARKS_A': icd_remarks_a,
                'ICD_REMARKS_D': icd_remarks_d,
                'Predefined ICD_code': predefined_icd,
                'Generated Matching ICD Code(s)': expected_matches,
                'Matched ICD Codes': matched_codes_str
            })
            logger.debug(f"Row {index}: Added result to list")

        except Exception as e:
            logger.error(f"Error processing row {index}: {e}")
            raise TrainingDataProcessingError(f"Error processing row {index}: {e}")

    # Save results
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