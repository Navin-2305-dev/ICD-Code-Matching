
import logging
import pandas as pd
from django.conf import settings
from icd_matcher.models import ICDCategory
from icd_matcher.utils import generate_patient_summary, find_best_icd_match, setup_fts_table

logger = logging.getLogger(__name__)

def get_icd_title(code):
    """Retrieve ICD title from ICDCategory or return a placeholder."""
    try:
        icd_entry = ICDCategory.objects.get(code=code)
        return icd_entry.title
    except ICDCategory.DoesNotExist:
        return f"Title for {code}"

def run():

    data_path = getattr(settings, 'TRAINING_DATA_PATH', 'Training_dataset.csv')
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded training data from {data_path} with {len(df)} records.")
    except FileNotFoundError:
        logger.error(f"Training data file not found at {data_path}")
        return
    except pd.errors.EmptyDataError:
        logger.error(f"Training data file at {data_path} is empty")
        return
    except Exception as e:
        logger.error(f"Error reading training data: {e}")
        return

    expected_columns = ['DISCHARGE_REMARKS', 'ICD_REMARKS_A', 'ICD_REMARKS_D', 
                       'Predefined ICD_code', 'Generated Matching ICD Code(s)']
    if not all(col in df.columns for col in expected_columns):
        logger.error(f"Training data must contain columns: {expected_columns}")
        return

    results = []

    for index, row in df.iterrows():
        try:
            discharge_remarks = row['DISCHARGE_REMARKS']
            icd_remarks_a = row['ICD_REMARKS_A']
            icd_remarks_d = row['ICD_REMARKS_D']
            predefined_icd = row['Predefined ICD_code']
            expected_matches = row['Generated Matching ICD Code(s)']

            medical_text = f"{discharge_remarks} {icd_remarks_a} {icd_remarks_d}"

            summary, conditions = generate_patient_summary(medical_text)
            logger.debug(f"Row {index}: Summary: {summary}, Conditions: {conditions}")

            matched_codes = []

            if not conditions or all(c.lower() == "no conditions identified" for c in conditions):
                logger.info(f"Row {index}: No valid conditions found for: {medical_text}")
                matched_codes.append("None (0.0%)")
            else:
                icd_matches = find_best_icd_match(conditions, medical_text)
                logger.debug(f"Row {index}: ICD Matches: {icd_matches}")

                for condition, matches in icd_matches.items():
                    if not matches:
                        logger.info(f"Row {index}: No ICD matches for condition: {condition}")
                        continue

                    for code, score in matches:
                        title = get_icd_title(code)
                        match_str = f"{code}: {title} ({score:.1f}%)"
                        matched_codes.append(match_str)

                        ICDCategory.objects.update_or_create(
                            code=code,
                            defaults={
                                'title': f"Matched for {condition}",
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

            results.append({
                'DISCHARGE_REMARKS': discharge_remarks,
                'ICD_REMARKS_A': icd_remarks_a,
                'ICD_REMARKS_D': icd_remarks_d,
                'Predefined ICD_code': predefined_icd,
                'Generated Matching ICD Code(s)': expected_matches,
                'Matched ICD Codes': matched_codes_str
            })

        except Exception as e:
            logger.error(f"Error processing row {index}: {e}")
            results.append({
                'DISCHARGE_REMARKS': discharge_remarks,
                'ICD_REMARKS_A': icd_remarks_a,
                'ICD_REMARKS_D': icd_remarks_d,
                'Predefined ICD_code': predefined_icd,
                'Generated Matching ICD Code(s)': expected_matches,
                'Matched ICD Codes': f"Error: {str(e)}"
            })
            print(results)
            continue

    try:
        output_df = pd.DataFrame(results)
        output_df.to_csv('Sample_output.csv', index=False)
        logger.info("Saved formatted results to matched_results_formatted.csv")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

    logger.info("ICD code matching process completed.")