import json
import logging
from django.core.management.base import BaseCommand
from django.db import connection
from icd_matcher.models import ICDCategory
from icd_matcher.utils.exceptions import FTSQueryError

logging.getLogger('').handlers = []
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Set up SQLite FTS5 table for ICD full-text search'

    def validate_entry(self, entry):
        if not isinstance(entry.pk, str):
            raise ValueError(f"Invalid pk type for code {entry.code}: {type(entry.pk)}")
        if entry.pk != entry.code:
            raise ValueError(f"Primary key {entry.pk} does not match code {entry.code}")
        if not isinstance(entry.title, str):
            raise ValueError(f"Invalid title type for code {entry.code}: {type(entry.title)}")
        if entry.definition is not None and not isinstance(entry.definition, str):
            raise ValueError(f"Invalid definition type for code {entry.code}: {type(entry.definition)}")
        inclusions = entry.inclusions
        exclusions = entry.exclusions
        clinical_notes = entry.clinical_notes
        if not isinstance(inclusions, list):
            try:
                inclusions = json.dumps(inclusions or [])
                logger.warning(f"Converted inclusions to string for code {entry.code}")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid inclusions for code {entry.code}: {str(e)}")
        if not isinstance(exclusions, list):
            try:
                exclusions = json.dumps(exclusions or [])
                logger.warning(f"Converted exclusions to string for code {entry.code}")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid exclusions for code {entry.code}: {str(e)}")
        if not isinstance(clinical_notes, list):
            try:
                clinical_notes = json.dumps(clinical_notes or [])
                logger.warning(f"Converted clinical_notes to string for code {entry.code}")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid clinical_notes for code {entry.code}: {str(e)}")
        return (
            entry.code,
            entry.title,
            entry.definition or '',
            inclusions,
            exclusions,
            clinical_notes
        )

    def handle(self, *args, **options):
        logger.info("Starting icd_fts table setup")

        try:
            with connection.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS icd_fts")
                cursor.execute("""
                    CREATE VIRTUAL TABLE icd_fts USING fts5(
                        code, title, definition, inclusions, exclusions, clinical_notes
                    )
                """)

            icd_entries = ICDCategory.objects.all()
            total_entries = icd_entries.count()
            logger.info(f"Populating icd_fts with {total_entries} ICD entries")

            batch_size = 1000
            inserted_rows = 0
            for i in range(0, total_entries, batch_size):
                batch = icd_entries[i:i + batch_size]
                values = []
                for entry in batch:
                    try:
                        values.append(self.validate_entry(entry))
                    except ValueError as ve:
                        logger.error(f"Validation error for entry {entry.code}: {str(ve)}")
                        raise FTSQueryError(f"Validation error for entry {entry.code}: {str(ve)}")

                try:
                    with connection.cursor() as cursor:
                        cursor.executemany("""
                            INSERT INTO icd_fts(code, title, definition, inclusions, exclusions, clinical_notes)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, values)
                        inserted_rows += len(values)
                        logger.debug(f"Inserted {len(values)} rows, total: {inserted_rows}")
                except Exception as e:
                    logger.error(f"Batch insertion failed at offset {i}: {str(e)}")
                    raise FTSQueryError(f"Batch insertion failed at offset {i}: {str(e)}")

            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM icd_fts")
                fts_count = cursor.fetchone()[0]
                if fts_count != total_entries:
                    raise FTSQueryError(
                        f"Mismatch in icd_fts population: expected {total_entries}, got {fts_count}"
                    )

            logger.info("Successfully set up icd_fts table")
        except Exception as e:
            logger.error(f"Error setting up icd_fts table: {str(e)}", exc_info=True)
            raise FTSQueryError(f"Error setting up icd_fts table: {str(e)}")