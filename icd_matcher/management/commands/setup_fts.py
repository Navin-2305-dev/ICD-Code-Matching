import logging
from django.core.management.base import BaseCommand
from django.db import connection
from icd_matcher.models import ICDCategory

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Set up the icd_fts table for full-text search'

    def handle(self, *args, **options):
        logger.info("Starting icd_fts table setup")
        try:
            # Drop and recreate icd_fts table
            with connection.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS icd_fts")
                cursor.execute("""
                    CREATE TABLE icd_fts (
                        code VARCHAR(10) PRIMARY KEY,
                        title TEXT,
                        search_vector TSVECTOR
                    )
                """)
                cursor.execute("""
                    CREATE INDEX icd_fts_idx ON icd_fts USING GIN(search_vector)
                """)
            
            # Populate icd_fts from ICDCategory
            icd_entries = ICDCategory.objects.all()
            total_entries = icd_entries.count()
            logger.info(f"Populating icd_fts with {total_entries} ICD entries")

            with connection.cursor() as cursor:
                for entry in icd_entries:
                    cursor.execute("""
                        INSERT INTO icd_fts (code, title, search_vector)
                        VALUES (%s, %s, to_tsvector('english', %s))
                    """, [entry.code, entry.title, entry.title])
            
            logger.info("icd_fts table setup completed successfully")
            self.stdout.write(self.style.SUCCESS("Successfully set up icd_fts table"))
        except Exception as e:
            logger.error(f"Error setting up icd_fts table: {e}")
            self.stdout.write(self.style.ERROR(f"Error: {e}"))
            raise