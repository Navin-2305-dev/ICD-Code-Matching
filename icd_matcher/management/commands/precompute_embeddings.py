import logging
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from icd_matcher.utils.embeddings import precompute_icd_embeddings
from icd_matcher.utils.knowledge_graph import rebuild_knowledge_graph
import asyncio

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Initialize ICD-10 data (embeddings and knowledge graph)'

    def handle(self, *args, **options):
        logger.info("Starting ICD-10 data initialization")
        try:
            # Ensure cache directory for file-based cache
            cache_dir = os.path.join(settings.BASE_DIR, 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Cache directory ensured: {cache_dir}")

            # Ensure logs directory (redundant but safe)
            log_dir = os.path.join(settings.BASE_DIR, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"Log directory ensured: {log_dir}")

            # Run embeddings
            self.stdout.write("Precomputing ICD embeddings...")
            asyncio.run(precompute_icd_embeddings())
            self.stdout.write(self.style.SUCCESS("ICD embeddings precomputed successfully"))

            # Run knowledge graph
            self.stdout.write("Rebuilding knowledge graph...")
            asyncio.run(rebuild_knowledge_graph())
            self.stdout.write(self.style.SUCCESS("Knowledge graph rebuilt successfully"))

            self.stdout.write(self.style.SUCCESS("Successfully initialized ICD-10 data"))
        except Exception as e:
            logger.error(f"Failed to initialize ICD-10 data: {e}", exc_info=True)
            self.stderr.write(self.style.ERROR(f"Error: {str(e)}"))
            raise