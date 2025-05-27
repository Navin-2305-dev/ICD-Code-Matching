import asyncio
from django.core.management.base import BaseCommand
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Rebuilds the ICD knowledge graph and caches it'

    def handle(self, *args, **options):
        from icd_matcher.utils.knowledge_graph import rebuild_knowledge_graph
        try:
            asyncio.run(rebuild_knowledge_graph())
            self.stdout.write(self.style.SUCCESS('Successfully rebuilt knowledge graph'))
        except Exception as e:
            logger.error(f"Failed to rebuild knowledge graph: {e}")
            self.stdout.write(self.style.ERROR(f'Error: {e}'))