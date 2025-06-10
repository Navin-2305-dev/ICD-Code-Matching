import json
import logging
import re
from pathlib import Path
from typing import List, Tuple, Dict
from django.core.management.base import BaseCommand
from django.db import connection, transaction
from icd_matcher.models import ICDCategory
from icd_matcher.utils.exceptions import ICDCategoryCreationError, JSONFileLoadError

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Load ICD-10 codes from JSON file into the database'

    def add_arguments(self, parser):
        parser.add_argument(
            'json_file',
            type=str,
            help='Name of the JSON file in data/icd_codes (e.g., icd_version_2019.json)'
        )
        parser.add_argument(
            '--incremental',
            action='store_true',
            help='Perform incremental update instead of clearing the database'
        )

    def expand_range(self, code_range: str, base_title: str) -> List[Tuple[str, str, List, List]]:
        """Expand an ICD code range (e.g., A00-A09) into individual codes."""
        if '-' not in code_range:
            return [(code_range, base_title, [], [])]

        start, end = code_range.split('-')
        start_match = re.match(r'([A-Za-z]+)(\d+)', start)
        end_match = re.match(r'([A-Za-z]+)(\d+)', end)

        if not (start_match and end_match):
            logger.warning(f"Invalid range format: {code_range}")
            return [(code_range, base_title, [], [])]

        start_letter, start_num = start_match.groups()
        end_letter, end_num = end_match.groups()

        if start_letter != end_letter:
            logger.warning(f"Mismatched letters in range: {code_range}")
            return [(code_range, base_title, [], [])]

        try:
            start_num, end_num = int(start_num), int(end_num)
        except ValueError:
            logger.warning(f"Non-numeric range values: {code_range}")
            return [(code_range, base_title, [], [])]

        codes = []
        for num in range(start_num, end_num + 1):
            code = f"{start_letter}{num:02d}"
            title = f"{base_title} ({code})"
            codes.append((code, title, [], []))
        return codes

    def collect_categories(self, item: Dict, parent_code: str = None, seen_codes: set = None) -> List[Dict]:
        """Flatten JSON hierarchy into a list of category records, avoiding duplicates."""
        if seen_codes is None:
            seen_codes = set()

        records = []
        code = item.get("code", "").strip()
        if not code:
            logger.warning("Empty code provided, skipping")
            return records

        # Initialize record for the current category
        record = {
            "code": code,
            "title": item.get("title", "No title provided"),
            "definition": item.get("definition", ""),
            "inclusions": item.get("inclusions", []),
            "exclusions": item.get("exclusions", []),
            "parent_code": parent_code,
            "is_range": False
        }

        # Handle code ranges
        if '-' in code and re.match(r'[A-Za-z]+\d+-[A-Za-z]+\d+', code):
            record["is_range"] = True
            if code not in seen_codes:
                records.append(record)
                seen_codes.add(code)

            expanded_codes = self.expand_range(code, record["title"])
            if len(expanded_codes) > 1:
                for exp_code, exp_title, exp_inclusions, exp_exclusions in expanded_codes:
                    if exp_code not in seen_codes:
                        records.append({
                            "code": exp_code,
                            "title": exp_title,
                            "definition": record["definition"],
                            "inclusions": exp_inclusions,
                            "exclusions": exp_exclusions,
                            "parent_code": code,
                            "is_range": True
                        })
                        seen_codes.add(exp_code)
                    else:
                        logger.warning(f"Duplicate code {exp_code} from range {code}, skipping")
        else:
            # Prioritize standalone entries
            if code not in seen_codes:
                records.append(record)
                seen_codes.add(code)
            else:
                # Update existing record if standalone has more specific data
                for existing in records:
                    if existing["code"] == code and existing["is_range"]:
                        existing.update({
                            "title": record["title"],
                            "definition": record["definition"],
                            "inclusions": record["inclusions"],
                            "exclusions": record["exclusions"],
                            "parent_code": record["parent_code"],
                            "is_range": False
                        })
                        logger.info(f"Updated code {code} with standalone data")
                        break
                else:
                    logger.warning(f"Duplicate code {code} as standalone, skipping")

        # Process subcategories
        for sub_item in item.get("subcategories", []) + item.get("sub_subcategories", []) + item.get("codes", []):
            records.extend(self.collect_categories(sub_item, parent_code=code, seen_codes=seen_codes))

        return records

    def create_icd_category(self, record: Dict):
        """Create or update an ICDCategory from a record."""
        code = record["code"]
        parent_code = record["parent_code"]
        parent = None

        if parent_code:
            try:
                parent = ICDCategory.objects.get(code=parent_code)
            except ICDCategory.DoesNotExist:
                logger.error(f"Parent code {parent_code} does not exist for {code}")
                raise ICDCategoryCreationError(f"Parent code {parent_code} does not exist for {code}")

        try:
            icd_entry, created = ICDCategory.objects.get_or_create(
                code=code,
                defaults={
                    "title": record["title"],
                    "definition": record["definition"],
                    "parent": parent,
                    "inclusions": json.dumps(record["inclusions"]),
                    "exclusions": json.dumps(record["exclusions"]),
                }
            )
            if not created:
                # Update existing record
                icd_entry.title = record["title"]
                icd_entry.definition = record["definition"]
                icd_entry.parent = parent
                icd_entry.inclusions = json.dumps(record["inclusions"])
                icd_entry.exclusions = json.dumps(record["exclusions"])
                icd_entry.save()
            self.stdout.write(f"{'Created' if created else 'Updated'}: {icd_entry} (Parent: {parent})")
            logger.debug(f"{'Created' if created else 'Updated'} ICDCategory: {code}")
        except Exception as e:
            logger.error(f"Failed to create ICDCategory for code {code}: {str(e)}", exc_info=True)
            self.stdout.write(self.style.ERROR(f"Failed to create ICDCategory for code {code}: {str(e)}"))
            raise ICDCategoryCreationError(f"Failed to create ICDCategory for code {code}: {str(e)}")

    def handle(self, *args, **options):
        """Handle the command execution."""
        # Configure logging to avoid duplicates
        logger.handlers = [logging.StreamHandler()]
        logger.setLevel(logging.DEBUG)

        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        data_dir = base_dir / 'data' / 'icd_codes'
        data_dir.mkdir(parents=True, exist_ok=True)

        json_filename = options['json_file']
        json_file = data_dir / json_filename
        incremental = options.get('incremental', False)

        if not json_file.exists():
            logger.error(f"JSON file not found: {json_file}")
            self.stdout.write(self.style.ERROR(f"File not found: {json_file}"))
            raise JSONFileLoadError(f"JSON file not found: {json_file}")

        try:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as e:
            logger.error(f"Failed to load JSON file: {str(e)}", exc_info=True)
            self.stdout.write(self.style.ERROR(f"Failed to load JSON file: {str(e)}"))
            raise JSONFileLoadError(f"Failed to load JSON file: {str(e)}")

        if not incremental:
            logger.info("Clearing existing ICDCategory data")
            ICDCategory.objects.all().delete()
            self.stdout.write("Cleared existing ICDCategory data")

        # Collect all categories
        categories = data.get("categories", data.get("codes", []))
        all_records = []
        for category in categories:
            all_records.extend(self.collect_categories(category))

        # Temporarily disable foreign key checks
        with connection.cursor() as cursor:
            cursor.execute('PRAGMA foreign_keys = OFF;')

        try:
            with transaction.atomic():
                # Create all records
                for record in all_records:
                    self.create_icd_category(record)
        except Exception as e:
            logger.error(f"Failed to load ICD data: {str(e)}", exc_info=True)
            raise
        finally:
            # Re-enable foreign key checks and validate
            with connection.cursor() as cursor:
                cursor.execute('PRAGMA foreign_keys = ON;')
                cursor.execute('PRAGMA foreign_key_check;')
                violations = cursor.fetchall()
                if violations:
                    logger.error(f"Foreign key violations detected: {violations}")
                    raise ICDCategoryCreationError(f"Foreign key violations detected: {violations}")

        self.stdout.write(self.style.SUCCESS("Successfully loaded ICD codes"))
        logger.info("Successfully loaded ICD codes")