import json
import re
from pathlib import Path
from django.core.management.base import BaseCommand
from icd_matcher.models import ICDCategory
from icd_matcher.utils.exceptions import ICDCategoryCreationError, JSONFileLoadError

class Command(BaseCommand):
    help = 'Load ICD-10 codes from JSON file into the database'

    def add_arguments(self, parser):
        parser.add_argument(
            'json_file', 
            type=str, 
            help='Name of the JSON file in data/icd_codes (e.g., icd_version_2019.json)'
        )

    def expand_range(self, code_range, base_title):
        """Expand a range like W00-W19 into individual codes, only if same letter prefix."""
        if '-' not in code_range:
            return [(code_range, base_title, [], [])]

        start, end = code_range.split('-')
        start_match = re.match(r'([A-Za-z]+)(\d+)', start)
        end_match = re.match(r'([A-Za-z]+)(\d+)', end)
        
        if not (start_match and end_match):
            return [(code_range, base_title, [], [])]
        
        start_letter, start_num = start_match.groups()
        end_letter, end_num = end_match.groups()
        
        if start_letter != end_letter:
            return [(code_range, base_title, [], [])]
        
        start_num, end_num = int(start_num), int(end_num)
        codes = []
        for num in range(start_num, end_num + 1):
            code = f"{start_letter}{num:02d}"
            title = f"{base_title} ({code})"
            codes.append((code, title, [], []))
        return codes

    def create_icd_category(self, item, parent=None):
        """Create or update an ICDCategory with its subcategories, including inclusions and exclusions."""
        code = item["code"]
        title = item.get("title", "No title")
        definition = item.get("definition", "")
        inclusions = json.dumps(item.get("inclusions", []))  # Convert list to JSON string
        exclusions = json.dumps(item.get("exclusions", []))  # Convert list to JSON string

        try:
            # Check if the code is a range and expandable
            if '-' in code and re.match(r'[A-Za-z]+\d+-[A-Za-z]+\d+', code):
                expanded_codes = self.expand_range(code, title)
                if len(expanded_codes) > 1:
                    icd_entry, created = ICDCategory.objects.update_or_create(
                        code=code,
                        defaults={
                            "title": title,
                            "definition": definition,
                            "parent": parent,
                            "inclusions": inclusions,
                            "exclusions": exclusions,
                        }
                    )
                    self.stdout.write(f"{'Created' if created else 'Updated'}: {icd_entry} (Parent: {parent})")
                    
                    # Create individual codes under the parent
                    for exp_code, exp_title, exp_inclusions, exp_exclusions in expanded_codes:
                        sub_entry, created = ICDCategory.objects.update_or_create(
                            code=exp_code,
                            defaults={
                                "title": exp_title,
                                "definition": definition,
                                "parent": icd_entry,
                                "inclusions": json.dumps(exp_inclusions),
                                "exclusions": json.dumps(exp_exclusions),
                            }
                        )
                        self.stdout.write(f"{'Created' if created else 'Updated'}: {sub_entry} (Parent: {icd_entry})")
                else:
                    icd_entry, created = ICDCategory.objects.update_or_create(
                        code=code,
                        defaults={
                            "title": title,
                            "definition": definition,
                            "parent": parent,
                            "inclusions": inclusions,
                            "exclusions": exclusions,
                        }
                    )
                    self.stdout.write(f"{'Created' if created else 'Updated'}: {icd_entry} (Parent: {parent})")
            else:
                icd_entry, created = ICDCategory.objects.update_or_create(
                    code=code,
                    defaults={
                        "title": title,
                        "definition": definition,
                        "parent": parent,
                        "inclusions": inclusions,
                        "exclusions": exclusions,
                    }
                )
                self.stdout.write(f"{'Created' if created else 'Updated'}: {icd_entry} (Parent: {parent})")

            # Recursively process nested categories
            for sub_item in item.get("subcategories", []):
                self.create_icd_category(sub_item, parent=icd_entry)
            for sub_item in item.get("sub_subcategories", []):
                self.create_icd_category(sub_item, parent=icd_entry)
            for code_item in item.get("codes", []):
                self.create_icd_category(code_item, parent=icd_entry)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to create ICDCategory for code {code}: {e}"))
            raise ICDCategoryCreationError(f"Failed to create ICDCategory for code {code}: {e}")

    def handle(self, *args, **options):
        """Load ICD-10 codes from a JSON file into the database."""
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        data_dir = base_dir / 'data' / 'icd_codes'
        
        data_dir.mkdir(parents=True, exist_ok=True)
        
        json_filename = options['json_file']
        json_file = data_dir / json_filename
        
        if not json_file.exists():
            self.stdout.write(self.style.ERROR(f"File not found: {json_file}"))
            self.stdout.write(self.style.ERROR("Please ensure the JSON file exists in data/icd_codes/"))
            raise JSONFileLoadError(f"JSON file not found: {json_file}")

        try:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to load JSON file: {e}"))
            raise JSONFileLoadError(f"Failed to load JSON file: {e}")
            
        ICDCategory.objects.all().delete()
        self.stdout.write("Cleared existing ICDCategory data")

        categories = data.get("categories", data.get("codes", []))
        for category in categories:
            self.create_icd_category(category)
        self.stdout.write(self.style.SUCCESS("Successfully loaded ICD codes"))