# icd_matcher/management/commands/load_icd_data.py
import json
import re
from django.core.management.base import BaseCommand
from icd_matcher.models import ICDCategory

class Command(BaseCommand):
    help = 'Load ICD-10 codes from JSON file into the database'

    def add_arguments(self, parser):
        parser.add_argument('json_file', type=str, help='icd_10\icd_version_2019.json')

    def expand_range(self, code_range, base_title):
        """Expand a range like W00-W19 into individual codes, only if same letter prefix."""
        if '-' not in code_range:
            return [(code_range, base_title)]

        start, end = code_range.split('-')
        start_match = re.match(r'([A-Za-z]+)(\d+)', start)
        end_match = re.match(r'([A-Za-z]+)(\d+)', end)
        
        if not (start_match and end_match):
            return [(code_range, base_title)]  # Treat as single code if not a valid range
        
        start_letter, start_num = start_match.groups()
        end_letter, end_num = end_match.groups()
        
        if start_letter != end_letter:
            # Skip expansion for ranges with different letters (e.g., V01-X59)
            return [(code_range, base_title)]
        
        start_num, end_num = int(start_num), int(end_num)
        codes = []
        for num in range(start_num, end_num + 1):
            code = f"{start_letter}{num:02d}"
            title = f"{base_title} ({code})"
            codes.append((code, title))
        return codes

    def create_icd_category(self, item, parent=None):
        code = item["code"]
        title = item.get("title", "No title")
        definition = item.get("definition", "")

        # Check if the code is a range and expandable
        if '-' in code and re.match(r'[A-Za-z]+\d+-[A-Za-z]+\d+', code):
            expanded_codes = self.expand_range(code, title)
            if len(expanded_codes) > 1:  # Range was expanded
                # Create the range as a parent category
                icd_entry, created = ICDCategory.objects.update_or_create(
                    code=code,
                    defaults={
                        "title": title,
                        "definition": definition,
                        "parent": parent,
                    }
                )
                print(f"{'Created' if created else 'Updated'}: {icd_entry} (Parent: {parent})")
                
                # Create individual codes under the parent
                for exp_code, exp_title in expanded_codes:
                    sub_entry, created = ICDCategory.objects.update_or_create(
                        code=exp_code,
                        defaults={
                            "title": exp_title,
                            "definition": definition,
                            "parent": icd_entry,
                        }
                    )
                    print(f"{'Created' if created else 'Updated'}: {sub_entry} (Parent: {icd_entry})")
            else:
                # Single entry (e.g., V01-X59)
                icd_entry, created = ICDCategory.objects.update_or_create(
                    code=code,
                    defaults={
                        "title": title,
                        "definition": definition,
                        "parent": parent,
                    }
                )
                print(f"{'Created' if created else 'Updated'}: {icd_entry} (Parent: {parent})")
        else:
            # Single code, no expansion
            icd_entry, created = ICDCategory.objects.update_or_create(
                code=code,
                defaults={
                    "title": title,
                    "definition": definition,
                    "parent": parent,
                }
            )
            print(f"{'Created' if created else 'Updated'}: {icd_entry} (Parent: {parent})")

        # Recursively process nested items
        for sub_item in item.get("subcategories", []):
            self.create_icd_category(sub_item, parent=icd_entry)
        for sub_item in item.get("sub_subcategories", []):
            self.create_icd_category(sub_item, parent=icd_entry)
        for code_item in item.get("codes", []):
            self.create_icd_category(code_item, parent=icd_entry)

    def handle(self, *args, **options):
        json_file = options['json_file']
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Clear existing data to avoid duplicates
        ICDCategory.objects.all().delete()
        self.stdout.write("Cleared existing ICDCategory data")

        categories = data.get("categories", data.get("codes", []))
        for category in categories:
            self.create_icd_category(category)
        self.stdout.write(self.style.SUCCESS("Successfully loaded ICD codes"))