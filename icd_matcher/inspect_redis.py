import os
import sys
import redis
import json
import django
from datetime import datetime, timedelta

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'icd_matcher_project.settings')
django.setup()

# Import Django model after setup
from icd_matcher.models import MedicalAdmissionDetails

def inspect_redis():
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        print("Connected to Redis")

        # List all keys
        print("\nAll Redis keys:")
        keys = r.keys('*')
        if not keys:
            print("No keys found")
        for key in keys:
            print(f"- {key}")

        # Inspect celery queue
        print("\nCelery queue contents:")
        queue_items = r.lrange('celery', 0, -1)
        if not queue_items:
            print("Queue is empty")
        for i, item in enumerate(queue_items, 1):
            try:
                decoded = json.loads(item)
                print(f"Task {i}: {json.dumps(decoded, indent=2)}")
            except json.JSONDecodeError:
                print(f"Task {i}: (Unparseable) {item}")

        # Inspect task results
        print("\nTask results:")
        task_keys = [k for k in keys if k.startswith('celery-task-meta-')]
        if not task_keys:
            print("No task results found")
        for key in task_keys:
            result = r.get(key)
            try:
                decoded = json.loads(result)
                print(f"{key}:")
                print(json.dumps(decoded, indent=2))
                ttl = r.ttl(key)
                print(f"TTL: {ttl} seconds")
                # Correlate with database
                task_id = decoded['task_id']
                date_done = decoded.get('date_done', '')
                if date_done:
                    try:
                        # Parse date_done (ISO format: 2025-05-05T06:46:04.542845+00:00)
                        date_done_dt = datetime.fromisoformat(date_done.replace('Z', '+00:00'))
                        # Filter records within a 1-hour range
                        start_date = date_done_dt - timedelta(hours=1)
                        end_date = date_done_dt + timedelta(hours=1)
                        records = MedicalAdmissionDetails.objects.filter(
                            created_at__range=(start_date, end_date)
                        )
                        if records:
                            print("Related database records:")
                            for record in records:
                                icd_codes = getattr(record, 'predicted_icd_codes', []) or []
                                print(f"- ID: {record.id}, ICD Codes: {icd_codes}")
                        else:
                            print("No related database records found")
                    except ValueError as e:
                        print(f"Error parsing date_done: {e}")
                else:
                    print("No date_done available for correlation")
            except json.JSONDecodeError:
                print(f"{key}: (Unparseable) {result}")

    except redis.ConnectionError as e:
        print(f"Redis connection failed: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_redis()