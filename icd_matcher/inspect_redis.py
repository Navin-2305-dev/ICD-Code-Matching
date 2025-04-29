# icd_matcher/inspect_redis.py
import os
import sys
import redis
import json
import django

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
                records = MedicalAdmissionDetails.objects.filter(created_at__contains=date_done[:10])
                if records:
                    print("Related database records:")
                    for r in records:
                        print(f"- ID: {r.id}, ICD Codes: {r.predicted_icd_codes}")
            except json.JSONDecodeError:
                print(f"{key}: (Unparseable) {result}")

    except redis.ConnectionError as e:
        print(f"Redis connection failed: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_redis()