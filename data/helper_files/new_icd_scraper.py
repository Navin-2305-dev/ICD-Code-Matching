import os
import json
import time
import requests
from datetime import datetime
import jwt
from random import uniform

class TokenManager:
    def __init__(self):
        self.token = None
        self.token_expiry = None
        self.token_url = "https://icdaccessmanagement.who.int/connect/token"
        self.client_id = "acd986cd-04b6-4162-b455-23a264c09826_8e6015ac-455f-4fe4-9492-885ab1c29fbd"
        self.client_secret = "J2leDQ78jBy0ydB8af/uI8p9SnXsZEXR0xs/Q4cFjIk="
        self.scope = "icdapi_access"
    
    def get_new_token(self):
        """
        Function to get a new bearer token from WHO API.
        Makes a request to the WHO ICD API token endpoint using client credentials.
        """
        if not self.client_id or not self.client_secret:
            raise ValueError("Missing WHO API credentials. Please set WHO_CLIENT_ID and WHO_CLIENT_SECRET environment variables.")
        
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': self.scope
        }
        
        try:
            response = requests.post(self.token_url, data=data)
            
            if response.status_code == 200:
                token_data = response.json()
                return token_data['access_token']
            else:
                raise Exception("Failed to obtain token")
                
        except Exception as e:
            raise
    
    def get_valid_token(self):
        """Returns a valid bearer token, refreshing if necessary."""
        current_time = datetime.now().timestamp()
        
        # If we don't have a token or if it's expired/about to expire (within 60 seconds)
        if (not self.token or 
            not self.token_expiry or 
            self.token_expiry - current_time <= 60):
            
            self.token = self.get_new_token()
            
            # Decode token to get expiry time
            try:
                decoded_token = jwt.decode(self.token, options={"verify_signature": False})
                self.token_expiry = decoded_token.get('exp')
            except Exception:
                self.token_expiry = current_time + 3500  # Set default expiry to 58 minutes
        
        return self.token


class IcdScraper(TokenManager):
    def __init__(self):
        super().__init__()
        
        self.BASE_URL = "https://id.who.int"
        self.JSON_DIR = "data/icd_codes"
        os.makedirs(self.JSON_DIR, exist_ok=True)
    
    def get_headers(self):
        """Returns headers with a valid bearer token."""
        return {
            "Authorization": f"Bearer {self.get_valid_token()}",
            "Accept": "*/*",
            "Accept-Language": "en",
            "API-Version": "v2"
        }
    
    def fetch_json(self, url, max_retries=5, initial_delay=2):
        """
        Helper function to fetch JSON data from a URL with retry logic.
        
        Args:
            url (str): The URL to fetch data from
            max_retries (int): Maximum number of retry attempts (default: 5)
            initial_delay (int): Initial delay in seconds between retries (default: 2)
        """
        RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.get_headers(), timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                
                elif response.status_code == 401:  # Unauthorized - token might be expired
                    self.token = None  # Force token refresh
                    response = requests.get(url, headers=self.get_headers(), timeout=30)
                    if response.status_code == 200:
                        return response.json()
                
                elif response.status_code in RETRY_STATUS_CODES:
                    # Get retry-after header or use default backoff
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        sleep_time = int(retry_after)
                    else:
                        # Exponential backoff with jitter
                        sleep_time = initial_delay * (2 ** attempt) + uniform(0, 1)
                    
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        time.sleep(sleep_time)
                    continue
                
                if attempt < max_retries - 1:
                    sleep_time = initial_delay * (2 ** attempt) + uniform(0, 1)
                    time.sleep(sleep_time)
                    
            except requests.exceptions.Timeout:
                pass
            except requests.exceptions.ConnectionError:
                pass
            except requests.exceptions.RequestException:
                pass
            except Exception:
                pass
            
            if attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt) + uniform(0, 1)
                time.sleep(sleep_time)
        
        return None 
    
    def save_to_json(self, data, filename):
        """Saves data to a JSON file."""
        file_path = os.path.join(self.JSON_DIR, filename)
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
    
    def fetch_icd_codes(self):
        """Fetches ICD-10 codes for 2019 release from WHO API, including inclusions and exclusions."""
        print("Starting ICD-10 code scraping for 2019 release...")
        release_url = f"{self.BASE_URL}/icd/release/10"
        releases_data = self.fetch_json(release_url)
        if not releases_data:
            print("Failed to fetch release data")
            return {}
        
        releases = releases_data.get("release", [])
        # Filter for 2019 release only
        release_2019 = [url for url in releases if url.endswith('/2019')]
        
        if not release_2019:
            print("2019 release not found")
            return {}
        
        release_url = release_2019[0]
        release_year = "2019"
        print(f"\nProcessing release: {release_year}")
        
        release_data = self.fetch_json(release_url)
        if not release_data:
            print(f"Failed to fetch data for release {release_year}")
            return {}
        
        release_entry = {"year": release_year, "categories": []}
        categories = release_data.get("child", [])
        print(f"Found {len(categories)} categories in release {release_year}")
        
        start_time = time.time()
        for cat_idx, category_url in enumerate(categories):
            try:
                # Calculate and display progress for categories
                cat_elapsed = time.time() - start_time
                cat_items_per_sec = (cat_idx + 1) / cat_elapsed if cat_elapsed > 0 else 0
                cat_eta_seconds = (len(categories) - (cat_idx + 1)) / cat_items_per_sec if cat_items_per_sec > 0 else 0
                cat_eta_min = cat_eta_seconds / 60
                
                print(f"Category [{cat_idx+1}/{len(categories)}] - {(cat_idx+1)/len(categories)*100:.1f}% - ETA: {cat_eta_min:.1f} min")
                
                category_data = self.fetch_json(category_url)
                if not category_data:
                    print(f"Failed to fetch category data for {category_url}")
                    continue
                
                category_entry = {
                    "code": category_data.get("code", ""),
                    "title": category_data.get("title", {}).get("@value", ""),
                    "subcategories": []
                }
                # Add inclusions and exclusions if they exist
                if "inclusion" in category_data:
                    category_entry["inclusions"] = [
                        inc.get("label", {}).get("@value", "") for inc in category_data["inclusion"]
                    ]
                if "exclusion" in category_data:
                    category_entry["exclusions"] = [
                        exc.get("label", {}).get("@value", "") for exc in category_data["exclusion"]
                    ]
                
                subcategories = category_data.get("child", [])
                print(f"Found {len(subcategories)} subcategories")
                
                subcat_start_time = time.time()
                for subcat_idx, subcategory_url in enumerate(subcategories):
                    # Calculate and display progress for subcategories
                    subcat_elapsed = time.time() - subcat_start_time
                    subcat_items_per_sec = (subcat_idx + 1) / subcat_elapsed if subcat_elapsed > 0 else 0
                    subcat_eta_seconds = (len(subcategories) - (subcat_idx + 1)) / subcat_items_per_sec if subcat_items_per_sec > 0 else 0
                    subcat_eta_min = subcat_eta_seconds / 60
                    
                    print(f"Subcategory [{subcat_idx+1}/{len(subcategories)}] - {(subcat_idx+1)/len(subcategories)*100:.1f}% - ETA: {subcat_eta_min:.1f} min", end="\r")
                    
                    subcategory_data = self.fetch_json(subcategory_url)
                    if not subcategory_data:
                        continue
                    
                    subcategory_entry = {
                        "code": subcategory_data.get("code", ""),
                        "title": subcategory_data.get("title", {}).get("@value", ""),
                        "sub_subcategories": []
                    }
                    # Add inclusions and exclusions if they exist
                    if "inclusion" in subcategory_data:
                        subcategory_entry["inclusions"] = [
                            inc.get("label", {}).get("@value", "") for inc in subcategory_data["inclusion"]
                        ]
                    if "exclusion" in subcategory_data:
                        subcategory_entry["exclusions"] = [
                            exc.get("label", {}).get("@value", "") for exc in subcategory_data["exclusion"]
                        ]
                    
                    sub_subcategories = subcategory_data.get("child", [])
                    if len(sub_subcategories) > 0:
                        print(f"\nFound {len(sub_subcategories)} sub-subcategories")
                    
                    sub_subcat_start_time = time.time()
                    for sub_subcat_idx, sub_subcategory_url in enumerate(sub_subcategories):
                        # Calculate and display progress for sub-subcategories
                        if len(sub_subcategories) > 0:
                            sub_subcat_elapsed = time.time() - sub_subcat_start_time
                            sub_subcat_items_per_sec = (sub_subcat_idx + 1) / sub_subcat_elapsed if sub_subcat_elapsed > 0 else 0
                            sub_subcat_eta_seconds = (len(sub_subcategories) - (sub_subcat_idx + 1)) / sub_subcat_items_per_sec if sub_subcat_items_per_sec > 0 else 0
                            sub_subcat_eta_min = sub_subcat_eta_seconds / 60
                            
                            print(f"Sub-subcategory [{sub_subcat_idx+1}/{len(sub_subcategories)}] - {(sub_subcat_idx+1)/len(sub_subcategories)*100:.1f}% - ETA: {sub_subcat_eta_min:.1f} min", end="\r")
                        
                        sub_subcategory_data = self.fetch_json(sub_subcategory_url)
                        if not sub_subcategory_data:
                            continue
                        
                        sub_subcategory_entry = {
                            "code": sub_subcategory_data.get("code", ""),
                            "title": sub_subcategory_data.get("title", {}).get("@value", ""),
                            "codes": []
                        }
                        # Add inclusions and exclusions if they exist
                        if "inclusion" in sub_subcategory_data:
                            sub_subcategory_entry["inclusions"] = [
                                inc.get("label", {}).get("@value", "") for inc in sub_subcategory_data["inclusion"]
                            ]
                        if "exclusion" in sub_subcategory_data:
                            sub_subcategory_entry["exclusions"] = [
                                exc.get("label", {}).get("@value", "") for exc in sub_subcategory_data["exclusion"]
                            ]
                        
                        codes = sub_subcategory_data.get("child", [])
                        if len(codes) > 0:
                            print(f"\nFound {len(codes)} codes")
                        
                        code_start_time = time.time()
                        for code_idx, code_url in enumerate(codes):
                            # Calculate and display progress for codes
                            if len(codes) > 0:
                                code_elapsed = time.time() - code_start_time
                                code_items_per_sec = (code_idx + 1) / code_elapsed if code_elapsed > 0 else 0
                                code_eta_seconds = (len(codes) - (code_idx + 1)) / code_items_per_sec if code_items_per_sec > 0 else 0
                                code_eta_min = code_eta_seconds / 60
                                
                                print(f"Code [{code_idx+1}/{len(codes)}] - {(code_idx+1)/len(codes)*100:.1f}% - ETA: {code_eta_min:.1f} min", end="\r")
                            
                            code_data = self.fetch_json(code_url)
                            if code_data:
                                code_entry = {
                                    "code": code_data.get("code", ""),
                                    "title": code_data.get("title", {}).get("@value", "")
                                }
                                # Add inclusions and exclusions if they exist
                                if "inclusion" in code_data:
                                    code_entry["inclusions"] = [
                                        inc.get("label", {}).get("@value", "") for inc in code_data["inclusion"]
                                    ]
                                if "exclusion" in code_data:
                                    code_entry["exclusions"] = [
                                        exc.get("label", {}).get("@value", "") for exc in code_data["exclusion"]
                                    ]
                                sub_subcategory_entry["codes"].append(code_entry)
                        
                        if len(codes) > 0:
                            print()  # Add a newline after codes progress
                        
                        subcategory_entry["sub_subcategories"].append(sub_subcategory_entry)
                    
                    if len(sub_subcategories) > 0:
                        print()  # Add a newline after sub-subcategories progress
                    
                    category_entry["subcategories"].append(subcategory_entry)
                
                print()  # Add a newline after subcategories progress
                release_entry["categories"].append(category_entry)
            except Exception as e:
                print(f"\nError processing category: {str(e)}")
                continue
        
        print(f"\nCompleted processing release {release_year}. Saving to file...")
        self.save_to_json(release_entry, f"icd_version_{release_year}.json")
        print(f"Saved release {release_year} data to {self.JSON_DIR}/icd_version_{release_year}.json")
        
        print("\nICD-10 code scraping for 2019 completed!")


def run():
    icd_scraper = IcdScraper()
    icd_scraper.fetch_icd_codes()

if __name__ == "__main__":
    run()