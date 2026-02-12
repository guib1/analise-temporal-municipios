import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

APPEEARS_API_URL = "https://appeears.earthdatacloud.nasa.gov/api"
USER = os.getenv("NASA_USER")
PASSWORD = os.getenv("NASA_PASSWORD")

def login():
    if not USER or not PASSWORD:
        print("Missing credentials.")
        return None
    try:
        response = requests.post(
            f"{APPEEARS_API_URL}/login", 
            auth=(USER, PASSWORD)
        )
        response.raise_for_status()
        return response.json()['token']
    except Exception as e:
        print(f"Login failed: {e}")
        return None

def dump_products():
    token = login()
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    
    try:
        response = requests.get(f"{APPEEARS_API_URL}/product", headers=headers)
        response.raise_for_status()
        products = response.json()
        
        with open("all_products_auth.txt", "w") as f:
            for p in products:
                f.write(f"{p['ProductAndVersion']} - {p['Description']}\n")
                
        print(f"Dumped {len(products)} products to all_products_auth.txt")
        
        # Check specific
        for p in products:
            if 'MCD19' in p['ProductAndVersion']:
                print(f"FOUND: {p['ProductAndVersion']} - {p['Description']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    dump_products()
