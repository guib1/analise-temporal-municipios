import requests
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

APPEEARS_API_URL = "https://appeears.earthdatacloud.nasa.gov/api"

def list_products():
    try:
        response = requests.get(f"{APPEEARS_API_URL}/product")
        response.raise_for_status()
        products = response.json()
        
        # Filter for AOD or MCD19
        found = []
        for p in products:
            if 'MCD19' in p['ProductAndVersion'] or 'AOD' in p['Description'] or 'MOD04' in p['ProductAndVersion']:
                found.append(p)
        
        print(f"Found {len(found)} products matching criteria:\n")
        for p in found:
            print(f"Product: {p['ProductAndVersion']}")
            print(f"Platform: {p['Platform']}")
            print(f"Description: {p['Description']}")
            print("-" * 40)
            
    except Exception as e:
        LOGGER.error(f"Error listing products: {e}")

if __name__ == "__main__":
    list_products()
