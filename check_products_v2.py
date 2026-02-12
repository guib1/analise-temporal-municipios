import requests
import json

APPEEARS_API_URL = "https://appeears.earthdatacloud.nasa.gov/api"

def debug_products():
    try:
        response = requests.get(f"{APPEEARS_API_URL}/product")
        response.raise_for_status()
        products = response.json()
        
        print(f"Total products: {len(products)}")
        
        search_terms = ['Aerosol', 'MOD04', 'MYD04', 'MCD19', 'MAIAC']
        
        for p in products:
            p_str = str(p)
            for term in search_terms:
                if term in p_str:
                    print(f"MATCH ({term}): {p['ProductAndVersion']} - {p['Description']}")
                    # print(json.dumps(p, indent=2))
                    break # avoid double printing
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_products()
