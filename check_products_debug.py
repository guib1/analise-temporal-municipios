import requests
import json

APPEEARS_API_URL = "https://appeears.earthdatacloud.nasa.gov/api"

def debug_products():
    try:
        response = requests.get(f"{APPEEARS_API_URL}/product")
        response.raise_for_status()
        products = response.json()
        
        if products:
            print("First product structure:")
            print(json.dumps(products[0], indent=2))
            
            # Count total
            print(f"\nTotal products: {len(products)}")
            
            # Simple search again with looser criteria
            print("\nSearching for 'MCD19' in any value...")
            for p in products:
                # Convert entire dict to string for search
                if 'MCD19' in str(p):
                    print(f"MATCH: {p}")
        else:
            print("No products returned.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_products()
