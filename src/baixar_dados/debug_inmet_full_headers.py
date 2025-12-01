import requests
import logging

logging.basicConfig(level=logging.DEBUG)

url = "https://apitempo.inmet.gov.br/estacao/2025-11-29/2025-11-30/A701"

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
    "Referer": "https://tempo.inmet.gov.br/",
    "Origin": "https://tempo.inmet.gov.br",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

s = requests.Session()
s.headers.update(headers)

# Visit homepage first to get cookies
print("Visiting homepage...")
try:
    s.get("https://tempo.inmet.gov.br/")
except Exception as e:
    print(f"Error visiting homepage: {e}")

print(f"Requesting {url}...")
try:
    r = s.get(url)
    print(f"Status Code: {r.status_code}")
    print(f"Headers: {r.headers}")
    print(f"Content: {r.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
