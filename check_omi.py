import earthaccess
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

def check_omi():
    auth = earthaccess.login(strategy="environment")
    if not auth:
        print("Auth failed")
        return

    # Check L3 Daily NO2 (0.25 deg)
    print("\n--- Searching for OMNO2d (Level 3 Daily) ---")
    results_l3 = earthaccess.search_data(
        short_name="OMNO2d",
        bounding_box=(-46.8, -24.0, -46.3, -23.0),
        temporal=("2023-01-01", "2023-01-05"),
        count=3
    )
    print(f"Found {len(results_l3)} granules for OMNO2d.")
    if results_l3:
        print(results_l3[0])
        print(results_l3[0].data_links(access="direct"))

    # Check L2 Swath NO2
    print("\n--- Searching for OMNO2 (Level 2 Swath) ---")
    results_l2 = earthaccess.search_data(
        short_name="OMNO2",
        bounding_box=(-46.8, -24.0, -46.3, -23.0),
        temporal=("2023-01-01", "2023-01-05"),
        count=3
    )
    print(f"Found {len(results_l2)} granules for OMNO2.")
    if results_l2:
        print(results_l2[0])
        print(results_l2[0].data_links(access="direct"))
        
     # Check L3 Ozone (OMTO3d)
    print("\n--- Searching for OMTO3d (Level 3 Daily Ozone) ---")
    results_o3 = earthaccess.search_data(
        short_name="OMTO3d",
        bounding_box=(-46.8, -24.0, -46.3, -23.0),
        temporal=("2023-01-01", "2023-01-05"),
        count=3
    )
    print(f"Found {len(results_o3)} granules for OMTO3d.")


if __name__ == "__main__":
    check_omi()
