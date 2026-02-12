import earthaccess
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()

def test_earthaccess():
    auth = earthaccess.login(strategy="environment") # Uses EARTHDATA_USERNAME/PASSWORD or NASA_USER/PASSWORD
    if not auth:
        print("Authentication failed!")
        return

    print("Authenticated successfully.")

    # Search for MCD19A2
    print("Searching for MCD19A2 (Jan 2023, Sao Paulo approx box)...")
    results = earthaccess.search_data(
        short_name="MCD19A2",
        bounding_box=(-46.8, -24.0, -46.3, -23.0), # Approx SP
        temporal=("2023-01-01", "2023-01-05"),
        count=5
    )
    
    print(f"Found {len(results)} granules.")
    if results:
        print("Downloading first granule to testing...")
        files = earthaccess.download(results[0], "testing")
        print(f"Downloaded: {files}")
        
        # Try opening
        import xarray as xr
        try:
            print("Attempting to open with xarray...")
            ds = xr.open_dataset(files[0])
            print("Success! (default engine)")
            print(ds)
        except Exception as e:
            print(f"Failed with default: {e}")
            try:
                print("Attempting with engine='netcdf4'...")
                ds = xr.open_dataset(files[0], engine='netcdf4')
                print("Success! (netcdf4)")
                print(ds)
            except Exception as e2:
                print(f"Failed with netcdf4: {e2}")
                try:
                    print("Attempting with engine='rasterio' (rioxarray)...")
                    # Force string conversion
                    file_path = str(files[0])
                    ds = xr.open_dataset(file_path, engine='rasterio')
                    print("Success! (rasterio)")
                    print(ds)
                except Exception as e3:
                    print(f"Failed with rasterio: {e3}")
                    # Try pyhdf
                    try:
                        from pyhdf.SD import SD, SDC
                        print("Attempting with pyhdf...")
                        # Force string conversion as earthaccess might return Path objects
                        file_path = str(files[0])
                        hdf = SD(file_path, SDC.READ)
                        print("Success! (pyhdf)")
                        print(hdf.datasets().keys())
                    except Exception as e4:
                         print(f"Failed with pyhdf: {e4}")

if __name__ == "__main__":
    # Ensure env vars match what earthaccess expects if they differ
    # earthaccess expects EARTHDATA_USERNAME / EARTHDATA_PASSWORD by default
    # map NASA_USER -> EARTHDATA_USERNAME if needed
    if "EARTHDATA_USERNAME" not in os.environ and "NASA_USER" in os.environ:
        os.environ["EARTHDATA_USERNAME"] = os.environ["NASA_USER"]
    if "EARTHDATA_PASSWORD" not in os.environ and "NASA_PASSWORD" in os.environ:
        os.environ["EARTHDATA_PASSWORD"] = os.environ["NASA_PASSWORD"]
        
    test_earthaccess()
