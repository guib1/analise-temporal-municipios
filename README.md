# analise-temporal-municipios

## Dependencies
- python3.11

Fedora: python3.11-devel
Ubuntu/Debian: python3.11-dev
- python3.11-distutils

## Configuration

### ERA5 Data Access (CDS API)

To download ERA5 data, you need a Copernicus Climate Data Store (CDS) account and API key.

1.  **Register/Login**: Go to [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/) and create an account or log in.
2.  **Accept Licenses**: **Crucial Step!** You must accept the terms of use for the "ERA5 hourly data on single levels from 1940 to present" dataset. Visit [https://cds.climate.copernicus.eu/cdsapp/#!/terms/licences-to-use-copernicus-products](https://cds.climate.copernicus.eu/cdsapp/#!/terms/licences-to-use-copernicus-products) and accept the licenses. If you skip this, the download will fail.
3.  **Get API Key**: Go to your user profile page (usually linked from the top right) to find your UID and API Key.
4.  **Setup Environment Variables**:
    Create a `.env` file in the root of the project (copy from `.env.example` if available) and add your credentials:

    ```ini
    CDSAPI_URL=https://cds.climate.copernicus.eu/api/v2
    CDSAPI_KEY=YOUR_UID:YOUR_API_KEY
    ```

    Alternatively, you can still use the `~/.cdsapirc` file method or pass arguments via command line, but the `.env` method is recommended for this project.