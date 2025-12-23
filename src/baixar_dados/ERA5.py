from __future__ import annotations
import os
import logging
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple

import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
import zipfile
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    gpd = None

LOGGER = logging.getLogger(__name__)


class ERA5Downloader:
    """
    Downloads ERA5 data for a given shapefile and processes it into a daily CSV.
    """

    def __init__(self):
        """
        Initializes the downloader using credentials from environment variables.
        """
        self.cds_url = os.getenv("CDSAPI_URL")
        self.cds_key = os.getenv("CDSAPI_KEY")

    def fetch_daily_data(
        self,
        shapefile_path: str,
        start: str,
        end: str,
        out_nc: str = 'data/output/era5/temp_era5.nc',
        out_csv: str = 'data/output/era5/era5_daily.csv',
        use_bbox: bool = False,
        variables: Optional[List[str]] = None,
        cds_padding_deg: float = 0.0,
    ) -> pd.DataFrame:
        """
        Wrapper to download ERA5 for a shapefile and produce daily CSV.

        Args:
            shapefile_path: path to .shp
            start/end: date strings 'DD/MM/YYYY' or 'YYYY-MM-DD'
            out_nc/out_csv: output filenames
            use_bbox: if True, process every grid cell in bbox (can be very large)
            variables: list of ERA5 variable names to request; default set used if None
            cds_padding_deg: add padding (degrees) to bbox to ensure full coverage

        Returns:
            A pandas DataFrame with the daily aggregated data.
        """
        start_date = self._parse_date(start)
        end_date = self._parse_date(end)
        if end_date < start_date:
            raise ValueError('end date must be after start date')

        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_nc) or '.', exist_ok=True)

        if variables is None:
            variables = ['surface_net_solar_radiation', 'surface_net_thermal_radiation', 'boundary_layer_height']

        bbox = self._bbox_from_shapefile(shapefile_path)
        ibge_code = self._get_ibge_code(shapefile_path)

        if cds_padding_deg > 0:
            bbox = [
                bbox[0] + cds_padding_deg,
                bbox[1] - cds_padding_deg,
                bbox[2] - cds_padding_deg,
                bbox[3] + cds_padding_deg,
            ]

        coords = None
        if not use_bbox:
            centroid = self._centroid_from_shapefile(shapefile_path)
            coords = [centroid]
            LOGGER.info(f'Using centroid as representative point: lat={centroid[0]}, lon={centroid[1]}')
        else:
            LOGGER.info('Using full bounding box; grid processing can be slow and memory-intensive.')

        temp_files = []
        try:
            for year, month in self._iter_months(start_date, end_date):
                days_in_month = self._get_days_in_month(year, month, start_date, end_date)
                if not days_in_month:
                    continue

                chunk_nc = f"{out_nc.replace('.nc', '')}_{year}{month:02d}.nc"
                LOGGER.info(f"Downloading chunk: {year}-{month:02d} ({len(days_in_month)} days)")

                self._request_era5_singlelevel(
                    output_nc=chunk_nc,
                    area=bbox,
                    years=[str(year)],
                    months=[f"{month:02d}"],
                    days=days_in_month,
                    variables=variables,
                )
                temp_files.append(chunk_nc)

            if not temp_files:
                LOGGER.warning("No data downloaded.")
                return pd.DataFrame()

            LOGGER.info(f"Merging {len(temp_files)} NetCDF files...")
            ds_list = [xr.open_dataset(f, engine='netcdf4') for f in temp_files]
            ds_merged = xr.concat(ds_list, dim="time")
            ds_merged.to_netcdf(out_nc, engine='netcdf4')
            ds_merged.close()

        except Exception as e:
            msg = str(e)
            if "dependencies may not be installed" in msg or "netcdf4" in msg.lower():
                LOGGER.error("It seems you are missing the required libraries to read NetCDF files.")
                LOGGER.error("Please install them by running: pip install netcdf4 h5netcdf")
            LOGGER.error(f"Processing error: {e}")
            raise
        finally:
            # Cleanup temp files
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)

        # Process to CSV
        try:
            return self._process_nc_to_daily_csv(out_nc, coords=coords, use_bbox=use_bbox, out_csv=out_csv, ibge_code=ibge_code)
        finally:
            if os.path.exists(out_nc):
                os.remove(out_nc)
                LOGGER.info(f"Removed temporary file: {out_nc}")

    def _request_era5_singlelevel(
        self,
        output_nc: str,
        area: List[float],
        years: List[str],
        months: List[str],
        days: List[str],
        variables: List[str],
    ):
        if not self.cds_url or not self.cds_key:
            LOGGER.error("CDSAPI_URL and/or CDSAPI_KEY not found in environment variables.")
            LOGGER.error("Please ensure you have a .env file with these variables defined.")
            raise ValueError("Missing CDS API credentials in .env file.")

        try:
            c = cdsapi.Client(url=self.cds_url, key=self.cds_key)
        except Exception as e:
            LOGGER.error("Error initializing CDS API client: %s", e)
            LOGGER.error("Ensure CDSAPI_URL and CDSAPI_KEY are correctly set in your .env file.")
            raise

        req = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variables,
            'year': years,
            'month': months,
            'day': days,
            'time': [f"{h:02d}:00" for h in range(24)],
            'area': area,
        }
        LOGGER.info(f"Submitting CDS request: {req}")
        try:
            # CDS API might return a ZIP file even if we ask for netcdf
            # We download to a temporary name first
            temp_download = output_nc + ".download"
            c.retrieve('reanalysis-era5-single-levels', req, temp_download)
            
            # Check if it is a zip file
            if zipfile.is_zipfile(temp_download):
                LOGGER.info("CDS returned a ZIP file. Extracting...")
                with zipfile.ZipFile(temp_download, 'r') as zip_ref:
                    # Assuming the zip contains one or more NC files, we extract the first one
                    # or merge them if multiple. For simplicity, let's assume one relevant file or extract all.
                    # We extract to a temp dir
                    extract_dir = output_nc + "_extracted"
                    os.makedirs(extract_dir, exist_ok=True)
                    zip_ref.extractall(extract_dir)
                    
                    extracted_files = [
                        os.path.join(extract_dir, f) 
                        for f in os.listdir(extract_dir) 
                        if f.endswith('.nc')
                    ]
                    
                    if not extracted_files:
                        raise RuntimeError("No .nc files found in the downloaded ZIP.")
                    
                    if len(extracted_files) == 1:
                        shutil.move(extracted_files[0], output_nc)
                    else:
                        # If multiple files, we might need to merge them, but usually CDS returns one per request structure
                        # or we just take the first one if it matches our expectation.
                        # Let's try to merge them just in case
                        LOGGER.info(f"Found {len(extracted_files)} files in ZIP. Merging...")
                        ds_list = [xr.open_dataset(f) for f in extracted_files]
                        ds_merged = xr.concat(ds_list, dim="time")
                        ds_merged.to_netcdf(output_nc)
                        ds_merged.close()
                        for ds in ds_list:
                            ds.close()
                    
                    # Cleanup extraction dir
                    shutil.rmtree(extract_dir)
            else:
                # It's likely a NetCDF file already
                shutil.move(temp_download, output_nc)
                
            if os.path.exists(temp_download):
                os.remove(temp_download)

        except Exception as e:
            if 'required licences not accepted' in str(e):
                LOGGER.error("Copernicus license not accepted.")
                LOGGER.error("Please visit https://cds.climate.copernicus.eu/cdsapp/#!/terms/licences-to-use-copernicus-products to accept the terms.")
            raise
        LOGGER.info("Download complete -> %s", output_nc)

    def _get_ibge_code(self, shapefile_path: str) -> Optional[int]:
        if gpd is not None:
            try:
                gdf = gpd.read_file(shapefile_path)
                # Check for common column names for IBGE code
                for col in ['code_muni', 'CD_MUN', 'CD_GEOCMU']:
                    if col in gdf.columns:
                        return int(gdf[col].iloc[0])
            except Exception as e:
                LOGGER.warning(f"Could not read shapefile for IBGE code: {e}")
        
        # Fallback: look for CSV in the same directory
        try:
            directory = os.path.dirname(shapefile_path)
            for f in os.listdir(directory):
                if f.endswith('_ibge.csv'):
                    path = os.path.join(directory, f)
                    df = pd.read_csv(path)
                    if 'codigo_ibge' in df.columns:
                        return int(df['codigo_ibge'].iloc[0])
        except Exception as e:
            LOGGER.warning(f"Could not read IBGE CSV: {e}")
            
        return None

    def _process_nc_to_daily_csv(
        self,
        nc_path: str,
        coords: Optional[List[Tuple[float, float]]],
        use_bbox: bool,
        out_csv: str,
        ibge_code: Optional[int] = None
    ) -> pd.DataFrame:
        ds = xr.open_dataset(nc_path, engine='netcdf4')

        # Map long names to potential short names in the NetCDF
        var_map = {
            'surface_net_solar_radiation': ['surface_net_solar_radiation', 'ssr'],
            'surface_net_thermal_radiation': ['surface_net_thermal_radiation', 'str'],
            'boundary_layer_height': ['boundary_layer_height', 'blh']
        }

        # Rename variables to standard long names if they exist as short names
        for standard_name, candidates in var_map.items():
            found = False
            for candidate in candidates:
                if candidate in ds:
                    if candidate != standard_name:
                        ds = ds.rename({candidate: standard_name})
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Variable {standard_name} (or aliases {candidates}) not found in NetCDF {nc_path}")

        # Handle potential dimension issues (e.g. 'valid_time' vs 'time', extra 'time' dimension from concat)
        if 'valid_time' in ds.coords:
            if 'time' in ds.dims and 'time' not in ds.coords:
                # We have an extra 'time' dimension (likely from concat) and 'valid_time' coordinate
                # Collapse the extra 'time' dimension, assuming data is split across it with NaNs
                ds = ds.max(dim='time')
            
            if 'valid_time' in ds.coords and 'time' not in ds.coords:
                ds = ds.rename({'valid_time': 'time'})

        # NOTE:
        # ERA5 radiation variables are accumulated energy (J/m²) over the time step.
        # The project reference dataset (data/raw_input_reference/*) uses these values
        # directly (i.e., NOT converted to W/m² by dividing by 3600).
        # In the project reference dataset, `rnet_*` corresponds to SSR (solar) and
        # `rterm_*` corresponds to STR (thermal). Keep them separate.
        ds['rnet'] = ds['surface_net_solar_radiation']
        ds['rterm'] = ds['surface_net_thermal_radiation']
        ds = ds[['rnet', 'rterm', 'boundary_layer_height']]

        if coords and not use_bbox:
            # Single point analysis (centroid)
            final = self._aggregate_point(ds, coords, ibge_code)
        else:
            # Full grid analysis
            final = self._aggregate_grid(ds, ibge_code)
        
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        final.to_csv(out_csv, index=False)
        LOGGER.info('CSV generated -> %s', out_csv)
        return final

    def _aggregate_point(self, ds: xr.Dataset, coords: List[Tuple[float, float]], ibge_code: Optional[int] = None) -> pd.DataFrame:
        frames = []
        for lat, lon in coords:
            sel = ds.sel(latitude=lat, longitude=lon, method='nearest')
            # Daily aggregates (one row per day)
            dmax = sel.resample(time='1D').max()
            dmin = sel.resample(time='1D').min()
            dmean = sel.resample(time='1D').mean()
            df = pd.DataFrame({
                'date': pd.to_datetime(dmean['time'].values),
                'rnet_max': dmax['rnet'].values,
                'rnet_min': dmin['rnet'].values,
                'rnet_mea': dmean['rnet'].values,
                'rterm_max': dmax['rterm'].values,
                'rterm_min': dmin['rterm'].values,
                'rterm_mea': dmean['rterm'].values,
                'pblhera5_max': dmax['boundary_layer_height'].values,
                'pblhera5_min': dmin['boundary_layer_height'].values,
                'pblhera5_mea': dmean['boundary_layer_height'].values,
            })
            if ibge_code:
                df['codigo_ibge'] = ibge_code
            else:
                df['lat'] = lat
                df['lon'] = lon
            frames.append(df)
        final = pd.concat(frames, ignore_index=True)
        
        cols = ['date', 'rnet_max', 'rnet_min', 'rnet_mea', 'rterm_max', 'rterm_min', 'rterm_mea', 'pblhera5_max', 'pblhera5_min', 'pblhera5_mea']
        if ibge_code:
            cols.insert(0, 'codigo_ibge')
        else:
            cols = ['lat', 'lon'] + cols
            
        return final[cols]

    def _aggregate_grid(self, ds: xr.Dataset, ibge_code: Optional[int] = None) -> pd.DataFrame:
        dmax = ds.resample(time='1D').max().compute()
        dmin = ds.resample(time='1D').min().compute()
        dmean = ds.resample(time='1D').mean().compute()
        
        records = []
        for t_idx, t in enumerate(dmean['time'].values):
            vmax = dmax.isel(time=t_idx)
            vmin = dmin.isel(time=t_idx)
            vmean = dmean.isel(time=t_idx)
            
            stacked = vmean['rnet'].stack(allpoints=('latitude', 'longitude'))
            
            data = {
                'date': pd.to_datetime(t),
                'rnet_max': vmax['rnet'].stack(allpoints=('latitude','longitude')).values,
                'rnet_min': vmin['rnet'].stack(allpoints=('latitude','longitude')).values,
                'rnet_mea': stacked.values,
                'rterm_max': vmax['rterm'].stack(allpoints=('latitude','longitude')).values,
                'rterm_min': vmin['rterm'].stack(allpoints=('latitude','longitude')).values,
                'rterm_mea': vmean['rterm'].stack(allpoints=('latitude','longitude')).values,
                'pblhera5_max': vmax['boundary_layer_height'].stack(allpoints=('latitude','longitude')).values,
                'pblhera5_min': vmin['boundary_layer_height'].stack(allpoints=('latitude','longitude')).values,
                'pblhera5_mea': vmean['boundary_layer_height'].stack(allpoints=('latitude','longitude')).values,
            }
            
            if ibge_code:
                data['codigo_ibge'] = ibge_code
            else:
                data['lat'] = stacked['latitude'].values
                data['lon'] = stacked['longitude'].values
                
            rec = pd.DataFrame(data)
            records.append(rec)
            
        return pd.concat(records, ignore_index=True)


    @staticmethod
    def _parse_date(s: str) -> date:
        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                pass
        raise ValueError(f"Invalid date format: {s}. Use DD/MM/YYYY or YYYY-MM-DD")

    @staticmethod
    def _iter_months(start_date: date, end_date: date):
        current = start_date.replace(day=1)
        while current <= end_date:
            yield current.year, current.month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

    @staticmethod
    def _get_days_in_month(year: int, month: int, start_date: date, end_date: date) -> List[str]:
        import calendar
        _, last_day = calendar.monthrange(year, month)
        days = []
        for d in range(1, last_day + 1):
            date_obj = date(year, month, d)
            if start_date <= date_obj <= end_date:
                days.append(f"{d:02d}")
        return days

    @staticmethod
    def _bbox_from_shapefile(shp_path: str) -> List[float]:
        if gpd is None:
            raise RuntimeError("geopandas is not installed; please install it to use shapefiles.")
        gdf = gpd.read_file(shp_path)
        minx, miny, maxx, maxy = gdf.total_bounds
        return [float(maxy), float(minx), float(miny), float(maxx)]

    @staticmethod
    def _centroid_from_shapefile(shp_path: str) -> Tuple[float, float]:
        if gpd is None:
            raise RuntimeError("geopandas is not installed; please install it to use shapefiles.")
        gdf = gpd.read_file(shp_path)
        try:
            union_geom = gdf.union_all()
        except AttributeError:
            union_geom = gdf.unary_union
        centroid = union_geom.centroid
        return float(centroid.y), float(centroid.x)


if __name__ == '__main__':
    # Basic manual test (requires network access and supporting libraries).
    logging.basicConfig(level=logging.INFO)
    
    # Example usage:
    downloader = ERA5Downloader()
    try:
        df_result = downloader.fetch_daily_data(
            shapefile_path='data/shapefiles/SP-São_Paulo/SP_São_Paulo.shp',
            start='2000-01-01',
            end='2000-02-01',
            out_nc='data/output/era5/sao_paulo_era5.nc',
            out_csv='data/output/era5/daily_sao_paulo_2000.csv',
            use_bbox=False, # Use centroid by default
        )
        print("\n--- Download and processing successful ---")
        print(df_result.head())
    except Exception as e:
        LOGGER.error("An error occurred during the main execution: %s", e)