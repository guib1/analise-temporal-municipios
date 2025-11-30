"""
era5_shapefile_download.py

Helper to download ERA5 (reanalysis-era5-single-levels) using a municipality shapefile
and produce a daily CSV with max/min/mean for requested variables.

Features:
- Accepts a shapefile path and a date range.
- Builds a CDS API request using the shapefile bounding box.
- By default uses the shapefile centroid as a representative point (fast).
- Optionally downloads the full bbox and masks/aggregates the grid (slower, larger).
- Produces a CSV with columns: lat, lon, date, rnet_max, rnet_min, rnet_mean, rterm_max, rterm_min, rterm_mean, pblhera5_max, pblhera5_min, pblhera5_mean

Requirements:
  pip install cdsapi xarray netcdf4 numpy pandas geopandas shapely
  Configure CDS API key (~/.cdsapirc) as described at https://cds.climate.copernicus.eu/api-how-to

Usage (from notebook):
  from baixar_dados.era5_shapefile_download import download_from_shapefile
  download_from_shapefile('shapefiles/SP_Diadema/SP_Diadema.shp', '01/01/2020', '31/01/2020', out_nc='diadema_jan2020.nc', out_csv='../data/raw/diadema_jan2020_daily.csv')

"""
from __future__ import annotations
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import cdsapi
import xarray as xr
import pandas as pd
import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:  # pragma: no cover
    gpd = None


def _parse_date(s: str) -> datetime.date:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    raise ValueError("Data inválida: use DD/MM/YYYY ou YYYY-MM-DD")


def _date_range_lists(start_date: datetime.date, end_date: datetime.date):
    current = start_date
    years = set()
    months = set()
    days = set()
    while current <= end_date:
        years.add(f"{current.year:04d}")
        months.add(f"{current.month:02d}")
        days.add(f"{current.day:02d}")
        current += timedelta(days=1)
    return sorted(list(years)), sorted(list(months)), sorted(list(days))


def _bbox_from_shapefile(shp_path: str) -> List[float]:
    if gpd is None:
        raise RuntimeError("geopandas não instalado; instale geopandas para usar shapefile")
    gdf = gpd.read_file(shp_path)
    minx, miny, maxx, maxy = gdf.total_bounds
    # cds area: [north, west, south, east]
    return [float(maxy), float(minx), float(miny), float(maxx)]


def _centroid_from_shapefile(shp_path: str) -> Tuple[float, float]:
    if gpd is None:
        raise RuntimeError("geopandas não instalado; instale geopandas para usar shapefile")
    gdf = gpd.read_file(shp_path)
    centroid = gdf.unary_union.centroid
    return float(centroid.y), float(centroid.x)


def request_era5_singlelevel(output_nc: str, area: Optional[List[float]], years: List[str], months: List[str], days: List[str], variables: List[str]):
    c = cdsapi.Client()
    times = [f"{h:02d}:00" for h in range(24)]
    req = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': variables,
        'year': years,
        'month': months,
        'day': days,
        'time': times,
    }
    if area is not None:
        req['area'] = area
    print("Enviando requisição CDS; aguarde. Parâmetros:")
    print(req)
    c.retrieve('reanalysis-era5-single-levels', req, output_nc)
    print("Download concluído ->", output_nc)


def process_nc_to_daily_csv(nc_path: str, coords: Optional[List[Tuple[float, float]]] = None, use_bbox: bool = False, out_csv: str = 'era5_daily.csv') -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)

    # variables expected
    # surface_net_solar_radiation (J m-2 per hour)
    # surface_net_thermal_radiation (J m-2 per hour)
    # boundary_layer_height (m)
    for v in ('surface_net_solar_radiation', 'surface_net_thermal_radiation', 'boundary_layer_height'):
        if v not in ds:
            raise RuntimeError(f"Variável {v} não encontrada no NetCDF; verifique o request e o arquivo {nc_path}")

    ds['rnet_Wm2'] = (ds['surface_net_solar_radiation'] + ds['surface_net_thermal_radiation']) / 3600.0
    ds['rterm_Wm2'] = ds['surface_net_thermal_radiation'] / 3600.0

    ds = ds[['rnet_Wm2', 'rterm_Wm2', 'boundary_layer_height']]

    if coords and not use_bbox:
        frames = []
        for lat, lon in coords:
            sel = ds.sel(latitude=lat, longitude=lon, method='nearest')
            dmax = sel.resample(time='1D').max()
            dmin = sel.resample(time='1D').min()
            dmean = sel.resample(time='1D').mean()
            df = pd.DataFrame({
                'date': pd.to_datetime(dmean['time'].values),
                'rnet_max': dmax['rnet_Wm2'].values,
                'rnet_min': dmin['rnet_Wm2'].values,
                'rnet_mean': dmean['rnet_Wm2'].values,
                'rterm_max': dmax['rterm_Wm2'].values,
                'rterm_min': dmin['rterm_Wm2'].values,
                'rterm_mean': dmean['rterm_Wm2'].values,
                'pblhera5_max': dmax['boundary_layer_height'].values,
                'pblhera5_min': dmin['boundary_layer_height'].values,
                'pblhera5_mean': dmean['boundary_layer_height'].values,
            })
            df['lat'] = lat
            df['lon'] = lon
            frames.append(df)
        final = pd.concat(frames, ignore_index=True)
        cols = ['lat', 'lon', 'date', 'rnet_max', 'rnet_min', 'rnet_mean', 'rterm_max', 'rterm_min', 'rterm_mean', 'pblhera5_max', 'pblhera5_min', 'pblhera5_mean']
        final = final[cols]
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        final.to_csv(out_csv, index=False)
        print('CSV gerado ->', out_csv)
        return final
    else:
        # aggregate whole grid (may be large)
        dmax = ds.resample(time='1D').max().compute()
        dmin = ds.resample(time='1D').min().compute()
        dmean = ds.resample(time='1D').mean().compute()
        records = []
        for t_idx, t in enumerate(dmean['time'].values):
            vmax = dmax.isel(time=t_idx)
            vmin = dmin.isel(time=t_idx)
            vmean = dmean.isel(time=t_idx)
            stacked = vmean['rnet_Wm2'].stack(allpoints=('latitude', 'longitude'))
            lat_vals = stacked['latitude'].values
            lon_vals = stacked['longitude'].values
            rec = pd.DataFrame({
                'lat': lat_vals,
                'lon': lon_vals,
                'date': pd.to_datetime(t),
                'rnet_max': vmax['rnet_Wm2'].stack(allpoints=('latitude','longitude')).values,
                'rnet_min': vmin['rnet_Wm2'].stack(allpoints=('latitude','longitude')).values,
                'rnet_mean': stacked.values,
                'rterm_max': vmax['rterm_Wm2'].stack(allpoints=('latitude','longitude')).values,
                'rterm_min': vmin['rterm_Wm2'].stack(allpoints=('latitude','longitude')).values,
                'rterm_mean': vmean['rterm_Wm2'].stack(allpoints=('latitude','longitude')).values,
                'pblhera5_max': vmax['boundary_layer_height'].stack(allpoints=('latitude','longitude')).values,
                'pblhera5_min': vmin['boundary_layer_height'].stack(allpoints=('latitude','longitude')).values,
                'pblhera5_mean': vmean['boundary_layer_height'].stack(allpoints=('latitude','longitude')).values,
            })
            records.append(rec)
        final = pd.concat(records, ignore_index=True)
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        final.to_csv(out_csv, index=False)
        print('CSV grade gerado ->', out_csv)
        return final


def download_from_shapefile(shapefile_path: str, start: str, end: str, out_nc: str = 'era5_out.nc', out_csv: str = '../data/raw/era5_daily.csv', use_bbox: bool = False, variables: Optional[List[str]] = None, cds_padding_deg: float = 0.0):
    """Wrapper to download ERA5 for a shapefile and produce daily CSV.

    shapefile_path: path to .shp
    start/end: date strings 'DD/MM/YYYY' or 'YYYY-MM-DD'
    out_nc/out_csv: output filenames
    use_bbox: if True, process every grid cell in bbox (can be very large)
    variables: list of ERA5 variable names to request; default set used if None
    cds_padding_deg: add padding (degrees) to bbox to ensure full coverage
    """
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    if end_date < start_date:
        raise ValueError('end < start')
    years, months, days = _date_range_lists(start_date, end_date)
    if variables is None:
        variables = ['surface_net_solar_radiation', 'surface_net_thermal_radiation', 'boundary_layer_height']

    bbox = _bbox_from_shapefile(shapefile_path)
    if cds_padding_deg and cds_padding_deg > 0:
        # bbox = [north, west, south, east]
        bbox = [bbox[0] + cds_padding_deg, bbox[1] - cds_padding_deg, bbox[2] - cds_padding_deg, bbox[3] + cds_padding_deg]

    coords = None
    if not use_bbox:
        centroid = _centroid_from_shapefile(shapefile_path)
        coords = [centroid]
        print(f'Usando centróide como ponto representativo: lat={centroid[0]}, lon={centroid[1]}')
    else:
        print('Usando bbox completa; processamento de grade pode ser lento e ocupar muita memória/disk')

    # Run request
    request_era5_singlelevel(out_nc, bbox, years, months, days, variables)

    # Process to CSV
    return process_nc_to_daily_csv(out_nc, coords=coords, use_bbox=use_bbox, out_csv=out_csv)


if __name__ == '__main__':
    # pequena CLI para testar localmente
    import argparse

    p = argparse.ArgumentParser(description='Download ERA5 a partir de shapefile e gerar CSV diário (max/min/mean)')
    p.add_argument('--shapefile', required=True, help='Caminho do shapefile (.shp)')
    p.add_argument('--start', required=True, help='Data inicial DD/MM/YYYY ou YYYY-MM-DD')
    p.add_argument('--end', required=True, help='Data final DD/MM/YYYY ou YYYY-MM-DD')
    p.add_argument('--out-nc', default='era5_out.nc')
    p.add_argument('--out-csv', default='../data/raw/era5_daily.csv')
    p.add_argument('--use-bbox', action='store_true', help='Processar grade inteira da bbox')
    p.add_argument('--pad', type=float, default=0.0, help='Padding em graus para bbox')
    args = p.parse_args()
    download_from_shapefile(args.shapefile, args.start, args.end, out_nc=args.out_nc, out_csv=args.out_csv, use_bbox=args.use_bbox, cds_padding_deg=args.pad)
