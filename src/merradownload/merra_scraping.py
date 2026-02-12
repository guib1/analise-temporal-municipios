# Imports
import os
import time
import re
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from calendar import monthrange
from .opendap_download.multi_processing_download import DownloadManager


def baixar_merra(
    username,
    password,
    years,
    field_id,
    field_name,
    database_name,
    database_id,
    locs,
    conversion_function,
    aggregator,
    start_date=None,
    end_date=None
):

    ####### CONSTANTS - DO NOT CHANGE BELOW THIS LINE #######
    lat_coords = np.arange(0, 361, dtype=int)
    lon_coords = np.arange(0, 576, dtype=int)
    database_url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/' + database_name + '.5.12.4/'
    NUMBER_OF_CONNECTIONS = 5

    ####### DOWNLOAD DATA #########
    # Translate lat/lon into coordinates that MERRA-2 understands
    def translate_lat_to_geos5_native(latitude):
        """
        The source for this formula is in the MERRA2 
        Variable Details - File specifications for GEOS pdf file.
        The Grid in the documentation has points from 1 to 361 and 1 to 576.
        The MERRA-2 Portal uses 0 to 360 and 0 to 575.
        latitude: float Needs +/- instead of N/S
        """
        return ((latitude + 90) / 0.5)

    def translate_lon_to_geos5_native(longitude):
        """See function above"""
        return ((longitude + 180) / 0.625)

    def find_closest_coordinate(calc_coord, coord_array):
        """
        Since the resolution of the grid is 0.5 x 0.625, the 'real world'
        coordinates will not be matched 100% correctly. This function matches 
        the coordinates as close as possible. 
        """
        # np.argmin() finds the smallest value in an array and returns its
        # index. np.abs() returns the absolute value of each item of an array.
        # To summarize, the function finds the difference closest to 0 and returns 
        # its index. 
        index = np.abs(coord_array-calc_coord).argmin()
        return coord_array[index]

    def translate_year_to_file_number(year, month=None):
        """
        The file names consist of a number and a meta data string. 
        The number changes over the years. 1980 until 1991 it is 100, 
        1992 until 2000 it is 200, 2001 until 2010 it is  300 
        and from 2011 until now it is 400.
        """
        file_number = ''
        
        if year >= 1980 and year < 1992:
            file_number = '100'
        elif year >= 1992 and year < 2001:
            file_number = '200'
        elif year >= 2001 and year < 2011:
            file_number = '300'
        elif year >= 2011:
            file_number = '400'
        else:
            raise Exception('The specified year is out of range.')
        return file_number

    def generate_url_params(parameter, time_para, lat_para, lon_para):
        """Creates a string containing all the parameters in query form"""
        parameter = map(lambda x: x + time_para, parameter)
        parameter = map(lambda x: x + lat_para, parameter)
        parameter = map(lambda x: x + lon_para, parameter)
        return ','.join(parameter)
        
    def generate_download_links(download_years, base_url, dataset_name, url_params, start_date=None, end_date=None):
        """
        Generates the links for the download. 
        download_years: The years you want to download as array. 
        dataset_name: The name of the data set. For example tavg1_2d_slv_Nx
        start_date: datetime object (optional)
        end_date: datetime object (optional)
        """
        urls = []

        # If strict dates are provided, iterate day by day
        if start_date and end_date:
            curr_date = start_date
            while curr_date <= end_date:
                y = curr_date.year
                m = curr_date.month
                d = curr_date.day
                
                y_str = str(y)
                m_str = str(m).zfill(2)
                d_str = str(d).zfill(2)
                
                file_num = translate_year_to_file_number(y)
                
                file_name = 'MERRA2_{num}.{name}.{y}{m}{d}.nc4'.format(
                    num=file_num, name=dataset_name, 
                    y=y_str, m=m_str, d=d_str)
                
                query = '{base}{y}/{m}/{name}.nc4?{params}'.format(
                    base=base_url, y=y_str, m=m_str, 
                    name=file_name, params=url_params)
                urls.append(query)
                
                curr_date += timedelta(days=1)
                
        else:
            # Legacy behavior: download entire years
            for y in download_years: 
                y_str = str(y)
                file_num = translate_year_to_file_number(y)
                for m in range(1,13):
                    m_str = str(m).zfill(2)
                    _, nr_of_days = monthrange(y, m)
                    for d in range(1,nr_of_days+1):
                        d_str = str(d).zfill(2)
                        # Create the file name string
                        file_name = 'MERRA2_{num}.{name}.{y}{m}{d}.nc4'.format(
                            num=file_num, name=dataset_name, 
                            y=y_str, m=m_str, d=d_str)
                        # Create the query
                        query = '{base}{y}/{m}/{name}.nc4?{params}'.format(
                            base=base_url, y=y_str, m=m_str, 
                            name=file_name, params=url_params)
                        urls.append(query)
        return urls

    print('DOWNLOADING DATA FROM MERRA')
    print('Predicted time: ' + str(len(years)*len(locs)*6) + ' minutes')
    print('=====================')
    for loc, lat, lon in locs:
        print('Downloading ' + field_name + ' data for ' + loc)
        # Translate the coordinates that define your area to grid coordinates.
        lat_coord = translate_lat_to_geos5_native(lat)
        lon_coord = translate_lon_to_geos5_native(lon)
        # Find the closest coordinate in the grid.
        lat_closest = find_closest_coordinate(lat_coord, lat_coords)
        lon_closest = find_closest_coordinate(lon_coord, lon_coords)
        # Generate URLs for scraping
        requested_lat = '[{lat}:1:{lat}]'.format(lat=lat_closest)
        requested_lon = '[{lon}:1:{lon}]'.format(lon=lon_closest)
        parameter = generate_url_params([field_id], '[0:1:23]', requested_lat, requested_lon)
        generated_URL = generate_download_links(years, database_url, database_id, parameter, start_date, end_date)
        download_manager = DownloadManager()
        download_manager.set_username_and_password(username, password)
        download_manager.download_path = field_name + '/' + loc
        download_manager.download_urls = generated_URL
        start = time.time()
        download_manager.start_download(NUMBER_OF_CONNECTIONS)
        end = time.time()
        print(f"Download completed in {end - start} seconds.")

    ######### OPEN, CLEAN, MERGE, MERGE DATA AND WRITE CSVS ##########
    def extract_date(data_set):
        """
        Extracts the date from the filename before merging the datasets. 
        """ 
        if 'HDF5_GLOBAL.Filename' in data_set.attrs:
            f_name = data_set.attrs['HDF5_GLOBAL.Filename']
        elif 'Filename' in data_set.attrs:
            f_name = data_set.attrs['Filename']
        else: 
            raise AttributeError('The attribute name has changed again!')
        # find a match between "." and ".nc4" that does not have "." .
        exp = r'(?<=\.)[^\.]*(?=\.nc4)'
        match = re.search(exp, f_name)
        if match is None:
            raise ValueError(f'Unable to extract date from filename metadata: {f_name}')
        res = match.group(0)
        # Extract the date. 
        y, m, d = res[0:4], res[4:6], res[6:8]
        date_str = ('%s-%s-%s' % (y, m, d))
        data_set = data_set.assign(date=date_str)
        return data_set

    # Open nc4 files as dataframes, perform aggregations and save as CSV files
    print('CLEANING AND MERGING DATA')
    print('Predicted time: ' + str(len(years)*len(locs)*0.1) + ' minutes')
    print('=====================')
    for loc, lat, lon in locs:
        print('Cleaning and merging ' + field_name + ' data for ' + loc)
        dfs = []
        failed_files = []
        folder_path = os.path.join(field_name, loc)

        if not os.path.exists(folder_path):
            raise RuntimeError(
                f'Download folder not found: {folder_path}. '
                f'The download step may have failed.'
            )

        nc4_files = sorted(f for f in os.listdir(folder_path) if '.nc4' in f)
        if not nc4_files:
            raise RuntimeError(
                f'No .nc4 files found in {folder_path}. '
                f'The download step may have failed or files were not saved.'
            )

        for file in nc4_files:
            file_path = os.path.join(folder_path, file)

            # Pre-validate: detect corrupted files (HTML error pages, tiny files)
            file_size = os.path.getsize(file_path)
            if file_size < 200:
                with open(file_path, 'rb') as fh:
                    raw = fh.read()
                msg = (
                    f'File {file} is only {file_size} bytes and appears corrupted: '
                    f'{raw[:80].decode("utf-8", errors="replace")}'
                )
                if b'Access denied' in raw or b'<html' in raw.lower():
                    msg += ' (likely authentication failure)'
                failed_files.append((file, msg))
                print(f'Skipping corrupted file {file}: {msg}')
                continue

            try:
                with xr.open_dataset(file_path) as ds:
                    ds = extract_date(ds)
                    dfs.append(ds.to_dataframe())
            except Exception as exc:
                failed_files.append((file, str(exc)))
                print(f'Issue with file {file}: {exc}')

        if not dfs:
            sample_errors = '; '.join(
                f"{name}: {err}" for name, err in failed_files[:5]
            ) or 'no error details captured'
            raise RuntimeError(
                'No valid NetCDF files could be processed for '
                f"{field_name}/{loc}. Total failures: {len(failed_files)}. "
                f'Sample errors: {sample_errors}'
            )

        df_hourly = pd.concat(dfs)
        df_hourly['time'] = df_hourly.index.get_level_values(level=2)
        
        # Robustly identify the data column (it's not 'date' or 'time')
        cols = list(df_hourly.columns)
        data_cols = [c for c in cols if c not in ['date', 'time']]
        if not data_cols:
             raise ValueError(f"Could not identify data column in {cols}")
        data_col = data_cols[0]
        
        # Rename data column to field_name if it's different
        if data_col != field_name:
            df_hourly = df_hourly.rename(columns={data_col: field_name})

        # Force numeric, coercing errors to NaN
        df_hourly[field_name] = pd.to_numeric(df_hourly[field_name], errors='coerce')
        
        # Remove any rows where data might be missing (optional but cleaner for aggregation)
        # df_hourly = df_hourly.dropna(subset=[field_name]) # Kept commented to match original behavior of mean() handling NaNs

        df_hourly[field_name] = df_hourly[field_name].apply(conversion_function)
        df_hourly['date'] = pd.to_datetime(df_hourly['date'])
        df_hourly.to_csv(field_name + '/' + loc + '_hourly.csv', header=[field_name, 'date', 'time'], index=False)
        df_hourly = pd.read_csv(field_name + '/' + loc + '_hourly.csv')
        
        # Aggregate ONLY the data column (ignore 'time' column which causes mean/agg failure)
        df_daily = df_hourly.groupby('date')[[field_name]].agg(aggregator)
        # df_daily = df_daily.drop('time', axis=1) # No longer needed as we selected only field_name
        
        df_daily['date'] = df_daily.index
        df_daily.to_csv(field_name + '/' + loc + '_daily.csv', header=[field_name, 'date'], index=False)
        df_weekly = df_daily
        df_weekly['Week'] = pd.to_datetime(df_weekly['date']).apply(lambda x: x.isocalendar()[1])
        df_weekly['Year'] = pd.to_datetime(df_weekly['date']).apply(lambda x: x.year)
        
        # Aggregate ONLY the data column
        df_weekly = df_weekly.groupby(['Year', 'Week'])[[field_name]].agg(aggregator)
        
        df_weekly['Year'] = df_weekly.index.get_level_values(0)
        df_weekly['Week'] = df_weekly.index.get_level_values(1)
        df_weekly.to_csv(field_name + '/' + loc + '_weekly.csv', index=False)

    print('FINISHED')