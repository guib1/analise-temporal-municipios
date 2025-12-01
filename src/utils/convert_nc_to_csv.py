import os
import sys
from pathlib import Path

# Add project root to path to allow imports
sys.path.append(os.getcwd())

from baixar_dados.ERA5 import ERA5Downloader

def convert_and_cleanup():
    # Find all .nc files in the current directory and subdirectories
    root_dir = Path('.')
    # Exclude .venv or hidden directories if necessary, but rglob is simple
    nc_files = list(root_dir.rglob('*.nc'))
    
    if not nc_files:
        print("Nenhum arquivo .nc encontrado.")
        return

    print(f"Encontrados {len(nc_files)} arquivos .nc para conversão.")
    
    # Instantiate downloader (no credentials needed for processing)
    downloader = ERA5Downloader()

    for nc_file in nc_files:
        # Skip if inside .venv
        if '.venv' in str(nc_file):
            continue
            
        print(f"Processando {nc_file}...")
        csv_file = nc_file.with_suffix('.csv')
        
        try:
            # We use _process_nc_to_daily_csv with use_bbox=True to trigger _aggregate_grid
            # This will process all points in the NC file.
            # If the NC file was a single point (centroid), it will result in a CSV with that single point.
            downloader._process_nc_to_daily_csv(
                nc_path=str(nc_file),
                coords=None,
                use_bbox=True,
                out_csv=str(csv_file)
            )
            
            print(f"Convertido para {csv_file}")
            
            # Delete the .nc file
            os.remove(nc_file)
            print(f"Arquivo original removido: {nc_file}")
            
        except Exception as e:
            print(f"Falha ao converter {nc_file}: {e}")

if __name__ == "__main__":
    convert_and_cleanup()
