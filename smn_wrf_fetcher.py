import s3fs
import xarray as xr
import datetime
import pandas as pd

def explore_smn_data():
    fs = s3fs.S3FileSystem(anon=True)
    
    # Get today's UTC date
    now = datetime.datetime.utcnow()
    # Try yesterday if today hasn't been uploaded yet
    for test_date in [now, now - datetime.timedelta(days=1)]:
        year = test_date.strftime("%Y")
        month = test_date.strftime("%m")
        day = test_date.strftime("%d")
        cycle = "00" # Usually runs at 00 UTC and 12 UTC
        
        prefix = f"smn-ar-wrf/DATA/WRF/DET/{year}/{month}/{day}/{cycle}/"
        print(f"Checking prefix: {prefix}")
        
        files = fs.ls(prefix)
        if files:
            print(f"Found {len(files)} files!")
            
            # Let's open the first forecast hour file (000)
            first_file = files[0]
            print(f"Opening: s3://{first_file}")
            
            # Open directly from S3 using xarray and h5netcdf
            s3_file_obj = fs.open(f"s3://{first_file}")
            ds = xr.open_dataset(s3_file_obj, engine='h5netcdf')
            
            print("\n--- DATASET STRUCTURE ---")
            print(ds.info())
            
            print("\n--- VARIABLES ---")
            for var in ds.data_vars:
                print(f"{var}: {ds[var].long_name if hasattr(ds[var], 'long_name') else 'N/A'}")
                
            # Target coords: Campo Marina Blasco
            target_lat = -36.2086
            target_lon = -61.87869
            
            # TODO: We will need to figure out which dimensions are lat/lon or y/x
            # to extract the nearest neighbor.
            
            return
            
    print("No files found for today or yesterday.")

if __name__ == "__main__":
    explore_smn_data()
