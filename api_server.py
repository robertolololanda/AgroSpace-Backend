from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Index
from sqlalchemy.orm import declarative_base, sessionmaker
import s3fs
import xarray as xr
import datetime
import numpy as np
import time
import os

# Database Setup
Base = declarative_base()
engine = create_engine('sqlite:///smn_cache.db', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Forecast(Base):
    __tablename__ = "forecasts"
    id = Column(Integer, primary_key=True, index=True)
    lat = Column(Float, index=True)
    lon = Column(Float, index=True)
    cycle = Column(String) # To know which forecast cycle this belongs to
    time = Column(DateTime, index=True)
    temperature_2m = Column(Float)
    relative_humidity_2m = Column(Float)
    precipitation = Column(Float)
    wind_speed_10m = Column(Float)
    wind_direction_10m = Column(Float)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AgroSpace SMN WRF 4km API")

# Setup CORS to allow the Hostinger dashboard to ping this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global lock/flag to prevent multiple concurrent downloads for the same coords
downloading_coords = set()

def fetch_smn_data_worker(target_lat: float, target_lon: float):
    coord_key = f"{target_lat:.4f}_{target_lon:.4f}"
    if coord_key in downloading_coords:
        return
    downloading_coords.add(coord_key)
    
    fs = s3fs.S3FileSystem(anon=True)
    now = datetime.datetime.utcnow()
    
    # 1. Detect latest cycle
    s3_prefix = None
    cycle_id = None
    for hour_offset in range(48):
        test_date = now - datetime.timedelta(hours=hour_offset)
        year = test_date.strftime("%Y")
        month = test_date.strftime("%m")
        day = test_date.strftime("%d")
        cc = "12" if test_date.hour >= 12 else "00"
        
        prefix = f"smn-ar-wrf/DATA/WRF/DET/{year}/{month}/{day}/{cc}/"
        try:
            files = fs.ls(prefix)
            if files:
                s3_prefix = prefix
                cycle_id = f"{year}{month}{day}_{cc}"
                break
        except FileNotFoundError:
            continue
            
    if not s3_prefix:
        downloading_coords.remove(coord_key)
        return
        
    print(f"[{coord_key}] Inicia descarga del ciclo {cycle_id}...")
    
    try:
        # 2. Delete old forecast for these coords
        db = SessionLocal()
        db.query(Forecast).filter(Forecast.lat == target_lat, Forecast.lon == target_lon).delete()
        db.commit()
        
        # 3. Process the 72 hours
        files = fs.ls(s3_prefix)
        files.sort() # Ensure temporal order 000 to 072
        
        # We find the nearest neighbor index ONLY ONCE using the first file
        y_idx, x_idx = None, None
        cycle_dt = datetime.datetime.strptime(cycle_id, "%Y%m%d_%H")
        
        for f in files:
            if not f.endswith(".nc"): continue
            
            # Extract forecast hour (000, 001, etc) from filename WRFDETAR_01H_20260321_00_000.nc
            try:
                hr_str = f.split('_')[-1].replace('.nc', '')
                hr_offset = int(hr_str)
            except: continue
            
            s3_file_obj = fs.open(f"s3://{f}")
            try:
                ds = xr.open_dataset(s3_file_obj, engine='h5netcdf')
            except Exception as e:
                print(f"Error opening {f}: {e}")
                continue
                
            if y_idx is None:
                # Map coords
                lats = ds['lat'].values
                lons = ds['lon'].values
                dist = np.sqrt((lats - target_lat)**2 + (lons - target_lon)**2)
                y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
                
            # Extract point
            t2 = ds['T2'].isel(y=y_idx, x=x_idx).values[0]
            rh = ds['HR2'].isel(y=y_idx, x=x_idx).values[0]
            pp = ds['PP'].isel(y=y_idx, x=x_idx).values[0]
            w_spd = ds['magViento10'].isel(y=y_idx, x=x_idx).values[0]
            w_dir = ds['dirViento10'].isel(y=y_idx, x=x_idx).values[0]
            
            # Time of this timestep
            valid_time = cycle_dt + datetime.timedelta(hours=hr_offset)
            
            # Save to DB
            new_record = Forecast(
                lat=target_lat,
                lon=target_lon,
                cycle=cycle_id,
                time=valid_time,
                temperature_2m=float(t2),
                relative_humidity_2m=float(rh),
                precipitation=float(pp),
                wind_speed_10m=float(w_spd),
                wind_direction_10m=float(w_dir)
            )
            db.add(new_record)
            
            # Periodically commit so frontend can see partial loads
            if hr_offset % 6 == 0:
                db.commit()
                print(f"[{coord_key}] Guardadas {hr_offset} horas...")
                
        db.commit()
        db.close()
        print(f"[{coord_key}] Carga completada ({len(files)} horas).")
        
    finally:
        downloading_coords.remove(coord_key)

@app.get("/")
def health_check():
    return {"status": "AgroSpace SMN WRF 4km API Online"}

@app.get("/v1/forecast")
def get_forecast(latitude: float, longitude: float, background_tasks: BackgroundTasks):
    db = SessionLocal()
    
    # Fix precision to match cache key
    lat_r = round(latitude, 4)
    lon_r = round(longitude, 4)
    
    # Query future records from DB
    records = db.query(Forecast).filter(
        Forecast.lat == lat_r,
        Forecast.lon == lon_r,
    ).order_by(Forecast.time).all()
    
    db.close()
    
    # If no records, trigger background fetch and return "Processing" status
    if not records:
        background_tasks.add_task(fetch_smn_data_worker, lat_r, lon_r)
        return {
            "status": "processing", 
            "message": "Datos de Alta Resolución SMN 4km en descarga desde AWS. El proceso completo demora ~8 minutos. Por favor reintente en breve."
        }
        
    # Check if the cache is older than 24 hours
    current_cycle = records[0].cycle
    cycle_dt = datetime.datetime.strptime(current_cycle, "%Y%m%d_%H")
    if (datetime.datetime.utcnow() - cycle_dt).total_seconds() > 86400:
        background_tasks.add_task(fetch_smn_data_worker, lat_r, lon_r) # Trigger stealth background update
        
    # Format identical to Open-Meteo
    return {
        "latitude": latitude,
        "longitude": longitude,
        "generationtime_ms": 0.0,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "timezone_abbreviation": "GMT",
        "hourly": {
            "time": [r.time.strftime("%Y-%m-%dT%H:00") for r in records],
            "temperature_2m": [round(r.temperature_2m, 1) for r in records],
            "relative_humidity_2m": [round(r.relative_humidity_2m, 1) for r in records],
            "precipitation": [round(r.precipitation, 2) for r in records],
            "wind_speed_10m": [round(r.wind_speed_10m, 1) for r in records],
            "wind_direction_10m": [round(r.wind_direction_10m, 0) for r in records]
        }
    }
