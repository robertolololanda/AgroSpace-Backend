import s3fs
import xarray as xr
import datetime
import numpy as np
import time

def extract_for_location(target_lat=-36.2086, target_lon=-61.87869):
    print(f"Buscando datos para Lat: {target_lat}, Lon: {target_lon}...")
    
    start_time = time.time()
    fs = s3fs.S3FileSystem(anon=True)
    
    now = datetime.datetime.utcnow()
    # Find latest available cycle (00 or 12)
    s3_path = None
    for hour_offset in range(48):
        test_date = now - datetime.timedelta(hours=hour_offset)
        year = test_date.strftime("%Y")
        month = test_date.strftime("%m")
        day = test_date.strftime("%d")
        cycle = "12" if test_date.hour >= 12 else "00"
        
        prefix = f"smn-ar-wrf/DATA/WRF/DET/{year}/{month}/{day}/{cycle}/"
        files = fs.ls(prefix)
        if files:
            s3_path = f"s3://{files[0]}"
            print(f"Cycle found: {year}-{month}-{day} {cycle}UTC")
            break
            
    if not s3_path:
        print("No se encontraron datos recientes en S3.")
        return

    print(f"Descargando/Abriendo el archivo: {s3_path}")
    s3_file_obj = fs.open(s3_path)
    
    # Open dataset using h5netcdf engine via memory buffer from S3
    ds = xr.open_dataset(s3_file_obj, engine='h5netcdf')
    
    # Obtenemos las matrices 2D de latitud y longitud
    lats = ds['lat'].values
    lons = ds['lon'].values
    
    print("Calculando vecino más cercano (Matriz Euclidiana)...")
    calc_start = time.time()
    # Calculamos la distancia euclidiana pitagórica
    distances = np.sqrt((lats - target_lat)**2 + (lons - target_lon)**2)
    
    # Encontramos el índice (y, x) del mínimo
    min_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    y_idx, x_idx = min_idx
    calc_time = time.time() - calc_start
    print(f"Índice encontrado: y={y_idx}, x={x_idx} en {calc_time:.3f} segundos.")
    
    # Coordenada real del modelo
    real_lat = lats[y_idx, x_idx]
    real_lon = lons[y_idx, x_idx]
    dist_error = distances[y_idx, x_idx] * 111 # aprox km
    print(f"Coordenada WRF más cercana: {real_lat:.4f}, {real_lon:.4f} (Error: {dist_error:.2f} km)")
    
    # Extraer variables para ese punto
    t2 = ds['T2'].isel(y=y_idx, x=x_idx).values[0] # T2m 
    pp = ds['PP'].isel(y=y_idx, x=x_idx).values[0] # Precipitación
    rh = ds['HR2'].isel(y=y_idx, x=x_idx).values[0] # Humedad
    
    print("\n=== CLIMA ACTUAL (Hora 0) ===")
    print(f"Temperatura: {t2:.1f} °C")
    print(f"Humedad Relativa: {rh:.1f} %")
    print(f"Lluvia Acumulada: {pp:.2f} mm")
    
    total_time = time.time() - start_time
    print(f"\nTiempo Total Ejecución: {total_time:.2f} segundos.")

if __name__ == "__main__":
    extract_for_location()
