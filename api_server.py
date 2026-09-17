from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import s3fs
import xarray as xr
import datetime
import numpy as np
import asyncio
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrospace")

# ─── In-memory forecast cache ──────────────────────────────────────────────
# Structure: { "lat_lon_key": { "expires": datetime, "data": {...} } }
CACHE: dict = {}
DOWNLOADING: set = set()

# Known client coordinates — pre-loaded on startup
KNOWN_CLIENTS = [
    {"lat": -36.2086, "lon": -61.87869},  # Campo Marina Blasco
]

# La corrida del SMN se nombra en UTC; la respuesta se declara en hora argentina
# (utc_offset_seconds = -10800) y la plataforma empareja estas horas con las de
# Open-Meteo, que vienen en hora local. Sin esta conversión cada valor quedaba
# rotulado 3 h más tarde de lo que correspondía.
ART_OFFSET = datetime.timedelta(hours=-3)

# El NetCDF declara magViento10 en "meter / second". La API lo publica como
# wind_speed_10m en km/h, la misma unidad que Open-Meteo y que muestra la
# plataforma al lado de las ráfagas.
MS_TO_KMH = 3.6


def build_coord_key(lat: float, lon: float) -> str:
    return f"{round(lat, 4)}_{round(lon, 4)}"

def find_nearest_idx(lats, lons, target_lat, target_lon):
    dist = np.sqrt((lats - target_lat) ** 2 + (lons - target_lon) ** 2)
    y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
    return int(y_idx), int(x_idx)

def fetch_smn_data_sync(target_lat: float, target_lon: float):
    """Synchronous worker — runs in a thread so it doesn't block the event loop."""
    coord_key = build_coord_key(target_lat, target_lon)
    if coord_key in DOWNLOADING:
        return
    DOWNLOADING.add(coord_key)

    try:
        fs = s3fs.S3FileSystem(anon=True)
        now = datetime.datetime.utcnow()

        # 1. Find latest available SMN cycle
        s3_prefix = None
        cycle_id = None
        for hour_offset in range(48):
            test_date = now - datetime.timedelta(hours=hour_offset)
            yy = test_date.strftime("%Y")
            mm = test_date.strftime("%m")
            dd = test_date.strftime("%d")
            cc = "12" if test_date.hour >= 12 else "00"
            prefix = f"smn-ar-wrf/DATA/WRF/DET/{yy}/{mm}/{dd}/{cc}/"
            try:
                files = fs.ls(prefix)
                if files:
                    s3_prefix = prefix
                    cycle_id = f"{yy}{mm}{dd}_{cc}"
                    break
            except FileNotFoundError:
                continue

        if not s3_prefix:
            logger.warning(f"[{coord_key}] No SMN cycle found in last 48h!")
            return

        logger.info(f"[{coord_key}] Processing cycle {cycle_id}...")

        # Cada corrida publica tres familias: 01H (horaria, con T2, HR2, PP y
        # viento), 10M (solo precipitación cada 10 min) y 24H (acumulados
        # diarios). Solo la horaria tiene lo que se extrae; las otras dos
        # sumaban 76 archivos y ~0,74 GB por corrida que se bajaban y fallaban.
        nc_files = sorted(f for f in fs.ls(s3_prefix)
                          if "WRFDETAR_01H_" in f and f.endswith(".nc"))
        logger.info(f"[{coord_key}] Found {len(nc_files)} hourly files")

        cycle_dt = datetime.datetime.strptime(cycle_id, "%Y%m%d_%H")
        y_idx, x_idx = None, None

        times, t2s, rhs, pps, wspds, wdirs = [], [], [], [], [], []

        for f in nc_files:
            try:
                hr_str = f.split("_")[-1].replace(".nc", "")
                hr_offset = int(hr_str)
            except Exception:
                continue

            try:
                with fs.open(f"s3://{f}") as fobj:
                    ds = xr.open_dataset(fobj, engine="h5netcdf")

                    if y_idx is None:
                        lats = ds["lat"].values
                        lons = ds["lon"].values
                        y_idx, x_idx = find_nearest_idx(lats, lons, target_lat, target_lon)
                        real_lat = float(lats[y_idx, x_idx])
                        real_lon = float(lons[y_idx, x_idx])
                        dist_km = float(np.sqrt((real_lat - target_lat)**2 + (real_lon - target_lon)**2) * 111)
                        logger.info(f"[{coord_key}] Grid match: ({real_lat:.4f}, {real_lon:.4f}) | Error: {dist_km:.2f} km")

                    t2 = float(ds["T2"].isel(y=y_idx, x=x_idx).values[0])
                    rh = float(ds["HR2"].isel(y=y_idx, x=x_idx).values[0])
                    pp = float(ds["PP"].isel(y=y_idx, x=x_idx).values[0])
                    wspd = float(ds["magViento10"].isel(y=y_idx, x=x_idx).values[0])
                    wdir = float(ds["dirViento10"].isel(y=y_idx, x=x_idx).values[0])
                    ds.close()

                valid_time = cycle_dt + datetime.timedelta(hours=hr_offset) + ART_OFFSET
                times.append(valid_time.strftime("%Y-%m-%dT%H:00"))
                t2s.append(round(t2, 1))
                rhs.append(round(rh, 1))
                pps.append(round(pp, 2))
                wspds.append(round(wspd * MS_TO_KMH, 1))
                wdirs.append(round(wdir, 0))

            except Exception as e:
                logger.warning(f"[{coord_key}] Error at hr={hr_offset}: {e}")
                continue

        if times:
            expires = datetime.datetime.utcnow() + datetime.timedelta(hours=12)
            CACHE[coord_key] = {
                "expires": expires,
                "cycle": cycle_id,
                "data": {
                    "latitude": target_lat,
                    "longitude": target_lon,
                    "generationtime_ms": 0.0,
                    "utc_offset_seconds": -10800,   # Argentina (UTC-3)
                    "timezone": "America/Argentina/Buenos_Aires",
                    "timezone_abbreviation": "ART",
                    "hourly": {
                        "time": times,
                        "temperature_2m": t2s,
                        "relative_humidity_2m": rhs,
                        "precipitation": pps,
                        "wind_speed_10m": wspds,
                        "wind_direction_10m": wdirs,
                    }
                }
            }
            logger.info(f"[{coord_key}] Cache built ✅ — {len(times)} hours loaded.")
        else:
            logger.error(f"[{coord_key}] No data extracted!")

    finally:
        DOWNLOADING.discard(coord_key)


def refresh_all_known_clients():
    """Runs in a background thread. Refreshes all known client coordinates."""
    for client in KNOWN_CLIENTS:
        coord_key = build_coord_key(client["lat"], client["lon"])
        entry = CACHE.get(coord_key)
        if entry and datetime.datetime.utcnow() < entry["expires"]:
            logger.info(f"[{coord_key}] Cache still valid, skipping refresh.")
            continue
        logger.info(f"[{coord_key}] Starting refresh...")
        fetch_smn_data_sync(client["lat"], client["lon"])


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """On startup: kick off a background thread to pre-load all known client data."""
    logger.info("=== AgroSpace Backend Starting — Pre-loading SMN WRF data ===")
    thread = threading.Thread(target=refresh_all_known_clients, daemon=True)
    thread.start()
    yield
    logger.info("=== AgroSpace Backend Shutdown ===")


# ─── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="AgroSpace SMN WRF 4km API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    cached_coords = list(CACHE.keys())
    return {
        "status": "AgroSpace SMN WRF 4km API Online",
        "version": "2.2",
        "cached_coordinates": cached_coords,
        "downloading": list(DOWNLOADING),
    }


@app.get("/v1/forecast")
def get_forecast(latitude: float, longitude: float):
    coord_key = build_coord_key(latitude, longitude)
    entry = CACHE.get(coord_key)

    if not entry:
        # Not a known client — trigger fetch and return partial status
        if coord_key not in DOWNLOADING:
            thread = threading.Thread(
                target=fetch_smn_data_sync, args=(latitude, longitude), daemon=True
            )
            thread.start()
        return {
            "status": "processing",
            "message": (
                "Datos SMN WRF 4km en descarga desde AWS (~8 min). "
                "Por favor, reintente en breve. "
                f"Coordenadas: {latitude}, {longitude}"
            ),
        }

    # Trigger background refresh if expired (serve stale data meanwhile)
    if datetime.datetime.utcnow() >= entry["expires"]:
        if coord_key not in DOWNLOADING:
            thread = threading.Thread(
                target=fetch_smn_data_sync, args=(latitude, longitude), daemon=True
            )
            thread.start()

    return entry["data"]


@app.get("/v1/status")
def get_status():
    status = {}
    for key, entry in CACHE.items():
        hours = len(entry["data"]["hourly"]["time"])
        expires_in = (entry["expires"] - datetime.datetime.utcnow()).total_seconds() / 3600
        status[key] = {
            "cycle": entry["cycle"],
            "hours_cached": hours,
            "expires_in_hours": round(expires_in, 1),
        }
    return {"cache": status, "downloading": list(DOWNLOADING)}
