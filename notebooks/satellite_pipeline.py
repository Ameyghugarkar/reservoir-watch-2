# =============================================================================
# satellite_pipeline.py  --  Incremental GEE Pipeline (run in Google Colab)
# ReservoirWatch: Pune Reservoir NDWI Water Surface Area
# =============================================================================
# KEY FEATURE: INCREMENTAL UPDATES
#   - First run: fetches everything from 2020-01-01 to TODAY
#   - Every subsequent run: only fetches NEW satellite passes since your last run
#   - Safe to re-run daily/weekly with zero wasted API calls
#
# HOW TO USE:
#   1. colab.research.google.com -> New notebook
#   2. Create one cell per ## CELL N section below
#   3. Run top to bottom
#   4. Run it again next week to get the latest passes automatically
# =============================================================================


## CELL 1 — Install (run once per session)
# !pip install earthengine-api geemap -q


## CELL 2 — Imports
import ee
import geemap
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from google.colab import files
import io, os

print("Imports done")


## CELL 3 — Authenticate & Initialise
ee.Authenticate()
ee.Initialize(project='big-elysium-428716-s2')
print("Earth Engine ready")

# TODAY is always the live end date — no hardcoding
TODAY = date.today().strftime('%Y-%m-%d')
FULL_START = '2020-01-01'    # Historical baseline start

print(f"Extraction window: {FULL_START}  -->  {TODAY}  (live)")


## CELL 4 — Reservoir Geometries
# Format: ee.Geometry.Rectangle([west_lon, south_lat, east_lon, north_lat])
#
# Khadakwasla: CONFIRMED (6.6-7.0 sq km, 39 dates on Day 1)
# Panshet / Varasgaon: research-estimated — run visualize_ndwi() to verify

RESERVOIRS = {
    'Khadakwasla': ee.Geometry.Rectangle([73.71, 18.40, 73.79, 18.46]),
    'Panshet':     ee.Geometry.Rectangle([73.55, 18.35, 73.65, 18.43]),  # VERIFY
    'Varasgaon':   ee.Geometry.Rectangle([73.58, 18.36, 73.68, 18.42]),  # VERIFY
}

# Colab path for the accumulating CSV
CSV_PATH = '/content/reservoir_data.csv'


## CELL 5 — Core Functions
def add_ndwi(image):
    """
    NDWI = (Green B3 - NIR B8) / (Green B3 + NIR B8)
    CRITICAL: always .rename('NDWI') — without it the band is called 'nd'
    and stats.get('NDWI').getInfo() silently returns None.
    """
    return image.addBands(
        image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    )


def get_area_for_image(image, geometry):
    """Compute water surface area (sq km) for a single Sentinel-2 image."""
    try:
        date_ms = image.get('system:time_start').getInfo()
        if date_ms is None: return None
        date_str    = datetime.utcfromtimestamp(date_ms / 1000).strftime('%Y-%m-%d')
        cloud_cover = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo() or 0.0

        water_mask = add_ndwi(image).select('NDWI').gt(0.2)
        stats = (water_mask
                 .multiply(ee.Image.pixelArea())
                 .reduceRegion(reducer=ee.Reducer.sum(), geometry=geometry,
                               scale=10, maxPixels=1e10))
        area_sqm = stats.get('NDWI').getInfo()
        if area_sqm is None: return None

        return {
            'date':               date_str,
            'surface_area_sqkm':  round(area_sqm / 1e6, 4),
            'cloud_cover_pct':    round(float(cloud_cover), 1),
        }
    except Exception as exc:
        print(f"    Warning: {exc}")
        return None


def get_last_date_in_csv(reservoir_name):
    """
    Read the local CSV and return the most recent date for this reservoir.
    Returns None if file doesn't exist or reservoir has no rows.
    """
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=['date'])
        sub = df[df['reservoir_name'] == reservoir_name]
        return sub['date'].max() if not sub.empty else None
    except FileNotFoundError:
        return None


def extract_range(geometry, name, start_date, end_date):
    """
    Pull Sentinel-2 data for one reservoir between two dates.
    Returns a DataFrame (may be empty if no cloud-free images).
    """
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )
    n = collection.size().getInfo()
    print(f"    {n} cloud-free images found")
    if n == 0:
        return pd.DataFrame()

    image_list = collection.toList(n)
    records    = []
    for i in range(n):
        result = get_area_for_image(ee.Image(image_list.get(i)), geometry)
        if result:
            result['reservoir_name'] = name
            records.append(result)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n} processed ...")

    if not records:
        return pd.DataFrame()

    return (pd.DataFrame(records)
              [['reservoir_name', 'date', 'surface_area_sqkm', 'cloud_cover_pct']]
              .assign(date=lambda d: pd.to_datetime(d['date']))
              .sort_values('date').reset_index(drop=True))


def extract_incremental(geometry, name):
    """
    SMART INCREMENTAL EXTRACTION:
    - Checks the CSV for the most recent date already extracted for this reservoir
    - If none found: fetches from FULL_START (2020-01-01) to TODAY
    - If found: fetches only from last_date+1 day to TODAY  (delta only!)
    - Merges and deduplicates into the CSV

    This means running the notebook again next week costs almost nothing —
    it only fetches the handful of new satellite passes since last time.
    """
    last_date = get_last_date_in_csv(name)

    if last_date is None:
        start = FULL_START
        print(f"\n  {name}: No existing data — full extraction from {start} to {TODAY}")
    else:
        start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        days_of_new = (date.today() - last_date.date()).days
        print(f"\n  {name}: Last obs {last_date.date()} | fetching {days_of_new} days of new data ({start} → {TODAY})")

    if start > TODAY:
        print(f"  {name}: Already fully up to date — skipping")
        return

    new_df = extract_range(geometry, name, start, TODAY)

    if new_df.empty:
        print(f"  {name}: No new cloud-free images in this window")
        return

    # Load existing CSV, append new rows, deduplicate, save
    try:
        existing = pd.read_csv(CSV_PATH, parse_dates=['date'])
    except FileNotFoundError:
        existing = pd.DataFrame()

    combined = (pd.concat([existing, new_df])
                  .drop_duplicates(subset=['reservoir_name', 'date'])
                  .sort_values(['reservoir_name', 'date'])
                  .reset_index(drop=True))
    combined.to_csv(CSV_PATH, index=False)

    n_new    = len(new_df)
    n_total  = len(combined[combined['reservoir_name'] == name])
    print(f"  {name}: +{n_new} new records | {n_total} total in CSV")


def visualize_ndwi(geometry, name):
    """
    Render true-colour + NDWI + water mask for visual verification.
    ALWAYS run this before trusting extract_incremental() numbers.
    Cyan pixels = detected water. They should tightly cover the reservoir.
    """
    print(f"Rendering verification map for {name} ...")
    image = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geometry)
        .filterDate('2024-02-01', '2024-06-01')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .first()
    )
    ndwi = add_ndwi(image).select('NDWI')
    M    = geemap.Map()
    M.centerObject(geometry, zoom=13)
    M.addLayer(image, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000},
               f'{name} True Colour')
    M.addLayer(ndwi,  {'min': -0.5, 'max': 0.5, 'palette': ['red', 'white', 'blue']},
               f'{name} NDWI')
    M.addLayer(ndwi.gt(0.2), {'min': 0, 'max': 1, 'palette': ['black', 'cyan']},
               f'{name} Water Mask')
    M.addLayerControl()
    return M

print("Functions loaded")


## CELL 6 — VISUAL VERIFICATION
# Run one at a time. Confirm cyan water shape matches the actual reservoir.
# If it doesn't, adjust the bounding box in CELL 4 and re-run.

# visualize_ndwi(RESERVOIRS['Khadakwasla'], 'Khadakwasla')  # already confirmed
# visualize_ndwi(RESERVOIRS['Panshet'],     'Panshet')       # VERIFY BEFORE EXTRACTING
# visualize_ndwi(RESERVOIRS['Varasgaon'],   'Varasgaon')     # VERIFY BEFORE EXTRACTING


## CELL 7 — RUN INCREMENTAL EXTRACTION (safe to re-run anytime)
print("=" * 55)
print(f"Incremental extraction run — {TODAY}")
print("=" * 55)

for reservoir_name, geometry in RESERVOIRS.items():
    extract_incremental(geometry, reservoir_name)

print("\nExtraction complete.")


## CELL 8 — SUMMARY & DOWNLOAD
try:
    final = pd.read_csv(CSV_PATH, parse_dates=['date'])
    print(f"\nTotal records: {len(final)}")
    print(final.groupby('reservoir_name').agg(
        count    = ('date', 'count'),
        from_date= ('date', 'min'),
        to_date  = ('date', 'max'),       # <-- this is your "live" latest date
        area_min = ('surface_area_sqkm', 'min'),
        area_max = ('surface_area_sqkm', 'max'),
    ).to_string())

    print("\nDownloading reservoir_data.csv ...")
    files.download(CSV_PATH)

except FileNotFoundError:
    print("No CSV found — check if extraction ran successfully above.")
