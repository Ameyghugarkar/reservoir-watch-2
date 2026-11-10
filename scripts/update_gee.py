"""
update_gee.py
Automatically fetches new Sentinel-2 data from Earth Engine and appends it to reservoir_data.csv
Supports both local OAuth credentials and GitHub Actions service account.
"""
import ee
import pandas as pd
import json
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path

# Parse optional --service-account argument (used by GitHub Actions)
parser = argparse.ArgumentParser()
parser.add_argument('--service-account', type=str, default=None,
                    help='Path to GEE service account JSON key file')
args = parser.parse_args()

# Initialize Earth Engine BEFORE defining any geometries!
print("Connecting to Google Earth Engine...")
try:
    if args.service_account:
        # GitHub Actions path: use service account JSON
        key = json.load(open(args.service_account))
        credentials = ee.ServiceAccountCredentials(
            email=key['client_email'], key_file=args.service_account
        )
        ee.Initialize(credentials, project='big-elysium-428716-s2')
        print("Authenticated via Service Account.")
    else:
        # Local path: use saved OAuth token
        ee.Initialize(project='big-elysium-428716-s2')
        print("Authenticated via local OAuth token.")
except Exception as e:
    print(f"Earth Engine authentication failed: {e}")
    print("Locally: run 'python -c \"import ee; ee.Authenticate()\"'")
    exit(1)

# Paths
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / 'data' / 'reservoir_data.csv'

# Just Khadakwasla for now (we can add the others later)
RESERVOIRS = {
    'Khadakwasla': ee.Geometry.Rectangle([73.71, 18.40, 73.79, 18.46]),
}

def add_ndwi(image):
    return image.addBands(image.normalizedDifference(['B3', 'B8']).rename('NDWI'))

def get_area_for_image(image, geometry):
    try:
        date_ms = image.get('system:time_start').getInfo()
        if not date_ms: return None
        date_str = datetime.utcfromtimestamp(date_ms / 1000).strftime('%Y-%m-%d')
        cloud_cover = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo() or 0.0

        water_mask = add_ndwi(image).select('NDWI').gt(0.2)
        area_sqm = water_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geometry, scale=10, maxPixels=1e10
        ).get('NDWI').getInfo()
        
        if area_sqm is None: return None
        return {
            'date': date_str, 
            'surface_area_sqkm': round(area_sqm / 1e6, 4), 
            'cloud_cover_pct': round(float(cloud_cover), 1)
        }
    except Exception as e: 
        print(f"Warning on image: {e}")
        return None

def update_reservoir(name, geometry):
    # Load existing data to find the last date
    try:
        existing = pd.read_csv(CSV_PATH, parse_dates=['date'])
        sub = existing[existing['reservoir_name'] == name]
        last_date = sub['date'].max() if not sub.empty else None
    except FileNotFoundError:
        existing = pd.DataFrame()
        last_date = None

    today = date.today().strftime('%Y-%m-%d')
    
    if last_date is None:
        start_date = '2020-01-01'
        print(f"{name}: First run, fetching from {start_date} to {today}")
    else:
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        if start_date >= today:
            print(f"{name}: Already up to date (Last obs: {last_date.strftime('%Y-%m-%d')}).")
            return
        print(f"{name}: Fetching new data from {start_date} to {today}...")

    # Fetch from GEE
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(geometry)
                  .filterDate(start_date, today)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    
    n = collection.size().getInfo()
    print(f"  Found {n} new cloud-free images.")
    if n == 0: return

    records = []
    image_list = collection.toList(n)
    for i in range(n):
        res = get_area_for_image(ee.Image(image_list.get(i)), geometry)
        if res:
            res['reservoir_name'] = name
            records.append(res)
            
    if not records: return
            
    new_df = pd.DataFrame(records)
    new_df['date'] = pd.to_datetime(new_df['date'])
    
    # Combine and save
    combined = (pd.concat([existing, new_df])
                .drop_duplicates(subset=['reservoir_name', 'date'])
                .sort_values(['reservoir_name', 'date'])
                .reset_index(drop=True))
    
    combined.to_csv(CSV_PATH, index=False)
    print(f"  Saved {len(new_df)} new records. Total is now {len(combined[combined['reservoir_name']==name])}.")

if __name__ == '__main__':
    for name, geom in RESERVOIRS.items():
        update_reservoir(name, geom)
    
    print("\nGEE Update Complete.")
