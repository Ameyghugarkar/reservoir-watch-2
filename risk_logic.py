"""
risk_logic.py  --  Risk Analysis Engine
ReservoirWatch: Pune Reservoir Drought Early Warning System

Reads data/reservoir_data.csv, computes depletion rates, classifies risk,
projects days of buffer remaining, and writes data/risk_analysis.csv.

Usage:
    python risk_logic.py                      # default paths
    python risk_logic.py --input my.csv       # custom input
    python risk_logic.py --rainfall           # include OWM rainfall (needs OWM_API_KEY in .env)
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta
import os
import sys
import argparse
import requests
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── CONFIG ───────────────────────────────────────────────────────────────────────

# Reservoir max surface areas (sq km) — from known capacities
RESERVOIR_MAX_AREAS = {
    'Khadakwasla': 7.5,
    'Panshet':     12.0,   # updated from satellite observation
    'Varasgaon':   18.5,   # updated from satellite observation
    'Temghar':     4.5,    # updated from satellite observation
}


# Risk thresholds
CRITICAL_AREA_PCT   = 35.0   # % of max capacity
CRITICAL_DEPLETION  = 4.0    # %/week
WATCH_AREA_PCT      = 55.0
WATCH_DEPLETION     = 2.0

# Minimum viable fill level — "days remaining" counts down to this
MIN_VIABLE_FRACTION = 0.20

# Rolling window for depletion rate regression
ROLLING_WINDOW_DAYS = 28

# OpenWeatherMap
OWM_API_KEY  = os.getenv('OWM_API_KEY', '')
PUNE_LAT     = 18.5204
PUNE_LON     = 73.8567


# ── DEPLETION RATE ───────────────────────────────────────────────────────────────

def calculate_depletion_rates(df_reservoir: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling linear regression over last ROLLING_WINDOW_DAYS to estimate
    weekly depletion rate as % of max capacity.

    Positive  = depleting (area shrinking)
    Negative  = filling   (monsoon recharge)
    """
    df = df_reservoir.sort_values('date').copy()
    reservoir_name = df['reservoir_name'].iloc[0]
    max_area = RESERVOIR_MAX_AREAS.get(reservoir_name, df['surface_area_sqkm'].max())

    rates = []
    for i in range(len(df)):
        current_date = df.iloc[i]['date']
        window_start = current_date - timedelta(days=ROLLING_WINDOW_DAYS)
        window = df[(df['date'] >= window_start) & (df['date'] <= current_date)]

        if len(window) >= 3:
            x = (window['date'] - window['date'].min()).dt.days.values.astype(float)
            y = window['surface_area_sqkm'].values.astype(float)
            slope, *_ = stats.linregress(x, y)
            # slope is sq km/day  ->  % of max per week, sign-flipped (positive = depletion)
            rate = -(slope * 7 / max_area) * 100
        else:
            rate = 0.0

        rates.append(round(rate, 4))

    df['depletion_rate_pct_per_week'] = rates
    return df


# ── RISK CLASSIFICATION ──────────────────────────────────────────────────────────

def classify_risk(row, max_area: float) -> str:
    """
    SAFE     -- area > 55% of max AND depletion < 2%/wk
    WATCH    -- area 35-55% of max OR depletion 2-4%/wk
    CRITICAL -- area < 35% of max OR depletion > 4%/wk
    Filling rapidly (depletion < -1%/wk) always returns SAFE.
    """
    area_pct   = (row['surface_area_sqkm'] / max_area) * 100
    depletion  = row['depletion_rate_pct_per_week']

    if depletion < -1.0:                                             # actively filling
        return 'SAFE'
    if area_pct < CRITICAL_AREA_PCT or depletion > CRITICAL_DEPLETION:
        return 'CRITICAL'
    if area_pct < WATCH_AREA_PCT    or depletion > WATCH_DEPLETION:
        return 'WATCH'
    return 'SAFE'


def days_remaining(row, max_area: float) -> int:
    """
    Days until reservoir drops below MIN_VIABLE_FRACTION of max capacity.
    Returns 999 if filling or depletion negligible.
    Returns 0   if already below minimum.
    """
    min_viable   = max_area * MIN_VIABLE_FRACTION
    current_area = row['surface_area_sqkm']
    depletion    = row['depletion_rate_pct_per_week']

    if current_area <= min_viable:
        return 0
    if depletion <= 0.1:
        return 999

    area_per_day = (depletion / 100 * max_area) / 7
    buffer       = current_area - min_viable
    return int(min(999, max(0, buffer / area_per_day)))


# ── RAINFALL (OPTIONAL) ──────────────────────────────────────────────────────────

def fetch_rainfall_pune() -> float | None:
    """Fetch last-7-day total rainfall (mm) for Pune via OpenWeatherMap."""
    if not OWM_API_KEY:
        return None
    try:
        import time
        total = 0.0
        for days_back in range(1, 8):
            ts  = int(time.time()) - days_back * 86400
            url = (
                f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
                f"?lat={PUNE_LAT}&lon={PUNE_LON}&dt={ts}&appid={OWM_API_KEY}&units=metric"
            )
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                for h in r.json().get('data', [{}]):
                    total += h.get('rain', {}).get('1h', 0)
        return round(total, 1)
    except Exception as exc:
        print(f"  Warning: rainfall fetch failed — {exc}")
        return None


# ── MAIN PIPELINE ────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str, include_rainfall: bool = False):
    print(f"Loading {input_path}")
    df = pd.read_csv(input_path, parse_dates=['date'])
    if df.empty:
        print("Input CSV is empty — nothing to process.")
        return None

    n_reservoirs = df['reservoir_name'].nunique()
    print(f"  {len(df)} rows | {n_reservoirs} reservoirs: {list(df['reservoir_name'].unique())}")

    rainfall_mm = None
    if include_rainfall:
        print("Fetching Pune rainfall from OpenWeatherMap...")
        rainfall_mm = fetch_rainfall_pune()
        print(f"  Last 7 days: {rainfall_mm} mm" if rainfall_mm is not None else "  Skipped (no API key or error)")

    all_results = []

    for name in df['reservoir_name'].unique():
        print(f"\nAnalysing {name} ...")
        sub = df[df['reservoir_name'] == name].copy()
        max_area = RESERVOIR_MAX_AREAS.get(name, sub['surface_area_sqkm'].max())

        sub = calculate_depletion_rates(sub)
        sub['risk_level']             = sub.apply(lambda r: classify_risk(r, max_area), axis=1)
        sub['days_remaining_estimate'] = sub.apply(lambda r: days_remaining(r, max_area), axis=1)

        if rainfall_mm is not None:
            sub['rainfall_mm_last7d'] = rainfall_mm

        latest = sub.iloc[-1]
        area_pct = latest['surface_area_sqkm'] / max_area * 100
        print(f"  Latest {latest['date'].date()} | "
              f"Area {latest['surface_area_sqkm']:.2f} sq km ({area_pct:.0f}%) | "
              f"Depletion {latest['depletion_rate_pct_per_week']:+.2f}%/wk | "
              f"Risk {latest['risk_level']} | "
              f"Buffer {latest['days_remaining_estimate']} days")

        all_results.append(sub)

    out_cols = ['reservoir_name', 'date', 'surface_area_sqkm',
                'depletion_rate_pct_per_week', 'risk_level', 'days_remaining_estimate']
    if rainfall_mm is not None:
        out_cols.append('rainfall_mm_last7d')

    final = (pd.concat(all_results)[out_cols]
               .sort_values(['reservoir_name', 'date'])
               .reset_index(drop=True))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    print(f"\nSaved {len(final)} records -> {output_path}")
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReservoirWatch Risk Analysis')
    parser.add_argument('--input',    default='data/reservoir_data.csv')
    parser.add_argument('--output',   default='data/risk_analysis.csv')
    parser.add_argument('--rainfall', action='store_true',
                        help='Fetch Pune rainfall via OpenWeatherMap (needs OWM_API_KEY in .env)')
    args = parser.parse_args()
    run(args.input, args.output, args.rainfall)
