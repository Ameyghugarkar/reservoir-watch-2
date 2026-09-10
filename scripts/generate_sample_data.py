"""
generate_sample_data.py
Generates realistic synthetic reservoir_data.csv for offline development.

Simulates Sentinel-2 observation cadence (~every 10-14 days after cloud filtering),
seasonal monsoon swells (Jul-Sep peak), dry-season depletion, and a slight
year-over-year declining trend to make the early-warning system meaningful.

Usage:
    python scripts/generate_sample_data.py

Once you have real GEE data, replace data/reservoir_data.csv with the real file.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import os

np.random.seed(42)

RESERVOIRS = {
    'Khadakwasla': {'max_area': 7.5,  'min_area': 2.8},
    'Panshet':     {'max_area': 11.0, 'min_area': 4.0},
    'Varasgaon':   {'max_area': 9.0,  'min_area': 3.2},
}

# Monthly seasonal fill factor (index 0=Jan ... 11=Dec)
# 1.0 = full (Sep peak), 0.0 = minimum (May pre-monsoon)
SEASONAL_FACTORS = [
    0.60,  # Jan
    0.50,  # Feb
    0.40,  # Mar
    0.30,  # Apr
    0.18,  # May — driest
    0.25,  # Jun — pre-monsoon
    0.58,  # Jul — monsoon onset
    0.82,  # Aug — monsoon peak
    1.00,  # Sep — maximum
    0.90,  # Oct — post-monsoon high
    0.76,  # Nov
    0.66,  # Dec
]

START_DATE = date(2023, 1, 1)
END_DATE   = date(2026, 9, 1)
BASE_YEAR  = 2023


def get_observation_dates(start, end):
    """Simulate Sentinel-2 revisit with cloud-cover gaps."""
    dates = []
    current = start
    while current < end:
        gap = int(np.clip(np.random.normal(12, 3), 5, 25))
        current = current + timedelta(days=gap)
        if current >= end:
            break
        # ~55% of Jul/Aug skipped due to monsoon cloud cover
        if current.month in [7, 8] and np.random.random() < 0.55:
            continue
        dates.append(current)
    return dates


def generate_area(obs_date, config, year_offset):
    """Realistic surface area with seasonal pattern, YoY decline, and noise."""
    m = obs_date.month - 1
    next_m = (m + 1) % 12
    day_frac = obs_date.day / 30.0
    factor = SEASONAL_FACTORS[m] + (SEASONAL_FACTORS[next_m] - SEASONAL_FACTORS[m]) * day_frac * 0.4
    year_factor = max(0.85, 1.0 - year_offset * 0.03)  # 3% YoY decline
    max_a, min_a = config['max_area'], config['min_area']
    amplitude = max_a - min_a
    noise = np.random.normal(0, 0.04) * amplitude
    area = min_a + amplitude * factor * year_factor + noise
    return round(float(np.clip(area, min_a * 0.88, max_a)), 4)


def main():
    records = []
    for name, config in RESERVOIRS.items():
        dates = get_observation_dates(START_DATE, END_DATE)
        for d in dates:
            records.append({
                'reservoir_name': name,
                'date': d.strftime('%Y-%m-%d'),
                'surface_area_sqkm': generate_area(d, config, d.year - BASE_YEAR),
                'cloud_cover_pct': round(float(np.random.uniform(0, 19)), 1),
            })

    df = pd.DataFrame(records).sort_values(['reservoir_name', 'date']).reset_index(drop=True)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'reservoir_data.csv')
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} records -> {os.path.abspath(out_path)}")
    print(df.groupby('reservoir_name').agg(
        records=('date', 'count'),
        from_date=('date', 'min'),
        to_date=('date', 'max'),
        area_min=('surface_area_sqkm', 'min'),
        area_max=('surface_area_sqkm', 'max'),
    ).to_string())


if __name__ == '__main__':
    main()
