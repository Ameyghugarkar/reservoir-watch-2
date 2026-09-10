<div align="center">
  <h1>💧 ReservoirWatch</h1>
  <p><b>AI-Powered Reservoir Water-Level & Drought Early-Warning System</b></p>
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
  [![Google Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-API-green.svg)](https://earthengine.google.com/)
  [![UN SDG 13](https://img.shields.io/badge/UN_SDG-13_Climate_Action-1f2937.svg)](#)
</div>

---

## 🌍 Overview
Built for the **PCCoE IGC Indradhanu Hackathon**, ReservoirWatch is a proactive climate adaptation tool. 

Currently, tracking drinking water reservoir levels relies on expensive ground sensors or delayed municipal reports, leading to vague, city-wide drought advisories. **ReservoirWatch replaces ground sensors with space-based optical imagery.** By dynamically extracting free Sentinel-2 satellite data via Google Earth Engine, we compute the water surface area, calculate weekly depletion rates, and classify drought risk levels (SAFE / WATCH / CRITICAL) in real-time.

## ✨ Key Features

- **🛰️ Space-Based Sensing (NDWI):** Uses the Normalized Difference Water Index to accurately map water bodies from Sentinel-2 Level-2A imagery at 10m resolution.
- **⚡ Automated Pipeline:** Click a button to securely connect to Google Earth Engine, fetch only *new* satellite passes, and update the dashboard automatically. No manual CSV handling required.
- **🧠 Intelligent Cloud-Cover Handling:** Standard satellites cannot see through monsoon clouds. Instead of reporting false dry-land readings, our AI filters out images with `>20%` cloud cover and triggers an intelligent **Stale Data Alert** to warn decision-makers not to rely on old visual data during heavy monsoons.
- **📉 Predictive Risk Engine:** Calculates a 28-day rolling regression to determine the exact `% capacity depletion per week`, forecasting the exact number of buffer days remaining before reaching critical crisis levels.

## 📊 Risk Classification Logic
| Risk Level | Area Condition | Depletion Rate | Action Required |
| :--- | :--- | :--- | :--- |
| 🟢 **SAFE** | > 55% of Max | < 2% / week | Normal operations |
| 🟡 **WATCH** | 35–55% of Max | 2–4% / week | Prepare targeted rationing |
| 🔴 **CRITICAL** | < 35% of Max | > 4% / week | Immediate crisis intervention |

## 🛠️ Tech Stack
* **Satellite Data:** ESA Copernicus Sentinel-2 (via Google Earth Engine API)
* **Backend Pipeline:** Python, Pandas, Numpy, Earth Engine Python API
* **Frontend Dashboard:** Streamlit, Plotly Graph Objects

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/reservoir-watch.git
   cd reservoir-watch
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Authenticate Google Earth Engine (One-time setup):**
   ```bash
   python -c "import ee; ee.Authenticate()"
   ```
4. **Launch the dashboard:**
   ```bash
   python -m streamlit run app.py
   ```

## 📍 Tracked Reservoirs (Mutha River Basin, Pune)
| Reservoir | River | Max Area | 
| :--- | :--- | :--- | 
| **Khadakwasla** | Mutha | ~7.5 sq km |
| **Panshet** *(Tanaji Sagar)* | Ambi | ~11.0 sq km |
| **Varasgaon** *(Veer Baaji Pasalkar)*| Moshi | ~9.0 sq km |

---
*Developed during the 4-day PCCoE IGC "Indradhanu" Hackathon — Aligning AI with Climate Action.*
