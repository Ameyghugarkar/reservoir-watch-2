# ============================================================
# ReservoirWatch Auto-Update Script
# Runs every 5 days via Windows Task Scheduler
# Fetches latest satellite data → computes risk → pushes to GitHub
# ============================================================

$ProjectDir = "C:\Users\Amey\Downloads\water reservoir"
$LogFile    = "$ProjectDir\scripts\auto_update.log"
$Python     = "python"

function Log($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp — $msg" | Tee-Object -FilePath $LogFile -Append
}

Log "====== Auto-Update Started ======"

# Step 1: Fetch latest satellite data from GEE
Log "Step 1: Fetching latest satellite data from Google Earth Engine..."
$result = & $Python "$ProjectDir\scripts\update_gee.py" 2>&1
Log $result

# Step 2: Recompute risk analysis
Log "Step 2: Running risk analysis..."
$result = & $Python "$ProjectDir\risk_logic.py" 2>&1
Log $result

# Step 3: Push updated data to GitHub (triggers Streamlit auto-redeploy)
Log "Step 3: Pushing updated data to GitHub..."
Set-Location $ProjectDir
git add data/reservoir_data.csv data/risk_analysis.csv
$commitMsg = "Auto-update: satellite data $(Get-Date -Format 'yyyy-MM-dd')"
git commit -m $commitMsg
git push origin main
Log "GitHub push complete. Streamlit will auto-redeploy within 2 minutes."

Log "====== Auto-Update Complete ======"
