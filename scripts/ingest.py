import requests
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path

# The Master File URL (Reports of non-emergency problems...)
DATA_URL = "https://seshat.datasd.org/get_it_done_reports/get_it_done_requests_open_datasd.csv"

def ingest_data():
    # 1. Setup paths
    today_str = datetime.now().strftime("%Y-%m-%d")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = raw_dir / f"get_it_done_{today_str}.csv"
    latest_path = raw_dir / "get_it_done_latest.csv"

    print(f"⬇️  Downloading Master Dataset from {DATA_URL}...")
    
    # 2. Download
    try:
        with requests.get(DATA_URL, stream=True) as r:
            r.raise_for_status()
            with open(archive_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print(f"✅ Download complete: {archive_path}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return

    # 3. "Light ETL" / Validation
    # Before we overwrite 'latest', let's check if the data isn't garbage.
    print("🔍 Running Health Checks...")
    try:
        # Read just the header and first few rows to validate schema
        df = pd.read_csv(archive_path, nrows=100)
        
        required_columns = ['service_request_id',
                            'service_request_parent_id',
                            'sap_notification_number',
                            'date_requested',
                            'case_age_days',
                            'case_record_type',
                            'service_name',
                            'service_name_detail',
                            'date_closed',
                            'status',
                            'lat',
                            'lng',
                            'street_address',
                            'zipcode',
                            'council_district',
                            'comm_plan_code',
                            'comm_plan_name',
                            'park_name',
                            'case_origin',
                            'referred',
                            'iamfloc',
                            'floc',
                            'public_description'
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"❌ CRITICAL: Source schema changed! Missing columns: {missing_cols}")
            
        # Check if we have data
        if len(df) == 0:
            raise ValueError("❌ CRITICAL: Downloaded file is empty!")

        print("✅ Schema Validation Passed")
        
        # 4. Success - Update the 'latest' pointer
        shutil.copy(archive_path, latest_path)
        print(f"✅ Pipeline ready: {latest_path} updated.")
        
    except Exception as e:
        print(f"⚠️ Validation Failed: {e}")
        print("Keeping previous 'latest.csv' to prevent breaking the dashboard.")

if __name__ == "__main__":
    ingest_data()


