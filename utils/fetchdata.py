import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

def fetchData(output_path=None):
  if output_path is None:
    # Go up from utils to root, then down to data/raw
    base_dir = os.path.dirname(os.path.dirname(__file__)) 
    output_path = os.path.join(base_dir, "data", "raw", "Chandigarh_Sector22_AirQuality_2025_2026.csv")
    # output_path = os.path.join(base_dir, "data", "raw", "Chandigarh_Sector53_AirQuality_2024_2025.csv")

  # Ensure the target directory exists before writing
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  
  # --- CONFIGURATION ---  
  load_dotenv()

  API_KEY = os.getenv("OPENAQ_API_KEY")
  LOCATION_ID = os.getenv("LOCATION_ID")

  # v3 strictly requires this exact date-time format
  # DATE_FROM = "2024-01-01T00:00:00Z"
  # DATE_TO = "2025-12-31T23:59:59Z"

  # DATE_FROM = "2024-01-01T00:00:00Z"
  # DATE_TO = "2024-12-31T23:59:59Z"

  DATE_FROM = "2025-01-01T00:00:00Z"
  DATE_TO = "2026-07-31T23:59:59Z"

  headers = {
    "X-API-Key": API_KEY,
    "Accept": "application/json"
  }

  print(f"Step 1: Finding sensors for Location {LOCATION_ID}...")
  
  # --- STEP 1: GET ALL SENSORS FOR THE LOCATION ---
  sensors_url = f"https://api.openaq.org/v3/locations/{LOCATION_ID}/sensors"
  sensors_response = requests.get(sensors_url, headers=headers)
  
  if sensors_response.status_code != 200:
    print(f"Error {sensors_response.status_code}: Could not load sensors. Check Location ID.")
    return
      
  sensors_data = sensors_response.json().get('results', [])
  if not sensors_data:
    print("No sensors found for this location.")
    return
    
  print(f"Found {len(sensors_data)} sensors. Starting hourly data pull...")
  all_records = []
  
  # --- STEP 2: LOOP THROUGH EACH SENSOR & FETCH DATA ---
  for sensor in sensors_data:
    sensor_id = sensor['id']
    param_name = sensor['parameter']['name']
    param_units = sensor['parameter']['units']
    
    # Combine them so our columns end up like "pm25_µg/m³"
    full_param_name = f"{param_name}_{param_units}"
    
    # # Strict Exception Rule: We only want mass (µg/m³) for core pollutants, 
    # # but we specifically WANT ppb for our traffic proxies (no, nox)
    # if param_units == "ppm" or param_units == "ppb":
    #   if param_name not in ["no", "nox"]:
    #     print(f"Skipping {full_param_name} (We strictly want mass for {param_name})")
    #     continue # Skip to the next sensor
            
    print(f"--- Fetching {full_param_name} (Sensor ID: {sensor_id}) ---")
    
    page = 1
    max_retries = 3

    while True:
      url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours?datetime_from={DATE_FROM}&datetime_to={DATE_TO}&limit=1000&page={page}"
      
      retry_count = 0
      success = False
      
      # Retry loop for unstable OpenAQ servers
      while retry_count < max_retries:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
          success = True
          break  # Exit the retry loop, we got the data!
          
        print(f"  Server error {response.status_code}. Retrying ({retry_count + 1}/{max_retries}) in 2 seconds...")
        retry_count += 1
        time.sleep(2)
      
      # If it failed 3 times in a row, give up on this specific sensor
      if not success:
        print(f"Failed to fetch page {page} after {max_retries} attempts. Stopping extraction for {full_param_name}.")
        break

      data = response.json()
      results = data.get('results', [])

      if not results:
        break 

      for row in results:
        dt = None
        if 'period' in row and 'datetimeFrom' in row['period']:
          dt = row['period']['datetimeFrom']['utc']
        elif 'datetimeFrom' in row:
          dt = row['datetimeFrom']['utc']
        elif 'datetime' in row:
          dt = row['datetime']['utc']

        val = row.get('value')
        if val is None and 'summary' in row:
          val = row['summary'].get('avg')

        if dt and val is not None:
          all_records.append({
            'date.utc': dt,
            'parameter': full_param_name, 
            'value': val
          })

      print(f"  Fetched page {page} for {full_param_name}...")
      page += 1
      time.sleep(1)

  # --- DATA CLEANING & EXPORT ---
  if all_records:
    print("Data extraction complete! Building CSV...")
    # Since we pre-formatted the records, we can throw them directly into pandas
    df = pd.DataFrame(all_records)
    
    # Pivot the table so each parameter (PM2.5, NO2, etc.) gets its own column
    df_pivot = df.pivot_table(index='date.utc', columns='parameter', values='value', aggfunc='mean')
    
    # Save to CSV
    df_pivot.to_csv(output_path, encoding='utf-8')
    print(f"Success! Saved {len(df_pivot)} hourly rows to: {output_path}")
  else:
    print("No valid data was retrieved for the specified dates.")