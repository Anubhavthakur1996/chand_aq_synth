import numpy as np
import pandas as pd
from pathlib import Path

def preprocess_data(input_path=None, output_path=None):
  """
  Cleans raw CPCB air quality data, converts gas concentrations to mass units (µg/m³),
  engineers cyclical diurnal time features, and handles missing sensor readings.
  """
  # 1. Resolve default input and output paths relative to project root
  base_dir = Path(__file__).resolve().parent.parent
  
  if input_path is None:
    input_path = base_dir / "data" / "raw" / "Chandigarh_Sector22_AirQuality_2025_2026.csv"
    # input_path = base_dir / "data" / "raw" / "Chandigarh_Sector53_AirQuality_2024_2025.csv"
  else:
    input_path = Path(input_path)
      
  if output_path is None:
    output_path = base_dir / "data" / "processed" / "chandigarh_sector22_processed.csv"
    # output_path = base_dir / "data" / "processed" / "chandigarh_sector53_processed.csv"
  else:
    output_path = Path(output_path)

  if not input_path.exists():
    raise FileNotFoundError(f"Raw data file not found at: {input_path}")

  print(f"Loading raw data from: {input_path}")
  df = pd.read_csv(input_path, encoding='utf-8')

  # 2. Parse Timestamps and Sort Chronologically
  df['date.utc'] = pd.to_datetime(df['date.utc'])
  df = df.sort_values(by='date.utc').reset_index(drop=True)

  # 3. Standardize Column Names (Mapping raw fetched columns to clean internal names)
  # This prevents special character encoding bugs in XGBoost/CTGAN
  rename_mapping = {}
  for col in df.columns:
    if 'co_' in col:
      rename_mapping[col] = 'co_ppb'
    elif 'no2_' in col:
      rename_mapping[col] = 'no2_ppb'
    elif 'so2_' in col:
      rename_mapping[col] = 'so2_ppb'
    elif 'no_' in col and 'nox' not in col:
      rename_mapping[col] = 'no_ppb'
    elif 'nox_' in col:
      rename_mapping[col] = 'nox_ppb'
    elif 'o3_' in col:
      rename_mapping[col] = 'o3'
    elif 'pm10_' in col:
      rename_mapping[col] = 'pm10'
    elif 'pm25_' in col:
      rename_mapping[col] = 'pm25'
    elif 'relativehumidity_' in col or col == 'relativehumidity':
      rename_mapping[col] = 'relative_humidity'
    elif 'temperature_' in col or col == 'temperature':
      rename_mapping[col] = 'temperature'
    elif 'wind_direction_' in col:
      rename_mapping[col] = 'wind_direction'
    elif 'wind_speed_' in col:
      rename_mapping[col] = 'wind_speed'

  df = df.rename(columns=rename_mapping)

  # 4. Standard Atmospheric Gas Conversions (ppb to µg/m³ at 25°C, 1 atm)
  # Formula: µg/m³ = ppb * (Molecular Weight / 24.45)
  print("Performing molecular mass conversions for CO, NO2, and SO2...")
  if 'co_ppb' in df.columns:
    df['co'] = df['co_ppb'] * (28.01 / 24.45) # CO MW = 28.01 g/mol
    df = df.drop(columns=['co_ppb'])
    
  if 'no2_ppb' in df.columns:
    df['no2'] = df['no2_ppb'] * (46.01 / 24.45) # NO2 MW = 46.01 g/mol
    df = df.drop(columns=['no2_ppb'])
      
  if 'so2_ppb' in df.columns:
    df['so2'] = df['so2_ppb'] * (64.06 / 24.45) # SO2 MW = 64.06 g/mol
    df = df.drop(columns=['so2_ppb'])

  if 'no_ppb' in df.columns:
    df['no'] = df['no_ppb'] * (30.01 / 24.45) # NO MW = 30.01 g/mol
    df = df.drop(columns=['no_ppb'])

  if 'nox_ppb' in df.columns:
    df['nox'] = df['nox_ppb'] * (46.01 / 24.45) # Use NO2 equivalent MW for NOx
    df = df.drop(columns=['nox_ppb'])

  # (Note: no_ppb and nox_ppb remain intact as non-mass traffic proxies)

  # 5. Cyclical Diurnal Time Encoding (Trigonometric Sine/Cosine representation)
  print("Engineering cyclical diurnal time features...")
  df['hour'] = df['date.utc'].dt.hour
  df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
  df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
  
  # Optional Day-of-Week encoding (captures weekend vs. weekday traffic shifts)
  df['dayofweek'] = df['date.utc'].dt.dayofweek
  df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
  df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

  # 6. Missing Value Handling
  # Short sensor dropouts (<= 3 consecutive hours) are linearly interpolated
  print("Interpolating short missing sensor intervals...")
  numeric_cols = df.select_dtypes(include=[np.number]).columns
  df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=3, limit_direction='both')

  # --- DIAGNOSTIC AUDIT ---
  print("\n--- MISSING DATA AUDIT ---")
  print(df.isnull().sum())
  print("--------------------------\n")

  # Drop any remaining unfillable rows (e.g., prolonged multi-day outages)
  initial_len = len(df)
  df_clean = df.dropna().reset_index(drop=True)
  dropped_count = initial_len - len(df_clean)
  print(f"Dropped {dropped_count} rows containing unrecoverable null intervals.")

  # 7. Export Processed Dataset
  output_path.parent.mkdir(parents=True, exist_ok=True)
  df_clean.to_csv(output_path, index=False)
  print(f"Preprocessing complete! Saved {len(df_clean)} cleaned rows to: {output_path}")

  return df_clean

if __name__ == "__main__":
  preprocess_data()