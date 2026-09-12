import numpy as np
import pandas as pd
from pathlib import Path

def preprocess_data(input_path=None, output_dir=None):
  """
  Cleans raw CPCB air quality data, converts gas concentrations to mass units (µg/m³),
  engineers cyclical diurnal time features, handles missing sensor readings, and 
  exports the 3 specific Tiered datasets for the DoppelGANger Ablation study.
  """
  # 1. Resolve default input and output paths relative to project root
  base_dir = Path(__file__).resolve().parent.parent
  
  if input_path is None:
    input_path = base_dir / "data" / "raw" / "Chandigarh_Sector22_AirQuality_2025_2026.csv"
  else:
    input_path = Path(input_path)
      
  if output_dir is None:
    output_dir = base_dir / "data" / "processed"
  else:
    output_dir = Path(output_dir)

  if not input_path.exists():
    raise FileNotFoundError(f"Raw data file not found at: {input_path}")

  print(f"Loading raw data from: {input_path}")
  df = pd.read_csv(input_path, encoding='utf-8')

  # 2. Parse Timestamps, Sort, and Set as Index
  df['date.utc'] = pd.to_datetime(df['date.utc'])
  df = df.sort_values(by='date.utc')
  df = df.set_index('date.utc') # Set as index so it is excluded from GAN feature tensors

  # 3. Standardize Column Names
  rename_mapping = {}
  for col in df.columns:
    if 'co_' in col: rename_mapping[col] = 'co_ppb'
    elif 'no2_' in col: rename_mapping[col] = 'no2_ppb'
    elif 'so2_' in col: rename_mapping[col] = 'so2_ppb'
    elif 'no_' in col and 'nox' not in col: rename_mapping[col] = 'no_ppb'
    elif 'nox_' in col: rename_mapping[col] = 'nox_ppb'
    elif 'o3_' in col: rename_mapping[col] = 'o3'
    elif 'pm10_' in col: rename_mapping[col] = 'pm10'
    elif 'pm25_' in col: rename_mapping[col] = 'pm25'
    elif 'relativehumidity_' in col or col == 'relativehumidity': rename_mapping[col] = 'rh'
    elif 'temperature_' in col or col == 'temperature': rename_mapping[col] = 'temp'
    elif 'wind_direction_' in col: rename_mapping[col] = 'wd'
    elif 'wind_speed_' in col: rename_mapping[col] = 'ws'

  df = df.rename(columns=rename_mapping)

  # 4. Standard Atmospheric Gas Conversions (ppb to µg/m³ at 25°C, 1 atm)
  print("Performing molecular mass conversions for CO, NO, NO2, and SO2...")
  if 'co_ppb' in df.columns:
    df['co'] = df['co_ppb'] * (28.01 / 24.45)
    df = df.drop(columns=['co_ppb'])
  if 'no2_ppb' in df.columns:
    df['no2'] = df['no2_ppb'] * (46.01 / 24.45)
    df = df.drop(columns=['no2_ppb'])
  if 'so2_ppb' in df.columns:
    df['so2'] = df['so2_ppb'] * (64.06 / 24.45)
    df = df.drop(columns=['so2_ppb'])
  if 'no_ppb' in df.columns:
    df['no'] = df['no_ppb'] * (30.01 / 24.45)
    df = df.drop(columns=['no_ppb'])
  if 'nox_ppb' in df.columns:
    df['nox'] = df['nox_ppb'] * (46.01 / 24.45) 
    df = df.drop(columns=['nox_ppb'])

  # 5. Cyclical Diurnal Time Encoding (Trigonometric Sine/Cosine)
  print("Engineering cyclical diurnal time features...")
  hour = df.index.hour
  df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
  df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
  
  dayofweek = df.index.dayofweek
  df['dow_sin'] = np.sin(2 * np.pi * dayofweek / 7.0)
  df['dow_cos'] = np.cos(2 * np.pi * dayofweek / 7.0)

  # 6. Missing Value Handling
  print("Interpolating short missing sensor intervals...")
  numeric_cols = df.select_dtypes(include=[np.number]).columns
  df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=3, limit_direction='both')

  # Drop any remaining unfillable rows
  initial_len = len(df)
  df_clean = df.dropna()
  dropped_count = initial_len - len(df_clean)
  print(f"Dropped {dropped_count} rows containing unrecoverable null intervals.")
  print(f"Final valid sequence length: {len(df_clean)} hours.")

  # 7. Route Features for the 3-Tier Ablation Study
  processed_dir = base_dir / "data" / "processed"
  processed_dir.mkdir(parents=True, exist_ok=True)
  
  # Always save the full master processed file where prep_experiments expects it
  master_output_path = processed_dir / "chandigarh_sector22_processed.csv"
  df_clean.to_csv(master_output_path)
  print(f"Master processed dataset saved to: {master_output_path}")

  # Target variables + Meteorology (No explicit traffic or combustion signatures)
  tier1_baseline = ['pm25', 'pm10', 'o3', 'co', 'so2', 'temp', 'rh', 'ws', 'wd']
  
  # Tier 2: Tier 1 + Nitrogen oxides (The Chemical Proxies for tailpipe emissions)
  tier2_chem_proxy = tier1_baseline + ['no', 'no2', 'nox']
  
  # Tier 3: Tier 2 + Trigonometric Time Waves (The Spatio-Temporal Proxy for city rhythm)
  tier3_full_proxy = tier2_chem_proxy + ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

  # 8. Export Generative Seeds
  output_dir.mkdir(parents=True, exist_ok=True)
  print("\n--- Exporting Ablation Seeds ---")
  df_clean[tier1_baseline].to_csv(output_dir / "tier1_baseline_seed.csv")
  print(f"Tier 1 (Baseline): Saved with {len(tier1_baseline)} features.")
  
  df_clean[tier2_chem_proxy].to_csv(output_dir / "tier2_chem_proxy_seed.csv")
  print(f"Tier 2 (Chemical): Saved with {len(tier2_chem_proxy)} features.")
  
  df_clean[tier3_full_proxy].to_csv(output_dir / "tier3_full_proxy_seed.csv")
  print(f"Tier 3 (Full Proxy): Saved with {len(tier3_full_proxy)} features.")

  return df_clean

if __name__ == "__main__":
  preprocess_data()