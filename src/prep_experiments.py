import pandas as pd
import os

def prep_all_experiments(input_csv, output_dir):
  print("Loading preprocessed dataset...")
  df = pd.read_csv(input_csv)
  os.makedirs(output_dir, exist_ok=True)

  # Ensure chronological order before slicing
  df = df.sort_values(by='date.utc').reset_index(drop=True)

  # 1. Define the 336-hour Sequential Split (2 Weeks)
  print("\nExecuting the 336-hour sequential split...")
  df_train = df.iloc[:336]
  df_test = df.iloc[336:]

  # 2. Define the exact feature spaces for the 3 Tiers
  # Drop all timestamps and linear time features from ALL tiers
  base_drop = ['date.utc', 'hour', 'dayofweek']
  
  # Target and weather columns (Adjust if our names differ)
  core_features = ['pm25', 'pm10', 'o3', 'co', 'so2', 'temp', 'rh', 'ws', 'wd']

  # TIER 1: BASELINE (Strictly Core Features, no Traffic, no Time Waves)
  tier1_cols = core_features
  
  # TIER 2: CHEMICAL PROXY (Core + NO, NO2, NOX)
  tier2_cols = core_features + ['no', 'no2', 'nox']
  
  # TIER 3: SPATIO-TEMPORAL PROXY (Tier 2 + Cyclical Time Waves)
  tier3_cols = tier2_cols + ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

  # 3. Filter and Save Train/Test sets for each Tier
  tiers = {
    "tier1_baseline": tier1_cols,
    "tier2_chemproxy": tier2_cols,
    "tier3_fullproxy": tier3_cols
  }

  print("\n--- Exporting Ablation Datasets ---")
  for tier_name, columns in tiers.items():
    # Ensure only columns that actually exist in the dataframe are kept
    valid_cols = [c for c in columns if c in df.columns]
    
    train_out = df_train[valid_cols]
    test_out = df_test[valid_cols]
    
    train_out.to_csv(f"{output_dir}/{tier_name}_train.csv", index=False)
    test_out.to_csv(f"{output_dir}/{tier_name}_test.csv", index=False)
    
    print(f"{tier_name.upper()}:")
    print(f"   - Features ({len(valid_cols)}): {valid_cols}")
    print(f"   - Train Shape: {train_out.shape} | Test Shape: {test_out.shape}")

if __name__ == "__main__":
  # Update paths to match our directory structure
  INPUT_CSV = 'data/processed/chandigarh_sector22_processed.csv'
  OUTPUT_DIR = 'data/experiments'
  
  prep_all_experiments(INPUT_CSV, OUTPUT_DIR)