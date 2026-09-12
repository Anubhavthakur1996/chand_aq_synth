import pandas as pd

def prep_baseline_dataset(input_csv=None, output_dir=None):
  print("Loading preprocessed dataset...")
  df = pd.read_csv(input_csv)

  # 1. Feature Filtration (Enforcing the Base Paper constraints)
  # We must explicitly drop our novel features so the baseline GAN cannot see them.
  # refinement_features = [
  #   'nox_ppb', 'no_ppb', 'wind_direction', 
  #   'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
  #   'date.utc', 'hour', 'dayofweek'
  # ]

  refinement_features = [
    'nox', 'no', 'wind_direction', 
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'date.utc', 'hour', 'dayofweek'
  ]
  
  cols_to_drop = [c for c in refinement_features if c in df.columns]
  if cols_to_drop:
    print(f"Stripping refinement features for baseline: {cols_to_drop}")
    df_baseline = df.drop(columns=cols_to_drop)
  else:
    df_baseline = df.copy()

  # Ensure we only have Base Paper features: 
  # PM2.5, PM10, CO, NO2, SO2, O3, Temp, Humidity, Wind Speed
  print(f"Baseline feature space locked. Columns: {df_baseline.columns.tolist()}")

  # 2. The Standard Sequential Split (Base Paper Methodology)
  print("\nExecuting the 336-hour sequential split...")
  
  # Take the very first contiguous 336 rows (exactly 2 weeks)
  df_seed = df_baseline.iloc[:336]
  
  # Lock the remaining rows as the untouchable test set
  df_test = df_baseline.iloc[336:]

  # 3. Save the static CSVs for DoppelGANger
  df_seed.to_csv(f"{output_dir}/baseline_seed.csv", index=False)
  df_test.to_csv(f"{output_dir}/untouched_test_set.csv", index=False)
  
  print("\nTier 2 Data Prep Complete!")
  print(f"Training Seed: {df_seed.shape[0]} rows.")
  print(f"Validation Set: {df_test.shape[0]} rows.")

if __name__ == "__main__":
  prep_baseline_dataset()