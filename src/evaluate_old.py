import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

def calculate_js_divergence(real_data, synth_data, bins=50):
  """Calculates Jensen-Shannon Divergence using probability distributions."""
  min_val = min(real_data.min(), synth_data.min())
  max_val = max(real_data.max(), synth_data.max())
  bin_edges = np.linspace(min_val, max_val, bins+1)
  
  p_real, _ = np.histogram(real_data, bins=bin_edges, density=True)
  p_synth, _ = np.histogram(synth_data, bins=bin_edges, density=True)
  
  p_real = np.clip(p_real, 1e-10, None)
  p_synth = np.clip(p_synth, 1e-10, None)
  
  return jensenshannon(p_real, p_synth)

def evaluate_synthetic_data(real_csv_path, synth_csv_path, output_dir):
  print("Loading datasets for Layer 1 & 2 validation...")
  real_df = pd.read_csv(real_csv_path).dropna()
  synth_df = pd.read_csv(synth_csv_path).dropna()
  
  if 'date.utc' in real_df.columns:
    real_df = real_df.drop(columns=['date.utc'])
      
  common_cols = [c for c in real_df.columns if c in synth_df.columns]
  real_df = real_df[common_cols]
  synth_df = synth_df[common_cols]
  
  os.makedirs(output_dir, exist_ok=True)
  
  print(f"{'Feature':<15} | {'KS Stat':<10} | {'JS Div':<10} | {'Real Mean':<10} | {'Synth Mean':<10}")
  print("-" * 65)
  
  for col in common_cols:
    ks_stat, _ = ks_2samp(real_df[col], synth_df[col])
    js_div = calculate_js_divergence(real_df[col], synth_df[col])
    r_mean, s_mean = real_df[col].mean(), synth_df[col].mean()
    print(f"{col:<15} | {ks_stat:<10.4f} | {js_div:<10.4f} | {r_mean:<10.2f} | {s_mean:<10.2f}")

  fig, axes = plt.subplots(1, 2, figsize=(18, 8))
  sns.heatmap(real_df.corr(), ax=axes[0], cmap='coolwarm', vmin=-1, vmax=1)
  axes[0].set_title("Real Data Correlation (Ground Truth)")
  sns.heatmap(synth_df.corr(), ax=axes[1], cmap='coolwarm', vmin=-1, vmax=1)
  axes[1].set_title("Synthetic Data Correlation")
  
  plt.tight_layout()
  corr_path = os.path.join(output_dir, 'correlation_paper_comparison.png')
  plt.savefig(corr_path)
  print(f"\nSaved correlation heatmap to: {corr_path}")
  plt.close()

def evaluate_predictive_utility(synth_csv_path, test_csv_path, target_col='pm25'):
  print("\n--- Layer 3: Predictive Utility Validation ---")
  synth_df = pd.read_csv(synth_csv_path).dropna()
  test_df = pd.read_csv(test_csv_path).dropna()
  
  feature_cols = [c for c in synth_df.columns if c != target_col]
  
  X_synth = synth_df[feature_cols]
  y_synth = synth_df[target_col]
  
  X_test = test_df[feature_cols]
  y_test = test_df[target_col]
  
  # 1. Random Forest Regressor
  rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
  rf_model.fit(X_synth, y_synth)
  rf_preds = rf_model.predict(X_test)
  rf_mae = mean_absolute_error(y_test, rf_preds)
  
  # 2. Gradient Boosting Machine (GBM)
  gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
  gbm_model.fit(X_synth, y_synth)
  gbm_preds = gbm_model.predict(X_test)
  gbm_mae = mean_absolute_error(y_test, gbm_preds)
  
  print(f"Target Variable: {target_col}")
  print(f"Random Forest MAE: {rf_mae:.4f}")
  print(f"GBM MAE:           {gbm_mae:.4f}")
  print("---------------------------------------------")
  
  return rf_mae, gbm_mae

if __name__ == "__main__":
  BASE_DIR = os.path.dirname(os.path.dirname(__file__))
  
  real_path = os.path.join(BASE_DIR, "data", "processed", "baseline_seed.csv")
  synth_path = os.path.join(BASE_DIR, "data", "synthetic", "synth_baseline_pt.csv")
  test_path = os.path.join(BASE_DIR, "data", "processed", "untouched_test_set.csv") 
  out_dir = os.path.join(BASE_DIR, "evaluation_outputs")
  
  # Run Layer 1 & 2
  evaluate_synthetic_data(real_path, synth_path, out_dir)
  
  # Run Layer 3
  if os.path.exists(test_path):
    evaluate_predictive_utility(synth_path, test_path, target_col='pm25')
  else:
    print(f"\nWarning: Could not find test set at {test_path} for Layer 3 validation.")