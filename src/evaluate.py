import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

def calculate_acf_error(real_data, synth_data, max_lag=24):
  """Calculates the Mean Absolute Error of the Autocorrelation Function (ACF) up to 24 hours."""
  real_acf = [pd.Series(real_data).autocorr(lag=i) for i in range(1, max_lag + 1)]
  synth_acf = [pd.Series(synth_data).autocorr(lag=i) for i in range(1, max_lag + 1)]
  return mean_absolute_error(np.nan_to_num(real_acf), np.nan_to_num(synth_acf))

def evaluate_synthetic_data(real_csv_path, synth_csv_path, output_dir):
  real_df = pd.read_csv(real_csv_path).dropna()
  synth_df = pd.read_csv(synth_csv_path).dropna()
  
  if 'date.utc' in real_df.columns:
    real_df = real_df.drop(columns=['date.utc'])
      
  common_cols = [c for c in real_df.columns if c in synth_df.columns]
  real_df = real_df[common_cols]
  synth_df = synth_df[common_cols]
  
  os.makedirs(output_dir, exist_ok=True)
  
  print(f"\n{'Feature':<15} | {'W-Dist (EMD)':<12} | {'JS Div':<10} | {'ΔACF (24h)':<10} | {'Real Mean':<10} | {'Synth Mean':<10}")
  print("-" * 80)
  
  for col in common_cols:
    w_dist = wasserstein_distance(real_df[col], synth_df[col])
    js_div = calculate_js_divergence(real_df[col], synth_df[col])
    acf_err = calculate_acf_error(real_df[col], synth_df[col], max_lag=24)
    
    r_mean, s_mean = real_df[col].mean(), synth_df[col].mean()
    print(f"{col:<15} | {w_dist:<12.4f} | {js_div:<10.4f} | {acf_err:<10.4f} | {r_mean:<10.2f} | {s_mean:<10.2f}")

  # Plot correlation heatmaps
  fig, axes = plt.subplots(1, 2, figsize=(18, 8))
  sns.heatmap(real_df.corr(), ax=axes[0], cmap='coolwarm', vmin=-1, vmax=1)
  axes[0].set_title("Real Data Correlation (Ground Truth)")
  sns.heatmap(synth_df.corr(), ax=axes[1], cmap='coolwarm', vmin=-1, vmax=1)
  axes[1].set_title("Synthetic Data Correlation")
  
  plt.tight_layout()
  corr_path = os.path.join(output_dir, 'correlation_paper_comparison.png')
  plt.savefig(corr_path)
  plt.close()

def create_supervised_lags(df, target_col, look_back=24):
  """
  Transforms a 2D time-series dataframe into a supervised learning dataset 
  where X contains `look_back` hours of all features, and y is the target at hour `t`.
  """
  X, y = [], []
  data = df.values
  target_idx = df.columns.get_loc(target_col)
  
  for i in range(len(data) - look_back):
    # Flatten the 24h window into a single 1D feature vector for RF/GBM
    X.append(data[i : i + look_back].flatten())
    y.append(data[i + look_back, target_idx])
      
  return np.array(X), np.array(y)

def calc_forecasting_metrics(y_true, y_pred):
  """Calculates the exact suite of metrics utilized in Morales-Garcia et al."""
  mae = mean_absolute_error(y_true, y_pred)
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  r2 = r2_score(y_true, y_pred)
  
  # Avoid division by zero for MAPE and CVRMSE
  denominator = np.where(y_true == 0, 1e-10, y_true)
  mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100
  cvrmse = (rmse / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else np.nan
  
  return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "CVRMSE": cvrmse}

def evaluate_predictive_utility(synth_csv_path, test_csv_path, target_col='pm25', real_train_csv_path=None):
  """
  Evaluates downstream predictive utility via TSTR (Train Synthetic, Test Real).
  If real_train_csv_path is provided, it also calculates the TRTR baseline to derive the Utility Ratio.
  """
  synth_df = pd.read_csv(synth_csv_path).dropna()
  test_df = pd.read_csv(test_csv_path).dropna()
  
  if 'date.utc' in synth_df.columns: synth_df = synth_df.drop(columns=['date.utc'])
  if 'date.utc' in test_df.columns: test_df = test_df.drop(columns=['date.utc'])
  
  # Create 24-hour look-back sequences for Time-Series forecasting
  X_synth, y_synth = create_supervised_lags(synth_df, target_col, look_back=24)
  X_test, y_test = create_supervised_lags(test_df, target_col, look_back=24)
  
  # 1. Train on Synthetic, Test on Real (TSTR)
  rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
  rf_model.fit(X_synth, y_synth)
  rf_preds = rf_model.predict(X_test)
  tstr_metrics_rf = calc_forecasting_metrics(y_test, rf_preds)
  
  # Using HistGradientBoosting for faster, highly optimized LightGBM-style execution
  gbm_model = HistGradientBoostingRegressor(random_state=42)
  gbm_model.fit(X_synth, y_synth)
  gbm_preds = gbm_model.predict(X_test)
  tstr_metrics_gbm = calc_forecasting_metrics(y_test, gbm_preds)
  
  print(f"\n--- TSTR Downstream Utility (Target: {target_col}) ---")
  print(f"{'Model':<15} | {'R2':<8} | {'RMSE':<8} | {'MAE':<8} | {'MAPE':<8} | {'CVRMSE':<8}")
  print("-" * 70)
  print(f"{'Random Forest':<15} | {tstr_metrics_rf['R2']:<8.4f} | {tstr_metrics_rf['RMSE']:<8.4f} | {tstr_metrics_rf['MAE']:<8.4f} | {tstr_metrics_rf['MAPE']:<8.4f} | {tstr_metrics_rf['CVRMSE']:<8.4f}")
  print(f"{'LightGBM':<15} | {tstr_metrics_gbm['R2']:<8.4f} | {tstr_metrics_gbm['RMSE']:<8.4f} | {tstr_metrics_gbm['MAE']:<8.4f} | {tstr_metrics_gbm['MAPE']:<8.4f} | {tstr_metrics_gbm['CVRMSE']:<8.4f}")

  # Optional: TRTR Ratio calculation if real training data is passed
  if real_train_csv_path and os.path.exists(real_train_csv_path):
    real_df = pd.read_csv(real_train_csv_path).dropna()
    if 'date.utc' in real_df.columns: real_df = real_df.drop(columns=['date.utc'])
    X_real, y_real = create_supervised_lags(real_df, target_col, look_back=24)
    
    rf_real = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_real, y_real)
    trtr_mae_rf = mean_absolute_error(y_test, rf_real.predict(X_test))
    
    # Utility Retention Ratio (TSTR Error / TRTR Error). 1.0 means perfect replication of real data utility.
    rf_utility_ratio = tstr_metrics_rf['MAE'] / trtr_mae_rf
    print(f"\n[Utility Gap] RF TRTR MAE: {trtr_mae_rf:.4f} -> TSTR MAE: {tstr_metrics_rf['MAE']:.4f}")
    print(f"Utility Retention Ratio (TSTR/TRTR Error): {rf_utility_ratio:.4f}x")

  return tstr_metrics_rf['MAE'], tstr_metrics_gbm['MAE']

if __name__ == "__main__":
  # Test execution
  BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  real_path = os.path.join(BASE_DIR, "data", "experiments", "tier1_baseline_train.csv")
  synth_path = os.path.join(BASE_DIR, "data", "experiments", "tier1_baseline_synthetic.csv")
  test_path = os.path.join(BASE_DIR, "data", "experiments", "tier1_baseline_test.csv") 
  out_dir = os.path.join(BASE_DIR, "evaluation_outputs", "tier1_baseline")
  
  if os.path.exists(real_path) and os.path.exists(synth_path):
    evaluate_synthetic_data(real_path, synth_path, out_dir)
    if os.path.exists(test_path):
      evaluate_predictive_utility(synth_path, test_path, target_col='pm25', real_train_csv_path=real_path)