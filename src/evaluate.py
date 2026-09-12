import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_js_divergence(real_data, synth_data, bins=50):
  min_val = min(real_data.min(), synth_data.min())
  max_val = max(real_data.max(), synth_data.max())
  bin_edges = np.linspace(min_val, max_val, bins+1)
  
  p_real, _ = np.histogram(real_data, bins=bin_edges, density=True)
  p_synth, _ = np.histogram(synth_data, bins=bin_edges, density=True)
  
  p_real = np.clip(p_real, 1e-10, None)
  p_synth = np.clip(p_synth, 1e-10, None)
  return jensenshannon(p_real, p_synth)

def calculate_acf_error(real_data, synth_data, max_lag=24):
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
  
  metrics_list = []
  
  print(f"\n{'Feature':<15} | {'W-Dist (EMD)':<12} | {'JS Div':<10} | {'ΔACF (24h)':<10} | {'Real Mean':<10} | {'Synth Mean':<10}")
  print("-" * 80)
  
  for col in common_cols:
    w_dist = wasserstein_distance(real_df[col], synth_df[col])
    js_div = calculate_js_divergence(real_df[col], synth_df[col])
    acf_err = calculate_acf_error(real_df[col], synth_df[col], max_lag=24)
    
    r_mean, s_mean = real_df[col].mean(), synth_df[col].mean()
    
    metrics_list.append({
      'Feature': col, 'W-Dist': w_dist, 'JS-Div': js_div, 'ACF_Err': acf_err
    })
    print(f"{col:<15} | {w_dist:<12.4f} | {js_div:<10.4f} | {acf_err:<10.4f} | {r_mean:<10.2f} | {s_mean:<10.2f}")

  metrics_df = pd.DataFrame(metrics_list)
  avg_w_dist = metrics_df['W-Dist'].mean()
  avg_js_div = metrics_df['JS-Div'].mean()
  avg_acf_err = metrics_df['ACF_Err'].mean()
  
  real_corr = real_df.corr().fillna(0).values
  synth_corr = synth_df.corr().fillna(0).values
  frob_norm = np.linalg.norm(real_corr - synth_corr, ord='fro')
  
  print("-" * 80)
  print(f"{'FINAL AVERAGE':<15} | {avg_w_dist:<12.4f} | {avg_js_div:<10.4f} | {avg_acf_err:<10.4f} | Frob Norm: {frob_norm:.4f}")
  
  metrics_df.to_csv(os.path.join(output_dir, 'column_metrics.csv'), index=False)
  
  with open(os.path.join(output_dir, 'final_aggregate_scores.txt'), 'w') as f:
    f.write("--- FINAL DATASET METRICS ---\n")
    f.write(f"Average W-Dist (EMD): {avg_w_dist:.4f}\n")
    f.write(f"Average JS Divergence: {avg_js_div:.4f}\n")
    f.write(f"Average ΔACF: {avg_acf_err:.4f}\n")
    f.write(f"Correlation Frobenius Norm: {frob_norm:.4f}\n")

  fig, axes = plt.subplots(1, 2, figsize=(18, 8))
  sns.heatmap(real_df.corr(), ax=axes[0], cmap='coolwarm', vmin=-1, vmax=1)
  axes[0].set_title("Real Data Correlation (Ground Truth)")
  sns.heatmap(synth_df.corr(), ax=axes[1], cmap='coolwarm', vmin=-1, vmax=1)
  axes[1].set_title("Synthetic Data Correlation")
  
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, 'correlation_paper_comparison.png'))
  plt.close()

def create_supervised_lags(df, target_col, look_back=24):
  X, y = [], []
  data = df.values
  target_idx = df.columns.get_loc(target_col)
  
  for i in range(len(data) - look_back):
    X.append(data[i : i + look_back].flatten())
    y.append(data[i + look_back, target_idx])
      
  return np.array(X), np.array(y)

def create_lstm_sequences(df, target_col, look_back=24):
  """Creates 3D tensors (Samples, Timesteps, Features) specifically for LSTM architectures."""
  data = df.drop(columns=[target_col]).values
  target = df[target_col].values
  target_idx = df.columns.get_loc(target_col)
  
  X, y = [], []
  for i in range(len(df) - look_back):
    X.append(df.iloc[i : i + look_back].values)
    y.append(target[i + look_back])
  return np.array(X), np.array(y)

class SimpleLSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim=64):
    super(SimpleLSTM, self).__init__()
    self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, 1)
      
  def forward(self, x):
    out, _ = self.lstm(x)
    out = self.fc(out[:, -1, :])
    return out.squeeze()

def train_and_evaluate_lstm(X_train, y_train, X_test, y_test, epochs=10):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  input_dim = X_train.shape[2]
  
  train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  
  model = SimpleLSTM(input_dim=input_dim).to(device)
  criterion = nn.L1Loss() # MAE Loss
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
  model.train()
  for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
      batch_x, batch_y = batch_x.to(device), batch_y.to(device)
      optimizer.zero_grad()
      preds = model(batch_x)
      loss = criterion(preds, batch_y)
      loss.backward()
      optimizer.step()
          
  model.eval()
  with torch.no_grad():
    test_x = torch.tensor(X_test, dtype=torch.float32).to(device)
    predictions = model(test_x).cpu().numpy()
      
  return predictions

def calc_forecasting_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    denominator = np.where(y_true == 0, 1e-10, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100
    cvrmse = (rmse / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else np.nan
    
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "CVRMSE": cvrmse}

def evaluate_predictive_utility(synth_csv_path, test_csv_path, target_col='pm25', real_train_csv_path=None, output_dir=None):
  synth_df = pd.read_csv(synth_csv_path).dropna()
  test_df = pd.read_csv(test_csv_path).dropna()
  
  if 'date.utc' in synth_df.columns: synth_df = synth_df.drop(columns=['date.utc'])
  if 'date.utc' in test_df.columns: test_df = test_df.drop(columns=['date.utc'])
  
  # 1. Tabular ML Baseline (Random Forest & HistGB)
  X_synth, y_synth = create_supervised_lags(synth_df, target_col, look_back=24)
  X_test, y_test = create_supervised_lags(test_df, target_col, look_back=24)
  
  rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
  rf_model.fit(X_synth, y_synth)
  rf_preds = rf_model.predict(X_test)
  tstr_metrics_rf = calc_forecasting_metrics(y_test, rf_preds)
  
  gbm_model = HistGradientBoostingRegressor(random_state=42)
  gbm_model.fit(X_synth, y_synth)
  gbm_preds = gbm_model.predict(X_test)
  tstr_metrics_gbm = calc_forecasting_metrics(y_test, gbm_preds)

  # 2. Deep Learning Baseline (LSTM)
  X_train_lstm, y_train_lstm = create_lstm_sequences(synth_df, target_col, look_back=24)
  X_test_lstm, y_test_lstm = create_lstm_sequences(test_df, target_col, look_back=24)
  lstm_preds = train_and_evaluate_lstm(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, epochs=10)
  tstr_metrics_lstm = calc_forecasting_metrics(y_test_lstm, lstm_preds)

  # 3. Shuffle-Synth Control Baseline (Negative Control Test)
  synth_df_shuffled = synth_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
  X_synth_shuf, y_synth_shuf = create_supervised_lags(synth_df_shuffled, target_col, look_back=24)
  rf_shuf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
  rf_shuf.fit(X_synth_shuf, y_synth_shuf)
  shuf_preds = rf_shuf.predict(X_test)
  tstr_metrics_shuf = calc_forecasting_metrics(y_test, shuf_preds)
  
  print(f"\n--- TSTR Downstream Utility (Target: {target_col}) ---")
  print(f"{'Model / Control':<20} | {'R2':<8} | {'RMSE':<8} | {'MAE':<8} | {'MAPE':<8} | {'CVRMSE':<8}")
  print("-" * 75)
  print(f"{'Random Forest':<20} | {tstr_metrics_rf['R2']:<8.4f} | {tstr_metrics_rf['RMSE']:<8.4f} | {tstr_metrics_rf['MAE']:<8.4f} | {tstr_metrics_rf['MAPE']:<8.4f} | {tstr_metrics_rf['CVRMSE']:<8.4f}")
  print(f"{'LightGBM':<20} | {tstr_metrics_gbm['R2']:<8.4f} | {tstr_metrics_gbm['RMSE']:<8.4f} | {tstr_metrics_gbm['MAE']:<8.4f} | {tstr_metrics_gbm['MAPE']:<8.4f} | {tstr_metrics_gbm['CVRMSE']:<8.4f}")
  print(f"{'LSTM (PyTorch)':<20} | {tstr_metrics_lstm['R2']:<8.4f} | {tstr_metrics_lstm['RMSE']:<8.4f} | {tstr_metrics_lstm['MAE']:<8.4f} | {tstr_metrics_lstm['MAPE']:<8.4f} | {tstr_metrics_lstm['CVRMSE']:<8.4f}")
  print(f"{'Shuffle-Synth (Ctrl)':<20} | {tstr_metrics_shuf['R2']:<8.4f} | {tstr_metrics_shuf['RMSE']:<8.4f} | {tstr_metrics_shuf['MAE']:<8.4f} | {tstr_metrics_shuf['MAPE']:<8.4f} | {tstr_metrics_shuf['CVRMSE']:<8.4f}")

  rf_utility_ratio = None
  if real_train_csv_path and os.path.exists(real_train_csv_path):
    real_df = pd.read_csv(real_train_csv_path).dropna()
    if 'date.utc' in real_df.columns: real_df = real_df.drop(columns=['date.utc'])
    X_real, y_real = create_supervised_lags(real_df, target_col, look_back=24)
    
    rf_real = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_real, y_real)
    trtr_mae_rf = mean_absolute_error(y_test, rf_real.predict(X_test))
    
    rf_utility_ratio = tstr_metrics_rf['MAE'] / trtr_mae_rf
    print(f"\n[Utility Gap] RF TRTR MAE: {trtr_mae_rf:.4f} -> TSTR MAE: {tstr_metrics_rf['MAE']:.4f}")
    print(f"Utility Retention Ratio (TSTR/TRTR Error): {rf_utility_ratio:.4f}x")

  if output_dir:
    with open(os.path.join(output_dir, 'predictive_scores.txt'), 'w') as f:
      f.write(f"--- TSTR DOWNSTREAM UTILITY (Target: {target_col}) ---\n")
      f.write(f"Random Forest        -> R2: {tstr_metrics_rf['R2']:.4f} | RMSE: {tstr_metrics_rf['RMSE']:.4f} | MAE: {tstr_metrics_rf['MAE']:.4f}\n")
      f.write(f"LightGBM             -> R2: {tstr_metrics_gbm['R2']:.4f} | RMSE: {tstr_metrics_gbm['RMSE']:.4f} | MAE: {tstr_metrics_gbm['MAE']:.4f}\n")
      f.write(f"LSTM (PyTorch)       -> R2: {tstr_metrics_lstm['R2']:.4f} | RMSE: {tstr_metrics_lstm['RMSE']:.4f} | MAE: {tstr_metrics_lstm['MAE']:.4f}\n")
      f.write(f"Shuffle-Synth (Ctrl) -> R2: {tstr_metrics_shuf['R2']:.4f} | RMSE: {tstr_metrics_shuf['RMSE']:.4f} | MAE: {tstr_metrics_shuf['MAE']:.4f}\n")
      if rf_utility_ratio:
        f.write(f"\n[Utility Gap] RF TRTR MAE: {trtr_mae_rf:.4f} -> TSTR MAE: {tstr_metrics_rf['MAE']:.4f}\n")
        f.write(f"Utility Retention Ratio: {rf_utility_ratio:.4f}x\n")

  return tstr_metrics_rf['MAE'], tstr_metrics_gbm['MAE']

if __name__ == "__main__":
  BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  tiers = ["tier1_baseline", "tier2_chemproxy", "tier3_fullproxy"]
  
  for tier in tiers:
    print(f"\n{'='*50}\nEvaluating {tier.upper()}\n{'='*50}")
    
    real_path = os.path.join(BASE_DIR, "data", "experiments", f"{tier}_train.csv")
    synth_path = os.path.join(BASE_DIR, "data", "experiments", f"{tier}_synthetic_golden.csv")
    test_path = os.path.join(BASE_DIR, "data", "experiments", f"{tier}_test.csv") 
    out_dir = os.path.join(BASE_DIR, "evaluation_outputs", tier)
    
    print(f"Looking for Real:  {real_path} | Exists: {os.path.exists(real_path)}")
    print(f"Looking for Synth: {synth_path} | Exists: {os.path.exists(synth_path)}")
    
    if os.path.exists(real_path) and os.path.exists(synth_path):
      print("Files found! Running evaluation...")
      evaluate_synthetic_data(real_path, synth_path, out_dir)
      if os.path.exists(test_path):
        evaluate_predictive_utility(synth_path, test_path, target_col='pm25', real_train_csv_path=real_path, output_dir=out_dir)
    else:
      print(f"Skipping {tier}: Paths do not match files on disk.")

  # Enhanced Master Compilation with Cross-Tier Summary Table
  master_out_path = os.path.join(BASE_DIR, "evaluation_outputs", "master_evaluation_results.txt")
  with open(master_out_path, 'w') as master_file:
    master_file.write("=== COMPLETE MASTER ABLATION RESULTS (WITH LSTM & SHUFFLE CONTROLS) ===\n\n")
    
    # Write individual tier breakdowns
    for tier in tiers:
      master_file.write(f"{'='*50}\n[{tier.upper()}]\n{'='*50}\n")
      
      stat_txt = os.path.join(BASE_DIR, "evaluation_outputs", tier, "final_aggregate_scores.txt")
      if os.path.exists(stat_txt):
        with open(stat_txt, 'r') as f:
          master_file.write(f.read())
      
      master_file.write("\n")
      
      pred_txt = os.path.join(BASE_DIR, "evaluation_outputs", tier, "predictive_scores.txt")
      if os.path.exists(pred_txt):
        with open(pred_txt, 'r') as f:
          master_file.write(f.read())
      
      master_file.write("\n")

    # Append a synthesized comparison summary
    master_file.write(f"{'='*50}\n[CROSS-TIER SYNTHESIS TABLE]\n{'='*50}\n")
    master_file.write(f"{'Tier':<18} | {'RF MAE':<8} | {'LightGBM MAE':<12} | {'LSTM MAE':<10} | {'Shuffle MAE':<10}\n")
    master_file.write("-" * 68 + "\n")
    
    for tier in tiers:
      pred_txt = os.path.join(BASE_DIR, "evaluation_outputs", tier, "predictive_scores.txt")
      if os.path.exists(pred_txt):
        with open(pred_txt, 'r') as f:
          lines = f.readlines()
          # Extract MAE values from the text file safely
          rf_mae, gbm_mae, lstm_mae, shuf_mae = "N/A", "N/A", "N/A", "N/A"
          for line in lines:
            if "Random Forest" in line: rf_mae = line.split("MAE:")[-1].strip()
            elif "LightGBM" in line: gbm_mae = line.split("MAE:")[-1].strip()
            elif "LSTM" in line: lstm_mae = line.split("MAE:")[-1].strip()
            elif "Shuffle-Synth" in line: shuf_mae = line.split("MAE:")[-1].strip()
          master_file.write(f"{tier:<18} | {rf_mae:<8} | {gbm_mae:<12} | {lstm_mae:<10} | {shuf_mae:<10}\n")

  print(f"\nMaster file compiled successfully at: {master_out_path}")