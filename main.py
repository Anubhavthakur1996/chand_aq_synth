import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
from utils.fetchdata import fetchData
from src.preprocess import preprocess_data
from src.prep_experiments import prep_all_experiments
from src.train_dgan import run_dgan_experiment
from src.evaluate import evaluate_synthetic_data, evaluate_predictive_utility

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

def set_seed(seed=42):
  """Locks all random states for strict reproducibility (PyTorch architecture)."""
  print(f"-> Enforcing deterministic computation for seed: {seed}")
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths for data stages
raw_data_path = os.path.join(BASE_DIR, "data", "raw", "Chandigarh_Sector22_AirQuality_2025_2026.csv")
processed_data_path = os.path.join(BASE_DIR, "data", "processed", "chandigarh_sector22_processed.csv")
experiments_dir = os.path.join(BASE_DIR, "data", "experiments")
eval_output_dir = os.path.join(BASE_DIR, "evaluation_outputs")
eda_output_dir = os.path.join(BASE_DIR, "eda_outputs")

# Re-create directory structure if it was deleted
os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
os.makedirs(experiments_dir, exist_ok=True)
os.makedirs(eval_output_dir, exist_ok=True)
os.makedirs(eda_output_dir, exist_ok=True)

# =====================================================================
# PHASE 1: DATA PIPELINE (Fetch, Clean, Split, EDA)
# =====================================================================
if not os.path.exists(raw_data_path):
  print("Raw dataset not found. Initializing data extraction...")
  fetchData(output_path=raw_data_path)

  if not os.path.exists(raw_data_path):
    raise SystemExit(f"CRITICAL ERROR: fetchData() finished, but no CSV was saved to {raw_data_path}. Check your API console output above to see why the download failed.")

print("\n--- Preprocessing Raw Data ---")
preprocess_data(input_path=raw_data_path, output_dir=experiments_dir)

print("\n--- Generating 3-Tier Ablation Splits ---")
prep_all_experiments(input_csv=processed_data_path, output_dir=experiments_dir)

print("\n--- Generating Exploratory Data Analysis (EDA) Artifacts ---")
os.makedirs(eda_output_dir, exist_ok=True)
processed_df = pd.read_csv(processed_data_path)
processed_df.describe().to_csv(os.path.join(eda_output_dir, "dataset_description.csv"))

if 'pm25' in processed_df.columns:
  plt.figure(figsize=(12, 4))
  plt.plot(processed_df['pm25'].head(336).values, color='b', linewidth=1.5)
  plt.title("Baseline Real Data: PM2.5 Trace (First 336 Hours)")
  plt.xlabel("Hours")
  plt.ylabel("Concentration (ug/m3)")
  plt.tight_layout()
  plt.savefig(os.path.join(eda_output_dir, "raw_timeseries_pm25.png"))
  plt.close()
print(f"EDA artifacts saved successfully to: {eda_output_dir}")

# =====================================================================
# PHASE 2: THE GOLDEN SEED (100,000 Epoch Replication Fidelity)
# =====================================================================
print("\n" + "="*70)
print(" PHASE 2: GOLDEN SEED REPLICATION (Seed 42 | 100,000 Epochs)")
print("="*70)

GOLDEN_SEED = 42
tiers = ['tier1_baseline', 'tier2_chemproxy', 'tier3_fullproxy']

# Configure individual epoch budgets per tier if needed
golden_epochs_map = {
  'tier1_baseline': 100000,
  'tier2_chemproxy': 100000,
  'tier3_fullproxy': 10000  # Our custom 
}

golden_rf_results = {}
golden_gbm_results = {}

for tier in tiers:
  print(f"\n>> EXECUTING GOLDEN RUN FOR: {tier.upper()}")
  set_seed(GOLDEN_SEED)

  train_csv_path = os.path.join(experiments_dir, f"{tier}_train.csv")
  test_csv_path = os.path.join(experiments_dir, f"{tier}_test.csv")
  synth_output_path = os.path.join(experiments_dir, f"{tier}_synthetic_golden.csv")
  
  tier_eval_dir = os.path.join(eval_output_dir, tier, "golden_seed")
  os.makedirs(tier_eval_dir, exist_ok=True)

  run_dgan_experiment(tier_name=tier, input_dir=experiments_dir, output_dir=experiments_dir, epochs=golden_epochs_map[tier])
  os.rename(os.path.join(experiments_dir, f"{tier}_synthetic.csv"), synth_output_path)

  print(f"\n[{tier.upper()}] Running Statistical Fidelity Evaluation...")
  evaluate_synthetic_data(real_csv_path=train_csv_path, synth_csv_path=synth_output_path, output_dir=tier_eval_dir)

  print(f"\n[{tier.upper()}] Running Predictive Utility Evaluation (TSTR)...")
  rf_mae, gbm_mae = evaluate_predictive_utility(synth_csv_path=synth_output_path, test_csv_path=test_csv_path, target_col='pm25')
  
  golden_rf_results[tier] = rf_mae
  golden_gbm_results[tier] = gbm_mae

# =====================================================================
# PHASE 3: ROBUSTNESS LOOP (Multi-Seed | 10,000 Epochs)
# =====================================================================
print("\n" + "="*70)
print(" PHASE 3: ROBUSTNESS LOOP (Multi-Seed | 10,000 Epochs)")
print("="*70)

# EPOCHS_ROBUST = 10000 
ROBUST_SEEDS = [123, 456, 789, 999]

# Configure individual robust epoch budgets per tier if needed
robust_epochs_map = {
  'tier1_baseline': 10000,
  'tier2_chemproxy': 10000,
  'tier3_fullproxy': 5500  # Adjust here independently for Tier 3 robustness
}

robust_rf_results = {tier: [golden_rf_results[tier]] for tier in tiers}
robust_gbm_results = {tier: [golden_gbm_results[tier]] for tier in tiers}

for tier in tiers:
  train_csv_path = os.path.join(experiments_dir, f"{tier}_train.csv")
  test_csv_path = os.path.join(experiments_dir, f"{tier}_test.csv")
  
  for seed in ROBUST_SEEDS:
    print(f"\n    --- Running {tier.upper()} | Seed: {seed} ---")
    set_seed(seed)
    
    synth_output_path = os.path.join(experiments_dir, f"{tier}_synthetic_seed{seed}.csv")
    tier_eval_dir = os.path.join(eval_output_dir, tier, f"robustness_seed_{seed}")
    os.makedirs(tier_eval_dir, exist_ok=True)

    run_dgan_experiment(tier_name=tier, input_dir=experiments_dir, output_dir=experiments_dir, epochs=robust_epochs_map[tier])
    os.rename(os.path.join(experiments_dir, f"{tier}_synthetic.csv"), synth_output_path)

    evaluate_synthetic_data(real_csv_path=train_csv_path, synth_csv_path=synth_output_path, output_dir=tier_eval_dir)

    rf_mae, gbm_mae = evaluate_predictive_utility(synth_csv_path=synth_output_path, test_csv_path=test_csv_path, target_col='pm25')
    
    robust_rf_results[tier].append(rf_mae)
    robust_gbm_results[tier].append(gbm_mae)

print("\n" + "="*70)
print(" FINAL 3-TIER ABLATION RESULTS (Target: PM2.5 MAE)")
print("="*70)

archival_path = os.path.join(eval_output_dir, "master_ablation_results.txt")
with open(archival_path, "w") as f:
  header_golden = "1. GOLDEN SEED REPLICATION (100,000 Epochs | Seed 42)"
  print(header_golden)
  f.write(header_golden + "\n" + "-"*55 + "\n")
  for tier in tiers:
    line = f"   {tier.upper()} -> RF: {golden_rf_results[tier]:.4f} | GBM: {golden_gbm_results[tier]:.4f}"
    print(line)
    f.write(line + "\n")
  
  header_robust = "\n2. ROBUSTNESS & CONVERGENCE CHECK (5-Seed Average)"
  print(header_robust)
  f.write(header_robust + "\n" + "-"*55 + "\n")
  for tier in tiers:
    rf_mean, rf_std = np.mean(robust_rf_results[tier]), np.std(robust_rf_results[tier])
    gbm_mean, gbm_std = np.mean(robust_gbm_results[tier]), np.std(robust_gbm_results[tier])
    line = f"   {tier.upper()} -> RF: {rf_mean:.4f} ± {rf_std:.4f} | GBM: {gbm_mean:.4f} ± {gbm_std:.4f}"
    print(line)
    f.write(line + "\n")