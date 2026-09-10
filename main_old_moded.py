import os
import numpy as np
import tensorflow as tf
import random
from utils.fetchdata import fetchData
from src.preprocess import preprocess_data
from src.prep_experiments import prep_all_experiments
from src.train_dgan import run_dgan_experiment
from src.evaluate import evaluate_synthetic_data, evaluate_predictive_utility

def set_seed(seed=42):
  """Locks all random states for strict reproducibility (TensorFlow architecture)."""
  print(f"-> Enforcing deterministic computation for seed: {seed}")
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  os.environ['TF_DETERMINISTIC_OPS'] = '1'

BASE_DIR = os.path.dirname(__file__)

# Define paths for data stages
raw_data_path = os.path.join(BASE_DIR, "data", "raw", "Chandigarh_Sector22_AirQuality_2025_2026.csv")
processed_data_path = os.path.join(BASE_DIR, "data", "processed", "chandigarh_sector22_processed.csv")
experiments_dir = os.path.join(BASE_DIR, "data", "experiments")
eval_output_dir = os.path.join(BASE_DIR, "evaluation_outputs")

# =====================================================================
# PHASE 1: DATA PIPELINE (Fetch, Clean, Split)
# =====================================================================
if not os.path.exists(raw_data_path):
  print("Raw dataset not found. Initializing data extraction...")
  fetchData(output_path=raw_data_path)
else:
  print(f"Raw dataset found at {raw_data_path}. Skipping download.")

# Clean raw data
print("\n--- Preprocessing Raw Data ---")
preprocess_data(input_path=raw_data_path, output_path=processed_data_path)

# Slice into Tier 1, Tier 2, and Tier 3 train/test seeds
print("\n--- Generating 3-Tier Ablation Splits ---")
prep_all_experiments(input_csv=processed_data_path, output_dir=experiments_dir)


# =====================================================================
# PHASE 2: GENERATION & EVALUATION LOOP (100,000 Epoch Replication)
# =====================================================================
# Lock execution to a single Golden Seed to make the 100k epoch runs reproducible
set_seed(42)

# The base paper enforces 100,000 epochs. Reduce this to ~5000 ONLY for quick debugging.
EPOCHS = 100000 
tiers = ['tier1_baseline', 'tier2_chemproxy', 'tier3_fullproxy']

# Dictionaries to store final downstream utility metrics for comparison
rf_results = {}
gbm_results = {}

for tier in tiers:
  print("\n" + "="*60)
  print(f" EXECUTING PIPELINE FOR: {tier.upper()}")
  print("="*60)

  # Dynamic paths for the current tier
  train_csv_path = os.path.join(experiments_dir, f"{tier}_train.csv")
  test_csv_path = os.path.join(experiments_dir, f"{tier}_test.csv")
  synth_output_path = os.path.join(experiments_dir, f"{tier}_synthetic.csv")
  tier_eval_dir = os.path.join(eval_output_dir, tier)
  os.makedirs(tier_eval_dir, exist_ok=True)

  # 1. DoppelGANger Generation
  run_dgan_experiment(
    tier_name=tier,
    input_dir=experiments_dir,
    output_dir=experiments_dir,
    epochs=EPOCHS
  )

  # 2. Statistical Fidelity Evaluation (Train vs. Synth)
  print(f"\n[{tier.upper()}] Running Statistical Fidelity Evaluation...")
  evaluate_synthetic_data(
    real_csv_path=train_csv_path,
    synth_csv_path=synth_output_path,
    output_dir=tier_eval_dir
  )

  # 3. Downstream Predictive Utility Evaluation (TSTR)
  # Trains models on the synthetic data, tests them on the untouched real test set
  print(f"\n[{tier.upper()}] Running Predictive Utility Evaluation (TSTR)...")
  rf_mae, gbm_mae = evaluate_predictive_utility(
    synth_csv_path=synth_output_path,
    test_csv_path=test_csv_path,
    target_col='pm25'
  )
  
  rf_results[tier] = rf_mae
  gbm_results[tier] = gbm_mae


# =====================================================================
# FINAL ABLATION RESULTS
# =====================================================================
print("\n" + "="*60)
print(" FINAL 3-TIER ABLATION RESULTS (Target: PM2.5 MAE)")
print("="*60)
print(f"Tier 1 (Baseline):      RF: {rf_results['tier1_baseline']:.4f} | GBM: {gbm_results['tier1_baseline']:.4f}")
print(f"Tier 2 (Chem Proxy):    RF: {rf_results['tier2_chemproxy']:.4f} | GBM: {gbm_results['tier2_chemproxy']:.4f}")
print(f"Tier 3 (Full Proxy):    RF: {rf_results['tier3_fullproxy']:.4f} | GBM: {gbm_results['tier3_fullproxy']:.4f}")
print("="*60)

# Save master results for archival
archival_path = os.path.join(eval_output_dir, "master_ablation_results.txt")
with open(archival_path, "w") as f:
  f.write("FINAL 3-TIER ABLATION RESULTS (Target: PM2.5 MAE)\n")
  f.write("-" * 50 + "\n")
  for tier in tiers:
    f.write(f"{tier.upper()} -> RF: {rf_results[tier]:.4f} | GBM: {gbm_results[tier]:.4f}\n")

print(f"Archived master ablation metrics to: {archival_path}")