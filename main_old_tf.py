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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
# PHASE 2: THE GOLDEN SEED (100,000 Epoch Replication Fidelity)
# =====================================================================
print("\n" + "="*70)
print(" PHASE 2: GOLDEN SEED REPLICATION (Seed 42 | 100,000 Epochs)")
print("="*70)

GOLDEN_SEED = 42
EPOCHS_GOLDEN = 100000 
tiers = ['tier1_baseline', 'tier2_chemproxy', 'tier3_fullproxy']

# Dictionaries to store final downstream utility metrics for comparison
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

  # 1. DoppelGANger Generation (Golden)
  run_dgan_experiment(
    tier_name=tier,
    input_dir=experiments_dir,
    output_dir=experiments_dir, # train_dgan.py will handle specific naming
    epochs=EPOCHS_GOLDEN
  )
  
  # Rename the output dynamically to mark it as the golden seed
  os.rename(os.path.join(experiments_dir, f"{tier}_synthetic.csv"), synth_output_path)

  # 2. Statistical Fidelity Evaluation (Train vs. Synth)
  print(f"\n[{tier.upper()}] Running Statistical Fidelity Evaluation...")
  evaluate_synthetic_data(
    real_csv_path=train_csv_path,
    synth_csv_path=synth_output_path,
    output_dir=tier_eval_dir
  )

  # 3. Downstream Predictive Utility Evaluation (TSTR)
  print(f"\n[{tier.upper()}] Running Predictive Utility Evaluation (TSTR)...")
  rf_mae, gbm_mae = evaluate_predictive_utility(
    synth_csv_path=synth_output_path,
    test_csv_path=test_csv_path,
    target_col='pm25'
  )
  
  golden_rf_results[tier] = rf_mae
  golden_gbm_results[tier] = gbm_mae


# =====================================================================
# PHASE 3: ROBUSTNESS LOOP (Convergence Check & Variance Reduction)
# =====================================================================
print("\n" + "="*70)
print(" PHASE 3: ROBUSTNESS LOOP (Multi-Seed | 10,000 Epochs)")
print("="*70)

# We use fewer epochs here to demonstrate that Tier 3 converges faster 
# and outperforms the baseline even without brute-force 100k memorization.
EPOCHS_ROBUST = 10000 
ROBUST_SEEDS = [123, 456, 789, 999]

# Initialize lists with the Golden Seed results to calculate a true 5-seed average
robust_rf_results = {tier: [golden_rf_results[tier]] for tier in tiers}
robust_gbm_results = {tier: [golden_gbm_results[tier]] for tier in tiers}

for tier in tiers:
  train_csv_path = os.path.join(experiments_dir, f"{tier}_train.csv")
  test_csv_path = os.path.join(experiments_dir, f"{tier}_test.csv")
  
  for seed in ROBUST_SEEDS:
    print(f"\n   --- Running {tier.upper()} | Seed: {seed} ---")
    set_seed(seed)
    
    synth_output_path = os.path.join(experiments_dir, f"{tier}_synthetic_seed{seed}.csv")
    tier_eval_dir = os.path.join(eval_output_dir, tier, f"robustness_seed_{seed}")
    os.makedirs(tier_eval_dir, exist_ok=True)

    run_dgan_experiment(
      tier_name=tier,
      input_dir=experiments_dir,
      output_dir=experiments_dir,
      epochs=EPOCHS_ROBUST
    )
    
    os.rename(os.path.join(experiments_dir, f"{tier}_synthetic.csv"), synth_output_path)

    evaluate_synthetic_data(
      real_csv_path=train_csv_path,
      synth_csv_path=synth_output_path,
      output_dir=tier_eval_dir
    )

    rf_mae, gbm_mae = evaluate_predictive_utility(
      synth_csv_path=synth_output_path,
      test_csv_path=test_csv_path,
      target_col='pm25'
    )
    
    robust_rf_results[tier].append(rf_mae)
    robust_gbm_results[tier].append(gbm_mae)


# =====================================================================
# FINAL ABLATION RESULTS EXPORT
# =====================================================================
print("\n" + "="*70)
print(" FINAL 3-TIER ABLATION RESULTS (Target: PM2.5 MAE)")
print("="*70)

archival_path = os.path.join(eval_output_dir, "master_ablation_results.txt")
with open(archival_path, "w") as f:
    
  # 1. Write Golden Seed Results (The 1:1 Replication)
  header_golden = "1. GOLDEN SEED REPLICATION (100,000 Epochs | Seed 42)"
  print(header_golden)
  f.write(header_golden + "\n" + "-"*55 + "\n")
  
  for tier in tiers:
    line = f"  {tier.upper()} -> RF: {golden_rf_results[tier]:.4f} | GBM: {golden_gbm_results[tier]:.4f}"
    print(line)
    f.write(line + "\n")
  
  # 2. Write Robustness Results (5-Seed Average)
  header_robust = "\n2. ROBUSTNESS & CONVERGENCE CHECK (5-Seed Average)"
  print(header_robust)
  f.write(header_robust + "\n" + "-"*55 + "\n")
  
  for tier in tiers:
    rf_mean, rf_std = np.mean(robust_rf_results[tier]), np.std(robust_rf_results[tier])
    gbm_mean, gbm_std = np.mean(robust_gbm_results[tier]), np.std(robust_gbm_results[tier])
    
    line = f"  {tier.upper()} -> RF: {rf_mean:.4f} ± {rf_std:.4f} | GBM: {gbm_mean:.4f} ± {gbm_std:.4f}"
    print(line)
    f.write(line + "\n")

print(f"\nArchived master ablation metrics to: {archival_path}")