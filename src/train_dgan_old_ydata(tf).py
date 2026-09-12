import pandas as pd
import numpy as np
import os
import argparse
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

def prepare_overlapping_dataframes(df, sequence_length=24):
  """
  ydata-synthetic DoppelGANger expects a list of DataFrames (each representing an independent sequence).
  We slice the continuous 336-hour chronological dataset into overlapping 24-hour DataFrames.
  """
  seq_list = []
  for i in range(len(df) - sequence_length + 1):
    # Extract 24-hour block and reset index so each block starts at step 0
    seq = df.iloc[i : i + sequence_length].reset_index(drop=True)
    seq_list.append(seq)
  return seq_list

def run_dgan_experiment(tier_name, input_dir, output_dir, epochs):
  train_csv_path = os.path.join(input_dir, f"{tier_name}_train.csv")
  output_synth_csv = os.path.join(output_dir, f"{tier_name}_synthetic.csv")
  model_save_path = os.path.join(output_dir, f"models/{tier_name}_dgan.pkl")

  print(f"[{tier_name.upper()}] Loading training seed from: {train_csv_path}")
  df = pd.read_csv(train_csv_path)
  feature_names = df.columns.tolist()
  
  # 1. Prepare Overlapping Sequences
  sequence_length = 24
  train_data_list = prepare_overlapping_dataframes(df, sequence_length=sequence_length)
  print(f"[{tier_name.upper()}] Prepared {len(train_data_list)} overlapping sequences of length {sequence_length}.")

  # 2. Configure WGAN-GP DoppelGANger Parameters
  gan_args = ModelParameters(
    batch_size=32,            # Small batch size introduces regularizing mini-batch noise
    lr=0.0001,                # WGAN-GP critic/generator learning rate
    betas=(0.5, 0.9),         # Adam optimizer betas optimized for WGAN
    latent_dim=20,            # Dimensions of the random noise injection (Z)
    gp_lambda=10,             # Standard Gradient Penalty weight for WGAN-GP
    pac=1                     # PacGAN mode (1 = standard DoppelGANger discriminator)
  )

  train_args = TrainParameters(
    epochs=epochs,
    sequence_length=sequence_length,
    sample_length=sequence_length
  )

  # 3. Initialize and Train
  print(f"\n[{tier_name.upper()}] Initializing Open-Source DoppelGANger (ydata-synthetic)...")
  model = TimeSeriesSynthesizer(modelname='doppelganger', model_parameters=gan_args)
  
  print(f"[{tier_name.upper()}] Training for {epochs} epochs...")
  model.fit(train_data_list, train_args, num_cols=feature_names, cat_cols=[])

  # 4. Save the Model Checkpoint
  os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
  model.save(model_save_path)
  print(f"\n[{tier_name.upper()}] Model saved to: {model_save_path}")

  # 5. Generate Synthetic Data
  print(f"[{tier_name.upper()}] Generating synthetic 2-week block...")
  # We need 14 distinct 24-hour sequences to replicate the 336-hour timeline
  synthetic_samples = model.sample(n_samples=14)
  
  # Flatten the generated samples into a continuous 2D dataframe
  synthetic_flat = []
  for seq in synthetic_samples:
    synthetic_flat.append(seq)
  synth_df = pd.concat(synthetic_flat, ignore_index=True)

  # 6. Apply Physical Constraints
  # Gases and particulate matter cannot mathematically be negative
  non_negative_cols = ['o3', 'pm10', 'pm25', 'co', 'so2', 'no', 'no2', 'nox', 'ws', 'relative_humidity']
  for col in non_negative_cols:
    if col in synth_df.columns:
      synth_df[col] = synth_df[col].clip(lower=0.0)
          
  if 'relative_humidity' in synth_df.columns:
    synth_df['relative_humidity'] = synth_df['relative_humidity'].clip(upper=100.0)

  # 7. Export Synthetic CSV
  os.makedirs(os.path.dirname(output_synth_csv), exist_ok=True)
  synth_df.to_csv(output_synth_csv, index=False)
  print(f"[{tier_name.upper()}] Synthetic dataset exported to: {output_synth_csv}\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run 100k-epoch DGAN generation for a specific ablation tier.")
  parser.add_argument('--tier', type=str, required=True, choices=['tier1_baseline', 'tier2_chemproxy', 'tier3_fullproxy'],
                      help="Specify which tier to train (e.g., tier1_baseline)")
  parser.add_argument('--epochs', type=int, default=100000, 
                      help="Number of training epochs (default: 100000 for strict replication)")
  args = parser.parse_args()

  # Paths matching our structure
  INPUT_DIR = 'data/experiments'
  OUTPUT_DIR = 'data/experiments'
  
  run_dgan_experiment(args.tier, INPUT_DIR, OUTPUT_DIR, args.epochs)