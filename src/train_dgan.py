import pandas as pd
import numpy as np
import os
import argparse
import torch
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType

def prepare_3d_sequences(df, sequence_length=24):
  """
  Converts a flat 2D dataframe into a 3D numpy array of overlapping sequences:
  (num_examples, sequence_length, num_features).
  """
  data = df.values
  sequences = []
  for i in range(len(data) - sequence_length + 1):
    sequences.append(data[i : i + sequence_length])
  return np.array(sequences)

def run_dgan_experiment(tier_name, input_dir, output_dir, epochs):
  train_csv_path = os.path.join(input_dir, f"{tier_name}_train.csv")
  output_synth_csv = os.path.join(output_dir, f"{tier_name}_synthetic.csv")
  model_save_path = os.path.join(output_dir, f"models/{tier_name}_dgan.pt")

  print(f"[{tier_name.upper()}] Loading training seed from: {train_csv_path}")
  df = pd.read_csv(train_csv_path)
  feature_names = df.columns.tolist()
  
  # 1. Prepare 3D Data for PyTorch DoppelGANger
  sequence_length = 24
  train_data = prepare_3d_sequences(df, sequence_length=sequence_length)
  print(f"[{tier_name.upper()}] Prepared 3D training data shape: {train_data.shape}")

  # 2. Configure PyTorch DoppelGANger
  config = DGANConfig(
    max_sequence_len=sequence_length,
    sample_len=sequence_length,
    # batch_size=min(32, train_data.shape[0]), 
    batch_size=train_data.shape[0], 
    apply_feature_scaling=True, 
    apply_example_scaling=False, # STRICT ACADEMIC REPLICATION: Prevents auto-hiding of physical bounds
    use_attribute_discriminator=False,
    generator_learning_rate=1e-4,
    discriminator_learning_rate=1e-4,
    epochs=epochs
  )

  # 3. Initialize and Train
  print(f"\n[{tier_name.upper()}] Initializing PyTorch DoppelGANger Engine...")
  model = DGAN(config)
  
  print(f"[{tier_name.upper()}] Training for {epochs} epochs...")
  model.train_numpy(
    features=train_data,
    feature_types=[OutputType.CONTINUOUS] * len(feature_names)
  )

  # 4. Save the Model Checkpoint
  os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
  model.save(model_save_path)
  print(f"\n[{tier_name.upper()}] Model saved to: {model_save_path}")

  # 5. Generate Synthetic Data
  print(f"[{tier_name.upper()}] Generating synthetic 2-week block...")
  # Generate 14 distinct 24-hour sequences to get our 336 hours
  synthetic_3d = model.generate_numpy(14)[1]
  
  # Flatten from (14, 24, num_features) to (336, num_features)
  synthetic_flat = synthetic_3d.reshape(-1, len(feature_names))
  synth_df = pd.DataFrame(synthetic_flat, columns=feature_names)

  # 6. Apply Physical Constraints
  non_negative_cols = ['o3', 'pm10', 'pm25', 'co', 'so2', 'no', 'no2', 'nox', 'ws', 'rh']
  for col in non_negative_cols:
    if col in synth_df.columns:
      synth_df[col] = synth_df[col].clip(lower=0.0)
          
  if 'rh' in synth_df.columns:
    synth_df['rh'] = synth_df['rh'].clip(upper=100.0)

  # 7. Export Synthetic CSV
  os.makedirs(os.path.dirname(output_synth_csv), exist_ok=True)
  synth_df.to_csv(output_synth_csv, index=False)
  print(f"[{tier_name.upper()}] Synthetic dataset exported to: {output_synth_csv}\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run DGAN generation for a specific ablation tier.")
  parser.add_argument('--tier', type=str, required=True, choices=['tier1_baseline', 'tier2_chemproxy', 'tier3_fullproxy'])
  parser.add_argument('--epochs', type=int, default=100000)
  args = parser.parse_args()

  INPUT_DIR = 'data/experiments'
  OUTPUT_DIR = 'data/experiments'
  
  run_dgan_experiment(args.tier, INPUT_DIR, OUTPUT_DIR, args.epochs)