import pandas as pd
import numpy as np
import os
import torch
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType

def prepare_3d_sequences(df, sequence_length=24):
  """
  Converts a flat 2D dataframe into a 3D numpy array of overlapping sequences:
  (num_examples, sequence_length, num_features).
  Overlapping sequences give the GAN far more examples to learn from.
  """
  data = df.values
  sequences = []
  for i in range(len(data) - sequence_length + 1):
    sequences.append(data[i : i + sequence_length])
  return np.array(sequences)

def run_pt_baseline(seed_csv_path, output_synth_csv, model_save_path, epochs=100):
  print(f"Loading seed data from: {seed_csv_path}")
  df = pd.read_csv(seed_csv_path)
  
  feature_names = df.columns.tolist()
  
  # 1. Prepare 3D Data for DoppelGANger
  sequence_length = 24
  train_data = prepare_3d_sequences(df, sequence_length=sequence_length)
  print(f"Prepared 3D training data shape: {train_data.shape}")

  # 2. Configure True DoppelGANger (via Gretel)
  config = DGANConfig(
    max_sequence_len=sequence_length,
    sample_len=sequence_length, 
    # batch_size=32,          # Kept small to conserve local system RAM
    batch_size=min(3 * 365, features.shape[0]),  # Dynamic full-batch or slice-batch
    apply_feature_scaling=True, # DGAN handles MinMax scaling internally
    # apply_example_scaling=True, # This is the crucial Auto-Normalization
    apply_example_scaling=False,  # CHANGED FROM True TO False Bec, Base Paper Code
    use_attribute_discriminator=False, # We have no static attributes right now
    generator_learning_rate=1e-4,
    discriminator_learning_rate=1e-4,
    epochs=epochs
  )

  # 3. Initialize and Train
  print("\nInitializing True DoppelGANger Engine (PyTorch)...")
  model = DGAN(config)
  
  print("Training model... (This will log progress automatically)")
  model.train_numpy(
    features=train_data,
    feature_types=[OutputType.CONTINUOUS] * len(feature_names) # Fixed enum
  )

  # 4. Save the true baseline checkpoint
  os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
  model.save(model_save_path)
  print(f"\nModel saved to: {model_save_path}")

  # 5. Generate Synthetic Data
  print("Generating synthetic 2-week block...")
  # We need 14 distinct 24-hour sequences to get our 336 hours
  synthetic_3d = model.generate_numpy(14)[1] # index 1 returns the continuous features
  
  # Flatten from (14, 24, num_features) back to (336, num_features)
  synthetic_flat = synthetic_3d.reshape(-1, len(feature_names))
  
  synth_df = pd.DataFrame(synthetic_flat, columns=feature_names)

  non_negative_cols = [
    'o3', 'pm10', 'pm25', 'co', 'no2', 'so2', 
    'wind_speed', 'relative_humidity'
  ]
  for col in non_negative_cols:
    if col in synth_df.columns:
      synth_df[col] = synth_df[col].clip(lower=0.0)
          
  if 'relative_humidity' in synth_df.columns:
    synth_df['relative_humidity'] = synth_df['relative_humidity'].clip(upper=100.0)

  os.makedirs(os.path.dirname(output_synth_csv), exist_ok=True)
  synth_df.to_csv(output_synth_csv, index=False)
  
  print(f"True baseline synthetic dataset exported to: {output_synth_csv}")