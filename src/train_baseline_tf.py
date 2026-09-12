import pandas as pd
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

def run_tf_baseline(input_csv, output_csv):
  print(f"Loading seed data from: {input_csv}")
  df = pd.read_csv(input_csv)
  
  # ydata-synthetic handles pure numerical data best natively
  if 'date.utc' in df.columns:
    df = df.drop(columns=['date.utc'])

  numerical_cols = df.columns.tolist()

  # 1. Define hyperparameters for DoppelGANger
  # We use a small batch size (32) because our training seed is only 336 rows
  model_args = ModelParameters(
    batch_size=32, 
    lr=0.001,
    latent_dim=20,
    gp_lambda=10,
    pac=1
  )

  # 2. Define sequence and training parameters
  # sequence_length=24 means we are teaching the GAN to generate in 24-hour daily chunks
  train_args = TrainParameters(
    epochs=100, # Keep it at 100 for a quick sanity check; can increase later
    sequence_length=24,
    sample_length=24, 
    rounds=1,
    measurement_cols=numerical_cols
  )

  print("\nInitializing TensorFlow DoppelGANger (ydata-synthetic)...")
  synth = TimeSeriesSynthesizer(modelname='doppelganger', model_parameters=model_args)

  print("Training the baseline GAN (this may take a few minutes)...")
  synth.fit(df, train_args, num_cols=numerical_cols)

  print("\nGenerating synthetic benchmark data...")
  # Generate an equivalent sequence block
  synth_df = synth.sample(n_samples=1) 
  
  # Save directly to our synthetic data folder
  synth_df.to_csv(output_csv, index=False)
  print(f"TensorFlow benchmark generated and saved to: {output_csv}")