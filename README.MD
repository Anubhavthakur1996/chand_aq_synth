# Spatiotemporal Proxy Injection in DoppelGANger: Accelerated Synthetic Data Generation for Multivariate Air Quality

## Description
This repository contains the codebase, training architecture, and evaluation framework for an ablation study on generating high-fidelity synthetic air quality data using sector 22, Chandigarh, India AQ data with the help of DoppelGANger architecture.

While existing literature often relies on brute-force, high-epoch training (e.g., 100,000 epochs) to force GAN convergence on time-series meteorological data, this study demonstrates that such approaches are prone to overfitting and mode collapse. By injecting specific physical and temporal anchors (spatiotemporal proxies), we prove that the GAN can achieve superior downstream predictive utility in a fraction of the compute time (10,000 epochs).

## Features
**The 3-Tier Ablation Architecture**

To isolate the impact of domain-specific features on GAN stability and data utility, the study is structured as a strict 3-tier ablation:

1. **Tier 1: Baseline (Physical Dispersion)**
   - **Features:** Pollutants (PM2.5, PM10, CO, etc.) + Meteorological data (Temp, RH, WS, WD).
   - **Rationale:** Replicates the baseline architectural performance. Captures basic physical dispersion (e.g., wind moving particulate matter) but struggles with high variance and requires massive epoch counts to stabilize.

2. **Tier 2: ChemProxy (Incomplete Photochemistry)**
   - **Features:** Tier 1 + Chemical precursors (NO, NO2, NOx).
   - **Rationale:** Introduces the chemical drivers of PM2.5 and ozone. However, because NOx reactions are governed by diurnal solar cycles, adding chemicals *without* a temporal clock confuses the model. This tier acts as a negative architectural control, demonstrating degraded downstream utility (higher TSTR MAE).

3. **Tier 3: FullProxy (Complete Physical Constraints)**
   - **Features:** Tier 2 + Spatiotemporal cyclical encodings (Sine/Cosine of Hour and Month).
   - **Rationale:** Provides the GAN with the explicit "clock" needed to contextualize the chemical precursors. The model can map NOx spikes and temperature shifts to diurnal and seasonal cycles. This tier converges significantly faster (10,000 golden epochs / 5,500 robustness epochs) and achieves the highest stability.

**Evaluation Methodology: TSTR & Negative Controls**

Data fidelity is evaluated using a Train-on-Synthetic, Test-on-Real (TSTR) paradigm, prioritizing actual machine learning utility over surface-level statistical similarity.

1. **Downstream Probes:** We use tree-based regressors (Random Forest, LightGBM) on 24-hour lookback sequences to predict PM2.5. *(Note: Vanilla LSTMs were tested but omitted from final reporting due to consistent negative R² scores across all tiers, proving them to be weak probes for small-sample TSTR utility).*

2. **Utility Retention Ratio:** Measures the gap between a model trained on real data (TRTR) vs synthetic data (TSTR). Tier 3 achieved an exceptional 1.18x retention ratio (MAE: 10.53), vastly outperforming Tier 1 (1.77x) and Tier 2 (1.96x).

3. **Shuffle-Synth Negative Control:** Synthetic time-series data is randomly shuffled across the time axis before TSTR evaluation. The resulting collapse in predictive utility (e.g., Tier 3 Shuffle MAE jumping to 23.10 with a negative R²) mathematically proves the GAN learned genuine temporal mechanics, not just marginal distributions.

```
==================================================
[TIER3_FULLPROXY]
==================================================
--- FINAL DATASET METRICS ---
Average W-Dist (EMD): 3.6857
Average JS Divergence: 0.3572
Average ΔACF: 0.2309
Correlation Frobenius Norm: 1.9900

--- TSTR DOWNSTREAM UTILITY (Target: pm25) ---
Random Forest        -> R2: 0.5233 | RMSE: 22.3674 | MAE: 10.5358
LightGBM             -> R2: 0.5259 | RMSE: 22.3055 | MAE: 10.3840
LSTM (PyTorch)       -> R2: -1.0812 | RMSE: 46.7353 | MAE: 33.6591
Shuffle-Synth (Ctrl) -> R2: -0.0016 | RMSE: 32.4215 | MAE: 23.1032

[Utility Gap] RF TRTR MAE: 8.8998 -> TSTR MAE: 10.5358
Utility Retention Ratio: 1.1838x 
```

## Installation
```bash
git clone <repository-url>
cd chand_aq_synth
pip install -r requirements.txt
```

## Usage
```python
# 1. Generate Synthetic Data (Training Orchestration)
# This executes the 100k-epoch baseline replication and the optimized 10k/5.5k Tier 3 runs.
python main.py

# 2. Run TSTR Evaluation
# Computes statistical distances, trains downstream RF/LightGBM models, 
# and compiles the master evaluation reports.
python src/evaluate.py
```

## Project Structure
```text
chand_aq_synth/
├── data/                    # Contains real, preprocessed, and generated synthetic CSVs
├── evaluation_outputs/      # Auto-generated directory for MAE logs, TXT reports, and heatmaps
├── models/                  # Saved model weights and training checkpoints
├── notebooks/               # Exploratory data analysis and experimental prototyping
├── src/                     # Core pipeline modules
│   ├── evaluate.py          # Standalone script for TSTR evaluation and statistical metrics
│   ├── prep_experiments.py  # Generates Tier 1, Tier 2, and Tier 3 experimental splits
│   ├── preprocess.py        # Cleans and formats raw meteorological and pollutant data
│   ├── train_baseline_pt.py # PyTorch baseline implementation
│   ├── train_baseline_tf.py # TensorFlow baseline implementation
│   └── train_dgan.py        # Main DoppelGANger training architecture for the ablation study
├── utils/                   # Auxiliary helper functions
│   └── fetchdata.py         # API integration for raw PM2.5 and weather data ingestion
├── main.py                  # Orchestrates data fetching, preprocessing, and 3-tier training loops
├── requirements.txt         # Project dependencies
└── README.md                # Repository documentation
```

## Abstract
Pollution in urban environments requires robust predictive modeling, but high-resolution meteorological and chemical time-series data is often scarce or incomplete. While Generative Adversarial Networks (GANs) can synthesize this data, standard high-epoch training approaches on complex multivariate atmospheric datasets often suffer from instability, mode collapse, and high computational cost. This paper introduces a physically-grounded ablation study using the DoppelGANger architecture, demonstrating that injecting spatiotemporal proxies (cyclical encodings of time and season) significantly accelerates convergence. By providing the GAN with the explicit temporal anchors necessary to contextualize diurnal photochemistry, our optimized architecture achieves superior downstream predictive utility (1.18x Train-on-Synthetic, Test-on-Real retention ratio) in just 10,000 epochs, vastly outperforming brute-force baselines.

## How to Cite
If you use this dataset, codebase, or methodology in your research, please cite our paper:

**Plain Text:**
> Anubhav Singh, T. (2026). *Spatiotemporal Proxy Injection in DoppelGANger: Accelerated Synthetic Data Generation for Multivariate Air Quality*. (Manuscript in preparation/Under review).

**BibTeX:**
```bibtex
@article{thakuranubhavsingh2026doppelganger,
  title={Spatiotemporal Proxy Injection in DoppelGANger for Synthetic Data Generation},
  author={[Singh], [Thakur Anubhav]},
  journal={TBD},
  year={2026}
}

## Requirements
- Python 3.8+
- PyTorch (CUDA recommended)
- pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, dotenv

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Thakur Anubhav Singh
(anubhavthakur1996@outlook.com)