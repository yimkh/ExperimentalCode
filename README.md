# ExperimentalCode
This photovoltaic power prediction code base only retains part of the key experimental code at present, and the detailed and complete code will be uploaded after the paper is published.

## File Description
### 1. ceemdan_sst.py
Implements the Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) algorithm, commonly used for non-stationary signal processing. It decomposes complex signals into several Intrinsic Mode Functions (IMFs) to facilitate subsequent analysis of the time-frequency characteristics of signals and feature extraction.
#### Input Data
- **Default path**: `data.csv` (modifiable inside the script)
- **Format**: CSV
- **Required columns**:
  - `timestamp` (datetime, format: `YYYY-MM-DD HH:MM:SS`)
  - `PV_Active_Power` (float, active power of PV system)
**Example input:**
```csv
timestamp,PV_Active_Power
2025-01-01 00:00:00,123.4
2025-01-01 00:05:00,120.8
2025-01-01 00:10:00,118.7
```
### 2. correlation.py
Focuses on correlation analysis, capable of calculating linear or non-linear correlation relationships between variables. 
### 3. pinn_mamba.py
Combines Physics-Informed Neural Networks (PINN) with the Mamba architecture, attempting to integrate emerging efficient sequence modeling structures into physical constraint learning scenarios. It can be used for solving partial differential equations, physical system simulation, etc., leveraging data-driven methods combined with physical prior knowledge to improve the model's ability to fit and predict physical laws.
### 4. pv_wgan_gp.py
Implements the Wasserstein Generative Adversarial Network (WGAN) with Gradient Penalty (GP) for the photovoltaic (PV) scenario data. It can be used to generate realistic PV power sequences and perform data augmentation, assisting in PV power forecasting, scenario simulation, and other tasks to alleviate issues of data scarcity or insufficient diversity.
