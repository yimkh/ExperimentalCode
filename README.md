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
- **Example input**:
```csv
timestamp,PV_Active_Power
2025-01-01 00:00:00,123.4
2025-01-01 00:05:00,120.8
2025-01-01 00:10:00,118.7
```
#### Preprocessing
- Missing/Inf values → interpolated and backfilled  
- All-NaN signals → forced to 0.0  
- Zero-variance signals → small Gaussian noise is added to ensure decomposability  
#### Decomposition Workflow
1. **CEEMDAN decomposition**
   - Parameters: `trials=20`, `max_imf=8`  
   - Minimum segment length: 500 points  
   - Supports multiprocessing-based chunked decomposition (default chunk size ≥5000)  
2. **SST refinement**
   - Based on `ssq_cwt` with wavelet = `cmhat`, `nv=8`  
   - Produces smoothed/refined IMFs  
3. **IMF selection**
   - Correlation threshold: `|corr| > 0.1` → retain IMF  
   - Energy ratio threshold: `> 0.01` → retain IMF  
#### Output Results
- **CSV files**
  - `imfs_full_refined.csv`: all refined IMFs  
  - `selected_imfs.csv`: selected IMFs after filtering  
- **Figures**
  - `imf_plots/full/full_segment_1.png`: segmented plots of all IMFs  
  - `imf_plots/selected/selected_segment_1.png`: segmented plots of selected IMFs  
**Example output (CSV):**
```csv
IMF1,IMF2,IMF3,IMF4
0.23,0.02,-0.15,0.00
0.21,0.03,-0.14,0.01
```
### 2. correlation.py
Focuses on correlation analysis, capable of calculating linear or non-linear correlation relationships between variables. 
### 3. pinn_mamba.py
Combines Physics-Informed Neural Networks (PINN) with the Mamba architecture, attempting to integrate emerging efficient sequence modeling structures into physical constraint learning scenarios. It can be used for solving partial differential equations, physical system simulation, etc., leveraging data-driven methods combined with physical prior knowledge to improve the model's ability to fit and predict physical laws.
### 4. pv_wgan_gp.py
Implements the Wasserstein Generative Adversarial Network (WGAN) with Gradient Penalty (GP) for the photovoltaic (PV) scenario data. It can be used to generate realistic PV power sequences and perform data augmentation, assisting in PV power forecasting, scenario simulation, and other tasks to alleviate issues of data scarcity or insufficient diversity.
