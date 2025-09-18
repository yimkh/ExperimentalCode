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
- **Example output (CSV):**
```csv
IMF1,IMF2,IMF3,IMF4
0.23,0.02,-0.15,0.00
0.21,0.03,-0.14,0.01
```
### 2. correlation.py
Focuses on correlation analysis, capable of calculating linear or non-linear correlation relationships between variables.
#### Input
- **File format**: Excel (`.xlsx`)  
- **Default file path**: `data.xlsx`  
- **Required columns:**
  - `PV_Active_Power`
  - `Temperature`
  - `Humidity`
  - `Global_Horizontal_Irradiance`
  - `Diffuse_Horizontal_Irradiance`
  - `Wind_Direction`
  - `Daily_Precipitation`
  - `Tilted_Global_Irradiance`
  - `Tilted_Diffuse_Irradiance`
#### Processing Steps
1. Read the Excel file (`pandas.read_excel`).  
2. Rename original Chinese column names into English for consistency.  
3. Compute a correlation matrix across all selected features.  
4. Extract and sort correlations between each variable and `PV_Active_Power`.  
5. Plot the correlation matrix as a heatmap using **seaborn** and **matplotlib**.
#### Output
- **Terminal output**: sorted correlation coefficients with respect to `PV_Active_Power`.  
- **Visualization**:  Heatmap saved as `correlation_heatmap.png` (300 dpi, suitable for publication-quality figures).
### 3. clustering.py
This script performs **unsupervised clustering** of PV power and meteorological variables using **PCA + HDBSCAN**, followed by visualization with scatterplots, t-SNE, and UMAP.  
#### Input Data
- **File format**: Excel (`.xlsx`)  
- **Default file**: `data.xlsx`  
- **Required columns**:
  - `timestamp` (`YYYY-MM-DD HH:MM:SS`)
  - `PV_Active_Power`
  - `Temperature`
  - `Humidity`
  - `Global_Horizontal_Radiation`
  - `Diffuse_Horizontal_Radiation`
  - `Wind_Direction`
  - `Daily_Precipitation`
  - `Tilted_Global_Radiation`
  - `Tilted_Diffuse_Radiation`
**Example row (`data.xlsx`):**
```csv
timestamp,PV_Active_Power,Temperature,Humidity,Global_Horizontal_Radiation,Diffuse_Horizontal_Radiation,Wind_Direction,Daily_Precipitation,Tilted_Global_Radiation,Tilted_Diffuse_Radiation
2025-01-01 00:00:00,123.4,18.5,70,420,180,90,0,350,160
```
#### Processing Steps
1. **Data Cleaning**
   - Missing values → replaced by column mean  
   - Remove outliers:
     - `PV_Active_Power` ∉ [−5, 400]  
     - `Global_Horizontal_Radiation` > 1400  
2. **Standardization**
   - Features scaled with `StandardScaler`  
3. **Dimensionality Reduction**
   - PCA, retaining 90% variance  
4. **Clustering**
   - HDBSCAN (`min_cluster_size=2500`, `min_samples=2`)  
   - Noise points labeled as `-1`  
5. **Visualization**
   - Scatterplots: PV Active Power vs each meteorological variable  
   - t-SNE (`perplexity=30, learning_rate=200`)  
   - UMAP (`n_components=2`)  
6. **Statistics**
   - Mean and standard deviation of features by cluster  
#### Outputs
- **Clustered data**  
  - `pv_data_clustered.xlsx` → includes original features + cluster labels  
- **Figures**  
  - `hdbscan_clustering_results.png` → scatterplots of PV Active Power vs other features  
  - `tsne_cluster_visualization.png` → t-SNE 2D visualization  
  - `umap_cluster_visualization.png` → UMAP 2D visualization  
- **Cluster statistics**  
  - `cluster_feature_stats.xlsx` → mean and standard deviation of each feature per cluster  
- **Example console output:**
```csv
PCA...
5
HDBSCAN...
12.34s
Clusters: 4
Noise points: 1023 (5.67%)
```
### 4. pv_wgan_gp.py
Implements the Wasserstein Generative Adversarial Network (WGAN) with Gradient Penalty (GP) for the photovoltaic (PV) scenario data. It can be used to generate realistic PV power sequences and perform data augmentation, assisting in PV power forecasting, scenario simulation, and other tasks to alleviate issues of data scarcity or insufficient diversity.
#### Input Data
- **File format**: Excel (`.xlsx`)  
- **Default path**: `data.xlsx`  
- **Required columns** (example: `Temperature`):
  - `timestamp` (format: `YYYY-MM-DD HH:MM:SS`)  
  - One target variable column (default = `"Temperature"`)  
- **Example input:**
```csv
timestamp,Temperature
2025-01-01 00:00:00,18.4
2025-01-01 00:05:00,18.6
2025-01-01 00:10:00,18.3
```
#### Processing Steps
1. **Preprocessing**
   - Target column values are clipped at zero, then transformed with `log1p`.
   - Data is normalized using `MinMaxScaler`.
   - Sequences of length 20 are created, predicting the next 4 time steps.
2. **Generator Model**
   - Input: `(seq_length=20, feature_size=1)`
   - BiLSTM layers: 512 → 256 units
   - Multi-Head Attention: 8 heads, key_dim=64
   - Dense layers with GaussianNoise
   - Output: `(4, 1)` sequence, sigmoid-activated
3. **Discriminator Model**
   - Conv1D layers (32, 64, 128 filters) + LeakyReLU
   - Dense layers with LeakyReLU
   - Final output: scalar score
4. **WGAN-GP Training**
   - Loss = Wasserstein + Gradient Penalty + MSE term (λ=2.0)
   - `n_critic = 3` updates per generator step
   - Optimizer: Adam (`lr=1e-4, beta1=0.9, beta2=0.999`)
   - Checkpoints saved every 15 epochs
5. **Evaluation**
   - Real vs Generated values rescaled back with `expm1`
   - RMSE computed
   - Results saved to Excel + plots saved as PNG
#### Outputs
- **Excel files**
  - `Save/dataTemperature/data_Full_results.xlsx` → Real vs Generated series
- **Figures**
  - `loss_plot.png` → Generator & Discriminator training losses
  - `data_Full_plot.png` → Real vs Generated time series comparison
**Example Excel output:**
```csv
Timestamp,Real,Generated
2025-01-01 00:00:00,18.4,18.3
2025-01-01 00:05:00,18.6,18.7
2025-01-01 00:10:00,18.3,18.4
```

### 5. pinn_mamba.py
Combines Physics-Informed Neural Networks (PINN) with the Mamba architecture, attempting to integrate emerging efficient sequence modeling structures into physical constraint learning scenarios. It can be used for solving partial differential equations, physical system simulation, etc., leveraging data-driven methods combined with physical prior knowledge to improve the model's ability to fit and predict physical laws.
#### Input Data

- **File**: `data.csv`
- **Required columns** (minimum):
  - `timestamp` (`YYYY-MM-DD HH:MM:SS`)
  - `PV_Active_Power`
  - Meteorology: `Temperature`, `Humidity`,
    `Global_Horizontal_Radiation`, `Diffuse_Horizontal_Radiation`,
    `Tilted_Global_Radiation`, `Tilted_Diffuse_Radiation`
  - CEEMDAN IMFs: `imfs0`, `imfs1`, `imfs2`, `imfs3`, `imfs4`
- **Lag features**: the script auto-creates `PV_Lag_1 … PV_Lag_10` from `PV_Active_Power`. :contentReference[oaicite:2]{index=2}
**Example rows (`data.csv`):**
```csv
timestamp,PV_Active_Power,Temperature,Humidity,Global_Horizontal_Radiation,Diffuse_Horizontal_Radiation,Tilted_Global_Radiation,Tilted_Diffuse_Radiation,imfs0,imfs1,imfs2,imfs3,imfs4
2025-01-01 00:00:00,123.4,18.5,70,420,180,350,160,0.12,-0.03,0.07,0.01,-0.02
2025-01-01 00:05:00,125.1,18.7,71,425,182,352,161,0.11,-0.02,0.06,0.02,-0.01
```
#### Processing Steps
1. **Lag construction**  
   - Build lag features `PV_Lag_1..10`  
   - Drop initial NaN rows  
2. **Feature set**  
   - Combine meteorological variables + CEEMDAN IMFs + lag features  
   - Target variable: `PV_Active_Power`  
3. **Scaling**  
   - Apply `MinMaxScaler` to `X` and `y`  
   - Split into **80% training / 20% validation**  
4. **Windowing**  
   - Sliding windows with `SEQ_LEN = 10` and `PRED_LEN = 1`  
5. **Model**  
   - Input → Linear projection  
   - **N × Mamba layers** (default 4)  
     - Each block: LayerNorm → Mamba → Dropout → Residual connection  
   - MLP head  
   - Final embedding from the **last time step** used for 1-step prediction  
6. **Loss function**  
   - `Total Loss = MSE(pred, y) + λ * physics_loss`  
   - Physics loss:  
     - `eta_temp = 1 - 0.004 * (T - 25)`  
     - `eta_humid = 1 - 0.002 * |H - 50|`  
     - `pv_theory = G_tilted * eta_temp * eta_humid`  
     - `physics_loss = MSE(pred, pv_theory)`  
   - Default λ = 0.1  
7. **Training loop**  
   - Optimizer: Adam (`lr = 1e-4`)  
   - Learning rate scheduler: `ReduceLROnPlateau`  
   - EarlyStopping (`patience = 20`, `min_delta = 1e-4`)  
8. **Checkpointing**  
   - Save best model (lowest validation RMSE) to:  
     ```
     checkpoints/mamba_best_<timestamp>.pth
     ```  
#### Outputs
- **Checkpoints**
  - `checkpoints/mamba_best_<timestamp>.pth` → best model (selected by lowest validation RMSE)
- **CSV results**
  - `train_prediction.csv`
  - `val_prediction.csv`  
  Each file has the format:index,true,predicted

- **Figures** (publication-ready, Times New Roman, 300 dpi)
- Segment plots (per 1000 points) under:
  - `data/3.0/Train/`
  - `data/3.0/Validation/`
- `results/plots/mamba_true_vs_pred.png` → True vs Predicted (first 2000 points)
- `results/plots/mamba_residual_hist_kde.png` → Residual histogram + KDE
- `results/plots/mamba_residual_over_time.png` → Residual time series
**Example of `val_prediction.csv`:**
```csv
index,true,predicted
0,132.41,129.87
1,130.02,130.55
2,128.77,128.60
```

