# Benchmarking Deep Spatiotemporal Models for Land Subsidence Projection Using SBAS-InSAR
### A High-Resolution Remote Sensing & GeoAI Pipeline for Coastal Hazard Forecasting in Sayung District, Demak Regency, Central Java, Indonesia

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow_2.x-orange.svg)](https://tensorflow.org/)
[![InSAR Engine](https://img.shields.io/badge/InSAR-ISCE2%20%7C%20MintPy-purple.svg)](https://github.com/insarlab/MintPy)
[![Satellite Data](https://img.shields.io/badge/SAR-Sentinel--1%20IW%20SLC-brightgreen.svg)](https://dataspace.copernicus.eu/)
[![Study Area](https://img.shields.io/badge/AOI-Sayung%2C%20Demak%20(64%C3%9764)-red.svg)](#study-area--geodetic-reference)

---

## 📌 Research Overview & Motivation
The coastal plain of Sayung District, Demak Regency experiences catastrophic land subsidence rates exceeding **150 mm/year**, accumulating over **810 mm** of vertical ground settlement between 2021 and 2026. This accelerated deformation has resulted in severe coastal retreat (>2 km shoreline erosion), permanent tidal flooding (*rob*), and critical infrastructure collapse.

This repository provides a reproducible, end-to-end geodetic and deep learning computational pipeline. The workflow converts **124 sequential Sentinel-1 Single Look Complex (SLC)** SAR acquisitions into a spatiotemporal **5D tensor** $(B, T=12, 64, 64, C=3)$ without artificial spatial interpolation over coastal aquaculture ponds. It presents a rigorous benchmark comparing **ConvLSTM**, **CNN-LSTM**, and **Pixel-Wise Linear Regression** across recursive multi-step forecasting horizons ($t+1$, $t+6$, and $t+17$) projecting through year **2030**.

---

## 🛰️ SAR Mission & InSAR Processing Parameters

| Parameter | Configuration / Specification | Technical Description |
| :--- | :--- | :--- |
| **Spaceborne Sensor** | Sentinel-1A / Sentinel-1B | C-band Synthetic Aperture Radar (SAR) ($f = 5.405\text{ GHz}, \lambda \approx 5.55\text{ cm}$) |
| **Acquisition Mode** | Interferometric Wide (IW) Swath | Single Look Complex (SLC), TOPSAR (*Terrain Observation with Progressive Scans*) |
| **Polarization** | **VV (Vertical-Vertical)** | Selected for superior phase stability over terrestrial and built-up land |
| **Flight Direction / Pass** | **Descending** | North-to-South morning overpass geometry |
| **Relative Orbit (Track/Path)**| **Track 40** (Descending) | Constant Line-of-Sight (LOS) look angle across the 2021–2026 baseline |
| **Subswath & Burst Coverage** | **IW2** (Targeted Burst Extraction) | Precise burst extraction covering Semarang, Sayung, and Demak coastal plains |
| **Temporal Coverage** | **January 2021 – April 2026** | **124 SLC acquisitions** sampled at a nominal 12-day revisit interval |
| **Orbital Ephemerides** | **AUX_POEORB** | ESA Precise Orbit Determination (POD) to eliminate orbital phase ramps |
| **Reference DEM** | **SRTM 30 m (1 arc-second)** | Used for topographic phase cancellation and geometric coregistration |
| **Spatial Baseline Constraint**| $B_\perp \le 100\text{ m}$ | Minimizes geometric decorrelation across interferometric pairs |
| **Temporal Baseline Constraint**| $\Delta t \le 48\text{ days}$ | Mitigates temporal decorrelation over dynamic intertidal and vegetated land |
| **Phase Filter & Unwrapping** | Adaptive Goldstein + **SNAPHU** | Unwrapped with a spatial coherence threshold $\gamma \ge 0.35$ |
| **Time-Series Inversion** | Weighted Least Squares (**WLS**) | Implemented in MintPy for robust displacement phase and velocity estimation |

---

## 📍 Study Area & Geodetic Reference

- **Geographic Bounding Matrix (Sayung Grid AOI)**:
  - North: `-6.859°`
  - South: `-6.987°`
  - West: `110.478°`
  - East: `110.566°`
  - Spatial Matrix: **$64 \times 64$ grid cells** ($\approx 100\text{ m}$ ground pixel resolution)
- **Regional Processing Bounding Box (InSAR Frame)**:
  - North: `-6.800°`, South: `-7.100°`, West: `110.100°`, East: `110.650°` (extended south into stable Semarang bedrock).
- **Geodetic Zero-Displacement Reference Station**:
  - **Station Name**: Lempongsari Bedrock GNSS Benchmark (Southern Semarang hills)
  - **Coordinates**: `Latitude: -6.992° S`, `Longitude: 110.412° E`
  - **Geological Setting**: Stable volcanic formation validated against regional geodetic networks.
  - **Mean Calibration Offset**: `-0.00488247 m`

---

## 🧠 GeoAI Architecture & Formulation

### 1. 5D Spatiotemporal Tensor Construction
The InSAR time-series is structured into a 5-dimensional tensor representation:
$$\mathbf{X} \in \mathbb{R}^{B \times T \times H \times W \times C} \implies (B, 12, 64, 64, 3)$$
- **$B$**: Batch size (4 sequences per parameter update).
- **$T = 12$**: Historical lookback sequence length (12 SAR timesteps $\approx 144$ days).
- **$H \times W = 64 \times 64$**: Spatial coordinates of Sayung District.
- **$C = 3$ Physical Channels**:
  - `Channel 0`: Cumulative vertical LOS ground displacement (mm).
  - `Channel 1`: Temporal interferometric coherence ($\gamma \in [0, 1]$).
  - `Channel 2`: Topographic elevation from SRTM DEM (meters).

### 2. Zero-Interpolation Coastal Masking & Masked Loss
Coastal Sayung is heavily fragmented by inundated aquaculture ponds and marine boundaries where radar coherence is lost. Instead of artificial spatial interpolation (which introduces artificial deformation artifacts), invalid pixels are retained as masked cells. Network optimization is conducted exclusively using a **Masked Mean Squared Error (MSE)** loss:
$$\mathcal{L}_{\text{masked}} = \frac{\sum_{i=1}^{H}\sum_{j=1}^{W} M_{i,j} \cdot (y_{i,j} - \hat{y}_{i,j})^2}{\sum_{i=1}^{H}\sum_{j=1}^{W} M_{i,j} + \epsilon}$$
where $M_{i,j} \in \{0, 1\}$ denotes the binary terrestrial validity mask ($\gamma \ge 0.35$).

### 3. Chronological Dataset Partitioning with Double Buffers
To prevent temporal data leakage and target autocorrelation:
- **Training Subset**: January 2021 – December 2024
- *Buffer 1*: January 2025 (1-month transition buffer excluded from training)
- **Validation Subset**: February 2025 – July 2025
- *Buffer 2*: August 2025 (1-month transition buffer excluded from training)
- **Testing Subset**: September 2025 – April 2026

---

## 📊 Benchmark Results (IEEE ICARES Table I)

Open-loop autoregressive recursive forecasting evaluated over three operational lead times:
- **$t+1$**: 12 days ahead (immediate single-step)
- **$t+6$**: 2.5 months / 72 days ahead (seasonal horizon)
- **$t+17$**: 7 months / 204 days ahead (long-term multi-step rollout)

| Lead Time ($t$) | Model Architecture | MAE (mm) | RMSE (mm) | Hazard Recall* | $R^2$ Score | Operational Characteristic |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **$t+1$** (12 Days) | Linear Regression (OLS) | 32.54 | 39.77 | 0.70 | 0.86 | Independent pixel-wise secular trend |
| | **ConvLSTM (Tanh)** | **25.59** | **29.37** | **0.95** | **0.92** | **Highest spatial fidelity & boundary sharpness** |
| | CNN-LSTM Hybrid | 60.69 | 71.62 | 0.87 | 0.54 | Spatial flattening impairs local context |
| **$t+6$** (2.5 Mos) | Linear Regression (OLS) | 31.48 | 39.38 | 0.77 | 0.88 | Constant slope, progressive Recall gain |
| | **ConvLSTM (Tanh)** | **31.14** | **35.73** | **0.97** | **0.90** | **Optimal seasonal projection fidelity** |
| | CNN-LSTM Hybrid | 75.39 | 87.59 | 0.89 | 0.41 | Compounding recursive residual drift |
| **$t+17$** (7 Mos) | **Linear Regression (OLS)** | **27.32** | **40.07** | **0.90** | **0.89** | **Best long-term secular stability** |
| | ConvLSTM (Tanh) | 51.30 | 59.88 | **0.95** | 0.75 | Edge smoothing & variance contraction |
| | CNN-LSTM Hybrid | 96.51 | 112.43 | 0.91 | 0.14 | *Severe autoregressive divergence* |

*\*Hazard Recall is calculated using the empirical median subsidence threshold ($-440.03\text{ mm}$) optimized against the highest validation F1-score.*

---

## 🔍 Critical Technical Insights & Failure Mode Analysis

1. **ConvLSTM Spatiotemporal Supremacy ($t+1$ to $t+6$)**:
   - Stacked ConvLSTM layers preserve local spatial connectivity by performing convolutions directly within the recurrent state transitions, yielding $R^2 = 0.92$ at $t+1$ and $R^2 = 0.90$ at $t+6$.
2. **Linear Regression Long-Term Robustness ($t+17$)**:
   - Over multi-month horizons, secular clay compaction in Demak is dominated by constant groundwater head decline. Linear OLS achieves $R^2 = 0.89$ at $t+17$, although its lower initial recall ($0.70$) fails to capture emerging local subsidence bowls.
3. **Instability of Batch Normalization & Unbounded Activations**:
   - Incorporating `BatchNormalization` and `ReLU` inside ConvLSTM cells triggered numerical divergence during recursive rollout ($R^2$ collapsed to $-11.8087$, $\text{MAE} = 367.16\text{ mm}$). Stability was achieved using **pure `tanh` activations** and **`sigmoid` recurrent gates** without normalization layers.
4. **Failure Modes of CNN-LSTM**:
   - Flattening 2D convolutional feature maps into 1D vectors prior to feeding the LSTM severed spatial coordinates, precipitating severe spatial drift during recursive inference.
5. **Recursive Edge Smoothing on Long Horizons (2030 Projections)**:
   - Autoregressive feeding leads to spatial smoothing near coastal boundaries by 2030 (e.g., Surodadi Village projected at only 47.7 mm settlement despite severe historical trends), defining the boundary conditions for recursive GeoAI models.

---

## 🗂️ Repository Directory Structure

```text
.
├── notebooks/
│   ├── 01_data_acquisition_and_burst_splitting.ipynb
│   ├── 02_insar_stack_processing.ipynb
│   ├── 03_timeseries_inversion_and_tensor_formatting.ipynb
│   └── 04_geoai_spatiotemporal_modeling.ipynb
├── data/                             # Raw & processed tensors (excluded via .gitignore)
├── geoai_models/                     # Model weights & evaluation metric summaries
├── .gitignore
├── requirements.txt
├── LICENSE                           # Apache License 2.0
└── README.md
```

### Notebook Modules
- **`01_data_acquisition_and_burst_splitting.ipynb`**: Automated Sentinel-1 catalog querying, descending Track 40 filtering, `AUX_POEORB` ephemeris application, and IW2 burst slicing.
- **`02_insar_stack_processing.ipynb`**: Coregistration via geometric back-geocoding with Enhanced Spectral Diversity (ESD), SBAS baseline graph creation, Goldstein filtering, and SNAPHU phase unwrapping.
- **`03_timeseries_inversion_and_tensor_formatting.ipynb`**: WLS time-series inversion, Lempongsari GNSS datum referencing, and 5D tensor generation $(B, 12, 64, 64, 3)$.
- **`04_geoai_spatiotemporal_modeling.ipynb`**: Masked MSE training, recursive multi-horizon benchmarking ($t+1, t+6, t+17$), residual error mapping, and year-2030 projection modeling.

---

## 🚀 Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)rasxelz/land-subsidence-sayung-geoai.git
cd sayung-insar-geoai-projection
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📄 Citation & Attribution

If this codebase, workflow, or dataset formatting contributes to your research, please cite:

```bibtex
@inproceedings{khu2026benchmarking,
  title={Benchmarking Deep Spatiotemporal Models for Land Subsidence Projection Using SBAS-InSAR in Sayung District, Demak, Indonesia},
  author={Khu, Rasxelz Cornelius and Irwansyah, Edy},
  booktitle={Proceedings of the IEEE International Conference on Aerospace Electronics and Remote Sensing Technology (ICARES)},
  year={2026},
  address={Indonesia}
}
```

---

## ⚖️ License

Distributed under the **Apache License 2.0**. Refer to the [LICENSE](LICENSE) file for complete rights, permissions, and patent clauses.