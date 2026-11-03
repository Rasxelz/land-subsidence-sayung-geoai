# Benchmarking Deep Spatiotemporal Models for Land Subsidence Projection Using SBAS-InSAR in Sayung District, Demak, Indonesia

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow_2.x-orange.svg)]()
[![InSAR Engine](https://img.shields.io/badge/InSAR-ISCE2%20%7C%20MintPy-purple.svg)]()
[![Satellite Data](https://img.shields.io/badge/SAR-Sentinel--1%20IW%20SLC-brightgreen.svg)](https://search.asf.alaska.edu/)
[![Study Area](https://img.shields.io/badge/AOI-Sayung%2C%20Demak%20(64x64)-red.svg)]()

---

## Research Overview

Land subsidence along the northern coastline of Central Java is facing severe land subsidence and socioeconomic crisis. In Sayung District, Demak Regency, subsidence rates reach up to 150 mm/year, resulting in over 810 mm of cumulative ground displacement between 2021 and 2026. This ongoing settlement has driven coastal retreat of up to 2 km, widespread tidal inundation, and structural damage to residences and the National North Coast Highway.

This repository provides a full pipeline documentation implementing Small Baseline Subset Interferometric Synthetic Aperture Radar (SBAS-InSAR) with spatiotemporal deep learning models. The workflow processes 124 Sentinel-1 Single Look Complex (SLC) scenes acquired from January 2021 to April 2026 into an un-interpolated 5D tensor, benchmarking Convolutional Long Short-Term Memory (ConvLSTM), CNN-LSTM, and Pixel-Wise Linear Regression across multi-horizon recursive projections ($t+1$, $t+6$, and $t+17$) through the year 2030.

---

## SAR Mission and InSAR Processing Parameters

| Parameter | Configuration / Specification | Technical Description |
| :--- | :--- | :--- |
| Acquisition Mode | Interferometric Wide (IW) Swath | Single Look Complex (SLC), TOPSAR acquisition mode |
| Polarization | Dual Polarization (VV + VH) | Co-polar and cross-polar channels for phase stability and surface scattering analysis |
| Flight Direction / Pass | Descending | North-to-South morning orbit geometry |
| Relative Orbit (Track / Path) | Path 76 | Consistent satellite orbital track over the Kendal-Semarang-Demak coastal belt |
| Frame | Frame 613 | Geodetic frame covering coastal Demak and northern Central Java |
| Subswath and Burst Coverage | IW2 (Targeted Burst Extraction) | Spatial cropping isolates bursts 1 to 3 in subswath IW2 |
| Temporal Coverage | January 2021 - April 2026 | 124 SLC acquisitions sampled at a nominal 12-day revisit interval |
| Orbital Ephemerides | AUX_POEORB | ESA Precise Orbit Determination (POD) to remove orbital phase ramps |
| Reference DEM | SRTM 30 m (1 arc-second) | Global SRTMGL1 stitched and converted to ellipsoidal WGS84 coordinates |
| Spatial Baseline Constraint | Bperp <= 100 m | Upper bound on perpendicular orbital baseline to limit spatial decorrelation |
| Temporal Baseline Constraint | Delta t <= 48 days | Revisit interval limit to minimize decorrelation over wetlands and dynamic land cover |
| Phase Filter and Unwrapping | Adaptive Goldstein + SNAPHU | Minimum spatial coherence threshold of 0.35 applied during statistical 2D unwrapping |
| Time-Series Inversion | Weighted Least Squares (WLS) | Implemented via ISCE2 and MintPy |

---

## Study Area and Geodetic Reference

- Spatial Processing Bounds (Kendal, Semarang, and Demak regional processing AOI):
  - North: -6.800 °
  - South: -7.100 °
  - West: 110.100 °
  - East: 110.650 °
- Target Modeling Bounds (Sayung District grid AOI):
  - North: -6.859 °
  - South: -6.987 °
  - West: 110.478 °
  - East: 110.566 °
  - Spatial Matrix: 64 x 64 grid cells (~100 m ground resolution per pixel)
- Geodetic Zero-Displacement Reference Point:
  - Station Name: Lempongsari Benchmark (Southern Semarang volcanic hills)
  - Coordinates: Latitude -6.992 deg S, Longitude 110.412 deg E
  - Geological Context: Stable bedrock formation verified against regional geodetic networks

---

## Spatiotemporal Tensor and GeoAI Methodology

### 1. 5D Spatiotemporal Tensor Construction
The InSAR-derived time-series data is structured into a 5D tensor without spatial interpolation:
$$\mathbf{X} \in \mathbb{R}^{B \times T \times H \times W \times C} \implies (B, 12, 64, 64, 3)$$
- $B$: Batch size of 4 sequences per parameter update.
- $T = 12$: Historical sequence length representing 12 consecutive SAR acquisitions (~144 days).
- $H \times W = 64 \times 64$: Spatial grid encompassing Sayung District.
- $C = 3$ Channels:
  - Channel 0: Cumulative vertical Line-of-Sight (LOS) ground displacement (mm).
  - Channel 1: Temporal interferometric coherence ($\gamma \in [0, 1]$).
  - Channel 2: Topographic elevation from SRTM DEM (meters).

### 2. Zero-Interpolation Masked Objective
Due to extensive inundated aquaculture ponds and coastal wetlands, invalid and decorrelated radar pixels are retained as masked coordinates rather than filled with interpolated values. Network loss is computed strictly over coherent land pixels:
$$\mathcal{L}_{\text{masked}} = \frac{\sum_{i=1}^{H}\sum_{j=1}^{W} M_{i,j} \cdot (y_{i,j} - \hat{y}_{i,j})^2}{\sum_{i=1}^{H}\sum_{j=1}^{W} M_{i,j} + \epsilon}$$
where $M_{i,j} \in \{0, 1\}$ represents the binary land validity mask ($\gamma \ge 0.35$).

### 3. Chronological Partitioning with Transition Buffers
To prevent temporal data leakage and short-term autocorrelation:
- Training Subset: January 2021 to December 2024
- Transition Buffer 1: January 2025 (one-month exclusion buffer)
- Validation Subset: February 2025 to July 2025
- Transition Buffer 2: August 2025 (one-month exclusion buffer)
- Testing Subset: September 2025 to April 2026

---

## Model Benchmark Results

Models were evaluated on the holdout test set using open-loop recursive forecasting, where each output is fed back as input for subsequent steps across three lead times:
- $t+1$: 12 days ahead (single-step immediate horizon)
- $t+6$: 2.5 months / 72 days ahead (seasonal horizon)
- $t+17$: 7 months / 204 days ahead (long-term multi-step horizon)

| Lead Time ($t$) | Model Architecture | MAE (mm) | RMSE (mm) | Hazard Recall* | $R^2$ Score | Operational Evaluation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| t+1 (12 Days) | Linear Regression (OLS) | 32.54 | 39.77 | 0.70 | 0.86 | Independent pixel-wise secular trend |
| | ConvLSTM (Tanh / No-BN) | 25.59 | 29.37 | 0.95 | 0.92 | Optimal spatial fidelity and edge sharpness |
| | CNN-LSTM Hybrid | 60.69 | 71.62 | 0.87 | 0.54 | Feature flattening breaks spatial context |
| t+6 (2.5 Months) | Linear Regression (OLS) | 31.48 | 39.38 | 0.77 | 0.88 | Constant slope, gradual increase in recall |
| | ConvLSTM (Tanh / No-BN) | 31.14 | 35.73 | 0.97 | 0.90 | Accurate retention of seasonal subsidence patterns |
| | CNN-LSTM Hybrid | 75.39 | 87.59 | 0.89 | 0.41 | Compounding recursive residual drift |
| t+17 (7 Months) | Linear Regression (OLS) | 27.32 | 40.07 | 0.90 | 0.89 | Highest stability for secular linear consolidation |
| | ConvLSTM (Tanh / No-BN) | 51.30 | 59.88 | 0.95 | 0.75 | High hazard recall with variance contraction |
| | CNN-LSTM Hybrid | 96.51 | 112.43 | 0.91 | 0.14 | Substantial autoregressive error accumulation |

*Hazard Recall evaluates the identification of severe subsidence exceeding the dataset median threshold of -440.03 mm, tuned dynamically to the highest validation F1-score.

---

## Key Findings and Discussion

1. ConvLSTM Spatiotemporal Retention ($t+1$ to $t+6$):
   - Embedding convolutional operations within recurrent memory cells preserves spatial relationships, achieving an $R^2$ of 0.92 at $t+1$ and 0.90 at $t+6$, with consistent hazard recall scores between 0.95 and 0.97.
2. Secular Trend Stability in Linear Regression ($t+17$):
   - Pixel-wise Ordinary Least Squares regression maintained an $R^2$ of 0.89 at $t+17$. This performance aligns with regional geodetic findings that deep clay compaction in Demak has progressed at a steady rate over the past five years. However, its lower initial recall (0.70 at $t+1$) indicates an inability to detect emerging non-linear acceleration at subsidence boundaries.
3. Numerical Stability in Recurrent Architectures:
   - Experimental configurations incorporating Batch Normalization and unbounded ReLU activations within recurrent cells suffered from numerical instability during open-loop multi-step forecasting ($R^2$ collapsed to -11.8087, MAE rose to 367.16 mm). Restricting activations to bounded hyperbolic tangent (tanh) functions and sigmoid recurrent gates without batch normalization ensured stable convergence.
4. Degradation of CNN-LSTM:
   - Decoupling spatial convolution from temporal recurrence via a 1D flattening step disrupted spatial adjacency, leading to error accumulation during multi-step recursive rollouts ($R^2 = 0.14$ at $t+17$).
5. Edge Smoothing in 2030 Projections:
   - Deploying ConvLSTM recursively through 2030 projects up to 820.3 mm of additional ground settlement across Sriwulan, Purwosari, Bedono, Sayung, and Loireng villages. However, Surodadi Village, located on the northern boundary of the modeling grid, projected only 47.7 mm of cumulative settlement due to convolution padding effects and recursive variance contraction, highlighting a limitation of recursive convolution at spatial boundaries.

---

### Notebook Modules
- `01_data_acquisition_and_burst_splitting.ipynb`: Automated downloading of 124 Sentinel-1 IW SLC scenes covering Path 76, Frame 613, descending orbit, dual polarization (VV + VH), and orbit file retrieval via `fetchOrbit.py`.
- `02_insar_stack_processing.ipynb`: Preparation of 30 m SRTM DEM, input validation, execution of `stackSentinel.py` for subswath IW2 bursts 1 to 3, coregistration via Geometric Back-Geocoding with Enhanced Spectral Diversity (ESD), baseline configuration ($B_\perp \le 100\text{ m}, \Delta t \le 48\text{ days}$), and SNAPHU phase unwrapping.
- `03_timeseries_inversion_and_tensor_formatting.ipynb`: Upstream audit of 245 unwrapped interferograms, perpendicular baseline harmonization via the triangle closure law, compilation of `ifgramStack.h5`, row-streaming SVD least-squares inversion into `timeseries_manual.h5`, phase-to-displacement conversion, and Lempongsari zero-referencing.
- `04_geoai_spatiotemporal_modeling.ipynb`: Construction of the 5D spatiotemporal tensor $(B, 12, 64, 64, 3)$, chronological data splitting with double transition buffers, masked loss formulation, benchmarking of ConvLSTM, CNN-LSTM, and Linear Regression across $t+1$, $t+6$, and $t+17$, and recursive spatial projection through 2030.