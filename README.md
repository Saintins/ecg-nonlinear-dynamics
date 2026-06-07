# ECG Nonlinear Dynamics & Chaos Analysis Framework

This repository contains a specialized computational framework written in Python for digital processing, algorithmic localization of R-peaks, and multi-dimensional topological reconstruction of electrocardiogram (ECG) data. The project leverages deterministic chaos theory and nonlinear dynamics to uncover hidden patterns in cardiac rhythms.

The empirical verification of the mathematical models is performed using the raw biological signals from the **MIT-BIH Arrhythmia Database**.

---

## Mathematical Foundations & Core Features

The processing pipeline completely transforms a one-dimensional discrete biological signal into a multidimensional phase space object through 4 sequential analytical steps:

1. **Piecewise-Cubic Hermite Spline Smoothing:** First-order analytical derivatives are calculated using second-order Hermite interpolation matrices to ensure robust smoothing and suppress high-frequency muscle artifacts.
2. **Adaptive R-Peak Detection:** QRS complexes are amplified via non-linear squaring, followed by the application of a dynamic threshold $\Gamma = 4.0 \cdot \mathbb{E}[\mathbf{E}^+]$ and a physiological refractory period filter to isolate exact time-stamps.
3. **Takens' Embedding Optimization:** The scalar time series of RR-intervals is mapped into a higher-dimensional state space $\mathbb{R}^m$. The optimal delay $\tau$ is localized using the first minimum of Average Mutual Information (AMI), while the embedding dimension $m$ is verified through the False Nearest Neighbors (FNN) criterion.
4. **Recurrence Plot Visualisation:** The multidimensional phase space trajectory is projected onto a 2D binary recurrence matrix $\mathbf{R}_{i,j}(\varepsilon) = \Theta(\varepsilon - \|\mathbf{y}_i - \mathbf{y}_j\|)$ to identify structural laminar states.

---

## Repository Structure

The framework is strictly decoupled into 4 modular computational blocks:

* `main.py` — The primary orchestration module that initializes configurations, manages pipeline dataflows, and executes the benchmark suite via the `run_analysis` controller.
* `core_processing.py` — Encapsulates mathematical routines for signal differentiation, vector squaring, and adaptive extreme values localization.
* `hidden_params.py` — Contains statistical information theory tools, including multidimensional histograms for AMI calculation and Kennel's criteria for FNN evaluation.
* `recurrence.py` — Realizes fast vectorized computations of pairwise Euclidean distance matrices and applies Heaviside thresholding to generate binary recurrence matrices.
* `plot_ecg.py` — Handles spatial rendering engines for time-series diagnostics, 3D chaotic strange attractors, and monochrome recurrence maps.

---

## Installation & Requirements

Ensure you have Python 3.8+ installed. Clone the repository and install the dependencies:

```bash
git clone [https://github.com/Saintins/ecg-nonlinear-dynamics.git](https://github.com/Saintins/ecg-nonlinear-dynamics.git)
cd ecg-nonlinear-dynamics
pip install numpy scipy matplotlib wfdb scikit-learn
