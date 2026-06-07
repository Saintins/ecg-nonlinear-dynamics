import os
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from mpl_toolkits.mplot3d import Axes3D


from core_processing import compute_hermite_derivative, process_derivative_squares, detect_r_peaks


def plot_patient_ecg(patient_id, base_dir="data", seconds_to_plot=6):
    """Read and visualize an ECG segment for a single patient.

    :param patient_id: Patient identifier (e.g., 122)
    :param base_dir: Folder containing .hea, .dat, .atr files
    :param seconds_to_plot: Duration of the segment to plot in seconds
    """
    # Create a single plot with the requested size
    fig, ax = plt.subplots(figsize=(14, 3))

    # Build the relative record path without extension (required by wfdb)
    record_path = os.path.join(base_dir, str(patient_id))

    try:
        # 1. Read numeric ECG signal (wfdb parses .hea and .dat automatically)
        signals, fields = wfdb.rdsamp(record_path)
        fs = fields["fs"]  # Sampling frequency (360 Hz for MIT-BIH)

        # Compute how many samples are needed for the requested seconds
        total_points = int(fs * seconds_to_plot)

        # Select the first lead/channel (index 0)
        ecg_signal = signals[:total_points, 0]

        # 2. Read expert R-peak annotations from the .atr file
        annotation = wfdb.rdann(record_path, "atr")
        # Filter reference R-peaks that fall within our time window
        true_peaks = [p for p in annotation.sample if p < total_points]

        # 3. Plot the signal
        ax.plot(
            ecg_signal,
            color="#1f77b4",
            linewidth=1.2,
            label=f"Patient #{patient_id} (Lead 1)",
        )

        # Overlay red markers for the original R-peaks for clarity
        ax.scatter(
            true_peaks,
            ecg_signal[true_peaks],
            color="red",
            marker="o",
            s=40,
            zorder=3,
            label="Reference R-peaks",
        )

        # Chart formatting (grid, axis labels)
        ax.set_title(
            f"ECG Patient #{patient_id} (Segment {seconds_to_plot}s, Fs: {fs} Hz)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Sample index", fontsize=9)
        ax.set_ylabel("Amplitude (mV)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right")

    except Exception as e:
        print(
            f"Error processing patient {patient_id}. Check that .hea, .dat, .atr files exist in '{base_dir}'."
        )
        print(f"Error details: {e}")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_RPeaks(patient_id, base_dir="data", seconds_to_plot=6):
    record_path = os.path.join(base_dir, str(patient_id))

    # 1. Read full patient data
    signals, fields = wfdb.rdsamp(record_path)
    raw_signal = signals[:, 0]  # Use the first channel
    fs = fields["fs"]

    # Read expert annotations
    annotation = wfdb.rdann(record_path, "atr")
    true_peaks = annotation.sample

    # 2. Run through the Krak-Stela mathematical kernel
    derived_signal = compute_hermite_derivative(raw_signal)
    optimized_squares = process_derivative_squares(derived_signal)

    # 3. Find R-peaks using our algorithm
    detected_peaks = detect_r_peaks(optimized_squares, raw_signal, fs=fs)

    # --- Basic accuracy evaluation for initial findings ---
    print(f"\n--- Processing results for patient #{patient_id} ---")
    print(f"Reference R-peaks (experts): {len(true_peaks)}")
    print(f"Detected by algorithm: {len(detected_peaks)}")

    # 4. Visualize a segment to check alignment
    total_points = int(fs * seconds_to_plot)

    plt.figure(figsize=(14, 5))
    plt.plot(
        raw_signal[:total_points],
        color="black",
        linewidth=1,
        label="ECG signal",
    )

    # Highlight reference peaks (large green circles)
    plot_true = [p for p in true_peaks if p < total_points]
    plt.scatter(
        plot_true,
        raw_signal[plot_true],
        color="green",
        edgecolors="black",
        marker="o",
        s=120,
        alpha=0.5,
        label="Reference peaks (PhysioNet)",
    )

    # Highlight detected peaks (red crosses)
    plot_detected = [p for p in detected_peaks if p < total_points]
    plt.scatter(
        plot_detected,
        raw_signal[plot_detected],
        color="red",
        marker="x",
        s=80,
        linewidths=2,
        zorder=4,
        label="Detected (Hermite algorithm)",
    )

    plt.title(
        f"Final R-peak localization check for Patient #{patient_id}",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_square_ecg(patient_id, base_dir="data", seconds_to_plot=6):
    record_path = os.path.join(base_dir, str(patient_id))

    # Read signal
    signals, fields = wfdb.rdsamp(record_path)
    fs = fields["fs"]
    total_points = int(fs * seconds_to_plot)
    raw_signal = signals[:total_points, 0]

    # --- Processing by the Krak-Stela algorithm ---
    # 1. Obtain analytic derivative via Hermite polynomials
    derived_signal = compute_hermite_derivative(raw_signal)

    # 2. Optimize: square ONLY the decreasing segments
    optimized_squares = process_derivative_squares(derived_signal)

    # --- Visualization of results ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Top plot: Original ECG signal
    ax1.plot(raw_signal, color="black", linewidth=1, label="Original ECG")
    ax1.set_title(
        f"Patient #{patient_id}: Input ECG (first {seconds_to_plot}s)"
    )
    ax1.set_ylabel("Amplitude (mV)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Bottom plot: Optimized squared derivatives
    ax2.plot(
        optimized_squares,
        color="darkred",
        linewidth=1.2,
        label="Squared decreasing derivatives",
    )
    ax2.set_title(
        "Mathematical kernel result: optimized squared decreasing derivatives"
    )
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Value ($S'^2$)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_new_diagnostics(ami_values, tau, fnn_pcts, m, patient_id):
    """
    Updated diagnostics visualization function.
    The left plot now shows Average Mutual Information (AMI) instead of the ACF.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Plot Average Mutual Information (AMI) for selecting TAU
    lags = np.arange(1, len(ami_values) + 1)
    ax1.plot(lags, ami_values, color='#2ca02c', linewidth=2, label='Average Mutual Information (AMI)')
    if tau <= len(ami_values):
        ax1.axvline(tau, color='red', linestyle=':', linewidth=2, label=f'Selected $\\tau = {tau}$')
        ax1.scatter(tau, ami_values[tau-1], color='red', s=80, zorder=5)
    
    ax1.set_title(f"Delay ($\\tau$) selection via AMI for #{patient_id}")
    ax1.set_xlabel("Lag")
    ax1.set_ylabel("Average mutual information")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # 2. FNN plot for embedding dimension m
    m_values = list(range(1, len(fnn_pcts) + 1))
    ax2.plot(m_values, fnn_pcts, marker='o', color='crimson', linewidth=2, label='False Nearest Neighbors (FNN)')
    
    if m is not None:
        ax2.axvline(m, color='green', linestyle=':', linewidth=2, label=f'Optimal $m = {m}$')
        ax2.scatter(m, fnn_pcts[m-1], color='green', s=100, zorder=5)
    
    ax2.set_title(f"False Nearest Neighbors (FNN) for #{patient_id}")
    ax2.set_xlabel("Embedding dimension ($m$)")
    ax2.set_ylabel("False neighbors (%)")
    ax2.set_xticks(np.arange(1, len(fnn_pcts) + 1, 1))
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def plot_attractor_3d(phase_matrix, patient_id):
    """Build a 3D projection of the attractor (first 3 coordinates)."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    x = phase_matrix[:, 0]
    y = phase_matrix[:, 1]
    z = phase_matrix[:, 2]
    
    ax.plot(x, y, z, color='#1f77b4', alpha=0.6, linewidth=0.7)
    ax.scatter(x, y, z, c=z, cmap='viridis', s=6, alpha=0.5)
    
    ax.set_title(f"3D Chaotic attractor (Patient #{patient_id})", fontsize=12, fontweight='bold')
    ax.set_xlabel("X(t)")
    ax.set_ylabel("X(t - tau)")
    ax.set_zlabel("X(t - 2*tau)")
    plt.tight_layout()
    plt.show()


def plot_recurrence(recurrence_matrix, eps_threshold, m, patient_id=""):
    """
    Visualize the recurrence matrix as a black-and-white image.
    """
    plt.figure(figsize=(8, 8))
    
    # Plot the matrix. cmap='Greys' maps 1 to black and 0 to white.
    # origin='lower' places (0,0) at the lower-left corner, as is common in mathematics.
    plt.imshow(recurrence_matrix, cmap='Greys', origin='lower')
    
    plt.title(f"Recurrence plot (Patient #{patient_id})\nEmbedding dim $m={m}$, Threshold $\\epsilon={eps_threshold:.2f}$", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Time index $i$", fontsize=12)
    plt.ylabel("Time index $j$", fontsize=12)
    
    plt.tight_layout()
    plt.show()

