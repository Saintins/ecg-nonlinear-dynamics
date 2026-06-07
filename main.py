import os
import numpy as np
import wfdb

from core_processing import compute_hermite_derivative, process_derivative_squares, detect_r_peaks
from recurrence import calculate_recurrence_matrix
from hidden_params import (
    calculate_average_mutual_information,
    find_optimal_delay_tau,
    calculate_false_nearest_neighbors,
    find_optimal_embedding_dimension_m,
    reconstruct_phase_space_takens
)
from plot_ecg import (
    plot_patient_ecg, 
    plot_RPeaks, 
    plot_square_ecg,
    plot_new_diagnostics,
    plot_attractor_3d,
    plot_recurrence
)


def run_analysis(patient_id, base_dir="data"):
    record_path = os.path.join(base_dir, str(patient_id))
    
    # 1. Load signal and detect peaks
    signals, fields = wfdb.rdsamp(record_path)
    raw_signal = signals[:, 0]
    fs = fields["fs"]
    
    derived_signal = compute_hermite_derivative(raw_signal)
    optimized_squares = process_derivative_squares(derived_signal)
    detected_peaks = detect_r_peaks(optimized_squares, raw_signal, fs=fs)
    
    # 2. R-R intervals series
    rr_intervals = np.diff(detected_peaks) / fs * 1000
    
    print(f"\n=== АНАЛІЗ ХАОТИЧНОЇ ДИНАМІКИ ДЛЯ ПАЦІЄНТА №{patient_id} ===")
    print(f"Довжина ряду: {len(rr_intervals)} відліків")
    plot_patient_ecg(patient_id)
    plot_square_ecg(patient_id)
    plot_RPeaks(patient_id)

    # 3. Determine TAU via Average Mutual Information (AMI)
    max_lag = 40
    ami_values = calculate_average_mutual_information(rr_intervals, max_lag=max_lag)
    tau = find_optimal_delay_tau(ami_values)
    print(f"Параметр затримки (tau) = {tau}")
    
    # 4. Determine embedding dimension m via False Nearest Neighbors (FNN)
    max_test_m = 10
    fnn_threshold = 2.0  # cutoff threshold
    r_thresh = 15.0      # neighbor divergence allowance
    
    fnn_pcts = calculate_false_nearest_neighbors(rr_intervals, tau, max_dim=max_test_m, r_thresh=r_thresh)
    optimal_m = find_optimal_embedding_dimension_m(fnn_pcts, fnn_threshold)
    
    # Handle case if FNN did not drop below threshold (fallback)
    if optimal_m is None:
        valid_fnn = [val for val in fnn_pcts if not np.isnan(val)]
        if valid_fnn:
            optimal_m = np.argmin(valid_fnn) + 1
            print(f"   * Попередження: FNN не опустився нижче {fnn_threshold}%. Обрано мінімум: m={optimal_m}")
        else:
            optimal_m = 3
            print("   * Помилка розрахунку FNN. Примусово встановлено m=3.")
    
    print(f"Розмірність простору (m) = {optimal_m}")
    
    # 5. Diagnostics visualization (AMI and FNN)
    plot_new_diagnostics(ami_values, tau, fnn_pcts, optimal_m, patient_id)
    
    # 6. Reconstruct phase space
    phase_space = reconstruct_phase_space_takens(rr_intervals, tau, max(3, optimal_m))
    
    if phase_space is not None:
        plot_attractor_3d(phase_space, patient_id)
                
        # 15% (0.15)
        rec_matrix, eps_used = calculate_recurrence_matrix(phase_space, eps_percentage=0.15)
        plot_recurrence(rec_matrix, eps_used, optimal_m, patient_id)


if __name__ == "__main__":
    
    run_analysis(patient_id=106)
    run_analysis(patient_id=113)
    run_analysis(patient_id=122)
    run_analysis(patient_id=208)
    run_analysis(patient_id=230)
    
    

  