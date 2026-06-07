
import numpy as np
from sklearn.metrics import mutual_info_score
import warnings


# --- Utilities for Takens' Theorem ---
def calculate_average_mutual_information(
    signal: np.ndarray, max_lag: int
) -> np.ndarray:
    """
    Calculates the average mutual information (AMI) for a time series
    for different time lags up to max_lag.
    """
    if signal.ndim > 1:
        signal = signal.flatten() # Ensure 1D signal
    
    # Ensure max_lag is feasible
    if len(signal) <= max_lag:
        max_lag = max(1, len(signal) - 2) # Max possible lag for at least 2 points in series_1/2
    
    if max_lag <= 0:
        warnings.warn("Signal too short or max_lag non-positive. Cannot calculate AMI. Returning empty array.")
        return np.array([])

    ami_values = []
    
    # Determine bins for discretization. Freedman-Diaconis is often a good start.
    try:
        bins = np.histogram_bin_edges(signal, bins='fd')
        if len(bins) < 3: # 'fd' might give too few bins for (nearly) constant signals
            bins = np.histogram_bin_edges(signal, bins='auto')
        if len(bins) < 3: # If 'auto' also fails, use a fixed number of bins as a fallback
            bins = np.linspace(np.min(signal), np.max(signal), 20) # 20 bins as a general fallback
    except Exception: 
        bins = np.linspace(np.min(signal), np.max(signal), 20)

    for lag in range(1, max_lag + 1):
        series_1 = signal[:-lag]
        series_2 = signal[lag:]

        if len(series_1) == 0: # Should be caught by max_lag adjustment, but as a safeguard
            break 

        # Digitize the series based on the calculated bins
        digitized_1 = np.digitize(series_1, bins)
        digitized_2 = np.digitize(series_2, bins)

        # mutual_info_score requires >1 unique labels. If all data falls in one bin, AMI is effectively 0.
        if len(np.unique(digitized_1)) <= 1 or len(np.unique(digitized_2)) <= 1:
            ami_values.append(0.0)
        else:
            try:
              ami_values.append(mutual_info_score(digitized_1, digitized_2))
            except Exception:
              # This can occur if sklearn's contingency matrix has issues.
              ami_values.append(0.0) # Treat as 0 AMI if calculation fails for the lag
              
    return np.array(ami_values)

def find_optimal_delay_tau(ami_values: np.ndarray) -> int:
    """
    Determines the optimal time delay (tau) from AMI values.
    The typical heuristic is the first local minimum of the AMI curve.
    """
    if len(ami_values) == 0:
        warnings.warn("AMI values array is empty. Defaulting tau to 1.")
        return 1 

    # Look for first local minimum: ami[i-1] > ami[i] < ami[i+1]
    # ami_values[k] corresponds to lag = k+1
    if len(ami_values) > 2:
        # local_minima_indices will store indices 'idx' such that ami_values[idx] is a local minimum.
        # We search in the slice ami_values[1:-1].
        # If ami_values[k] (where k is an index in original ami_values) is a local minimum,
        # its index in the slice ami_values[1:-1] is (k-1).
        local_minima_slice_indices = np.where(
            (ami_values[:-2] > ami_values[1:-1]) & (ami_values[1:-1] < ami_values[2:])
        )[0]
        
        if len(local_minima_slice_indices) > 0:
            # Smallest index in the slice corresponds to the first local minimum.
            first_min_slice_idx = local_minima_slice_indices[0]
            # Convert slice index back to original ami_values index: original_idx = slice_idx + 1
            # Lag = original_idx + 1
            optimal_tau = (first_min_slice_idx + 1) + 1
            return optimal_tau
    
    # Fallback: if no local minimum found or AMI array is too short, use the global minimum.
    if len(ami_values) > 0:
        # np.argmin gives index in ami_values. Lag = index + 1.
        optimal_tau = np.argmin(ami_values) + 1
        warnings.warn(f"No clear first local minimum in AMI. Using global minimum at lag {optimal_tau}.")
        return optimal_tau
        
    warnings.warn("Could not determine tau from AMI. Defaulting tau to 1.")
    return 1 # Default tau if all else fails

def calculate_false_nearest_neighbors(
    signal: np.ndarray, tau: int, max_dim: int, r_thresh: float
) -> list[float]:
    """
    Calculates the percentage of false nearest neighbors (FNN) for different
    embedding dimensions (m), from 1 to max_dim.
    This implementation follows the standard FNN algorithm structure.
    """
    N_total = len(signal)
    fnn_percentages = []

    for m_dim in range(1, max_dim + 1):
        # Number of points for which the (m+1)-th component can be constructed
        # This is len(signal) - m_dim * tau
        # These points form m_dim-dimensional vectors Y_i = (s_i, s_{i+tau}, ..., s_{i+(m_dim-1)tau})
        # and their (m_dim+1)-th component is s_{i+m_dim*tau}
        
        num_points_for_m_plus_1_check = N_total - m_dim * tau
        
        if num_points_for_m_plus_1_check <= 1: # Need at least two points to find a neighbor and compare
            # If m_dim is such that not enough points are left, append NaN or stop.
            # NaN indicates FNN could not be computed for this dimension.
            fnn_percentages.append(np.nan) 
            warnings.warn(f"Signal too short for m_dim={m_dim}, tau={tau} to compute FNN. Appending NaN.")
            continue

        # Construct m_dim-dimensional delay vectors Y
        # Y_m_vectors[i] = [signal[i], signal[i+tau], ..., signal[i+(m_dim-1)*tau]]
        Y_m_vectors = np.array([
            signal[i : i + m_dim * tau : tau] for i in range(num_points_for_m_plus_1_check)
        ])

        if Y_m_vectors.shape[0] <= 1: # Should be caught by num_points_for_m_plus_1_check <= 1 already
            fnn_percentages.append(np.nan)
            continue
            
        # Calculate pairwise distances in m_dim-dimensional space
        dists_m = np.linalg.norm(Y_m_vectors[:, None, :] - Y_m_vectors[None, :, :], axis=2)
        np.fill_diagonal(dists_m, np.inf) # Exclude self-matches (distance to self is 0)
        
        if dists_m.shape[1] == 0: # Only one point in Y_m_vectors after all
            fnn_percentages.append(np.nan)
            continue

        # Find nearest neighbor indices and distances in m_dim space
        nearest_neighbor_indices_m = np.argmin(dists_m, axis=1)
        dist_to_neighbor_m = np.linalg.norm(Y_m_vectors - Y_m_vectors[nearest_neighbor_indices_m], axis=1)

        # Get the (m_dim+1)-th components for all points and their neighbors
        # z_components are signal[i + m_dim*tau]
        # z_neighbor_components are signal[nearest_neighbor_indices_m[i] + m_dim*tau]
        
        # Indices for the original points that form Y_m_vectors
        original_indices_for_Y = np.arange(num_points_for_m_plus_1_check)
        
        z_components = signal[original_indices_for_Y + m_dim * tau]
        z_neighbor_components = signal[nearest_neighbor_indices_m + m_dim * tau]
        
        # Calculate the increase in distance if we add the (m_dim+1)-th component
        # This is |z_component - z_neighbor_component|
        dist_increase_m_plus_1 = np.abs(z_components - z_neighbor_components)
        
        # Calculate FNN ratio R_i(m_dim) = dist_increase_m_plus_1 / dist_to_neighbor_m
        # Handle division by zero: if dist_to_neighbor_m is zero (identical points in m-dim space):
        #  - If dist_increase_m_plus_1 is also zero, they remain true neighbors.
        #  - If dist_increase_m_plus_1 is non-zero, they become false neighbors.
        
        is_fnn = np.full(num_points_for_m_plus_1_check, False, dtype=bool)
        
        # Valid distances (non-zero denominator)
        valid_denominator_indices = dist_to_neighbor_m > 1e-12 # Use a small epsilon
        
        if np.any(valid_denominator_indices):
            R_values = np.full(num_points_for_m_plus_1_check, np.inf) # Default to Inf for zero denominators
            R_values[valid_denominator_indices] = \
                dist_increase_m_plus_1[valid_denominator_indices] / dist_to_neighbor_m[valid_denominator_indices]
            is_fnn[valid_denominator_indices] = R_values[valid_denominator_indices] > r_thresh
        
        # Handle zero denominator cases (dist_to_neighbor_m == 0)
        zero_denominator_indices = ~valid_denominator_indices
        if np.any(zero_denominator_indices):
            # If numerator (dist_increase_m_plus_1) is non-zero, it's an FNN
            is_fnn[zero_denominator_indices] = dist_increase_m_plus_1[zero_denominator_indices] > 1e-12 

        fnn_percentage = np.sum(is_fnn) / num_points_for_m_plus_1_check * 100.0
        fnn_percentages.append(fnn_percentage)
        
    return fnn_percentages

def find_optimal_embedding_dimension_m(
    fnn_percentages: list[float], fnn_threshold: float
) -> int | None:
    """
    Determines the optimal embedding dimension (m) from FNN percentages.
    'm' is typically the first dimension where FNN percentage drops below a threshold.
    """
    if not fnn_percentages: 
        warnings.warn("FNN percentages list is empty. Cannot determine m.")
        return None

    # Dimension 'd' corresponds to fnn_percentages[d-1]
    for i, fnn_val in enumerate(fnn_percentages):
        if not np.isnan(fnn_val) and fnn_val < fnn_threshold:
            return i + 1 # Dimension is index + 1

    warnings.warn(f"FNN percentage did not drop below threshold {fnn_threshold}%.")
    # Optionally, one could return the dimension with the minimum FNN if no threshold is crossed.
    # For this refactoring, we stick to None if threshold condition isn't met, as per original's implication.
    return None

def reconstruct_phase_space_takens(
    signal: np.ndarray, tau: int, m: int
) -> np.ndarray | None:
    """
    Reconstructs the phase space using Takens' delay embedding theorem.
    Returns a matrix where each row is a state vector:
    (signal[k+(m-1)tau], signal[k+(m-2)tau], ..., signal[k])
    which is equivalent to (x(t), x(t-tau), ..., x(t-(m-1)tau)) if t corresponds to k+(m-1)tau.
    """
    if m <= 0 :
        warnings.warn(f"Embedding dimension m must be positive, got {m}. Cannot reconstruct.")
        return None
    if tau <= 0:
        warnings.warn(f"Time delay tau must be positive, got {tau}. Cannot reconstruct.")
        return None
        
    min_required_length = (m - 1) * tau + 1 # Smallest index is 0, largest is (m-1)*tau for first vector
    if len(signal) < min_required_length:
        warnings.warn(
            f"Signal too short for m={m} and tau={tau}. "
            f"Need at least {min_required_length} points, got {len(signal)}."
        )
        return None

    # Number of reconstructed points (vectors)
    num_reconstructed_points = len(signal) - (m - 1) * tau
    reconstructed_space = np.zeros((num_reconstructed_points, m))

    for i in range(num_reconstructed_points):
        # The "latest" component of the delay vector for point 'i' in the reconstructed space
        # corresponds to signal[i + (m - 1) * tau]. This is x(t_current).
        current_signal_time_idx = i + (m - 1) * tau
        for j in range(m): # j-th component of the delay vector (0-indexed)
            # reconstructed_space[i, j] = x(t_current - j*tau)
            reconstructed_space[i, j] = signal[current_signal_time_idx - j * tau]
            
    return reconstructed_space

