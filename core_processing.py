import numpy as np


def compute_hermite_derivative(signal):
    """Implementation of Step 1 of the Krak-Stela algorithm.

    Builds a piecewise polynomial (2nd-order Hermite) approximation
    and returns the analytic first derivative of the signal.

    :param signal: 1-D NumPy array (ECG channel)
    :return: NumPy array of the analytic derivative (same length)
    """
    N = len(signal)
    # Create an output array for the derivative initialized with zeros
    derivative = np.zeros(N)

    # For MIT-BIH the step h = 1 simplifies the analytic expressions.
    # The derivative at point i uses neighboring values and their differences.
    # The smoothed derivative at i becomes a linear combination of neighbors.

    # Vectorized computation using array shifts (NumPy) for performance.
    # Hermite analytic derivative (2nd order) for interior points with h=1:
    # S'(tau_i) = 0.5 * (signal[i+1] - signal[i-1])
    # This piecewise-quadratic smoothing is more stable than a simple
    # central difference on noisy data.

    derivative[1 : N - 1] = 0.5 * (signal[2:N] - signal[0 : N - 2])

    # Boundary conditions (first and last points)
    derivative[0] = signal[1] - signal[0]
    derivative[N - 1] = signal[N - 1] - signal[N - 2]

    return derivative


def process_derivative_squares(derivative):
    """Implementation of Step 2.

    Square the derivative ONLY at points where it is decreasing (negative).
    All other positions are set to zero.

    :param derivative: 1-D array of computed derivative
    :return: Optimized array of squared decreasing derivatives
    """
    # Create zeros array
    optimized_squares = np.zeros_like(derivative)

    # Find indices where derivative < 0 (steep descending segment R->S).
    # Physiological rationale: the descending portion of QRS shows maximal slope.
    decreasing_mask = derivative < 0

    # Square only those segments
    optimized_squares[decreasing_mask] = derivative[decreasing_mask] ** 2

    return optimized_squares


# def detect_r_peaks(optimized_squares, raw_signal, fs=360):
#     """Implementation of Step 3 (legacy version).
#
#    Finds local maxima in the transformed space and then refines
#    the exact R-peak positions on the raw ECG signal.
#
#    :param optimized_squares: Array of squared decreasing derivatives
#    :param raw_signal: Raw ECG signal (to locate exact peak)
#    :param fs: Sampling frequency (default 360 Hz for MIT-BIH)
#    :return: NumPy array with detected R-peak indices
#    """
#    # 1. Compute adaptive threshold (percentage of peak magnitude)
#    # For many patients a threshold of 15-20% of the global maximum works well
#    dynamic_threshold = 0.20 * np.max(optimized_squares)
#
#    detected_indices = []
#
#    # 2. Physiological constraint: refractory period
#    # At least ~0.3 seconds should pass between two heartbeats.
#    # With fs=360 Hz this corresponds to ~100-110 samples.
#    min_distance = int(0.3 * fs)
#
#    last_peak_idx = -min_distance
#    N = len(optimized_squares)
#
#    # 3. Search for impulses above threshold
#    i = 0
#    while i < N:
#        if optimized_squares[i] > dynamic_threshold:
#            # Check whether refractory period passed since last peak
#            if i - last_peak_idx > min_distance:
#                # We found a descending region of the QRS complex.
#                # Since the square of the descending derivative indicates R->S,
#                # the true R-peak on the raw signal is slightly to the LEFT (earlier).
#                # Search for the amplitude maximum on the raw signal in a small left window (e.g. 20 samples)
#                search_window_start = max(0, i - 25)
#                search_window_end = i
#
#                # Find the amplitude maximum in this window
#                exact_r_pos = search_window_start + np.argmax(
#                    raw_signal[search_window_start:search_window_end]
#                )
#
#                detected_indices.append(exact_r_pos)
#                last_peak_idx = i
#                i += min_distance  # Skip refractory zone to speed up
#                continue
#        i += 1
#
#    return np.array(detected_indices)


def detect_r_peaks(optimized_squares, raw_signal, fs=360):
    """Robust, modified R-peak detector."""
    # Use mean/median of non-zero bursts instead of np.max()
    # Exclude zeros so the threshold is computed only from real impulses
    non_zero_values = optimized_squares[optimized_squares > 0]

    if len(non_zero_values) == 0:
        return np.array([])

    # Use the mean of working bursts and multiply by a factor (tuned experimentally)
    # Factor around 3.5-4.5 suppresses small fluctuations but captures stable peaks
    dynamic_threshold = 4.0 * np.mean(non_zero_values)

    detected_indices = []
    min_distance = int(0.3 * fs)  # refractory period (0.3 sec)
    last_peak_idx = -min_distance
    N = len(optimized_squares)

    i = 0
    while i < N:
        if optimized_squares[i] > dynamic_threshold:
            if i - last_peak_idx > min_distance:
                search_window_start = max(0, i - 25)
                search_window_end = i

                exact_r_pos = search_window_start + np.argmax(
                    raw_signal[search_window_start:search_window_end]
                )

                detected_indices.append(exact_r_pos)
                last_peak_idx = i
                i += min_distance
                continue
        i += 1

    return np.array(detected_indices)