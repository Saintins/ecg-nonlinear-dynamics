import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def calculate_recurrence_matrix(phase_matrix, eps_threshold=None, eps_percentage=0.1):
    """
    Build a binary recurrence matrix for a given phase space.

    :param phase_matrix: Phase space trajectory matrix of shape (N, m)
    :param eps_threshold: Absolute distance threshold (epsilon). If None, computed automatically.
    :param eps_percentage: Percentage of the maximum distance in phase space (used if eps_threshold=None).
    :return: recurrence_matrix (binary matrix), eps_threshold (used threshold)
    """
    # Compute the distance matrix between all pairs of phase-space points
    # cdist efficiently computes Euclidean distances for each pair (i, j)
    distances = cdist(phase_matrix, phase_matrix, metric='euclidean')

    # If threshold not provided, take a percentage of the maximum distance in the attractor
    if eps_threshold is None:
        max_dist = np.max(distances)
        eps_threshold = max_dist * eps_percentage

    # Build binary matrix: 1 if distance <= eps (black dot), else 0 (white)
    recurrence_matrix = np.where(distances <= eps_threshold, 1, 0)

    return recurrence_matrix, eps_threshold

