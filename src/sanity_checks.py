import numpy as np

def check_boundaries(point, margin=0.001):
    """Check if any dimension is within margin of 0 or 1."""
    for val in point:
        if val < margin or val > 1 - margin:
            return True
    return False

def nearest_neighbour_check(candidate, X_train, y_train, radius=0.05, percentile=25):
    """Check if candidate is too close to a poor output."""
    distances = np.linalg.norm(X_train - candidate, axis=1)
    nearest_idx = np.argmin(distances)
    nearest_dist = distances[nearest_idx]
    nearest_output = y_train[nearest_idx]
    bottom_25 = np.percentile(y_train, percentile)
    if nearest_dist < radius and nearest_output < bottom_25:
        return False
    return True
