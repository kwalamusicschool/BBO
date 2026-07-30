import numpy as np

def y_weighted_centroid(X_train, y_train, top_k=3):
    """Compute Y-weighted centroid of the top-k points."""
    sorted_idx = np.argsort(y_train)[::-1]
    top_idx = sorted_idx[:top_k]
    top_X = X_train[top_idx]
    top_y = y_train[top_idx]
    if np.min(top_y) < 0:
        weights = top_y - np.min(top_y) + 0.001
    else:
        weights = top_y
    centroid = np.average(top_X, axis=0, weights=weights)
    return np.round(centroid, 6)

def tiny_perturbation(best_point, step=0.001):
    """Generate a tiny perturbation around the best point."""
    dim = len(best_point)
    pert = np.random.uniform(-step, step, size=dim)
    return np.clip(best_point + pert, 0, 1)
