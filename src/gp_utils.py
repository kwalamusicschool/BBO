import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def fit_gp(X_train, y_train, alpha=1e-6, length_scale=1.0, normalize_y=True):
    """Fit a Gaussian Process with an RBF kernel."""
    kernel = RBF(length_scale=length_scale)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=normalize_y,
        random_state=42
    )
    gp.fit(X_train, y_train)
    return gp

def predict_gp(gp, X_test):
    """Predict mean and std from a trained GP."""
    return gp.predict(X_test, return_std=True)
