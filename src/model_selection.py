import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.base import clone

def loocv_score(model, X, y):
    """Leave-One-Out Cross-Validation score (negative MSE)."""
    n = len(X)
    scores = []
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i].reshape(1, -1)
        y_test = y[i]
        model_copy = clone(model)
        model_copy.fit(X_train, y_train)
        y_pred = model_copy.predict(X_test)[0]
        scores.append(-(y_pred - y_test)**2)
    return np.mean(scores)

def get_models():
    """Return a dictionary of models with default hyperparameters."""
    return {
        'Ridge': Ridge(alpha=1.0),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'GP': GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0),
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
    }
