import numpy as np
from scipy.stats import norm

def ucb(mean, std, kappa=2.5):
    """Upper Confidence Bound acquisition function."""
    return mean + kappa * std

def expected_improvement(mean, std, f_best, xi=0.01):
    """Expected Improvement acquisition function."""
    improvement = mean - f_best - xi
    Z = improvement / (std + 1e-9)
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
    ei[std < 1e-9] = 0
    return ei
