"""debiasforest: bias correction for random forest regression.

Implements the methods of Chen, Withana Gamage & Ryan (2022), "Debias Random Forest
Regression Predictors", J. Stat. Research 56(2):115-131, on top of scikit-learn forests.
"""
from ._forest import BaggedEnsemble, DefaultForestRegressor, default_forest
from ._oob import brute_force_oob_predictions, oob_predictions
from .boosted import BoostedForestRegressor, paper_label
from .conditional import ConditionalBiasForestRegressor
from .datasets import DatasetUnavailableError, load_yacht, simulate
from .evaluation import (predict_all_methods, repeated_splits, select_n_iterations,
                         simulation_study)
from .plotting import bias_plot
from .smoothing import SmoothedForestRegressor, SmoothingSpline, smooth_spline

__version__ = "0.1.0"

__all__ = [
    "BaggedEnsemble", "DefaultForestRegressor", "default_forest", "oob_predictions",
    "brute_force_oob_predictions", "BoostedForestRegressor", "paper_label",
    "ConditionalBiasForestRegressor", "SmoothedForestRegressor", "SmoothingSpline",
    "smooth_spline", "simulate", "load_yacht", "DatasetUnavailableError",
    "predict_all_methods", "select_n_iterations", "simulation_study", "repeated_splits",
    "bias_plot",
]
