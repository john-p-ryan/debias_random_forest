"""
    DebiasForest

Bias correction for random-forest regression (Chen, Withana Gamage & Ryan, 2022,
*J. Stat. Research* 56(2):115-131): boosted OOB-residual forests (`Bf`),
univariate smoothing of the forest predictions (`De/Qu/Cu/Sp`) and the conditional
bias correction (`Cond`), built on DecisionTree.jl trees with our own bagging loop.
"""
module DebiasForest

using Random
using Statistics
using LinearAlgebra
using Downloads
import DecisionTree
import StatsAPI
import StatsAPI: fit, predict, nobs

include("spline.jl")
include("bagging.jl")
include("boosted.jl")
include("smoothing.jl")
include("conditional.jl")
include("datasets.jl")
include("evaluation.jl")
include("mlj.jl")

export BaggedForest, BoostedForest, SmoothedForest, ConditionalBiasForest, SmoothSpline,
       BoostedForestRegressor, SmoothedForestRegressor, ConditionalBiasForestRegressor,
       fit, predict, nobs, fit_forest, staged_predict, oob_prediction, oob_staged_prediction,
       leaf_matrix, bias_estimate, smooth_spline, paper_label, METHOD_LABELS,
       predict_all_methods, select_n_iterations, simulation_study, repeated_splits,
       simulate, simulation_names, simulation_function, true_mean, load_yacht,
       DatasetUnavailableError

end
