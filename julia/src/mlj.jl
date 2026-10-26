#  MLJModelInterface models (Deterministic, Table(Continuous) -> Continuous target).
#  No dependency on MLJ/MLJBase itself: with MLJ loaded, `@load BoostedForestRegressor
#  pkg=DebiasForest` / `using DebiasForest` makes the models available.

import MLJModelInterface as MMI
import Tables

const _MLJ_COMMON_DOC = """
Hyper-parameters: `n_trees=500`, `mtry=0` (0 means `max(p÷3, 1)`), `nodesize=5`,
`seed=-1` (negative: draw a random seed from the global RNG).
"""

MMI.@mlj_model mutable struct BoostedForestRegressor <: MMI.Deterministic
    n_trees::Int = 500::(_ > 0)
    mtry::Int = 0::(_ >= 0)
    nodesize::Int = 5::(_ > 0)
    n_iterations::Int = 1::(_ >= 0)
    stopping::Symbol = :fixed::(_ in (:fixed, :oob))
    seed::Int = (-1)::(true)
end

MMI.@mlj_model mutable struct SmoothedForestRegressor <: MMI.Deterministic
    n_trees::Int = 500::(_ > 0)
    mtry::Int = 0::(_ >= 0)
    nodesize::Int = 5::(_ > 0)
    method::Symbol = :linear::(_ in (:linear, :quadratic, :cubic, :spline))
    seed::Int = (-1)::(true)
end

MMI.@mlj_model mutable struct ConditionalBiasForestRegressor <: MMI.Deterministic
    n_trees::Int = 500::(_ > 0)
    mtry::Int = 0::(_ >= 0)
    nodesize::Int = 5::(_ > 0)
    seed::Int = (-1)::(true)
end

const _MLJ_MODELS = Union{BoostedForestRegressor,SmoothedForestRegressor,ConditionalBiasForestRegressor}

_forest_kwargs(m::_MLJ_MODELS, p::Int) =
    (; n_trees=m.n_trees, mtry=m.mtry == 0 ? default_mtry(p) : min(m.mtry, p), nodesize=m.nodesize,
       seed=m.seed < 0 ? nothing : m.seed)

function MMI.fit(m::BoostedForestRegressor, verbosity::Int, X, y)
    Xm = Tables.matrix(X)
    fr = StatsAPI.fit(BoostedForest, Xm, collect(Float64, y); n_iterations=m.n_iterations,
                      stopping=m.stopping, _forest_kwargs(m, size(Xm, 2))...)
    return fr, nothing, (; n_iterations_used=fr.n_iterations, oob_mse=fr.oob_mse)
end

function MMI.fit(m::SmoothedForestRegressor, verbosity::Int, X, y)
    Xm = Tables.matrix(X)
    fr = StatsAPI.fit(SmoothedForest, Xm, collect(Float64, y); method=m.method, _forest_kwargs(m, size(Xm, 2))...)
    return fr, nothing, NamedTuple()
end

function MMI.fit(m::ConditionalBiasForestRegressor, verbosity::Int, X, y)
    Xm = Tables.matrix(X)
    fr = StatsAPI.fit(ConditionalBiasForest, Xm, collect(Float64, y); _forest_kwargs(m, size(Xm, 2))...)
    return fr, nothing, NamedTuple()
end

MMI.predict(::_MLJ_MODELS, fitresult, Xnew) = predict(fitresult, Tables.matrix(Xnew))

for (M, name) in ((BoostedForestRegressor, "BoostedForestRegressor"),
                  (SmoothedForestRegressor, "SmoothedForestRegressor"),
                  (ConditionalBiasForestRegressor, "ConditionalBiasForestRegressor"))
    MMI.metadata_model(M;
        input_scitype=MMI.Table(MMI.Continuous),
        target_scitype=AbstractVector{MMI.Continuous},
        supports_weights=false,
        load_path="DebiasForest.$name")
end

MMI.metadata_pkg.((BoostedForestRegressor, SmoothedForestRegressor, ConditionalBiasForestRegressor);
    name="DebiasForest", uuid="a57183db-4835-43ad-9d27-816e70dd1c96",
    julia=true, license="MIT", is_wrapper=false)

MMI.docstring(::Type{BoostedForestRegressor}) =
    "Boosted (iterated OOB-residual) random forest; `n_iterations=1` is the paper's `Bf`. " * _MLJ_COMMON_DOC
MMI.docstring(::Type{SmoothedForestRegressor}) =
    "Random forest with a univariate (linear/quadratic/cubic/spline) bias correction of its predictions. " * _MLJ_COMMON_DOC
MMI.docstring(::Type{ConditionalBiasForestRegressor}) =
    "Random forest with the conditional (leaf-wise OOB) bias correction of Lu & Hardin. " * _MLJ_COMMON_DOC
