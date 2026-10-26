#  Univariate bias correction: regress y on the OOB prediction (linear,
#  quadratic, cubic or smoothing spline) and apply the map to forest predictions.

const SMOOTHING_METHODS = (:linear, :quadratic, :cubic, :spline)

"""
    SmoothedForest

Forest whose predictions are post-processed by a smooth map `g` fitted to
`(oob_prediction, y)`.  Fields: `forest`, `method`, the fitted map
(`coef` for polynomial methods, `spline` for `:spline`) and the corrected OOB
predictions `g(yhat_oob)`.
"""
struct SmoothedForest
    forest::BaggedForest
    method::Symbol
    degree::Int
    center::Float64
    scale::Float64
    coef::Vector{Float64}                       # polynomial coefficients in z = (u-center)/scale
    spline::Union{Nothing,SmoothSpline}
    oob_corrected::Vector{Float64}
end

Base.show(io::IO, s::SmoothedForest) = print(io, "SmoothedForest(method=:", s.method, ")")

function _poly_design(z::AbstractVector, degree::Int)
    A = Matrix{Float64}(undef, length(z), degree + 1)
    for d in 0:degree
        @. A[:, d + 1] = z^d
    end
    return A
end

_apply_map(s::SmoothedForest, u::AbstractVector{<:Real}) =
    s.method === :spline ? predict(s.spline, u) :
    _poly_design((u .- s.center) ./ s.scale, s.degree) * s.coef

"""
    fit(SmoothedForest, X, y; method=:linear, n_trees=500, mtry, nodesize=5, seed, rng)
    fit(SmoothedForest, forest::BaggedForest; method=:linear)

Fit `y ~ g(ŷ_oob)` by least squares (`:linear`, `:quadratic`, `:cubic`; paper's
De, Qu, Cu) or with the R-compatible smoothing spline (`:spline`, Sp).  Predictions are
`g(forest prediction)`.  The second form re-uses a fitted base forest.
"""
function StatsAPI.fit(::Type{SmoothedForest}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                      method::Symbol=:linear, kwargs...)
    forest = fit_forest(X, y; kwargs...)
    return StatsAPI.fit(SmoothedForest, forest; method)
end

function StatsAPI.fit(::Type{SmoothedForest}, forest::BaggedForest; method::Symbol=:linear)
    method in SMOOTHING_METHODS || throw(ArgumentError("method must be one of $SMOOTHING_METHODS"))
    u = oob_prediction(forest)
    y = forest.y
    if method === :spline
        sp = smooth_spline(u, y)
        return SmoothedForest(forest, method, 0, 0.0, 1.0, Float64[], sp, predict(sp, u))
    end
    degree = method === :linear ? 1 : method === :quadratic ? 2 : 3
    c = mean(u); sc = std(u)
    sc = sc > 0 ? sc : 1.0
    A = _poly_design((u .- c) ./ sc, degree)
    coef = A \ y
    s = SmoothedForest(forest, method, degree, c, sc, coef, nothing, Float64[])
    return SmoothedForest(forest, method, degree, c, sc, coef, nothing, _apply_map(s, u))
end

StatsAPI.predict(s::SmoothedForest, X::AbstractMatrix{<:Real}) = _apply_map(s, predict(s.forest, X))
StatsAPI.nobs(s::SmoothedForest) = nobs(s.forest)
oob_prediction(s::SmoothedForest) = s.oob_corrected
