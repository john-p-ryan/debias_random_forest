#  Breiman's iterated (boosted) bias correction: each stage fits a fresh forest
#  to the OOB residuals of the previous stages.

"""
    BoostedForest

Residual-boosted forest (paper's `Bf` / `Bf(k)`).

* `stages[1]` is the base forest fitted to `y`; stage `j+1` is a forest fitted to
  the OOB residual `r⁽ʲ⁾ = r⁽ʲ⁻¹⁾ − oobpred⁽ʲ⁾` (with `r⁽⁰⁾ = y`).
* `n_iterations` = number of residual forests after the base forest that are used
  (0 → `Un`, 1 → `Bf`, k+1 → `Bf(k)`); `length(stages) == n_iterations + 1`.
* `predict` = sum of the stage predictions; `oob_prediction` = sum of the stage
  OOB predictions.
* `oob_mse[k+1]` = OOB MSE of `y` against the cumulative OOB prediction of stages
  `1:k+1` (so `oob_mse[1]` is the base forest's OOB MSE).  With
  `stopping = :oob` it is recorded for every stage that was grown, including one
  that triggered the stop (and was then discarded), see [`fit`](@ref).
"""
struct BoostedForest
    stages::Vector{BaggedForest}
    n_iterations::Int
    stopping::Symbol
    oob_mse::Vector{Float64}
    oob_stage_predictions::Matrix{Float64}   # n × (n_iterations+1), per-stage OOB preds
end

Base.show(io::IO, b::BoostedForest) =
    print(io, "BoostedForest(n_iterations=", b.n_iterations, ", stopping=:", b.stopping,
          ", n_trees=", b.stages[1].n_trees, ")")

"""
    paper_label(k)

Paper notation for `k` residual iterations: 0 → "Un", 1 → "Bf", k → "Bf(k-1)".
"""
paper_label(k::Integer) = k < 0 ? throw(ArgumentError("k must be >= 0")) :
                          k == 0 ? "Un" : k == 1 ? "Bf" : "Bf($(k - 1))"

"""
    fit(BoostedForest, X, y; n_iterations=1, stopping=:fixed, n_trees=500, mtry, nodesize=5,
        seed=nothing, rng=default_rng())
    fit(BoostedForest, forest::BaggedForest; n_iterations=1, stopping=:fixed, seed=nothing, ...)

Fit the boosted forest.  The second form re-uses a pre-fitted base forest (which
stores its own `X`, `y`) as stage 1; residual forests copy its `n_trees`, `mtry`
and `nodesize`.

`stopping = :fixed` fits exactly `n_iterations` residual forests.
`stopping = :oob` treats `n_iterations` as a maximum: stages are added while the
OOB MSE does not exceed `1.1 ×` the minimum so far, then the model is truncated to the stage
with the smallest OOB MSE; the number used is `n_iterations` of the result.

Stage `j` (j ≥ 1) is seeded by a seed derived from `seed`, the base forest by `seed`
itself, so `fit(BoostedForest, X, y; n_iterations=0, seed=s)` equals
`fit_forest(X, y; seed=s)`.
"""
function StatsAPI.fit(::Type{BoostedForest}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                      n_iterations::Integer=1, stopping::Symbol=:fixed,
                      n_trees::Integer=500, mtry::Integer=default_mtry(size(X, 2)),
                      nodesize::Integer=5, seed::Union{Nothing,Integer}=nothing,
                      rng::Random.AbstractRNG=Random.default_rng(), threaded::Bool=true)
    s = _resolve_seed(seed, rng)
    base = fit_forest(X, y; n_trees, mtry, nodesize, seed=s, threaded)
    return StatsAPI.fit(BoostedForest, base; n_iterations, stopping, seed=s, threaded)
end

function StatsAPI.fit(::Type{BoostedForest}, base::BaggedForest;
                      n_iterations::Integer=1, stopping::Symbol=:fixed,
                      seed::Union{Nothing,Integer}=nothing,
                      rng::Random.AbstractRNG=Random.default_rng(), threaded::Bool=true)
    n_iterations >= 0 || throw(ArgumentError("n_iterations must be >= 0"))
    stopping in (:fixed, :oob) || throw(ArgumentError("stopping must be :fixed or :oob"))
    s = seed === nothing ? base.seed : _resolve_seed(seed, rng)
    X, y = base.X, base.y
    n = length(y)

    stages = BaggedForest[base]
    oobp = Vector{Vector{Float64}}()
    push!(oobp, oob_prediction(base))
    cum = copy(oobp[1])
    mse = Float64[mean(abs2, y .- cum)]
    resid = y .- cum
    best = 1; best_mse = mse[1]
    for j in 1:n_iterations
        f = fit_forest(X, copy(resid); n_trees=base.n_trees, mtry=base.mtry, nodesize=base.nodesize,
                       seed=_derive_seed(s, j), threaded)
        o = oob_prediction(f)
        push!(stages, f); push!(oobp, o)
        cum .+= o
        resid .-= o
        push!(mse, mean(abs2, y .- cum))
        if mse[end] < best_mse
            best = length(mse); best_mse = mse[end]
        end
        if stopping === :oob && mse[end] > 1.1 * best_mse
            break
        end
    end
    keep = stopping === :oob ? best : length(stages)
    stages = stages[1:keep]
    O = Matrix{Float64}(undef, n, keep)
    for k in 1:keep
        O[:, k] = oobp[k]
    end
    return BoostedForest(stages, keep - 1, stopping, mse, O)
end

StatsAPI.nobs(b::BoostedForest) = nobs(b.stages[1])

"""
    staged_predict(model::BoostedForest, X) -> Matrix{Float64}

Cumulative predictions for `k = 0, …, n_iterations` residual iterations: column
`k+1` is the prediction of `Bf(k-1)` in paper notation (column 1 = `Un`).
"""
function staged_predict(b::BoostedForest, X::AbstractMatrix{<:Real})
    out = Matrix{Float64}(undef, size(X, 1), length(b.stages))
    acc = zeros(size(X, 1))
    for (k, f) in enumerate(b.stages)
        acc .+= predict(f, X)
        out[:, k] = acc
    end
    return out
end

StatsAPI.predict(b::BoostedForest, X::AbstractMatrix{<:Real}) = staged_predict(b, X)[:, end]

"""
    oob_prediction(model)

Bias-corrected out-of-bag predictions of the training rows (for `BoostedForest`:
the sum of the stage OOB predictions).
"""
oob_prediction(b::BoostedForest) = vec(sum(b.oob_stage_predictions; dims=2))

"""
    oob_staged_prediction(model::BoostedForest) -> Matrix

Cumulative OOB predictions, one column per stage (columns `k+1` = after `k` iterations).
"""
oob_staged_prediction(b::BoostedForest) = cumsum(b.oob_stage_predictions; dims=2)
