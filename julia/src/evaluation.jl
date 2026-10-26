#  Comparison of all methods: single fits, repeated splits, simulation studies.

"""
    METHOD_LABELS

Method labels in paper order: `"Un","De","Qu","Cu","Sp","Bf","Bf(1)","Bf(2)","Bf(3)","Cond"`.
"""
const METHOD_LABELS = ["Un", "De", "Qu", "Cu", "Sp", "Bf", "Bf(1)", "Bf(2)", "Bf(3)", "Cond"]

function _labels(max_boost::Integer)
    return vcat(["Un", "De", "Qu", "Cu", "Sp"], [paper_label(k) for k in 1:(max_boost + 1)], ["Cond"])
end

# n_test × n_methods matrix of predictions (columns follow `_labels(max_boost)`).
function _all_methods_matrix(Xtrain, ytrain, Xtest; n_trees, mtry, nodesize, max_boost, seed, threaded)
    base = fit_forest(Xtrain, ytrain; n_trees, mtry, nodesize, seed, threaded)
    cols = Vector{Vector{Float64}}()
    push!(cols, predict(base, Xtest))
    for m in (:linear, :quadratic, :cubic, :spline)
        sm = StatsAPI.fit(SmoothedForest, base; method=m)
        push!(cols, predict(sm, Xtest))
    end
    bf = StatsAPI.fit(BoostedForest, base; n_iterations=max_boost + 1, seed=_derive_seed(base.seed, 1), threaded)
    st = staged_predict(bf, Xtest)
    for k in 2:size(st, 2)
        push!(cols, st[:, k])
    end
    cond = StatsAPI.fit(ConditionalBiasForest, base)
    push!(cols, predict(cond, Xtest))
    return reduce(hcat, cols)
end

"""
    predict_all_methods(Xtrain, ytrain, Xtest; n_trees=500, mtry, nodesize=5, max_boost=3,
                        seed=nothing, rng) -> Dict{String,Vector{Float64}}

Fit every method on the training data (one shared base forest for Un, De, Qu, Cu, Sp,
Cond; a boosted chain for Bf, Bf(1), ..., Bf(`max_boost`)) and return the test-set
predictions keyed by label (see [`METHOD_LABELS`](@ref) for the order).
De/Qu/Cu/Sp are the linear/quadratic/cubic/spline univariate corrections.
"""
function predict_all_methods(Xtrain::AbstractMatrix{<:Real}, ytrain::AbstractVector{<:Real},
                             Xtest::AbstractMatrix{<:Real};
                             n_trees::Integer=500, mtry::Integer=default_mtry(size(Xtrain, 2)),
                             nodesize::Integer=5, max_boost::Integer=3,
                             seed::Union{Nothing,Integer}=nothing,
                             rng::Random.AbstractRNG=Random.default_rng(), threaded::Bool=true)
    s = _resolve_seed(seed, rng)
    P = _all_methods_matrix(Xtrain, ytrain, Xtest; n_trees, mtry, nodesize, max_boost, seed=s, threaded)
    return Dict{String,Vector{Float64}}(l => P[:, j] for (j, l) in enumerate(_labels(max_boost)))
end

_mspe(P::AbstractMatrix, y::AbstractVector) = [mean(abs2, y .- view(P, :, j)) for j in 1:size(P, 2)]

# Run `f(rep)` for rep in 1:n, in parallel when threads are available.
function _parallel_map(f, n::Int)
    out = Vector{Any}(undef, n)
    if Threads.nthreads() > 1 && n > 1
        Threads.@threads for r in 1:n
            out[r] = f(r)
        end
    else
        for r in 1:n
            out[r] = f(r)
        end
    end
    return out
end

"""
    simulation_study(name; n=200, n_test=2000, sigma=nothing, n_reps=200, n_trees=500,
                     mtry, nodesize=5, max_boost=3, seed=nothing)

Monte-Carlo comparison of all methods on simulation design `name` (see [`simulate`](@ref)).
The test covariates (`n_test` rows) are drawn once and held fixed; each replication
draws a fresh training set and fresh test noise.  Returns a `NamedTuple`

* `labels`     method labels,
* `mspe_mean`, `mspe_sd`  mean / SD over replications of the test MSPE (vs the noisy test response),
* `bias2`      `mean_i (mean_k ŷ_k(x_i) − f(x_i))²` (squared bias, same fits as the MSPE),
* `mspe`       raw `n_reps × n_methods` matrix of MSPEs,

plus the inputs (`name, n, n_test, sigma, n_reps`).  Replications run on separate
threads, each with its own seeded RNG, so results do not depend on the thread count.
"""
function simulation_study(name; n::Integer=200, n_test::Integer=2000, sigma::Union{Nothing,Real}=nothing,
                          n_reps::Integer=200, n_trees::Integer=500, mtry::Union{Nothing,Integer}=nothing,
                          nodesize::Integer=5, max_boost::Integer=3, seed::Union{Nothing,Integer}=nothing,
                          rng::Random.AbstractRNG=Random.default_rng())
    p = simulation_nfeatures(name)
    sig = sigma === nothing ? simulation_sigma(name) : Float64(sigma)
    mt = mtry === nothing ? default_mtry(p) : Int(mtry)
    s = _resolve_seed(seed, rng)
    Xtest = rand(Random.Xoshiro(_derive_seed(s, 0)), n_test, p)
    ftest = true_mean(name, Xtest)
    labels = _labels(max_boost)
    inner_threaded = n_reps < Threads.nthreads()
    res = _parallel_map(n_reps) do r
        rs = _derive_seed(s, r)
        r_rng = Random.Xoshiro(rs)
        Xtr, ytr, _ = simulate(name, n; sigma=sig, rng=r_rng)
        ytest = ftest .+ sig .* randn(r_rng, n_test)
        P = _all_methods_matrix(Xtr, ytr, Xtest; n_trees, mtry=mt, nodesize, max_boost,
                                seed=_derive_seed(rs, 1), threaded=inner_threaded)
        (_mspe(P, ytest), P)
    end
    M = reduce(vcat, (r[1]' for r in res))
    Pbar = sum(r[2] for r in res) ./ n_reps
    bias2 = [mean(abs2, view(Pbar, :, j) .- ftest) for j in 1:size(Pbar, 2)]
    return (; labels, mspe_mean=vec(mean(M; dims=1)), mspe_sd=vec(std(M; dims=1)), bias2, mspe=M,
            name=String(name), n=Int(n), n_test=Int(n_test), sigma=sig, n_reps=Int(n_reps))
end

function _split_indices(rng, n::Int, test_size::Real)
    # ceil matches scikit-learn's ShuffleSplit, keeping test sizes identical across languages
    nt = clamp(ceil(Int, n * test_size), 1, n - 1)
    perm = Random.randperm(rng, n)
    return perm[(nt + 1):end], perm[1:nt]
end

"""
    repeated_splits(X, y; n_splits=100, test_size=1/3, n_trees=500, mtry, nodesize=5,
                    max_boost=3, seed=nothing)

Real-data comparison: `n_splits` random train/test splits, all methods fitted on each
training part.  Returns a `NamedTuple` `(labels, mspe_mean, mspe_sd, mspe, n_splits)`
(`mspe` is `n_splits × n_methods`).  No squared bias is available without the true
regression function.
"""
function repeated_splits(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; n_splits::Integer=100,
                         test_size::Real=1 / 3, n_trees::Integer=500,
                         mtry::Union{Nothing,Integer}=nothing, nodesize::Integer=5,
                         max_boost::Integer=3, seed::Union{Nothing,Integer}=nothing,
                         rng::Random.AbstractRNG=Random.default_rng())
    Xm = Matrix{Float64}(X); yv = Vector{Float64}(y)
    mt = mtry === nothing ? default_mtry(size(Xm, 2)) : Int(mtry)
    s = _resolve_seed(seed, rng)
    inner_threaded = n_splits < Threads.nthreads()
    res = _parallel_map(n_splits) do r
        rs = _derive_seed(s, r)
        tr, te = _split_indices(Random.Xoshiro(rs), size(Xm, 1), test_size)
        P = _all_methods_matrix(Xm[tr, :], yv[tr], Xm[te, :]; n_trees, mtry=mt, nodesize, max_boost,
                                seed=_derive_seed(rs, 1), threaded=inner_threaded)
        _mspe(P, yv[te])
    end
    M = reduce(vcat, (r' for r in res))
    return (; labels=_labels(max_boost), mspe_mean=vec(mean(M; dims=1)), mspe_sd=vec(std(M; dims=1)),
            mspe=M, n_splits=Int(n_splits))
end

"""
    select_n_iterations(X, y; max_iterations=6, n_splits=20, test_size=1/3, n_trees=500,
                        mtry, nodesize=5, seed=nothing)

Paper's stopping rule.  On `n_splits` random splits, fit a `BoostedForest` with
`max_iterations` residual forests and evaluate the test MSPE of every staged
prediction `k = 0, ..., max_iterations`.  The selected `k` is the first one whose mean
MSPE is lower than that at `k+1` (i.e. stop when the mean MSPE first increases), or
`max_iterations` if it never increases.  Returns `(n_iterations, mean_mspe, sd_mspe)`
with the curves indexed by `k+1`.
"""
function select_n_iterations(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                             max_iterations::Integer=6, n_splits::Integer=20, test_size::Real=1 / 3,
                             n_trees::Integer=500, mtry::Union{Nothing,Integer}=nothing,
                             nodesize::Integer=5, seed::Union{Nothing,Integer}=nothing,
                             rng::Random.AbstractRNG=Random.default_rng())
    Xm = Matrix{Float64}(X); yv = Vector{Float64}(y)
    mt = mtry === nothing ? default_mtry(size(Xm, 2)) : Int(mtry)
    s = _resolve_seed(seed, rng)
    inner_threaded = n_splits < Threads.nthreads()
    res = _parallel_map(n_splits) do r
        rs = _derive_seed(s, r)
        tr, te = _split_indices(Random.Xoshiro(rs), size(Xm, 1), test_size)
        bf = StatsAPI.fit(BoostedForest, Xm[tr, :], yv[tr]; n_iterations=max_iterations, n_trees,
                          mtry=mt, nodesize, seed=_derive_seed(rs, 1), threaded=inner_threaded)
        _mspe(staged_predict(bf, Xm[te, :]), yv[te])
    end
    M = reduce(vcat, (r' for r in res))
    m = vec(mean(M; dims=1))
    sd = n_splits > 1 ? vec(std(M; dims=1)) : zeros(length(m))
    k = max_iterations
    for j in 1:max_iterations
        if m[j] < m[j + 1]
            k = j - 1
            break
        end
    end
    return (; n_iterations=k, mean_mspe=m, sd_mspe=sd, labels=[paper_label(j) for j in 0:max_iterations])
end
