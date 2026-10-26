#  Conditional bias correction (Lu & Hardin 2021, `forestError`).

"""
    ConditionalBiasForest

For every tree `t` and leaf `L`: `S_t[L] = Σ (ŷ_oob,i − y_i)` and `C_t[L] = #` over the
training rows `i` that are out-of-bag for tree `t` and fall in leaf `L`.  The bias
at `x₀` is `b(x₀) = Σ_t S_t[leaf_t(x₀)] / Σ_t C_t[leaf_t(x₀)]` (0 if the
denominator is 0) and the prediction is `forest(x₀) − b(x₀)`.

`oob_prediction(model)`: row `j` uses `ŷ_oob,j − b₋ⱼ(x_j)`, where `b₋ⱼ` is the same
ratio with row `j`'s own contribution removed (it enters every tree in which it is
OOB, with numerator term `ŷ_oob,j − y_j` and count 1), so the bias estimate for a
training row never uses its own error.
"""
struct ConditionalBiasForest
    forest::BaggedForest
    S::Vector{Vector{Float64}}
    C::Vector{Vector{Float64}}
    oob_base::Vector{Float64}
    oob_corrected::Vector{Float64}
end

Base.show(io::IO, ::ConditionalBiasForest) = print(io, "ConditionalBiasForest()")

"""
    fit(ConditionalBiasForest, X, y; n_trees=500, mtry, nodesize=5, seed, rng)
    fit(ConditionalBiasForest, forest::BaggedForest)
"""
function StatsAPI.fit(::Type{ConditionalBiasForest}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; kwargs...)
    return StatsAPI.fit(ConditionalBiasForest, fit_forest(X, y; kwargs...))
end

function StatsAPI.fit(::Type{ConditionalBiasForest}, forest::BaggedForest)
    n = nobs(forest)
    ob = oob_prediction(forest)
    e = ob .- forest.y
    T = forest.n_trees
    S = [zeros(n_leaves(t)) for t in forest.trees]
    C = [zeros(n_leaves(t)) for t in forest.trees]
    noob = zeros(Int, n)
    for t in 1:T
        St = S[t]; Ct = C[t]
        @inbounds for i in 1:n
            if forest.inbag[i, t] == 0
                l = forest.leaves[i, t]
                St[l] += e[i]; Ct[l] += 1
                noob[i] += 1
            end
        end
    end
    corr = similar(ob)
    @inbounds for j in 1:n
        num = 0.0; den = 0.0
        for t in 1:T
            l = forest.leaves[j, t]
            num += S[t][l]; den += C[t][l]
        end
        num -= noob[j] * e[j]
        den -= noob[j]
        corr[j] = ob[j] - (den > 0 ? num / den : 0.0)
    end
    return ConditionalBiasForest(forest, S, C, ob, corr)
end

"""
    bias_estimate(model::ConditionalBiasForest, X) -> Vector{Float64}

Estimated conditional bias `b(x₀)` for each row of `X`.
"""
function bias_estimate(m::ConditionalBiasForest, X::AbstractMatrix{<:Real})
    L = leaf_matrix(m.forest, X)
    n = size(L, 1)
    b = zeros(n)
    @inbounds for i in 1:n
        num = 0.0; den = 0.0
        for t in 1:m.forest.n_trees
            l = L[i, t]
            num += m.S[t][l]; den += m.C[t][l]
        end
        b[i] = den > 0 ? num / den : 0.0
    end
    return b
end

StatsAPI.predict(m::ConditionalBiasForest, X::AbstractMatrix{<:Real}) =
    predict(m.forest, X) .- bias_estimate(m, X)
StatsAPI.nobs(m::ConditionalBiasForest) = nobs(m.forest)
oob_prediction(m::ConditionalBiasForest) = m.oob_corrected
