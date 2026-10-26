#  Bagged regression forest built from DecisionTree.jl trees with our own
#  bootstrap loop, so that in-bag counts, OOB predictions and leaf ids are
#  available.

# --- deterministic seeding -------------------------------------------------

@inline function _splitmix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

# seed of the j-th derived stream (j = 0 returns `seed` itself)
_derive_seed(seed::UInt64, j::Integer) = j == 0 ? seed : _splitmix64(seed ⊻ _splitmix64(UInt64(j)))

function _resolve_seed(seed, rng)
    seed === nothing && return rand(rng, UInt64)
    return UInt64(seed % UInt64)
end

# --- flat trees --------------------------------------------------------------

"""
    FlatTree

A regression tree flattened into arrays (pre-order).  Node `k` is a leaf iff
`feature[k] == 0`; then `left[k]` holds the leaf index (`1:n_leaves`) and
`leaf_value[leaf]` the prediction.  Internal nodes send `x` left when
`x[feature] < threshold` (DecisionTree.jl's rule).
"""
struct FlatTree
    feature::Vector{Int32}
    threshold::Vector{Float64}
    left::Vector{Int32}
    right::Vector{Int32}
    leaf_value::Vector{Float64}
end

n_leaves(t::FlatTree) = length(t.leaf_value)

function _flatten(root)
    feature = Int32[]; threshold = Float64[]; left = Int32[]; right = Int32[]
    leaf_value = Float64[]
    function rec(node)
        k = length(feature) + 1
        push!(feature, 0); push!(threshold, 0.0); push!(left, 0); push!(right, 0)
        if node isa DecisionTree.Leaf
            push!(leaf_value, node.majority)
            left[k] = length(leaf_value)
        else
            feature[k] = node.featid
            threshold[k] = node.featval
            l = rec(node.left)
            r = rec(node.right)
            left[k] = l; right[k] = r
        end
        return k
    end
    rec(root isa DecisionTree.Root ? root.node : root)
    return FlatTree(feature, threshold, left, right, leaf_value)
end

@inline function _leaf_index(t::FlatTree, X::AbstractMatrix, i::Int)
    node = 1
    @inbounds while true
        f = t.feature[node]
        f == 0 && return Int(t.left[node])
        node = X[i, f] < t.threshold[node] ? Int(t.left[node]) : Int(t.right[node])
    end
end

"""
    apply(tree::FlatTree, X) -> Vector{Int}

Leaf index (1-based, within the tree) of every row of `X`.
"""
apply(t::FlatTree, X::AbstractMatrix) = [_leaf_index(t, X, i) for i in 1:size(X, 1)]

StatsAPI.predict(t::FlatTree, X::AbstractMatrix) = [t.leaf_value[_leaf_index(t, X, i)] for i in 1:size(X, 1)]

# --- the forest ---------------------------------------------------------------

"""
    BaggedForest

Bagged regression forest.  Fields: `trees::Vector{FlatTree}`, `inbag` (n × n_trees
bootstrap multiplicities, `Int32`), `leaves` (n × n_trees training-row leaf
indices), the training data `X`, `y` (stored by reference), the settings
`n_trees, mtry, nodesize` and the master `seed`.

Build with [`fit_forest`](@ref); `fit(T, forest)` for the bias-correcting
estimators `T` re-uses it.
"""
struct BaggedForest
    trees::Vector{FlatTree}
    inbag::Matrix{Int32}
    leaves::Matrix{Int32}
    X::Matrix{Float64}
    y::Vector{Float64}
    n_trees::Int
    mtry::Int
    nodesize::Int
    seed::UInt64
end

Base.length(f::BaggedForest) = f.n_trees
nobs(f::BaggedForest) = length(f.y)
Base.show(io::IO, f::BaggedForest) =
    print(io, "BaggedForest(n=", length(f.y), ", p=", size(f.X, 2), ", n_trees=", f.n_trees,
          ", mtry=", f.mtry, ", nodesize=", f.nodesize, ")")

default_mtry(p::Integer) = max(p ÷ 3, 1)

"""
    fit_forest(X, y; n_trees=500, mtry=max(p÷3, 1), nodesize=5, seed=nothing,
               rng=Random.default_rng(), threaded=true) -> BaggedForest

Random forest regression mirroring R's `randomForest` defaults: bootstrap of `n`
draws with replacement, `mtry` candidate features per split, and trees grown
until nodes hold at most `nodesize` draws.

DecisionTree.jl semantics: a node is split only if it has at least
`min_samples_split` draws and each child has at least `min_samples_leaf` draws.
R's rule "a node with `<= nodesize` draws is terminal" is therefore
`min_samples_split = nodesize + 1`, `min_samples_leaf = 1`.  Bootstrap
duplicates are kept as separate rows (as in R), so counts are in draws.
`mtry` is re-drawn at every split (DecisionTree.jl `n_subfeatures`).

Every tree gets its own RNG derived from `seed` (or one drawn from `rng`), so the
result is identical for any number of Julia threads.
"""
function fit_forest(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                    n_trees::Integer=500, mtry::Integer=default_mtry(size(X, 2)),
                    nodesize::Integer=5, seed::Union{Nothing,Integer}=nothing,
                    rng::Random.AbstractRNG=Random.default_rng(), threaded::Bool=true)
    Xm = X isa Matrix{Float64} ? X : Matrix{Float64}(X)
    yv = y isa Vector{Float64} ? y : Vector{Float64}(y)
    n, p = size(Xm)
    length(yv) == n || throw(DimensionMismatch("X has $n rows but y has $(length(yv)) entries"))
    n_trees >= 1 || throw(ArgumentError("n_trees must be >= 1"))
    1 <= mtry <= p || throw(ArgumentError("mtry must be in 1:$p"))
    nodesize >= 1 || throw(ArgumentError("nodesize must be >= 1"))
    s = _resolve_seed(seed, rng)

    trees = Vector{FlatTree}(undef, n_trees)
    inbag = zeros(Int32, n, n_trees)
    build = function (t::Int)
        trng = Random.Xoshiro(_derive_seed(s, t))
        idx = rand(trng, 1:n, n)
        @inbounds for i in idx
            inbag[i, t] += 1
        end
        tree = DecisionTree.build_tree(yv[idx], Xm[idx, :], mtry, -1, 1, nodesize + 1, 0.0;
                                       rng=trng, impurity_importance=false)
        trees[t] = _flatten(tree)
    end
    if threaded && Threads.nthreads() > 1
        Threads.@threads for t in 1:n_trees
            build(t)
        end
    else
        for t in 1:n_trees
            build(t)
        end
    end
    forest = BaggedForest(trees, inbag, Matrix{Int32}(undef, n, n_trees), Xm, yv,
                          Int(n_trees), Int(mtry), Int(nodesize), s)
    _fill_leaves!(forest.leaves, forest.trees, Xm, threaded)
    return forest
end

function _fill_leaves!(L::Matrix{Int32}, trees::Vector{FlatTree}, X::AbstractMatrix, threaded::Bool=true)
    n = size(X, 1)
    work = function (t::Int)
        tr = trees[t]
        @inbounds for i in 1:n
            L[i, t] = _leaf_index(tr, X, i)
        end
    end
    if threaded && Threads.nthreads() > 1
        Threads.@threads for t in eachindex(trees)
            work(t)
        end
    else
        for t in eachindex(trees)
            work(t)
        end
    end
    return L
end

"""
    leaf_matrix(forest, X) -> Matrix{Int32}

`n × n_trees` matrix of leaf indices (within each tree) for the rows of `X`.
"""
function leaf_matrix(f::BaggedForest, X::AbstractMatrix{<:Real})
    size(X, 2) == size(f.X, 2) || throw(DimensionMismatch("X must have $(size(f.X, 2)) columns"))
    Xm = X isa Matrix{Float64} ? X : Matrix{Float64}(X)
    return _fill_leaves!(Matrix{Int32}(undef, size(Xm, 1), f.n_trees), f.trees, Xm)
end

"""
    predict(forest::BaggedForest, X) -> Vector{Float64}

Ordinary forest prediction (average over all trees).
"""
function StatsAPI.predict(f::BaggedForest, X::AbstractMatrix{<:Real})
    size(X, 2) == size(f.X, 2) || throw(DimensionMismatch("X must have $(size(f.X, 2)) columns"))
    Xm = X isa Matrix{Float64} ? X : Matrix{Float64}(X)
    n = size(Xm, 1)
    out = zeros(n)
    T = f.n_trees
    # parallelise over row blocks (each task owns its rows -> deterministic)
    nblk = min(max(Threads.nthreads(), 1) * 2, max(n, 1))
    blocks = collect(Iterators.partition(1:n, cld(max(n, 1), nblk)))
    Threads.@threads for b in eachindex(blocks)
        for i in blocks[b]
            s = 0.0
            @inbounds for t in 1:T
                tr = f.trees[t]
                s += tr.leaf_value[_leaf_index(tr, Xm, i)]
            end
            out[i] = s / T
        end
    end
    return out
end

"""
    n_oob(forest) -> Vector{Int}

Number of trees for which each training row is out-of-bag.
"""
n_oob(f::BaggedForest) = [count(==(0), @view f.inbag[i, :]) for i in 1:nobs(f)]

"""
    oob_prediction(forest; warn=true) -> Vector{Float64}

Out-of-bag prediction of each training row (mean over the trees that did not see
it).  Rows that are in-bag for every tree fall back to the full-forest
prediction, with a warning.
"""
function oob_prediction(f::BaggedForest; warn::Bool=true)
    n = nobs(f)
    s = zeros(n); c = zeros(Int, n)
    @inbounds for t in 1:f.n_trees
        lv = f.trees[t].leaf_value
        for i in 1:n
            if f.inbag[i, t] == 0
                s[i] += lv[f.leaves[i, t]]
                c[i] += 1
            end
        end
    end
    miss = findall(==(0), c)
    out = s ./ max.(c, 1)
    if !isempty(miss)
        warn && @warn "$(length(miss)) training row(s) were never out-of-bag; using the full-forest prediction for them (increase n_trees)."
        full = predict(f, f.X[miss, :])
        out[miss] .= full
    end
    return out
end
