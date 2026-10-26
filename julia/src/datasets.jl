#  Simulation designs of Chen, Withana Gamage & Ryan (2022), pp. 122-123, and
#  the UCI yacht hydrodynamics data.

# name => (n_features, default sigma, f(x::AbstractVector))
const _SIM_FUNCTIONS = Dict{String,Tuple{Int,Float64,Function}}(
    # Fig. 1
    "fig1" => (4, 0.5, x -> 3x[1] + x[2]^2 + sin(x[3]) - sqrt(x[4])),
    # Function 1 (Friedman 1)
    "f1" => (5, 1.0, x -> 10sin(pi * x[1] * x[2]) + 20(x[3] - 0.5)^2 + 10x[4] + 5x[5]),
    # Function 2 (as printed in the paper: 3 x2 term)
    "f2" => (10, 1.0, x -> 0.1exp(4x[1]) + 4 / (1 + exp(-20x[2] + 10)) + 3x[2] + 2x[4] + x[5]),
    # Function 3 (linear; x6..x10 are noise variables)
    "f3" => (10, 1.0, x -> 10(0.878x[1] + 0.642x[2] + 0.766x[3] + 0.666x[4] + 0.933x[5])),
    # Function 4 (polynomial)
    "f4" => (5, 1.0, x -> 10x[1] + 0.5x[2]^2 + 15x[2] - 25sqrt(x[3]) + 0.05x[4]),
    # Function 5 (non-linear)
    "f5" => (5, 1.0, x -> 10cos(pi * x[1] * x[3]) + 10x[3]^2 + 10(x[1] + x[4])),
)

_canon_name(name::Union{AbstractString,Symbol}) = begin
    s = lowercase(String(name))
    s == "friedman1" ? "f1" : s
end

"""
    simulation_names()

Names accepted by [`simulate`](@ref): "fig1", "f1" (alias "friedman1"), "f2", "f3", "f4", "f5".
"""
simulation_names() = ["fig1", "f1", "f2", "f3", "f4", "f5"]

function _sim_entry(name)
    haskey(_SIM_FUNCTIONS, _canon_name(name)) ||
        throw(ArgumentError("unknown simulation \"$name\"; choose from $(simulation_names())"))
    return _SIM_FUNCTIONS[_canon_name(name)]
end

"""
    simulation_function(name) -> f

Regression function `f(x::AbstractVector)` of simulation `name`.
"""
simulation_function(name) = _sim_entry(name)[3]

"""
    simulation_nfeatures(name), simulation_sigma(name)

Number of covariates and default noise SD of simulation `name`.
"""
simulation_nfeatures(name) = _sim_entry(name)[1]
simulation_sigma(name) = _sim_entry(name)[2]

"""
    true_mean(name, X) -> Vector{Float64}

`f(x_i)` for every row of `X`.
"""
function true_mean(name, X::AbstractMatrix{<:Real})
    f = simulation_function(name)
    return [f(view(X, i, :)) for i in 1:size(X, 1)]
end

"""
    simulate(name, n; sigma=nothing, rng=Random.default_rng()) -> (X, y, f)

Draw `n` observations from the paper's simulation design `name`
(`"fig1"`, `"f1"`/`"friedman1"`, `"f2"`, ..., `"f5"`): `X ~ U(0,1)^p`,
`y = f(X) + sigma * N(0,1)`, and also return the noiseless `f = f(X)`.
`sigma` defaults to 0.5 for "fig1" and 1 otherwise.
"""
function simulate(name::Union{AbstractString,Symbol}, n::Integer; sigma::Union{Nothing,Real}=nothing,
                  rng::Random.AbstractRNG=Random.default_rng())
    p, sig0, _ = _sim_entry(name)
    sig = sigma === nothing ? sig0 : Float64(sigma)
    X = rand(rng, n, p)
    f = true_mean(name, X)
    y = f .+ sig .* randn(rng, n)
    return X, y, f
end

# --- yacht hydrodynamics ----------------------------------------------------

"""
    DatasetUnavailableError

Thrown by [`load_yacht`](@ref) when the data are neither cached nor downloadable.
"""
struct DatasetUnavailableError <: Exception
    msg::String
end
Base.showerror(io::IO, e::DatasetUnavailableError) = print(io, "DatasetUnavailableError: ", e.msg)

const YACHT_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
    "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/yacht/data/data.txt",
]

default_cache_dir() = get(ENV, "DEBIAS_FOREST_DATA", joinpath(homedir(), ".cache", "DebiasForest"))

function _parse_yacht(path::AbstractString)
    rows = Vector{Vector{Float64}}()
    for line in eachline(path)
        s = split(strip(line))
        length(s) == 7 || continue
        v = tryparse.(Float64, s)
        any(isnothing, v) && continue
        push!(rows, Float64.(v))
    end
    length(rows) == 308 || error("unexpected yacht file: $(length(rows)) valid rows (expected 308)")
    M = reduce(hcat, rows)'
    return Matrix{Float64}(M[:, 1:6]), Vector{Float64}(M[:, 7])
end

"""
    load_yacht(; cache_dir=default_cache_dir(), urls=YACHT_URLS, timeout=30) -> (X, y)

UCI Yacht Hydrodynamics data (308 × 6 features, residuary resistance as `y`).  The
file is cached in `cache_dir` (default `\$DEBIAS_FOREST_DATA` or `~/.cache/DebiasForest`);
otherwise the UCI URL and then a GitHub mirror are tried.  Throws
[`DatasetUnavailableError`](@ref) when offline, so callers can skip gracefully.
"""
function load_yacht(; cache_dir::AbstractString=default_cache_dir(), urls=YACHT_URLS, timeout::Real=30)
    path = joinpath(cache_dir, "yacht_hydrodynamics.data")
    if isfile(path)
        try
            return _parse_yacht(path)
        catch
            rm(path; force=true)
        end
    end
    mkpath(cache_dir)
    errs = String[]
    for url in urls
        tmp = path * ".part"
        try
            Downloads.download(url, tmp; timeout=timeout)
            res = _parse_yacht(tmp)
            mv(tmp, path; force=true)
            return res
        catch e
            push!(errs, "$url: $(sprint(showerror, e))")
            rm(tmp; force=true)
        end
    end
    throw(DatasetUnavailableError("could not obtain the yacht data (offline?): " * join(errs, " | ")))
end
