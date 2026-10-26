using Test
using Random, Statistics, LinearAlgebra, DelimitedFiles
using DebiasForest
import DecisionTree
import StatsAPI
import MLJModelInterface as MMI

const DF = DebiasForest
const DATA = joinpath(@__DIR__, "data")

# small reproducible regression problem
function toy(n=120; seed=1, p=5, sigma=0.5)
    X, y, f = simulate("f1", n; sigma, rng=Xoshiro(seed))
    return X, y, f
end

@testset "DebiasForest" begin

@testset "flat trees match DecisionTree.apply_tree" begin
    X, y, _ = toy(150)
    tree = DecisionTree.build_tree(y, X, 2, -1, 1, 6, 0.0; rng=Xoshiro(3))
    ft = DF._flatten(tree)
    Xt = rand(Xoshiro(9), 200, 5)
    ref = DecisionTree.apply_tree(tree, Xt)
    @test predict(ft, Xt) == ref
    # leaf ids are consistent with leaf values
    @test ft.leaf_value[DF.apply(ft, Xt)] == ref
    # a stump / single-leaf tree
    leaf = DecisionTree.build_tree(fill(1.0, 4), rand(4, 2), 1, -1, 1, 6, 0.0)
    fl = DF._flatten(leaf)
    @test DF.n_leaves(fl) == 1 && predict(fl, rand(3, 2)) == fill(1.0, 3)

    # trees inside a forest equal DecisionTree trees grown with the same seeds
    f = fit_forest(X, y; n_trees=5, seed=11, threaded=false)
    for t in 1:5
        rng = Xoshiro(DF._derive_seed(UInt64(11), t))
        idx = rand(rng, 1:150, 150)
        tr = DecisionTree.build_tree(y[idx], X[idx, :], f.mtry, -1, 1, 6, 0.0; rng, impurity_importance=false)
        @test predict(f.trees[t], Xt) == DecisionTree.apply_tree(tr, Xt)
        @test f.inbag[:, t] == [count(==(i), idx) for i in 1:150]
    end
    # nodesize semantics: no terminal node splits when it has <= nodesize draws
    @test all(f.trees[1].feature .>= 0)
end

@testset "bagged forest, OOB, leaves" begin
    X, y, _ = toy(100)
    f = fit_forest(X, y; n_trees=60, seed=5)
    @test size(f.inbag) == (100, 60) && all(sum(f.inbag; dims=1) .== 100)
    @test f.mtry == 1 && f.nodesize == 5          # p = 5 -> max(5 ÷ 3, 1) = 1
    Xt = rand(Xoshiro(2), 30, 5)
    brute = [mean(predict(t, Xt)[k] for t in f.trees) for k in 1:30]
    @test predict(f, Xt) ≈ brute
    # brute-force OOB
    ob = oob_prediction(f)
    for i in 1:100
        vals = [predict(f.trees[t], X[i:i, :])[1] for t in 1:60 if f.inbag[i, t] == 0]
        @test ob[i] ≈ mean(vals)
    end
    # leaf matrix agrees with training leaves and with per-tree apply
    @test leaf_matrix(f, X) == f.leaves
    @test f.leaves[:, 3] == DF.apply(f.trees[3], X)
    # never-OOB rows: fallback + warning
    small = fit_forest(X[1:6, :], y[1:6]; n_trees=2, seed=1, mtry=2)
    nev = findall(i -> all(small.inbag[i, :] .> 0), 1:6)
    @test !isempty(nev)
    local o
    @test_logs (:warn, r"never out-of-bag") (o = oob_prediction(small))
    @test o[nev] ≈ predict(small, X[1:6, :])[nev]
    @test_logs oob_prediction(small; warn=false)
    @test_throws DimensionMismatch predict(f, rand(3, 4))
end

@testset "determinism" begin
    X, y, _ = toy(100)
    a = fit_forest(X, y; n_trees=30, seed=42)
    b = fit_forest(X, y; n_trees=30, seed=42)
    c = fit_forest(X, y; n_trees=30, seed=42, threaded=false)   # serial == threaded
    d = fit_forest(X, y; n_trees=30, seed=43)
    Xt = rand(Xoshiro(1), 20, 5)
    @test predict(a, Xt) == predict(b, Xt) == predict(c, Xt)
    @test a.inbag == c.inbag
    @test predict(a, Xt) != predict(d, Xt)
    e = fit_forest(X, y; n_trees=25, rng=Xoshiro(1)); g = fit_forest(X, y; n_trees=25, rng=Xoshiro(1))
    @test predict(e, Xt) == predict(g, Xt)
    ba = fit(BoostedForest, X, y; n_iterations=2, n_trees=30, seed=1)
    bb = fit(BoostedForest, X, y; n_iterations=2, n_trees=30, seed=1, threaded=false)
    @test predict(ba, Xt) == predict(bb, Xt)
    # thread-count independence (fresh process with one thread)
    if Threads.nthreads() > 1
        script = """
        using DebiasForest, Random
        X, y, _ = simulate("f1", 100; sigma=0.5, rng=Xoshiro(1))
        f = fit_forest(X, y; n_trees=30, seed=42)
        println(sum(predict(f, rand(Xoshiro(1), 20, 5))))
        """
        proj = dirname(Base.active_project())
        out = read(`$(Base.julia_cmd()) -t 1 --project=$proj -e $script`, String)
        Xs, ys, _ = simulate("f1", 100; sigma=0.5, rng=Xoshiro(1))
        fs = fit_forest(Xs, ys; n_trees=30, seed=42)
        @test parse(Float64, strip(out)) == sum(predict(fs, rand(Xoshiro(1), 20, 5)))
    end
end

@testset "boosted forest" begin
    X, y, _ = toy(120)
    Xt = rand(Xoshiro(7), 40, 5)
    plain = fit_forest(X, y; n_trees=40, seed=3)
    b0 = fit(BoostedForest, X, y; n_iterations=0, n_trees=40, seed=3)
    @test b0.n_iterations == 0
    @test predict(b0, Xt) == predict(plain, Xt)
    @test oob_prediction(b0) == oob_prediction(plain)

    b3 = fit(BoostedForest, X, y; n_iterations=3, n_trees=40, seed=3)
    st = staged_predict(b3, Xt)
    @test size(st) == (40, 4)
    @test st[:, end] == predict(b3, Xt)
    @test st[:, 1] == predict(plain, Xt)
    @test st[:, 2] ≈ predict(plain, Xt) .+ predict(b3.stages[2], Xt)
    @test oob_prediction(b3) ≈ sum(oob_prediction(s) for s in b3.stages)
    # stage 2 is trained on the OOB residual of stage 1
    @test b3.stages[2].y ≈ y .- oob_prediction(plain)
    @test b3.stages[3].y ≈ b3.stages[2].y .- oob_prediction(b3.stages[2])
    @test b3.oob_mse[end] ≈ mean(abs2, y .- oob_prediction(b3))
    @test b3.oob_mse[1] ≈ mean(abs2, y .- oob_prediction(plain))
    @test oob_staged_prediction(b3)[:, 3] ≈ oob_prediction(fit(BoostedForest, plain; n_iterations=2, seed=3))
    # re-using a base forest == fitting from scratch
    @test predict(fit(BoostedForest, plain; n_iterations=3, seed=3), Xt) == predict(b3, Xt)
    @test nobs(b3) == 120

    # OOB stopping picks the argmin of the recorded OOB MSE path
    for (Xs, ys) in ((X, y), (X, randn(Xoshiro(5), 120)))
        bo = fit(BoostedForest, Xs, ys; n_iterations=6, stopping=:oob, n_trees=40, seed=2)
        @test bo.n_iterations == argmin(bo.oob_mse) - 1
        @test length(bo.stages) == bo.n_iterations + 1
        @test all(bo.oob_mse[1:(bo.n_iterations + 1)] .>= bo.oob_mse[bo.n_iterations + 1])
        @test length(bo.oob_mse) <= 7
        if length(bo.oob_mse) < 7     # stopped early: last recorded value triggered the rule
            @test bo.oob_mse[end] > 1.1 * minimum(bo.oob_mse)
        end
        @test mean(abs2, ys .- oob_prediction(bo)) ≈ bo.oob_mse[bo.n_iterations + 1]
    end
    # pure noise: boosting never helps -> stops at the base forest
    bn = fit(BoostedForest, X, randn(Xoshiro(8), 120); n_iterations=6, stopping=:oob, n_trees=40, seed=2)
    @test bn.n_iterations <= 1
    @test_throws ArgumentError fit(BoostedForest, X, y; stopping=:foo)
    @test_throws ArgumentError fit(BoostedForest, X, y; n_iterations=-1)
    @test paper_label(0) == "Un" && paper_label(1) == "Bf" && paper_label(4) == "Bf(3)"
end

@testset "univariate smoothing" begin
    X, y, _ = toy(150)
    Xt = rand(Xoshiro(4), 50, 5)
    base = fit_forest(X, y; n_trees=50, seed=9)
    u = oob_prediction(base); ut = predict(base, Xt)
    for (m, d) in ((:linear, 1), (:quadratic, 2), (:cubic, 3))
        s = fit(SmoothedForest, base; method=m)
        A = hcat((u .^ k for k in 0:d)...)
        beta = (A' * A) \ (A' * y)            # closed-form normal equations
        At = hcat((ut .^ k for k in 0:d)...)
        @test predict(s, Xt) ≈ At * beta rtol = 1e-6
        @test oob_prediction(s) ≈ A * beta rtol = 1e-6
        @test nobs(s) == 150
    end
    # linear fit = simple regression slope/intercept
    s = fit(SmoothedForest, base; method=:linear)
    slope = cov(u, y) / var(u); icpt = mean(y) - slope * mean(u)
    @test predict(s, Xt) ≈ icpt .+ slope .* ut
    sp = fit(SmoothedForest, base; method=:spline)
    ref = smooth_spline(u, y)
    @test predict(sp, Xt) ≈ predict(ref, ut)
    @test oob_prediction(sp) ≈ predict(ref, u)
    @test_throws ArgumentError fit(SmoothedForest, base; method=:foo)
    # convenience form
    s2 = fit(SmoothedForest, X, y; method=:quadratic, n_trees=20, seed=1)
    @test length(predict(s2, Xt)) == 50
end

@testset "smooth.spline vs R fixtures" begin
    data, _ = readdlm(joinpath(DATA, "smooth_spline_data.csv"), ','; header=true)
    pred, _ = readdlm(joinpath(DATA, "smooth_spline_pred.csv"), ','; header=true)
    meta, _ = readdlm(joinpath(DATA, "smooth_spline_meta.csv"), ','; header=true)
    @test size(meta, 1) >= 6
    for r in eachrow(meta)
        name = r[1]
        @testset "$name" begin
            m = data[:, 1] .== name
            x = Float64.(data[m, 2]); y = Float64.(data[m, 3])
            f = smooth_spline(x, y)
            @test length(x) == r[2]
            @test length(f.x) == r[3]
            @test length(f.coef) - 2 == r[4]
            @test isapprox(f.spar, r[5]; atol=1e-5)
            @test isapprox(f.lambda, r[6]; rtol=1e-4)
            @test isapprox(f.df, r[7]; atol=1e-4)
            @test isapprox(f.crit, r[8]; rtol=1e-6)
            pm = pred[:, 1] .== name
            px = Float64.(pred[pm, 2]); pr = Float64.(pred[pm, 3])
            p = predict(f, px)
            @test maximum(abs.(p .- pr) ./ (abs.(pr) .+ 1e-3 * maximum(abs.(pr)))) < 1e-5
        end
    end
    # linear extrapolation outside the data range
    x = collect(range(0, 1; length=40)); y = sin.(3 .* x) .+ 0.1 .* cos.(40 .* x)
    f = smooth_spline(x, y)
    xe = [-0.5, -0.3, -0.1, 1.1, 1.3, 1.5]
    pe = predict(f, xe)
    @test pe[2] - pe[1] ≈ pe[3] - pe[2] atol = 1e-10
    @test pe[5] - pe[4] ≈ pe[6] - pe[5] atol = 1e-10
    h = 1e-6
    @test (predict(f, h) - predict(f, 0.0)) / h ≈ predict(f, 0.0; deriv=1) rtol = 1e-3
    @test predict(f, -0.1) ≈ predict(f, 0.0) + predict(f, 0.0; deriv=1) * (-0.1) atol = 1e-8
    # ties: aggregating = fitting the means with count weights (same lambda up to weight scaling)
    xt = repeat([0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0], inner=3)
    yt = sin.(5 .* xt) .+ 0.1 .* repeat([1.0, -1.0, 0.5], 7)
    ft = smooth_spline(xt, yt)
    @test length(ft.x) == 7 && all(ft.w .== 3)
    yb = [mean(yt[3i-2:3i]) for i in 1:7]
    fw = smooth_spline(unique(xt), yb; w=fill(3.0, 7), lambda=ft.lambda / 3)  # weights are rescaled to mean 1
    @test fw.df ≈ ft.df rtol = 1e-8
    @test predict(ft, unique(xt)) ≈ predict(fw, unique(xt)) rtol = 1e-6
    @test_throws ArgumentError smooth_spline([1.0, 2, 3], [1.0, 2, 3])
    @test DF.R_nknots(30) == 30 && DF.R_nknots(120) == 69 && DF.R_nknots(900) == 142
    # fixed lambda / spar paths
    @test isfinite(smooth_spline(x, y; lambda=1e-4).df)
    @test smooth_spline(x, y; spar=1.0).spar == 1.0
end

@testset "conditional bias" begin
    X, y, _ = toy(100)
    base = fit_forest(X, y; n_trees=40, seed=6)
    cb = fit(ConditionalBiasForest, base)
    ob = oob_prediction(base)
    e = ob .- y
    Xt = rand(Xoshiro(3), 25, 5)
    Lt = leaf_matrix(base, Xt); L = base.leaves
    oobmask = base.inbag .== 0
    # brute force from the paper formula
    for k in 1:25
        num = 0.0; den = 0.0
        for t in 1:40, i in 1:100
            if oobmask[i, t] && L[i, t] == Lt[k, t]
                num += e[i]; den += 1
            end
        end
        b = den > 0 ? num / den : 0.0
        @test bias_estimate(cb, Xt)[k] ≈ b atol = 1e-10
        @test predict(cb, Xt)[k] ≈ predict(base, Xt)[k] - b atol = 1e-10
    end
    # OOB prediction excludes the row's own contribution
    for j in 1:100
        num = 0.0; den = 0.0
        for t in 1:40, i in 1:100
            if i != j && oobmask[i, t] && L[i, t] == L[j, t]
                num += e[i]; den += 1
            end
        end
        @test oob_prediction(cb)[j] ≈ ob[j] - (den > 0 ? num / den : 0.0) atol = 1e-10
    end
    @test nobs(cb) == 100
    @test length(predict(fit(ConditionalBiasForest, X, y; n_trees=25, seed=1), Xt)) == 25
end

@testset "datasets" begin
    for name in simulation_names()
        X, y, f = simulate(name, 30; rng=Xoshiro(1))
        @test size(X, 1) == 30 && length(y) == 30 && length(f) == 30
        @test size(X, 2) == DF.simulation_nfeatures(name)
        @test all(0 .<= X .<= 1)
        @test f ≈ true_mean(name, X)
    end
    X, y, f = simulate("friedman1", 10; sigma=0, rng=Xoshiro(1))
    @test y == f
    x = [0.5, 0.5, 0.5, 0.5, 0.5]
    @test simulation_function("f1")(x) ≈ 10 * sin(pi * 0.25) + 0 + 5 + 2.5
    @test simulation_function("f5")(x) ≈ 10cos(pi * 0.25) + 2.5 + 10
    @test simulation_function("fig1")([0.5, 0.5, 0.5, 0.5]) ≈ 1.5 + 0.25 + sin(0.5) - sqrt(0.5)
    @test_throws ArgumentError simulate("nope", 5)
    # yacht: parse a local copy, and fail gracefully when nothing can be fetched
    mktempdir() do dir
        rows = [join(round.(rand(Xoshiro(i), 7); digits=3), "  ") for i in 1:308]
        write(joinpath(dir, "yacht_hydrodynamics.data"), join(rows, "\n") * "\n")
        Xy, yy = load_yacht(; cache_dir=dir)
        @test size(Xy) == (308, 6) && length(yy) == 308
    end
    mktempdir() do dir
        @test_throws DatasetUnavailableError load_yacht(; cache_dir=dir, urls=["file:///nonexistent/yacht"])
    end
end

@testset "evaluation" begin
    X, y, _ = toy(90)
    Xt, yt, _ = toy(40; seed=2)
    P = predict_all_methods(X, y, Xt; n_trees=30, seed=1)
    @test Set(keys(P)) == Set(METHOD_LABELS)
    @test all(length(v) == 40 for v in values(P))
    base = fit_forest(X, y; n_trees=30, seed=1)
    @test P["Un"] == predict(base, Xt)
    @test P["Cond"] ≈ predict(fit(ConditionalBiasForest, base), Xt)
    @test P["De"] ≈ predict(fit(SmoothedForest, base; method=:linear), Xt)
    @test P["Bf"] ≈ predict(fit(BoostedForest, base; n_iterations=1, seed=DF._derive_seed(base.seed, 1)), Xt)
    @test P["Bf(3)"] ≈ predict(fit(BoostedForest, base; n_iterations=4, seed=DF._derive_seed(base.seed, 1)), Xt)

    r = repeated_splits(X, y; n_splits=3, n_trees=20, seed=3)
    @test r.labels == METHOD_LABELS && size(r.mspe) == (3, 10)
    @test r.mspe_mean ≈ vec(mean(r.mspe; dims=1)) && all(r.mspe_sd .>= 0)
    r2 = repeated_splits(X, y; n_splits=3, n_trees=20, seed=3)
    @test r.mspe == r2.mspe

    s = select_n_iterations(X, y; max_iterations=3, n_splits=3, n_trees=20, seed=4)
    @test 0 <= s.n_iterations <= 3 && length(s.mean_mspe) == 4 && length(s.sd_mspe) == 4
    m = s.mean_mspe
    k = findfirst(j -> m[j] < m[j + 1], 1:3)
    @test s.n_iterations == (k === nothing ? 3 : k - 1)

    ss = simulation_study("f1"; n=80, n_test=200, n_reps=4, n_trees=20, seed=5)
    @test ss.labels == METHOD_LABELS && size(ss.mspe) == (4, 10)
    @test length(ss.mspe_mean) == 10 && length(ss.bias2) == 10 && all(ss.bias2 .>= 0)
    ss2 = simulation_study("f1"; n=80, n_test=200, n_reps=4, n_trees=20, seed=5)
    @test ss.mspe == ss2.mspe && ss.bias2 == ss2.bias2      # thread-independent per-rep seeds
    @test ss.sigma == 1.0 && simulation_study("fig1"; n=60, n_test=50, n_reps=2, n_trees=25, seed=1).sigma == 0.5
end

@testset "MLJ interface" begin
    X, y, _ = toy(100)
    Xt = rand(Xoshiro(3), 20, 5)
    tab = (; (Symbol("x", j) => X[:, j] for j in 1:5)...)
    tabt = (; (Symbol("x", j) => Xt[:, j] for j in 1:5)...)
    mb = BoostedForestRegressor(n_trees=30, n_iterations=2, seed=3)
    fr, cache, rep = MMI.fit(mb, 0, tab, y)
    @test MMI.predict(mb, fr, tabt) == predict(fit(BoostedForest, X, y; n_iterations=2, n_trees=30, seed=3), Xt)
    @test rep.n_iterations_used == 2
    ms = SmoothedForestRegressor(n_trees=30, method=:cubic, seed=3)
    fr, _, _ = MMI.fit(ms, 0, tab, y)
    @test MMI.predict(ms, fr, tabt) == predict(fit(SmoothedForest, X, y; method=:cubic, n_trees=30, seed=3), Xt)
    mc = ConditionalBiasForestRegressor(n_trees=30, seed=3)
    fr, _, _ = MMI.fit(mc, 0, tab, y)
    @test MMI.predict(mc, fr, tabt) == predict(fit(ConditionalBiasForest, X, y; n_trees=30, seed=3), Xt)
    @test MMI.input_scitype(BoostedForestRegressor) == MMI.Table(MMI.Continuous)
    @test MMI.target_scitype(SmoothedForestRegressor) == AbstractVector{MMI.Continuous}
    @test MMI.is_pure_julia(ConditionalBiasForestRegressor)
    @test BoostedForestRegressor() isa MMI.Deterministic
    @test_logs (:warn, r"Constraint") BoostedForestRegressor(stopping=:nope)
end

@testset "bias reduction smoke test (Friedman 1)" begin
    mu = zeros(2); K = 4
    for k in 1:K
        X, y, _ = simulate("f1", 200; rng=Xoshiro(100 + k))
        Xt, _, ft = simulate("f1", 1000; sigma=0, rng=Xoshiro(200 + k))
        P = predict_all_methods(X, y, Xt; n_trees=150, seed=k)
        mu .+= [mean(abs2, P["Un"] .- ft), mean(abs2, P["Bf"] .- ft)] ./ K
        @test mean(abs2, P["De"] .- ft) < mean(abs2, P["Un"] .- ft)
    end
    @test mu[2] < mu[1]
end

end
