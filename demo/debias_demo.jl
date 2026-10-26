# =============================================================================
#  Demonstration of DebiasForest.jl
#  Chen, Withana Gamage & Ryan (2022), "Debias Random Forest Regression
#  Predictors", J. Stat. Research 56(2):115-131.
#
#  Run (from the repo root or from demo/):
#      export JULIA_PKG_SERVER=""            # only if the Julia pkg server is blocked
#      julia -t 4 --project=demo demo/debias_demo.jl
#  First time only (Manifest.toml is not tracked):
#      julia --project=demo -e 'using Pkg; Pkg.develop(path="julia"); Pkg.instantiate()'
#  (the script does this automatically if DebiasForest cannot be found).
#
#  Full ("thorough") mode: K = 200 replications, 100 yacht splits.
#  Quick mode:  DEBIAS_DEMO_QUICK=1  ->  K = 10, 10 splits (smoke test).
#
#  Outputs: tables  -> demo/results/julia/*.csv, *.md
#           figures -> demo/figures/julia/*.png
#           data cache -> demo/data/   (git-ignored)
# =============================================================================

const DEMO_DIR = @__DIR__
const REPO_DIR = dirname(DEMO_DIR)

# --- make sure the demo environment can see DebiasForest --------------------
import Pkg
if Base.find_package("DebiasForest") === nothing
    Pkg.activate(DEMO_DIR)
    Pkg.develop(path = joinpath(REPO_DIR, "julia"))
    Pkg.instantiate()
end

ENV["GKSwstype"] = "100"          # headless GR: draw to files only
using DebiasForest
using Plots
using Printf, Statistics, Random, DelimitedFiles
gr()

# --- settings ----------------------------------------------------------------
const QUICK     = get(ENV, "DEBIAS_DEMO_QUICK", "0") == "1"
const K_REPS    = QUICK ? 10 : 200      # Monte-Carlo replications (paper: 1000)
const N_SPLITS  = QUICK ? 10 : 100      # random yacht splits (paper: 1000)
const N_TRAIN   = 200
const N_TEST    = 2000
const SEED      = 20220101
const MAX_ITER  = 6                     # iteration curve: k = 0..6
const RES_DIR   = joinpath(DEMO_DIR, "results", "julia")
const FIG_DIR   = joinpath(DEMO_DIR, "figures", "julia")
const DATA_DIR  = joinpath(DEMO_DIR, "data")
mkpath.((RES_DIR, FIG_DIR, DATA_DIR))

const LABELS = METHOD_LABELS            # Un De Qu Cu Sp Bf Bf(1) Bf(2) Bf(3) Cond
const UNIVARIATE = ["De", "Qu", "Cu", "Sp"]

println("DebiasForest demo   mode = ", QUICK ? "QUICK" : "full",
        "   K = $K_REPS   yacht splits = $N_SPLITS   threads = $(Threads.nthreads())")
T0 = time()
timer(label, t) = @printf("  [%s] %.1f s\n", label, time() - t)

# --- small helpers -------------------------------------------------------------
f2(x) = @sprintf("%.2f", x)

# Write a matrix of strings as CSV (header + rows) and as a markdown table.
function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join(r, ","))
        end
    end
end
function write_md(path, header, rows; title = nothing, note = nothing)
    open(path, "w") do io
        title === nothing || println(io, "### ", title, "\n")
        println(io, "| ", join(header, " | "), " |")
        println(io, "|", join(fill("---", length(header)), "|"), "|")
        for r in rows
            println(io, "| ", join(r, " | "), " |")
        end
        note === nothing || println(io, "\n", note)
    end
end

# --- paper reference values (Tables 1 and 2, n = 200 rows) ---------------------
# Order of columns: Un De Qu Cu Sp Bf Bf(1) Bf(2) Bf(3) Cond
const PAPER = Dict(
    ("f1", 1.0) => (mspe = [6.44, 3.92, 3.79, 3.81, 3.83, 3.39, 2.95, 3.13, 3.13, 3.92],
                    sd   = [0.42, 0.41, 0.34, 0.34, 0.46, 0.29, 0.23, 0.26, 0.33, 0.34],
                    bias2 = [4.76, 1.44, 1.43, 1.42, 1.43, 1.46, 1.46, 0.74, 0.53, 2.01]),
    ("f1", 2.0) => (mspe = [9.42, 7.11, 7.15, 7.19, 7.20, 6.78, 6.78, 6.87, 7.71, 7.20],
                    sd   = [0.71, 0.5, 0.51, 0.51, 0.57, 0.46, 0.46, 0.44, 0.54, 0.52],
                    bias2 = [4.36, 1.53, 1.52, 1.53, 1.52, 1.38, 1.39, 0.77, 0.57, 1.96]),
    ("f1", 4.0) => (mspe = [21.90, 20.58, 20.67, 20.82, 20.81, 20.85, 20.85, 22.94, 26.45, 20.58],
                    sd   = [1.03, 0.91, 0.93, 0.98, 1.09, 0.90, 0.90, 1.14, 1.63, 0.90],
                    bias2 = [4.39, 1.75, 1.73, 1.79, 1.75, 1.22, 1.22, 0.64, 0.45, 1.90]),
    ("f2", 1.0) => (mspe = [2.01, 1.50, 1.49, 1.50, 1.52, 1.41, 1.41, 1.47, 1.65, 1.49],
                    sd   = [0.21, 0.12, 0.12, 0.15, 0.12, 0.12, 0.12, 0.11, 0.14, 0.13],
                    bias2 = [0.79, 0.15, 0.14, 0.13, 0.14, 0.11, 0.11, 0.02, 0.01, 0.21]),
    ("f3", 1.0) => (mspe = [8.68, 4.12, 4.15, 4.18, 4.19, 3.49, 3.49, 2.40, 2.46, 5.04],
                    sd   = [0.78, 0.58, 0.59, 0.59, 0.61, 0.40, 0.41, 0.21, 0.17, 0.61],
                    bias2 = [6.50, 0.78, 0.76, 0.74, 0.75, 1.34, 1.34, 0.18, 0.06, 2.56]),
    ("f4", 1.0) => (mspe = [8.83, 4.15, 4.12, 4.14, 4.17, 3.06, 3.07, 2.59, 2.96, 3.76],
                    sd   = [1.18, 0.67, 0.67, 0.68, 0.73, 0.46, 0.46, 0.27, 0.29, 0.65],
                    bias2 = [7.51, 1.30, 1.25, 1.22, 1.23, 1.01, 1.00, 0.22, 0.21, 1.80]),
    ("f5", 1.0) => (mspe = [2.79, 2.20, 2.20, 2.15, 2.12, 1.68, 1.68, 1.61, 1.80, 1.67],
                    sd   = [0.18, 0.19, 0.19, 0.18, 0.19, 0.10, 0.10, 0.07, 0.08, 0.11],
                    bias2 = [1.45, 0.77, 0.76, 0.67, 0.67, 0.29, 0.30, 0.08, 0.04, 0.37]),
)
# Paper Table 4, Yacht row: (mean, sd); Bf(3) not reported.
const PAPER_YACHT = Dict("Un" => (12.89, 4.65), "De" => (5.75, 1.58), "Qu" => (5.36, 1.45),
                         "Cu" => (4.52, 1.63), "Sp" => (4.08, 1.62), "Bf" => (1.69, 0.99),
                         "Bf(1)" => (1.55, 0.69), "Bf(2)" => (2.19, 0.81), "Cond" => (3.46, 2.24))

# =============================================================================
# 1. Fig. 1 replica: OOB prediction vs true f(x), Un vs Bf   (n = 500, sigma = 0.5)
# =============================================================================
println("\n[1] Fig. 1 replica")
t = time()
X1, y1, f1v = simulate("fig1", 500; sigma = 0.5, rng = Xoshiro(SEED))
un1 = fit_forest(X1, y1; seed = SEED)                      # uncorrected forest
bf1 = fit(BoostedForest, un1; n_iterations = 1)            # Breiman's bias correction
oob_un = oob_prediction(un1)
oob_bf = oob_prediction(bf1)

# Calibration slope of f(x) on the OOB prediction would be 1 for an unbiased predictor;
# regressing the OOB prediction on f(x) shows the shrinkage (<1) of the uncorrected forest.
slope(a, b) = cov(a, b) / var(a)
fig1_rows = [[string(i), @sprintf("%.6f", f1v[i]), @sprintf("%.6f", oob_un[i]), @sprintf("%.6f", oob_bf[i])]
             for i in eachindex(f1v)]
write_csv(joinpath(RES_DIR, "fig1.csv"), ["i", "f_true", "oob_Un", "oob_Bf"], fig1_rows)
write_md(joinpath(RES_DIR, "fig1_summary.md"), ["Predictor", "OOB MSE vs f(x)", "OOB MSE vs y", "slope of OOB pred on f(x)"],
         [["Un", f2(mean(abs2, oob_un .- f1v)), f2(mean(abs2, oob_un .- y1)), f2(slope(f1v, oob_un))],
          ["Bf", f2(mean(abs2, oob_bf .- f1v)), f2(mean(abs2, oob_bf .- y1)), f2(slope(f1v, oob_bf))]];
         title = "Fig. 1 replica (n = 500, sigma = 0.5)",
         note = "A slope below 1 means the OOB predictions are pulled towards the mean (bias at the extremes).")

lim = (minimum(f1v) - 0.2, maximum(f1v) + 0.2)
panel(p, title) = scatter(f1v, p; xlabel = "true f(x)", ylabel = "OOB prediction", title = title,
                          ms = 3, alpha = 0.6, legend = false, xlims = lim, ylims = lim,
                          markerstrokewidth = 0, aspect_ratio = :equal)
p1 = panel(oob_un, "Un: uncorrected forest")
plot!(p1, [lim...], [lim...]; color = :black, ls = :dash)
p2 = panel(oob_bf, "Bf: boosted (1 iteration)")
plot!(p2, [lim...], [lim...]; color = :black, ls = :dash)
savefig(plot(p1, p2; layout = (1, 2), size = (900, 450), margin = 4Plots.mm, left_margin = 9Plots.mm),
        joinpath(FIG_DIR, "fig1_oob_vs_true.png"))
@printf("  OOB MSE vs f:  Un %.3f   Bf %.3f\n", mean(abs2, oob_un .- f1v), mean(abs2, oob_bf .- f1v))
timer("fig1", t)

# =============================================================================
# 2. Simulation tables (Tables 1-2 style), n = 200, n_test = 2000
#    Note: test X is fixed across replications and the test noise is fresh in each
#    replication, so MSPE (vs noisy y) and squared bias come from the same fits.
# =============================================================================
println("\n[2] Simulation tables (K = $K_REPS)")
t = time()
designs = [("f1", 1.0), ("f1", 2.0), ("f1", 4.0), ("f2", 1.0), ("f3", 1.0), ("f4", 1.0), ("f5", 1.0)]
sim_csv = Vector{Vector{String}}()
sim_md = Vector{Vector{String}}()
for (di, (name, sigma)) in enumerate(designs)
    td = time()
    res = simulation_study(name; n = N_TRAIN, n_test = N_TEST, sigma = sigma, n_reps = K_REPS,
                           seed = SEED + di)
    ref = PAPER[(name, sigma)]
    dlabel = @sprintf("%s (sigma=%g)", uppercase(name), sigma)
    ours_cell = [@sprintf("%.2f (%.2f)", res.mspe_mean[j], res.mspe_sd[j]) for j in eachindex(LABELS)]
    ref_cell  = [@sprintf("%.2f (%.2f)", ref.mspe[j], ref.sd[j]) for j in eachindex(LABELS)]
    push!(sim_md, [dlabel, "MSPE (SD), this run", ours_cell...])
    push!(sim_md, [dlabel, "MSPE (SD), paper", ref_cell...])
    push!(sim_md, [dlabel, "bias^2, this run", f2.(res.bias2)...])
    push!(sim_md, [dlabel, "bias^2, paper", f2.(ref.bias2)...])
    for (stat, ours, theirs) in (("mspe", res.mspe_mean, ref.mspe), ("sd", res.mspe_sd, ref.sd),
                                 ("bias2", res.bias2, ref.bias2))
        push!(sim_csv, [name, string(sigma), stat, "julia", [@sprintf("%.4f", v) for v in ours]...])
        push!(sim_csv, [name, string(sigma), stat, "paper", [@sprintf("%.2f", v) for v in theirs]...])
    end
    timer("$dlabel", td)
end
write_csv(joinpath(RES_DIR, "simulation_table.csv"), ["function", "sigma", "statistic", "source", LABELS...], sim_csv)
write_md(joinpath(RES_DIR, "simulation_table.md"), ["Design", "Row", LABELS...], sim_md;
         title = "Simulation study, n = $N_TRAIN, n_test = $N_TEST, K = $K_REPS (paper: K = 1000)",
         note = "Paper values: Chen, Withana Gamage & Ryan (2022), Tables 1-2 (n = 200 rows). " *
                "MSPE is measured against the noisy test response; bias^2 uses the average prediction over replications.")
timer("simulation tables", t)

# =============================================================================
# 3. Iteration curve (Fig. 5 style): mean test MSPE vs number of residual forests k = 0..6
#    (paper labels Un, Bf, Bf(1), ..., Bf(5)).  Yacht is added in section 4.
# =============================================================================
println("\n[3] Iteration curves (K = $K_REPS)")
t = time()
curve_labels = [paper_label(k) for k in 0:MAX_ITER]

# Mean (over K replications) test MSPE of the staged boosted predictions.
function iteration_curve(name, sigma; n = N_TRAIN, n_test = N_TEST, reps = K_REPS, seed)
    p = DebiasForest.simulation_nfeatures(name)
    Xt = rand(Xoshiro(seed), n_test, p)
    ft = true_mean(name, Xt)
    out = Vector{Vector{Float64}}(undef, reps)
    Threads.@threads for r in 1:reps
        rng = Xoshiro(seed + 1000r)
        Xtr, ytr, _ = simulate(name, n; sigma = sigma, rng = rng)
        yt = ft .+ sigma .* randn(rng, n_test)
        bf = fit(BoostedForest, Xtr, ytr; n_iterations = MAX_ITER, seed = seed + 1000r + 1,
                 threaded = reps < Threads.nthreads())
        S = staged_predict(bf, Xt)
        out[r] = [mean(abs2, yt .- view(S, :, k)) for k in 1:size(S, 2)]
    end
    M = reduce(hcat, out)'
    return vec(mean(M; dims = 1))
end

curve_designs = [("f1", 1.0), ("f3", 1.0)]
curves = Dict{String,Vector{Float64}}()
for (i, (name, sigma)) in enumerate(curve_designs)
    curves[@sprintf("%s (sigma=%g)", uppercase(name), sigma)] =
        iteration_curve(name, sigma; seed = SEED + 100 + i)
end
timer("iteration curves", t)

# =============================================================================
# 4. Yacht hydrodynamics (UCI): 100 random 2/3-1/3 splits
# =============================================================================
println("\n[4] Yacht data (splits = $N_SPLITS)")
t = time()
yacht = try
    load_yacht(; cache_dir = DATA_DIR)
catch e
    e isa DatasetUnavailableError || rethrow()
    @warn "Yacht data unavailable; skipping section 4" exception = e
    nothing
end

if yacht !== nothing
    Xy, yy = yacht
    println("  loaded yacht data: ", size(Xy, 1), " rows, ", size(Xy, 2), " features")

    # 4a. MSPE table for all methods
    ty = time()
    rs = repeated_splits(Xy, yy; n_splits = N_SPLITS, seed = SEED + 200)
    rows_csv = Vector{Vector{String}}(); rows_md = Vector{Vector{String}}()
    for (j, l) in enumerate(rs.labels)
        pv = get(PAPER_YACHT, l, nothing)
        paper_cell = pv === nothing ? "-" : @sprintf("%.2f (%.2f)", pv...)
        push!(rows_csv, [l, @sprintf("%.4f", rs.mspe_mean[j]), @sprintf("%.4f", rs.mspe_sd[j]),
                         pv === nothing ? "" : string(pv[1]), pv === nothing ? "" : string(pv[2])])
        push!(rows_md, [l, @sprintf("%.2f (%.2f)", rs.mspe_mean[j], rs.mspe_sd[j]), paper_cell])
    end
    write_csv(joinpath(RES_DIR, "yacht_mspe.csv"), ["method", "mspe_mean", "mspe_sd", "paper_mean", "paper_sd"], rows_csv)
    write_md(joinpath(RES_DIR, "yacht_mspe.md"), ["Method", "MSPE (SD), this run", "MSPE (SD), paper Table 4"], rows_md;
             title = "Yacht hydrodynamics: test MSPE over $N_SPLITS random 2/3-1/3 splits (paper: 1000)")
    timer("yacht splits", ty)

    # 4b. selected number of iterations (paper's stopping rule)
    ts = time()
    sel = select_n_iterations(Xy, yy; max_iterations = MAX_ITER, n_splits = N_SPLITS, seed = SEED + 300)
    write_md(joinpath(RES_DIR, "yacht_select_n_iterations.md"), ["k", "label", "mean MSPE", "SD"],
             [[string(k - 1), sel.labels[k], f2(sel.mean_mspe[k]), f2(sel.sd_mspe[k])] for k in eachindex(sel.labels)];
             title = "Yacht: select_n_iterations (max_iterations = $MAX_ITER, $N_SPLITS splits)",
             note = "Selected number of residual forests: **k = $(sel.n_iterations)** ($(paper_label(sel.n_iterations))).")
    @printf("  select_n_iterations -> k = %d (%s)\n", sel.n_iterations, paper_label(sel.n_iterations))
    curves["Yacht"] = sel.mean_mspe
    timer("select_n_iterations", ts)

    # 4c. predicted vs observed for one split (Fig. 3 style): Un, best univariate, Bf
    best_uni = UNIVARIATE[argmin([rs.mspe_mean[findfirst(==(u), rs.labels)] for u in UNIVARIATE])]
    rng = Xoshiro(SEED + 400)
    perm = randperm(rng, size(Xy, 1)); nte = round(Int, size(Xy, 1) / 3)
    te, tr = perm[1:nte], perm[(nte + 1):end]
    base = fit_forest(Xy[tr, :], yy[tr]; seed = SEED + 401)
    preds = Dict("Un" => predict(base, Xy[te, :]),
                 best_uni => predict(fit(SmoothedForest, base; method = Dict("De" => :linear, "Qu" => :quadratic,
                                         "Cu" => :cubic, "Sp" => :spline)[best_uni]), Xy[te, :]),
                 "Bf" => predict(fit(BoostedForest, base; n_iterations = 1), Xy[te, :]))
    order = ["Un", best_uni, "Bf"]
    yl = (min(minimum(yy), 0) - 1, maximum(yy) + 1)
    ps = [begin
              scatter(yy[te], preds[l]; xlabel = "observed", ylabel = "predicted", legend = false, ms = 4,
                      alpha = 0.7, markerstrokewidth = 0, xlims = yl, ylims = yl, aspect_ratio = :equal,
                      title = @sprintf("%s  (MSPE %.2f)", l, mean(abs2, yy[te] .- preds[l])))
              plot!([yl...], [yl...]; color = :black, ls = :dash)
          end for l in order]
    savefig(plot(ps...; layout = (1, 3), size = (1250, 430), margin = 4Plots.mm, left_margin = 9Plots.mm),
            joinpath(FIG_DIR, "yacht_pred_vs_obs.png"))
    println("  predicted-vs-observed plot: Un vs $best_uni vs Bf")
else
    write_md(joinpath(RES_DIR, "yacht_mspe.md"), ["Note"], [["Yacht data unavailable (offline); section skipped."]])
end
timer("yacht", t)

# --- iteration-curve output (needs Yacht curve from section 4) ------------------
cnames = sort(collect(keys(curves)); by = s -> s == "Yacht" ? "zzz" : s)
write_csv(joinpath(RES_DIR, "iteration_curve.csv"), ["k", "label", cnames...],
          [[string(k), curve_labels[k + 1], [@sprintf("%.4f", curves[c][k + 1]) for c in cnames]...] for k in 0:MAX_ITER])
write_md(joinpath(RES_DIR, "iteration_curve.md"), ["k", "label", cnames...],
         [[string(k), curve_labels[k + 1], [f2(curves[c][k + 1]) for c in cnames]...] for k in 0:MAX_ITER];
         title = "Mean test MSPE vs number of residual forests k",
         note = "Simulations: K = $K_REPS replications, n = $N_TRAIN. Yacht: $N_SPLITS random splits.")
cp = [plot(0:MAX_ITER, curves[c]; marker = :circle, legend = false, title = c, xlabel = "iterations k",
           ylabel = "mean MSPE", xticks = (0:MAX_ITER, curve_labels), xrotation = 30, lw = 2) for c in cnames]
savefig(plot(cp...; layout = (1, length(cp)), size = (400 * length(cp), 400), margin = 5Plots.mm),
        joinpath(FIG_DIR, "iteration_curve.png"))

@printf("\nDone in %.1f s.  Tables: %s\n      Figures: %s\n", time() - T0, RES_DIR, FIG_DIR)
