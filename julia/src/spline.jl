#  Native port of R's `stats::smooth.spline` (cubic smoothing spline, GCV).
#
#  Follows R 4.3 (`smooth.spline` + Fortran/C `rbart`/`sbart`/`sslvrg`):
#    * ties in x (tolerance 1e-6 * IQR(x)) are aggregated (weights, weighted mean of y);
#    * x is rescaled to [0, 1];
#    * cubic B-spline basis on R's knot subset (`.nknots.smspl`);
#    * penalty Gram matrix  Sigma_ij = int B_i'' B_j'' (with R's 0.3330 constant);
#    * lambda = r * 256^(3 spar - 1),  r = tr(X'WX) / tr(Sigma) over coefficients 3..nk-3;
#    * GCV criterion as in `sslvrg` (rss + within-tie SS, df = sum of leverages);
#    * Brent / golden-section search over spar in [-1.5, 1.5] (control.spar defaults);
#    * linear extrapolation outside the data range.
#
#  Dense linear algebra is used for the (banded) normal equations; the maths is
#  identical to R's banded Cholesky / `sinerp` route.

"""
    R_nknots(n)

Number of inner knots R uses by default (`stats:::.nknots.smspl`).
"""
function R_nknots(n::Integer)
    n < 50 && return Int(n)
    a1 = log2(50.0); a2 = log2(100.0); a3 = log2(140.0); a4 = log2(200.0)
    v = if n < 200
        2^(a1 + (a2 - a1) * (n - 50) / 150)
    elseif n < 800
        2^(a2 + (a3 - a2) * (n - 200) / 600)
    elseif n < 3200
        2^(a3 + (a4 - a3) * (n - 800) / 2400)
    else
        200 + (n - 3200)^0.2
    end
    return trunc(Int, v)
end

# --- B-spline helpers (order 4, 1-based knot vector `t`) ------------------

# B_{i,k}(x) restricted to knot span `span` (t[span] <= x <= t[span+1]).
function _bsp(t::Vector{Float64}, i::Int, k::Int, x::Float64, span::Int)
    if k == 1
        return i == span ? 1.0 : 0.0
    end
    r = 0.0
    d1 = t[i + k - 1] - t[i]
    if d1 > 0
        r += (x - t[i]) / d1 * _bsp(t, i, k - 1, x, span)
    end
    d2 = t[i + k] - t[i + 1]
    if d2 > 0
        r += (t[i + k] - x) / d2 * _bsp(t, i + 1, k - 1, x, span)
    end
    return r
end

# d-th derivative of B_{i,k} at x (right/left limit determined by `span`).
function _bspd(t::Vector{Float64}, i::Int, k::Int, d::Int, x::Float64, span::Int)
    d == 0 && return _bsp(t, i, k, x, span)
    r = 0.0
    d1 = t[i + k - 1] - t[i]
    if d1 > 0
        r += _bspd(t, i, k - 1, d - 1, x, span) / d1
    end
    d2 = t[i + k] - t[i + 1]
    if d2 > 0
        r -= _bspd(t, i + 1, k - 1, d - 1, x, span) / d2
    end
    return (k - 1) * r
end

# Knot span index mu in 4:nk with t[mu] <= x < t[mu+1] (x == 1 -> last span).
function _span(t::Vector{Float64}, nk::Int, x::Float64)
    mu = searchsortedlast(t, x)
    return clamp(mu, 4, nk)
end

"""
    SmoothSpline

Fitted cubic smoothing spline (R-compatible).  Fields: `x` (unique sorted
inputs), `y` (fitted values there), `w` (aggregated weights), `lev`, `df`,
`spar`, `lambda`, `crit` (R's `cv.crit`, i.e. the GCV score), `ratio`, `knots`,
`coef`, `xmin`, `range`.  Evaluate with `predict(fit, xnew)`.
"""
struct SmoothSpline
    x::Vector{Float64}
    y::Vector{Float64}
    w::Vector{Float64}
    lev::Vector{Float64}
    df::Float64
    spar::Float64
    lambda::Float64
    crit::Float64
    ratio::Float64
    knots::Vector{Float64}
    coef::Vector{Float64}
    xmin::Float64
    range::Float64
end

function _basis_eval(t::Vector{Float64}, coef::Vector{Float64}, xs::Float64, deriv::Int)
    nk = length(coef)
    mu = _span(t, nk, xs)
    s = 0.0
    for j in (mu - 3):mu
        s += coef[j] * _bspd(t, j, 4, deriv, xs, mu)
    end
    return s
end

"""
    predict(fit::SmoothSpline, xnew; deriv=0)

Evaluate the spline (or its first derivative, `deriv=1`) at `xnew`; outside the
training range the fit is extended linearly (as in R, `deriv=0` only).
"""
function StatsAPI.predict(fit::SmoothSpline, xnew::AbstractVector{<:Real}; deriv::Int=0)
    deriv in (0, 1, 2, 3) || throw(ArgumentError("deriv must be in 0:3"))
    out = Vector{Float64}(undef, length(xnew))
    t = fit.knots
    e0 = _basis_eval(t, fit.coef, 0.0, 0)
    e1 = _basis_eval(t, fit.coef, 1.0, 0)
    s0 = _basis_eval(t, fit.coef, 0.0, 1)
    s1 = _basis_eval(t, fit.coef, 1.0, 1)
    for (k, xv) in enumerate(xnew)
        xs = (xv - fit.xmin) / fit.range
        if 0.0 <= xs <= 1.0
            out[k] = _basis_eval(t, fit.coef, xs, deriv)
        elseif deriv == 0
            out[k] = xs < 0 ? e0 + s0 * xs : e1 + s1 * (xs - 1)
        elseif deriv == 1
            out[k] = (xs < 0 ? s0 : s1) / fit.range
        else
            out[k] = 0.0
        end
        if deriv > 0 && 0.0 <= xs <= 1.0
            out[k] /= fit.range^deriv
        end
    end
    return out
end
StatsAPI.predict(fit::SmoothSpline, x::Real; kwargs...) = predict(fit, [float(x)]; kwargs...)[1]

"""
    smooth_spline(x, y; w=nothing, spar=nothing, lambda=nothing, nknots=R_nknots,
                  all_knots=false, df_offset=0.0, penalty=1.0, control_spar=(;),
                  tol=1e-6 * IQR(x)) -> SmoothSpline

Port of R's `smooth.spline` with GCV (`cv = FALSE`).  Give `spar` or `lambda`
to skip the smoothing-parameter search.  `control_spar` may override
`low`, `high`, `tol`, `eps`, `maxit` (R defaults -1.5, 1.5, 1e-4, 2e-8, 500).
"""
function smooth_spline(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                       w::Union{Nothing,AbstractVector{<:Real}}=nothing,
                       spar::Union{Nothing,Real}=nothing,
                       lambda::Union{Nothing,Real}=nothing,
                       nknots=R_nknots, all_knots::Bool=false,
                       df_offset::Real=0.0, penalty::Real=1.0,
                       control_spar=(;), tol::Union{Nothing,Real}=nothing)
    n = length(x)
    length(y) == n || throw(DimensionMismatch("x and y must have the same length"))
    all(isfinite, x) && all(isfinite, y) ||
        throw(ArgumentError("missing or infinite values in inputs are not allowed"))
    spar === nothing || lambda === nothing ||
        throw(ArgumentError("must not specify both `spar` and `lambda`"))
    cs = merge((low=-1.5, high=1.5, tol=1e-4, eps=2e-8, maxit=500), NamedTuple(control_spar))
    xv = Float64.(x); yv = Float64.(y)

    wn = if w === nothing
        nothing
    else
        length(w) == n || throw(DimensionMismatch("lengths of x and w must match"))
        any(<(0), w) && throw(ArgumentError("all weights should be non-negative"))
        all(==(0), w) && throw(ArgumentError("some weights should be positive"))
        Float64.(w) .* (count(>(0), w) / sum(w))
    end
    tolv = tol === nothing ? 1e-6 * (quantile(xv, 0.75) - quantile(xv, 0.25)) : Float64(tol)
    (isfinite(tolv) && tolv > 0) || throw(ArgumentError("'tol' must be strictly positive and finite"))

    # --- aggregate ties --------------------------------------------------
    xx = round.((xv .- mean(xv)) ./ tolv)
    ord = sortperm(xv)            # stable
    xxs = xx[ord]
    nd = trues(n)
    for i in 2:n
        nd[i] = xxs[i - 1] < xxs[i]
    end
    group = cumsum(nd)             # group id for sorted position
    nx = group[end]
    nx <= 3 && throw(ArgumentError("need at least four unique 'x' values"))
    ux = xv[ord][nd]
    gw = zeros(nx); gy = zeros(nx); gyy = zeros(nx)
    xo = sortperm(ord)             # sorted position of each original element
    gid = group[xo]                # group of each original observation
    for i in 1:n
        wi = wn === nothing ? 1.0 : wn[i]
        g = gid[i]
        gw[g] += wi; gy[g] += wi * yv[i]; gyy[g] += wi * yv[i]^2
    end
    wbar = gw
    ybar = gy ./ ifelse.(wbar .> 0, wbar, 1.0)
    yssw = sum(gyy .- wbar .* ybar .^ 2)

    r_ux = ux[nx] - ux[1]
    xbar = (ux .- ux[1]) ./ r_ux

    # --- knots -------------------------------------------------------------
    nkn = all_knots ? nx : (nknots isa Function ? nknots(nx) : Int(nknots))
    (1 <= nkn <= nx) || throw(ArgumentError("invalid number of knots"))
    inner = if all_knots || nkn == nx
        xbar
    else
        by = (nx - 1) / (nkn - 1)
        idx = [trunc(Int, 1 + i * by) for i in 0:(nkn - 1)]
        idx[end] = nx
        xbar[idx]
    end
    knots = vcat(fill(xbar[1], 3), inner, fill(xbar[nx], 3))
    nk = nkn + 2

    # --- design (4 nonzeros per row), X'WX and penalty --------------------
    spans = Vector{Int}(undef, nx)
    Bv = Matrix{Float64}(undef, nx, 4)
    for i in 1:nx
        mu = _span(knots, nk, xbar[i])
        spans[i] = mu
        for (c, j) in enumerate((mu - 3):mu)
            Bv[i, c] = _bsp(knots, j, 4, xbar[i], mu)
        end
    end
    H = zeros(nk, nk)
    xwy = zeros(nk)
    for i in 1:nx
        mu = spans[i]
        for a in 1:4, b in 1:4
            H[mu - 4 + a, mu - 4 + b] += wbar[i] * Bv[i, a] * Bv[i, b]
        end
        for a in 1:4
            xwy[mu - 4 + a] += wbar[i] * ybar[i] * Bv[i, a]
        end
    end
    Sg = zeros(nk, nk)
    for mu in 4:nk
        h = knots[mu + 1] - knots[mu]
        h > 0 || continue
        lo = [_bspd(knots, j, 4, 2, knots[mu], mu) for j in (mu - 3):mu]
        hi = [_bspd(knots, j, 4, 2, knots[mu + 1], mu) for j in (mu - 3):mu]
        for a in 1:4, b in 1:4
            # R's `sgram.f` integrates the product of the two linear second
            # derivatives with the constant 0.3330 in place of 1/3; reproduced
            # here so that lambda and the fit agree with R.
            da = hi[a] - lo[a]; db = hi[b] - lo[b]
            Sg[mu - 4 + a, mu - 4 + b] +=
                h * (lo[a] * lo[b] + (da * lo[b] + db * lo[a]) * 0.5 + da * db * 0.333)
        end
    end
    t1 = 0.0; t2 = 0.0
    for i in 3:(nk - 3)
        t1 += H[i, i]; t2 += Sg[i, i]
    end
    ratio = t1 / t2
    spar_is_lambda = lambda !== nothing

    # --- criterion at a given smoothing parameter ---------------------------
    A = similar(H)
    function evaluate(sp::Float64)
        lam = spar_is_lambda ? sp : ratio * 16.0^(sp * 6.0 - 2.0)
        @. A = H + lam * Sg
        ch = cholesky(Symmetric(A); check=false)
        issuccess(ch) || return (Inf, lam, Float64[], Float64[], Float64[])
        coef = ch \ xwy
        Ainv = inv(ch)
        sz = Vector{Float64}(undef, nx)
        lev = Vector{Float64}(undef, nx)
        for i in 1:nx
            mu = spans[i]
            s = 0.0; l = 0.0
            for a in 1:4
                s += coef[mu - 4 + a] * Bv[i, a]
                for b in 1:4
                    l += Bv[i, a] * Bv[i, b] * Ainv[mu - 4 + a, mu - 4 + b]
                end
            end
            sz[i] = s
            lev[i] = l * wbar[i]
        end
        rss = yssw; df = 0.0; sumw = 0.0
        for i in 1:nx
            rss += wbar[i] * (ybar[i] - sz[i])^2
            df += lev[i]
            sumw += wbar[i]
        end
        crit = (rss / sumw) / (1 - (df_offset + penalty * df) / sumw)^2
        return (crit, lam, coef, sz, lev)
    end

    local best_spar, lam_used, coefv, szv, levv
    if spar !== nothing || lambda !== nothing
        sp = Float64(spar !== nothing ? spar : lambda)
        _, lam_used, coefv, szv, levv = evaluate(sp)
        isempty(coefv) && throw(ErrorException("smoothing parameter value too extreme"))
        best_spar = spar_is_lambda ? NaN : sp
    else
        # Brent's fmin, transcribed from R's sbart.c
        c_gold = 0.381966011250105151795413165634
        BIG = 1e100
        a = Float64(cs.low); b = Float64(cs.high)
        v = a + c_gold * (b - a); w_ = v; xb = v
        d = 0.0; e = 0.0
        res = evaluate(xb)
        fx = res[1]; fv = fx; fw = fx
        last = res
        iter = 0
        while true
            xm = (a + b) * 0.5
            tol1 = cs.eps * abs(xb) + cs.tol / 3
            tol2 = tol1 * 2
            iter += 1
            if abs(xb - xm) <= tol2 - (b - a) * 0.5 || iter > cs.maxit
                break
            end
            golden = true
            if abs(e) > tol1 && isfinite(fx) && isfinite(fv) && isfinite(fw)
                r = (xb - w_) * (fx - fv)
                q = (xb - v) * (fx - fw)
                p = (xb - v) * q - (xb - w_) * r
                q = (q - r) * 2
                q > 0 && (p = -p)
                q = abs(q)
                r = e
                e = d
                if !(abs(p) >= abs(0.5 * q * r) || q == 0.0) &&
                   !(p <= q * (a - xb) || p >= q * (b - xb))
                    d = p / q
                    u = xb + d
                    if u - a < tol2 || b - u < tol2
                        d = copysign(tol1, xm - xb)
                    end
                    golden = false
                end
            end
            if golden
                e = xb >= xm ? a - xb : b - xb
                d = c_gold * e
            end
            u = xb + (abs(d) >= tol1 ? d : copysign(tol1, d))
            res = evaluate(u)
            last = res
            fu = res[1]
            isfinite(fu) || (fu = 2 * BIG)
            if fu <= fx
                u >= xb ? (a = xb) : (b = xb)
                v = w_; fv = fw
                w_ = xb; fw = fx
                xb = u; fx = fu
            else
                u < xb ? (a = u) : (b = u)
                if fu <= fw || w_ == xb
                    v = w_; fv = fw
                    w_ = u; fw = fu
                elseif fu <= fv || v == xb || v == w_
                    v = u; fv = fu
                end
            end
        end
        # As in R, spar is the best abscissa while coefficients, leverages and
        # lambda are those of the last function evaluation.
        best_spar = xb
        _, lam_used, coefv, szv, levv = last
        isempty(coefv) && throw(ErrorException("smoothing spline: singular system"))
    end

    df = sum(levv)
    # R-level GCV score on the original (un-aggregated) observations.
    gx = gid
    wfull = wn === nothing ? ones(n) : wn
    resid2 = (yv .- szv[gx]) .^ 2
    wm = wn === nothing ? mean(resid2) : sum(wfull .* resid2) / sum(wfull)
    cvcrit = wm / (1 - (df_offset + penalty * df) / n)^2

    return SmoothSpline(ux, szv, wbar, levv, df, best_spar, lam_used, cvcrit, ratio,
                        knots, coefv, ux[1], r_ux)
end
