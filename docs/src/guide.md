```@meta
CurrentModule = SystemProjectionsIV
```

# Guide

This guide walks through the public surface of the package on synthetic data
with a known structural coefficient β, so each step can be sanity-checked
against the truth. A fuller, text-only script lives at `examples/demo.jl` in the
repository.

## A synthetic design

We simulate `y_t = β·Y_t + u_t` where `Y` responds to an exogenous instrument
`ε` through a known impulse response and is endogenous (it loads on the
structural error `u`).

```@example guide
using SystemProjectionsIV, Random

function make_dgp(; T, β_true, a, ρ, loading = 1.0, seed = 20240530)
    Random.seed!(seed)
    ε = randn(T)                       # exogenous instrument / shock
    u = 0.3 .* randn(T)                # structural error
    ν = 0.2 .* randn(T)                # extra Y noise
    Yv = zeros(T)
    for t in 1:T, j in 0:(length(a) - 1)
        t - j ≥ 1 && (Yv[t] += loading * a[j + 1] * ε[t - j])
    end
    Yv .+= ρ .* u .+ ν                 # ρ ≠ 0 ⇒ endogeneity
    yv = β_true .* Yv .+ u
    return yv, Yv, ε
end

T, H, β_true = 4000, 4, 0.7
a = [1.0, 0.6, 0.3, 0.15]              # IRF of Y to the shock, horizons 0..H-1
y, Y, Z = make_dgp(; T = T, β_true = β_true, a = a, ρ = 0.8)
nothing # hide
```

## LP variant (default)

[`spiv`](@ref) defaults to the local-projections variant ([`SPIVwithLP`](@ref)).
Inputs are `y` (length `T`), `Y` (`T × K` endogenous), and `Z` (`T × Nz`
instruments) — single series can be plain vectors. An intercept is included in
the controls by default.

```@example guide
res_lp = spiv(y, Y, Z; H = H)
coef(res_lp)        # β̂ — compare to the true β = 0.7
```

Additional controls enter through keywords, and the sample is adjusted
automatically (Stata/R-style): leading and trailing rows containing `NaN` — the
padding the exported [`lag`](@ref)/[`lags`](@ref) helpers produce for lags and
leads — are dropped from every input (`NaN` inside the sample is an error). The
trim is applied to all inputs jointly, so an instrument that ends a few periods
before `y` also shortens the outcome sample (the Fieldhouse–Mertens code keeps
those outcome leads; the difference is at most `max(h)` observations).
`xlags = p` controls for `p` lags of all the variables `[y Y Z]` (the same
information set as the VAR variant below), while `X = ...` passes arbitrary
extra columns (`intercept = false` drops the automatic constant; collinear
columns, e.g. a constant in `X`, are dropped with a warning and `Nx` becomes their
rank — the dof corrections then differ slightly from code that counts every column;
instruments spanned by the controls are an error):

```@example guide
res_ctrl = spiv(y, Y, Z; H = H, xlags = 4)     # lags of [y Y Z] as controls
coef(res_ctrl)
```

```@example guide
res_lagX = spiv(y, Y, Z; H = H, X = lags(Y, 4))  # your own lagged controls
coef(res_lagX)
```

The result supports the usual StatsBase accessors (re-exported by the package):

```@example guide
coeftable(res_lp)
```

`confint`, `coeftable`, the robust set, and the IRF bands all use the `level`
passed to [`spiv`](@ref) (default `0.95`). Wald intervals use normal quantiles;
the Fieldhouse–Mertens MATLAB code uses Student-`t` with `dof_residual` degrees of
freedom, a difference that vanishes with `T_eff`.

`summary` prints the full estimation table, R-style:

```@example guide
summary(res_lp)
```

## Choosing horizons

`H` sets the horizons stacked in the estimation. An integer means `h = 0, …, H-1`;
a strictly increasing vector of horizons `≥ 0` stacks exactly those. Fieldhouse &
Mertens, for instance, stack every fourth quarter, `H = 3:4:59`. The effective
sample loses `maximum(horizons)` observations at the end, and [`horizons`](@ref)
returns what was used:

```@example guide
res_sub = spiv(y, Y, Z; H = [0, 2, 3])
(horizons(res_sub), coef(res_sub))
```

Row `i` of every IRF block corresponds to `horizons(r)[i]`.

## VAR variant

Passing [`SPIVwithVAR`](@ref) fits a VAR(`p`) on the stacked `[Y; y; z]` and
builds forecast errors from its MA representation; the controls (`X`,
`intercept`, `xlags`) are ignored with a warning because the VAR's own lags
subsume them. On the same data it should give a comparable estimate:

```@example guide
res_var = spiv(y, Y, Z, SPIVwithVAR(1); H = H)   # VAR(1); SPIVwithVAR(p) for p lags
(lp = coef(res_lp)[1], var = coef(res_var)[1])
```

## Serial correlation and controls

The strong-identification standard errors carry no HAR correction: they are valid
when the controls absorb the serial correlation of the residualised instrument
`z⊥` (`docs/technical.md` §5.4). The default controls are intercept-only
(`xlags = 0`), which suits a serially uncorrelated shock series as instrument —
as in the design above. With a persistent instrument, add lagged controls via
`xlags` or `X`; `xlags = p` conditions on the same information as
`SPIVwithVAR(p)`.

## Forward-transformed outcomes

Outcomes are often transformed with leads, e.g. a forward moving average
`ȳ_t = (y_t + y_{t+1} + y_{t+2} + y_{t+3})/4`, built with [`lag`](@ref) and a
negative shift. The trailing `NaN` padding is dropped automatically:

```@example guide
ma4(x) = (x .+ lag(x, -1) .+ lag(x, -2) .+ lag(x, -3)) ./ 4
res_ma = spiv(ma4(y), ma4(Y), Z; H = 0:4:8, X = lags(hcat(y, Y, Z), 4))
coef(res_ma)
```

`xlags` lags the series *exactly as passed*, so with forward-transformed `y`/`Y`
it would put future values into the controls. Build the controls from the raw
series with `X = lags(...)` instead, as above. Compare:

```@example guide
coef(spiv(ma4(y), ma4(Y), Z; H = 0:4:8, xlags = 4))   # biased: controls see the future
```

## Weak-instrument diagnostic

[`weak_iv_test`](@ref) returns a [`WeakIVDiagnostic`](@ref) flagging weak
instruments, with bias tolerance `tau` (default `0.10`) and size `1 - level`. The
design above is strongly identified:

```@example guide
weak_iv_test(res_lp)
```

Shrinking the instrument loading produces a weak design where `is_weak` flips to
`true`:

```@example guide
yw, Yw, Zw = make_dgp(; T = T, β_true = β_true, a = a, ρ = 0.8,
    loading = 0.02, seed = 11)
weak_iv_test(spiv(yw, Yw, Zw; H = H))
```

## Robust confidence sets

Under weak identification, invert the Anderson–Rubin (`robust = :AR`, default) or
Kleibergen LM (`robust = :KLM`) statistic over a grid. [`robust_inference`](@ref)
returns a [`RobustInference`](@ref) with per-parameter bounds; finite endpoints
are refined by bisection between the last accepted and first rejected grid point.

```@example guide
robust_inference(spiv(y, Y, Z; H = H, robust = :AR)).bounds
```

```@example guide
robust_inference(spiv(y, Y, Z; H = H, robust = :KLM)).bounds
```

**AR or KLM.** AR (`χ²` with `H·Nz` df) is fully robust and its sets are easy to
read. KLM (`χ²` with `K` df) is more powerful when `H·Nz ≫ K`, but as a score test
it can also accept points far from β̂, producing disconnected sets.

**Search region.** `grid` sets where the test is inverted: a number `c` means
`β̂ ± c·se` (default `10`), a tuple `(lo, hi)` applies to every parameter, and a
`K × 2` matrix gives per-parameter `[lo hi]` rows. `grid_length` sets the points
per parameter (default 201 for `K = 1`, 101 for `K = 2`, 31 otherwise). Under weak
instruments the strong-identification `se` is unreliable, so the default region
can be too narrow. A bound of `±Inf` means the accepted set reaches the edge of
the region — it is unbounded, or extends beyond the grid, in that direction:

```@example guide
res_w = spiv(yw, Yw, Zw; H = H, robust = :KLM)    # default grid: β̂ ± 10·se
robust_inference(res_w).bounds
```

Passing an explicit range resolves the upper end:

```@example guide
res_w = spiv(yw, Yw, Zw; H = H, robust = :KLM, grid = (-10, 10))
rb = robust_inference(res_w)
(bounds = rb.bounds, disconnected = rb.disconnected)
```

When `disconnected` is `true` the accepted points form more than one piece;
`bounds` then report the hull (a conservative interval spanning the gaps), and the
pieces can be read off `rb.confidence_set`. The printed summary flags both cases.

## Impulse responses

The result carries the implied IRFs with Newey–West HAC standard errors as
[`IRFBlock`](@ref)s, returned by [`irf`](@ref): the outcome IRF (`H × Nz`,
the default) and the endogenous-regressor IRF (`H × K × Nz`,
`response = :endogenous`). Row `i` is horizon `horizons(r)[i]`.

**Units.** Column `n` is the response to a one-standard-deviation shock to the
`n`-th symmetrically whitened instrument `(Z⊥Z⊥'/T)^{-1/2} z⊥_t` (eq. 11); with a
single instrument, a one-s.d. shock to `z⊥`. This is the scaling of the paper
and of the Fieldhouse–Mertens code.

**Standard errors** are a package choice — the paper and the Fieldhouse–Mertens
code report point IRFs only. They are per-horizon Newey–West HAC sandwiches
applied to a delta-method influence function that accounts for the estimated
whitening matrix (treating it as known under-covers).

```@example guide
irf(res_lp).point               # outcome IRF point estimates
```

```@example guide
irf(res_lp).se                  # HAC standard errors
```

The HAC bandwidth defaults to a fixed lag truncation equal to the horizon `h`
(`hac = :fixed`; horizon 0 is White/HC0); pass
`hac = :neweywest` or `hac = :andrews` for automatic-bandwidth selection. The
Bartlett weights down-weight the overlap autocovariances, so when those are large
(e.g. an outcome driven by several lags of the shock) the bands can under-cover
somewhat at short horizons. With
`Plots` loaded, `plot(result)` draws the outcome IRF bands against
`horizons(result)` and
`plot(result; response = :endogenous)` the endogenous IRFs (see
`examples/plots.jl`).
