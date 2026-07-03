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
using SystemProjectionsIV, StatsBase, Random

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
automatically (Stata/R-style): leading rows containing `NaN` — the burn-in the
exported [`lags`](@ref) helper pads with — are dropped from every input.
`xlags = p` controls for `p` lags of all the variables `[y Y Z]` (the same
information set as the VAR variant below), while `X = ...` passes arbitrary
extra columns (`intercept = false` drops the automatic constant):

```@example guide
res_ctrl = spiv(y, Y, Z; H = H, xlags = 4)     # lags of [y Y Z] as controls
coef(res_ctrl)
```

```@example guide
res_lagX = spiv(y, Y, Z; H = H, X = lags(Y, 4))  # your own lagged controls
coef(res_lagX)
```

The result supports the usual StatsBase accessors:

```@example guide
[coef(res_lp) stderror(res_lp) confint(res_lp)]
```

and `summary` prints the full estimation table, R-style:

```@example guide
summary(res_lp)
```

## VAR variant

Passing [`SPIVwithVAR`](@ref) fits a VAR(`p`) on the stacked `[Y; y; z]` and
builds forecast errors from its MA representation; the controls (`X`,
`intercept`, `xlags`) are ignored with a warning because the VAR's own lags
subsume them. On the same data it should give a comparable estimate:

```@example guide
res_var = spiv(y, Y, Z, SPIVwithVAR(); H = H, p = 1)
(lp = coef(res_lp)[1], var = coef(res_var)[1])
```

## Weak-instrument diagnostic

[`weak_iv_test`](@ref) returns a [`WeakIVDiagnostic`](@ref) flagging weak
instruments. The design above is strongly identified:

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

Under weak identification, invert the Anderson–Rubin (`weak_iv = :AR`) or
Kleibergen LM (`weak_iv = :KLM`) statistic over a grid. [`robust_inference`](@ref)
returns a [`RobustInference`](@ref) with per-parameter bounds:

```@example guide
robust_inference(spiv(y, Y, Z; H = H, weak_iv = :AR)).bounds
```

```@example guide
robust_inference(spiv(y, Y, Z; H = H, weak_iv = :KLM)).bounds
```

## Impulse responses

The result carries the implied IRFs with Newey–West HAC standard errors as
[`IRFBlock`](@ref)s, returned by [`irf`](@ref): the outcome IRF (`H × Nz`,
the default) and the endogenous-regressor IRF (`H × K × Nz`,
`response = :endogenous`).

```@example guide
irf(res_lp).point               # outcome IRF point estimates
```

```@example guide
irf(res_lp).se                  # HAC standard errors
```

The HAC bandwidth defaults to a fixed lag truncation (`hac = :fixed`); pass
`hac = :neweywest` or `hac = :andrews` for automatic-bandwidth selection. With
`Plots` loaded, `plot(result)` draws the outcome IRF bands and
`plot(result; response = :endogenous)` the endogenous IRFs (see
`examples/plots.jl`).
