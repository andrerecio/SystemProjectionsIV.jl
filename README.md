# SystemProjectionsIV.jl

[![CI](https://github.com/andrerecio/SystemProjectionsIV.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/andrerecio/SystemProjectionsIV.jl/actions/workflows/CI.yml)
[![Docs](https://github.com/andrerecio/SystemProjectionsIV.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/andrerecio/SystemProjectionsIV.jl/actions/workflows/Documentation.yml)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://andrerecio.github.io/SystemProjectionsIV.jl/)
[![codecov](https://codecov.io/gh/andrerecio/SystemProjectionsIV.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/andrerecio/SystemProjectionsIV.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A Julia implementation of the **System Projections IV (SP-IV)** estimator of Lewis & Mertens (2024). SP-IV identifies structural macroeconomic relationships (such as the Phillips Curve) by regressing impulse responses to external instruments on each other, estimated with local projections or VARs. Because identification comes from the responses at multiple horizons, a rich set of lagged controls can be included — so only contemporaneous and lead exogeneity of the instruments is required — without shrinking the instruments' explanatory power to that of the contemporaneous shock, as happens in IV with distributed lags of shocks as instruments.

> **Status:** alpha, under active development. The API is not yet stable.

> **Note:** This is an independent, unofficial implementation of the SP-IV methodology developed by Lewis & Mertens (2024). It is not affiliated with or endorsed by the paper's authors, and any implementation errors are the package author's own.

## Installation

The package is not registered. Install directly from the repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/andrerecio/SystemProjectionsIV.jl")
```

## Quick start

```julia
using SystemProjectionsIV, Random

# Simulate a small SP-IV design: y_t = β·Y_t + u_t, where Y responds to an
# external instrument ε through a known impulse response and is endogenous (Y
# correlates with the structural error u).
Random.seed!(1234)
T, H, β = 400, 8, 0.5
ε = randn(T)                          # external instrument / shock
u = 0.3 .* randn(T)
Y = zeros(T)
for t in 1:T, j in 0:3
    t - j ≥ 1 && (Y[t] += 0.8^j * ε[t - j])
end
Y .+= 0.7 .* u .+ 0.2 .* randn(T)     # endogeneity

y = β .* Y .+ u

# y outcome, Y endogenous regressor(s), ε instrument(s); vectors or matrices
# both work, and an intercept is included by default.
result = spiv(y, Y, ε; H = H, robust = :AR)

result                  # pretty summary: β̂, weak-IV verdict, robust set, IRF shapes
summary(result)         # the same table, R-style
coef(result)            # β̂
confint(result)         # strong-identification CI at `level` (default 0.95)
coeftable(result)       # estimate, std. error, z, p-value, CI
weak_iv_test(result)    # g statistic, critical value, is_weak
robust_inference(result)            # AR confidence set + per-parameter bounds
horizons(result)                    # stacked horizons, here 0:7
irf(result).point                   # outcome IRF point estimates (H × Nz)
irf(result; response = :endogenous) # endogenous-regressor IRFs (H × K × Nz)
```

The StatsBase accessors (`coef`, `vcov`, `stderror`, `confint`, `coeftable`, `coefnames`, `nobs`, `dof_residual`, `residuals`) are re-exported, so `using SystemProjectionsIV` alone is enough.

`spiv` returns an `SPIVResult` carrying the structural estimate β̂, its sandwich covariance under strong identification, residuals, weak-IV diagnostics, robust (AR or KLM) confidence sets, and the implied IRFs with HAC standard errors. A single `level` keyword (default `0.95`) sets the Wald intervals, the robust set, and the IRF bands. `H = 8` stacks horizons `0:7`; a vector stacks exactly those horizons (e.g. `H = 3:4:59`, as in Fieldhouse & Mertens). Pass `hac = :neweywest` (or `:andrews`) for automatic-bandwidth Newey–West IRF standard errors instead of the default fixed lag truncation. With `Plots` loaded, `plot(result)` draws the outcome IRF bands (`plot(result; response = :endogenous)` for the endogenous IRFs).

Controls enter through keywords, with the sample adjusted automatically (Stata/R-style): leading and trailing rows containing NaN — the padding that the exported `lag`/`lags` helpers produce for lags and leads — are dropped from every input. The default controls are intercept-only (`xlags = 0`), which suits a serially uncorrelated shock series as instrument. `xlags = p` controls for `p` lags of all the variables (the same information set as `SPIVwithVAR(p)`), and `X = ...` adds arbitrary extra columns beyond the automatic intercept (`intercept = false` drops the constant; collinear columns are dropped with a warning):

```julia
result = spiv(y, Y, ε; H = H, xlags = 4)             # lags of [y Y ε] as controls
result = spiv(y, Y, ε; H = H, X = lags(Y, 4))        # your own lagged controls
result = spiv(y, Y, ε, SPIVwithVAR(4); H = H)        # VAR(4) forecast errors
```

## Examples

Runnable scripts live under `examples/`:

- `examples/demo.jl` — a text-only walkthrough of the whole API (LP and VAR variants, weak-IV diagnostic, AR/KLM robust sets, IRFs, an overidentified case). Run it in the package environment:

  ```bash
  julia --project examples/demo.jl
  ```

- `examples/plots.jl` — renders the IRF figures via the `Plots` recipe. The package itself depends only on `RecipesBase`, so `Plots` is kept out of its dependencies and lives in a separate `examples/` environment instead:

  ```bash
  julia -e 'using Pkg; Pkg.activate("examples"); Pkg.develop(path="."); Pkg.add("Plots")'
  julia --project=examples examples/plots.jl
  ```

## What it estimates

The structural equation

```
y_t = β' Y_t + u_t
```

is estimated by regressing the **impulse responses** of `y_t` to the external shocks `Z` on the impulse responses of `Y_t` to the same shocks, after residualising on lagged controls `X_{t-1}`. Relative to 2SLS with distributed lags of shocks as instruments, this imposes weaker exogeneity requirements and can improve efficiency and increase effective instrument strength. Instruments can still be weak, hence the weak-IV test and robust confidence sets below. The motivating application is the hybrid New Keynesian Phillips Curve, `y_t = π_t`, `Y_t = (π_{t-1}, π^e_{t+1}, gap_t)`, instrumented with external shocks such as monetary-policy shocks (the paper's U.S. application uses the Angeletos et al. (2020) main business-cycle shock).

## Inference

- **Strong identification.** Sandwich asymptotic variance (eqs. 18–19) with normal confidence intervals. No HAR adjustment is needed *when* the controls include enough lags to make the residualised instrument `z⊥` serially uncorrelated; with the default intercept-only controls this requires a serially uncorrelated shock series, so add `xlags`/`X` for a persistent instrument.
- **Weak identification.** A bias-based first-stage test (bias tolerance `tau`, default `0.10`; size `1 − level`) with a conservative (moment-matched) critical value flags weak instruments. Robust confidence sets are available via either the Anderson–Rubin (`robust = :AR`, default; `χ²` with `H·Nz` df) or Kleibergen KLM (`robust = :KLM`; `χ²` with `K` df, better power when `H·Nz > K`) statistic, computed by grid search with bisection-refined endpoints. A bound of `±Inf` means the set reaches the edge of the search region; under weak instruments pass an explicit range, e.g. `grid = (-5, 5)`, since the default `β̂ ± 10·se` relies on the strong-identification standard error.
- **IRFs.** Responses to a one-standard-deviation shock to the (symmetrically whitened) instruments, with Newey–West HAC standard errors from a delta-method influence function (a package choice — IRF standard errors are not part of the paper's inference theory) and a fixed (`hac = :fixed`, default) or automatic (`:neweywest` / `:andrews`) bandwidth.

See `docs/technical.md` for the full mathematical specification.

## References

- Lewis, D. J., & Mertens, K. (2024). *Dynamic Identification Using System Projections on Instrumental Variables*. Federal Reserve Bank of Dallas Working Paper No. 2204 (revised July 2024). [doi:10.24149/wp2204r3](https://doi.org/10.24149/wp2204r3)
- Fieldhouse, A. J., & Mertens, K. *The Returns to Government R&D: Evidence from U.S. Appropriations Shocks.* Its MIT-licensed MATLAB SP-IV code (vendored under `docs/ext/fieldhouse_mertens/`) is the implementation reference for this package.

If you use this package, please cite the paper and the package — see [`CITATION.bib`](CITATION.bib).

## License

MIT — see [LICENSE](LICENSE).
