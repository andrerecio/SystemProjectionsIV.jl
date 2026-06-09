# SystemProjectionsIV.jl

[![CI](https://github.com/andrerecio/SystemProjectionsIV.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/andrerecio/SystemProjectionsIV.jl/actions/workflows/CI.yml)
[![Docs](https://github.com/andrerecio/SystemProjectionsIV.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/andrerecio/SystemProjectionsIV.jl/actions/workflows/Documentation.yml)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://andrerecio.github.io/SystemProjectionsIV.jl/)
[![codecov](https://codecov.io/gh/andrerecio/SystemProjectionsIV.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/andrerecio/SystemProjectionsIV.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A Julia implementation of the **System Projections IV (SP-IV)** estimator of Lewis & Mertens (2024). SP-IV identifies structural macroeconomic relationships (such as the Phillips Curve) by exploiting external instruments through impulse responses rather than raw variables, conditioning on a rich set of predetermined controls so that many lags of instruments can be used without losing identifying power.

> **Status:** alpha, under active development. The API is not yet stable.

## Installation

The package is not registered. Install directly from the repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/andrerecio/SystemProjectionsIV.jl")
```

## Quick start

```julia
using SystemProjectionsIV, StatsBase, Random

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

# y (T,), Y (T,K), X (T,Nx) controls, Z (T,Nz) instruments, H leads.
result = spiv(y, reshape(Y, T, 1), ones(T, 1), reshape(ε, T, 1); H = H, weak_iv = :AR)

result                  # pretty summary: β̂, weak-IV verdict, robust set, IRF shapes
coef(result)            # β̂
confint(result)         # strong-identification 95% CI
weak_iv_test(result)    # g statistic, critical value, is_weak
robust_inference(result)            # AR confidence set + per-parameter bounds
result.irf_outcome.point            # outcome IRF point estimates (H × Nz)
```

`spiv` returns an `SPIVResult` carrying the structural estimate β̂, its sandwich covariance under strong identification, residuals, weak-IV diagnostics, robust (AR or KLM) confidence sets, and the implied IRFs with HAC standard errors. Pass `hac = :neweywest` (or `:andrews`) for automatic-bandwidth Newey–West IRF standard errors instead of the default fixed lag truncation. With `Plots` loaded, `plot(result)` draws the outcome IRF bands (`plot(result; response = :endogenous)` for the endogenous IRFs).

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

is estimated by regressing the **impulse responses** of `y_t` to the external shocks `Z` on the impulse responses of `Y_t` to the same shocks, after residualising on lagged controls `X_{t-1}`. This bypasses the weak-instrument problem that afflicts standard 2SLS with distributed-lag instruments. The canonical empirical application is the Phillips Curve, where `y_t = π_t`, `Y_t = (π_{t-1}, E_t π_{t+1}, gap_t)`, and `Z_t` is a vector of monetary-policy shocks.

## Inference

- **Strong identification.** Sandwich asymptotic variance with no HAR adjustment (lagged controls absorb the autocorrelation); normal confidence intervals always reported.
- **Weak identification.** A bias-based first-stage test with a conservative (moment-matched) critical value flags weak instruments. Robust confidence sets are available via either the Anderson–Rubin (`weak_iv = :AR`) or Kleibergen LM (`weak_iv = :KLM`) statistic, computed by grid search.
- **IRFs.** Newey–West HAC standard errors on the implied outcome and endogenous impulse responses, with a fixed (`hac = :fixed`, default) or automatic (`:neweywest` / `:andrews`) bandwidth.

See `docs/technical.md` for the full mathematical specification.

## References

- Lewis, D. J., & Mertens, K. (2024). *System Projections IV*. Federal Reserve Bank of Dallas Working Paper.

## License

MIT — see [LICENSE](LICENSE).
