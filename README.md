# SystemProjectionsIV.jl

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
using SystemProjectionsIV

# y :: Vector{Float64}              outcome variable             (T,)
# Y :: Matrix{Float64}              endogenous regressors        (T, K)
# X :: Matrix{Float64}              predetermined controls       (T, Nx)
# Z :: Matrix{Float64}              external instruments         (T, Nz)
# H :: Int                          forecast horizon (number of leads)

result = spiv(y, Y, X, Z; H = 10, weak_iv = :AR)

summary(result)
```

`spiv` returns an `SPIVResult` carrying the structural estimate β̂, its sandwich covariance under strong identification, residuals, weak-IV diagnostics, robust (AR or KLM) confidence sets, and the implied IRFs with HAC standard errors.

## What it estimates

The structural equation

```
y_t = β' Y_t + u_t
```

is estimated by regressing the **impulse responses** of `y_t` to the external shocks `Z` on the impulse responses of `Y_t` to the same shocks, after residualising on lagged controls `X_{t-1}`. This bypasses the weak-instrument problem that afflicts standard 2SLS with distributed-lag instruments. The canonical empirical application is the Phillips Curve, where `y_t = π_t`, `Y_t = (π_{t-1}, E_t π_{t+1}, gap_t)`, and `Z_t` is a vector of monetary-policy shocks.

## Inference

- **Strong identification.** Sandwich asymptotic variance with no HAR adjustment (lagged controls absorb the autocorrelation); normal confidence intervals always reported.
- **Weak identification.** A bias-based first-stage test with an Imhof-bounded critical value flags weak instruments. Robust confidence sets are available via either the Anderson–Rubin (`weak_iv = :AR`) or Kleibergen LM (`weak_iv = :KLM`) statistic, computed by grid search.
- **IRFs.** Newey–West HAC standard errors on the implied outcome and endogenous impulse responses.

See `docs/technical.md` for the full mathematical specification.

## References

- Lewis, D. J., & Mertens, K. (2024). *System Projections IV*. Federal Reserve Bank of Dallas Working Paper.

## License

MIT — see [LICENSE](LICENSE).
