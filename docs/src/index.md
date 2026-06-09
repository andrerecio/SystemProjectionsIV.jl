```@meta
CurrentModule = SystemProjectionsIV
```

# SystemProjectionsIV.jl

A Julia implementation of the **System Projections IV (SP-IV)** estimator of
Lewis & Mertens (2024). SP-IV identifies structural macroeconomic relationships
(such as the Phillips Curve) by exploiting external instruments through impulse
responses rather than raw variables, conditioning on a rich set of predetermined
controls so that many lags of instruments can be used without losing identifying
power.

The structural equation

```math
y_t = \beta' Y_t + u_t
```

is estimated by regressing the impulse responses of ``y_t`` to the external
shocks ``Z`` on the impulse responses of ``Y_t`` to the same shocks, after
residualising on lagged controls ``X_{t-1}``. This bypasses the weak-instrument
problem that afflicts standard 2SLS with distributed-lag instruments.

## Installation

The package is not registered. Install directly from the repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/andrerecio/SystemProjectionsIV.jl")
```

## Quick start

```@example quickstart
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
```

The returned [`SPIVResult`](@ref) carries the structural estimate, its sandwich
covariance under strong identification, residuals, the weak-IV diagnostic, the
robust (AR or KLM) confidence set, and the implied IRFs with HAC standard errors:

```@example quickstart
coef(result)            # β̂
```

```@example quickstart
confint(result)         # strong-identification 95% CI
```

```@example quickstart
weak_iv_test(result)    # g statistic, critical value, is_weak
```

```@example quickstart
result.irf_outcome.point   # outcome IRF point estimates (H × Nz)
```

See the [Guide](@ref) for the LP and VAR variants, weak-instrument-robust
inference, and IRF options, and the [API](@ref) for the full reference. The
complete mathematical specification lives in `docs/technical.md` in the
repository.

## References

- Lewis, D. J., & Mertens, K. (2024). *System Projections IV*. Federal Reserve
  Bank of Dallas Working Paper.
