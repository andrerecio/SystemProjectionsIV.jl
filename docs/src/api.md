```@meta
CurrentModule = SystemProjectionsIV
```

# API

Complete reference for the exported public API.

## Estimator

```@docs
spiv
```

## Specs

```@docs
AbstractSPIVSpec
SPIVwithLP
SPIVwithVAR
```

## Result types

```@docs
SPIVResult
WeakIVDiagnostic
RobustInference
IRFBlock
```

## Accessors

The result supports the StatsBase accessors `coef`, `vcov`, `stderror`,
`confint`, `coeftable`, `coefnames`, `nobs`, `dof_residual`, and `residuals`,
re-exported so that `using SystemProjectionsIV` alone is enough. `confint` and
`coeftable` default to the `level` passed to [`spiv`](@ref):

```@docs
confint(::SPIVResult)
coeftable(::SPIVResult)
```

SP-IV-specific accessors:

```@docs
horizons
weak_iv_test
robust_inference
spec
irf
Base.summary(::IO, ::SPIVResult)
```

## Data helpers

Utilities for building lagged-control matrices for the `X` keyword of
[`spiv`](@ref). Both pad unavailable observations with `NaN` (leading for lags,
trailing for leads, `n < 0`); [`spiv`](@ref) drops that padding from all inputs
automatically. Integer input is promoted to floating point.

```@docs
lag
lags
```

## Index

```@index
```
