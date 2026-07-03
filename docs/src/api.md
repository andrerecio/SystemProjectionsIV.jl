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
ForecastErrors
```

## Accessors

The result supports the StatsBase accessors `coef`, `vcov`, `stderror`,
`confint`, `nobs`, `dof_residual`, and `residuals`, plus the SP-IV-specific
accessors below.

```@docs
horizon
weak_iv_test
robust_inference
spec
irf
Base.summary(::IO, ::SPIVResult)
```

## Data helpers

Utilities for building lagged-control matrices for the `X` keyword of
[`spiv`](@ref). Both pad unavailable initial observations with `NaN`;
[`spiv`](@ref) drops that burn-in from all inputs automatically.

```@docs
lag
lags
```

## Index

```@index
```
