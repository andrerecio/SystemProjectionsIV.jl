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
```

## Index

```@index
```
