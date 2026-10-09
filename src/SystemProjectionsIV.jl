module SystemProjectionsIV

using LinearAlgebra: diag, qr, ColumnNorm, Symmetric, I, inv, kron, tr, eigvals, eigen, dot,
                     norm,
                     mul!
using StatsBase
using StatsBase: coef, vcov, stderror, confint, coeftable, coefnames, nobs, dof_residual,
                 residuals
using StatsFuns: norminvcdf, normccdf, chisqinvcdf
using CovarianceMatrices: aVar, Bartlett, NeweyWest, Andrews
using Printf: @printf, @sprintf
using RecipesBase

include("types.jl")
include("utilities.jl")   # lag/create_lags/companion_form (gragusa-cited), used by var.jl
include("inference.jl")
include("irf.jl")
include("var.jl")         # VAR-variant forecast errors (consumes utilities.jl)
include("spiv.jl")
include("show.jl")
include("recipes.jl")

export AbstractSPIVSpec, SPIVwithLP, SPIVwithVAR
export WeakIVDiagnostic, RobustInference, IRFBlock, SPIVResult
export horizons, weak_iv_test, robust_inference, spec, irf
# StatsBase generics extended for SPIVResult, re-exported so `using SystemProjectionsIV`
# alone is enough.
export coef, vcov, stderror, confint, coeftable, coefnames, nobs, dof_residual, residuals
export spiv
export lag, lags

end # module SystemProjectionsIV
