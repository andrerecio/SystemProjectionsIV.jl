module SystemProjectionsIV

using LinearAlgebra: diag, qr, Symmetric, I, inv, kron, tr, eigvals, dot, mul!
using StatsBase
using StatsFuns: norminvcdf, chisqinvcdf
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
export ForecastErrors, WeakIVDiagnostic, RobustInference, IRFBlock, SPIVResult
export horizon, weak_iv_test, robust_inference, spec, irf
export spiv
export lag, lags

end # module SystemProjectionsIV
