module SystemProjectionsIV

using LinearAlgebra: diag, qr, Symmetric, I, inv, kron, tr, eigvals, dot
using StatsBase
using StatsFuns: norminvcdf, chisqinvcdf
using CovarianceMatrices: aVar, Bartlett, NeweyWest, Andrews
using Printf: @printf, @sprintf
using RecipesBase

include("types.jl")
include("inference.jl")
include("irf.jl")
include("spiv.jl")
include("show.jl")
include("recipes.jl")

export AbstractSPIVSpec, SPIVwithLP, SPIVwithVAR
export ForecastErrors, WeakIVDiagnostic, RobustInference, IRFBlock, SPIVResult
export horizon, weak_iv_test, robust_inference, spec
export spiv

# src/utilities.jl holds lag/matrix/companion utilities adapted (and cited) from
# gragusa MacroEconometricsTools.jl. It is intentionally not `include`d yet — no
# code needs it until the Phase 5 VAR variant.

end # module SystemProjectionsIV
