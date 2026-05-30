module SystemProjectionsIV

using LinearAlgebra: diag, qr, Symmetric, I, inv, kron
using StatsBase
using StatsFuns: norminvcdf

using MacroEconometricTools
using LocalProjections
using Regress

include("types.jl")
include("spiv.jl")

export AbstractSPIVSpec, SPIVwithLP, SPIVwithVAR
export ForecastErrors, WeakIVDiagnostic, RobustInference, IRFBlock, SPIVResult
export horizon, weak_iv_test, robust_inference, spec
export spiv

# src/utilities.jl is a working copy of utilities adapted from MacroEconometricTools.
# Still not included — see TODO.md Phase 5 audit.

end # module SystemProjectionsIV
