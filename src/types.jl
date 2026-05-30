# Core types for SP-IV.
# See docs/technical.md §3 for the math behind ForecastErrors and SPIVResult fields.

# ---------------------------------------------------------------------------
# Spec hierarchy — used for type-based dispatch between the LP and VAR variants
# of SP-IV (technical.md, appendix A).
# ---------------------------------------------------------------------------

abstract type AbstractSPIVSpec end

struct SPIVwithLP <: AbstractSPIVSpec end

struct SPIVwithVAR <: AbstractSPIVSpec end

# ---------------------------------------------------------------------------
# ForecastErrors: residualised stacked matrices used by the estimator.
# Dimensions follow docs/technical.md §3:
#   y_perp :: H × T_eff
#   Y_perp :: (H·K) × T_eff
#   Z_perp :: Nz × T_eff
# ---------------------------------------------------------------------------

struct ForecastErrors{T<:AbstractFloat}
    y_perp::Matrix{T}
    Y_perp::Matrix{T}
    Z_perp::Matrix{T}
    H::Int
    K::Int
    Nz::Int
end

function ForecastErrors(y_perp::AbstractMatrix{T}, Y_perp::AbstractMatrix{T},
                        Z_perp::AbstractMatrix{T}; H::Int, K::Int, Nz::Int) where {T<:AbstractFloat}
    size(y_perp, 1) == H || throw(DimensionMismatch("y_perp must have H = $H rows, got $(size(y_perp, 1))"))
    size(Y_perp, 1) == H * K || throw(DimensionMismatch("Y_perp must have H·K = $(H*K) rows, got $(size(Y_perp, 1))"))
    size(Z_perp, 1) == Nz || throw(DimensionMismatch("Z_perp must have Nz = $Nz rows, got $(size(Z_perp, 1))"))
    T_eff = size(y_perp, 2)
    size(Y_perp, 2) == T_eff && size(Z_perp, 2) == T_eff ||
        throw(DimensionMismatch("y_perp, Y_perp, Z_perp must share the same number of columns (T_eff)"))
    return ForecastErrors{T}(Matrix(y_perp), Matrix(Y_perp), Matrix(Z_perp), H, K, Nz)
end

# ---------------------------------------------------------------------------
# WeakIVDiagnostic: result of the first-stage weak-IV test (Proposition 7).
# ---------------------------------------------------------------------------

struct WeakIVDiagnostic{T<:AbstractFloat}
    g_min::T
    critical_value::T
    is_weak::Bool
end

WeakIVDiagnostic{T}() where {T<:AbstractFloat} =
    WeakIVDiagnostic{T}(T(NaN), T(NaN), false)

# ---------------------------------------------------------------------------
# RobustInference: AR / KLM robust confidence set output (eqs. 20–21).
# `confidence_set` stores only the accepted grid points (rows = points, cols = parameters).
# `bounds` stores per-parameter (lower, upper) extracted as min/max over the set.
# ---------------------------------------------------------------------------

struct RobustInference{T<:AbstractFloat}
    method::Symbol            # :AR or :KLM (or :none for the Phase 1 stub)
    critical_value::T
    bounds::Matrix{T}         # K × 2
    confidence_set::Matrix{T} # N_accepted × K
    degrees_freedom::Int
end

function RobustInference{T}(K::Int) where {T<:AbstractFloat}
    bounds = fill(T(NaN), K, 2)
    cs = Matrix{T}(undef, 0, K)
    return RobustInference{T}(:none, T(NaN), bounds, cs, 0)
end

# ---------------------------------------------------------------------------
# IRFBlock: point estimate, SE, and confidence band for an IRF.
# Parameterised on dimensionality N:
#   N = 2 for the outcome IRF (H × Nz)
#   N = 3 for the endogenous IRF (H × K × Nz)
# ---------------------------------------------------------------------------

struct IRFBlock{T<:AbstractFloat, N}
    point::Array{T, N}
    se::Array{T, N}
    lower::Array{T, N}
    upper::Array{T, N}

    function IRFBlock{T, N}(point::AbstractArray{T, N}, se::AbstractArray{T, N},
                            lower::AbstractArray{T, N}, upper::AbstractArray{T, N}) where {T<:AbstractFloat, N}
        size(point) == size(se) == size(lower) == size(upper) ||
            throw(DimensionMismatch("IRFBlock components must share the same shape"))
        return new{T, N}(Array(point), Array(se), Array(lower), Array(upper))
    end
end

IRFBlock(point::AbstractArray{T, N}, se::AbstractArray{T, N},
         lower::AbstractArray{T, N}, upper::AbstractArray{T, N}) where {T<:AbstractFloat, N} =
    IRFBlock{T, N}(point, se, lower, upper)

# NaN-filled stub with a given shape — used by SPIVResult before Phase 4 populates IRFs.
function IRFBlock{T}(dims::Vararg{Int, N}) where {T<:AbstractFloat, N}
    z = fill(T(NaN), dims...)
    return IRFBlock{T, N}(z, copy(z), copy(z), copy(z))
end

# ---------------------------------------------------------------------------
# SPIVResult: the bundle returned by spiv().
# Concrete fields throughout — no Union{T, Nothing}. Phase 2 fills β/vcov/residuals;
# Phase 3 fills weak_iv/robust; Phase 4 fills the IRF blocks. Until then, NaN.
# ---------------------------------------------------------------------------

struct SPIVResult{T<:AbstractFloat, S<:AbstractSPIVSpec}
    spec::S
    β::Vector{T}
    vcov::Matrix{T}
    residuals::Matrix{T}       # u_H^⊥ residuals, shape H × T_eff
    weak_iv::WeakIVDiagnostic{T}
    robust::RobustInference{T}
    irf_outcome::IRFBlock{T, 2}
    irf_endogenous::IRFBlock{T, 3}
    H::Int
    K::Int
    Nz::Int
    Nx::Int
    T_eff::Int
end

# Stub constructor — fills inference / IRF fields with NaN content given β + vcov + residuals.
function SPIVResult(spec::S, β::AbstractVector{T}, vcov::AbstractMatrix{T},
                    residuals::AbstractMatrix{T};
                    H::Int, K::Int, Nz::Int, Nx::Int,
                    T_eff::Int = size(residuals, 2)) where {T<:AbstractFloat, S<:AbstractSPIVSpec}
    length(β) == K || throw(DimensionMismatch("β must have length K = $K"))
    size(vcov) == (K, K) || throw(DimensionMismatch("vcov must be K × K = $K × $K"))
    size(residuals, 1) == H || throw(DimensionMismatch("residuals must have H = $H rows"))
    size(residuals, 2) == T_eff || throw(DimensionMismatch("residuals column count must match T_eff = $T_eff"))
    return SPIVResult{T, S}(
        spec,
        Vector(β),
        Matrix(vcov),
        Matrix(residuals),
        WeakIVDiagnostic{T}(),
        RobustInference{T}(K),
        IRFBlock{T}(H, Nz),
        IRFBlock{T}(H, K, Nz),
        H, K, Nz, Nx, T_eff,
    )
end

# ---------------------------------------------------------------------------
# Accessors. StatsBase methods are extended explicitly; SP-IV-specific accessors
# (horizon, weak_iv_test) live in the SystemProjectionsIV namespace.
# ---------------------------------------------------------------------------

StatsBase.coef(r::SPIVResult) = r.β
StatsBase.vcov(r::SPIVResult) = r.vcov
StatsBase.nobs(r::SPIVResult) = r.T_eff
StatsBase.dof_residual(r::SPIVResult) = r.T_eff - r.Nx - r.K
StatsBase.residuals(r::SPIVResult) = r.residuals
StatsBase.stderror(r::SPIVResult) = sqrt.(diag(r.vcov))

function StatsBase.confint(r::SPIVResult; level::Real = 0.95)
    α = 1 - level
    z = norminvcdf(1 - α / 2)
    se = StatsBase.stderror(r)
    return hcat(r.β .- z .* se, r.β .+ z .* se)
end

horizon(r::SPIVResult) = r.H
weak_iv_test(r::SPIVResult) = r.weak_iv
robust_inference(r::SPIVResult) = r.robust
spec(r::SPIVResult) = r.spec
