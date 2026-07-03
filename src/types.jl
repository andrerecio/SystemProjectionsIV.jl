# Core types for SP-IV.
# See docs/technical.md §3 for the math behind ForecastErrors and SPIVResult fields.

# ---------------------------------------------------------------------------
# Spec hierarchy — used for type-based dispatch between the LP and VAR variants
# of SP-IV (technical.md, appendix A).
# ---------------------------------------------------------------------------

"""
    AbstractSPIVSpec

Abstract supertype tagging which variant of the SP-IV estimator to run. The
first step of [`spiv`](@ref) — constructing the residualised forecast errors —
is dispatched on this type; the remaining estimation, inference, and IRF steps
are shared. Concrete subtypes: [`SPIVwithLP`](@ref) and [`SPIVwithVAR`](@ref).
"""
abstract type AbstractSPIVSpec end

"""
    SPIVwithLP() <: AbstractSPIVSpec

Local-projections variant of SP-IV. Forecast errors are built by projecting the
leads `y_{t+h}` and `Y_{t+h}` on the instruments `z_t` and lagged controls
`X_{t-1}` via direct local projections (technical.md §3, eq. A.1). This is the
default spec for [`spiv`](@ref).
"""
struct SPIVwithLP <: AbstractSPIVSpec end

"""
    SPIVwithVAR() <: AbstractSPIVSpec

VAR variant of SP-IV (forecast-error route). A VAR(`p`) is fit by OLS on the
stacked vector `[Y_t; y_t; z_t]` and the `h`-step forecast errors are formed
from its companion MA representation (technical.md §4.2, eqs. A.2–A.3). The
VAR's own lags replace the LP control-residualisation, so the controls `X`
passed to [`spiv`](@ref) are ignored. Select the lag order with the `p` keyword.
"""
struct SPIVwithVAR <: AbstractSPIVSpec end

# ---------------------------------------------------------------------------
# ForecastErrors: residualised stacked matrices used by the estimator.
# ---------------------------------------------------------------------------

"""
    ForecastErrors{T}

Residualised, stacked forecast errors that feed the shared SP-IV estimator
(technical.md §3). Built internally by the variant-specific first step; rarely
constructed by hand. Time runs along the columns.

# Fields
- `y_perp::Matrix{T}`: outcome forecast errors, `H × T_eff`.
- `Y_perp::Matrix{T}`: endogenous-regressor forecast errors, `(H·K) × T_eff`.
- `Z_perp::Matrix{T}`: instrument forecast errors, `Nz × T_eff`.
- `H::Int`: number of leads (horizons).
- `K::Int`: number of endogenous regressors.
- `Nz::Int`: number of instruments.

    ForecastErrors(y_perp, Y_perp, Z_perp; H, K, Nz)

Validate the row dimensions (`H`, `H·K`, `Nz`) and a common column count
(`T_eff`), then store dense copies.
"""
struct ForecastErrors{T <: AbstractFloat}
    y_perp::Matrix{T}
    Y_perp::Matrix{T}
    Z_perp::Matrix{T}
    H::Int
    K::Int
    Nz::Int
end

function ForecastErrors(
        y_perp::AbstractMatrix{T},
        Y_perp::AbstractMatrix{T},
        Z_perp::AbstractMatrix{T};
        H::Int,
        K::Int,
        Nz::Int
) where {T <: AbstractFloat}
    size(y_perp, 1) == H ||
        throw(DimensionMismatch("y_perp must have H = $H rows, got $(size(y_perp, 1))"))
    size(Y_perp, 1) == H * K || throw(
        DimensionMismatch("Y_perp must have H·K = $(H*K) rows, got $(size(Y_perp, 1))"),
    )
    size(Z_perp, 1) == Nz ||
        throw(DimensionMismatch("Z_perp must have Nz = $Nz rows, got $(size(Z_perp, 1))"))
    T_eff = size(y_perp, 2)
    size(Y_perp, 2) == T_eff && size(Z_perp, 2) == T_eff || throw(
        DimensionMismatch(
        "y_perp, Y_perp, Z_perp must share the same number of columns (T_eff)",
    ),
    )
    return ForecastErrors{T}(Matrix(y_perp), Matrix(Y_perp), Matrix(Z_perp), H, K, Nz)
end

# ---------------------------------------------------------------------------
# WeakIVDiagnostic: result of the first-stage weak-IV test (Proposition 7).
# ---------------------------------------------------------------------------

"""
    WeakIVDiagnostic{T}

Outcome of the first-stage weak-instrument test (technical.md §6, Proposition 7
and Theorem 1). Retrieve it from a result with [`weak_iv_test`](@ref).

# Fields
- `g_min::T`: the minimum `g` statistic; reduces to the first-stage `F` in the
  just-identified scalar case.
- `critical_value::T`: the (moment-matched, conservative) critical value it is
  compared against.
- `is_weak::Bool`: `true` when `g_min` falls below `critical_value`, flagging
  weak instruments.
"""
struct WeakIVDiagnostic{T <: AbstractFloat}
    g_min::T
    critical_value::T
    is_weak::Bool
end

function WeakIVDiagnostic{T}() where {T <: AbstractFloat}
    WeakIVDiagnostic{T}(T(NaN), T(NaN), false)
end

# ---------------------------------------------------------------------------
# RobustInference: AR / KLM robust confidence set output (eqs. 20–21).
# ---------------------------------------------------------------------------

"""
    RobustInference{T}

Weak-identification-robust confidence set, obtained by inverting the
Anderson–Rubin (eq. 20) or Kleibergen KLM (eq. 21) statistic over a grid
(technical.md §7). Retrieve it from a result with [`robust_inference`](@ref).

# Fields
- `method::Symbol`: `:AR`, `:KLM`, or `:none` (the unpopulated stub).
- `critical_value::T`: the chi-squared critical value used to accept grid points.
- `bounds::Matrix{T}`: `K × 2` per-parameter `(lower, upper)`, the min/max over
  the accepted set.
- `confidence_set::Matrix{T}`: the accepted grid points, `N_accepted × K`
  (rows = points, columns = parameters).
- `degrees_freedom::Int`: degrees of freedom of the reference distribution.
"""
struct RobustInference{T <: AbstractFloat}
    method::Symbol            # :AR or :KLM (or :none for the Phase 1 stub)
    critical_value::T
    bounds::Matrix{T}         # K × 2
    confidence_set::Matrix{T} # N_accepted × K
    degrees_freedom::Int
end

function RobustInference{T}(K::Int) where {T <: AbstractFloat}
    bounds = fill(T(NaN), K, 2)
    cs = Matrix{T}(undef, 0, K)
    return RobustInference{T}(:none, T(NaN), bounds, cs, 0)
end

# ---------------------------------------------------------------------------
# IRFBlock: point estimate, SE, and confidence band for an IRF.
# ---------------------------------------------------------------------------

"""
    IRFBlock{T, N}

A block of impulse responses with HAC standard errors and confidence bands
(technical.md §3.3). Parameterised on dimensionality `N`:

- `N = 2` for the outcome IRF, shape `H × Nz`;
- `N = 3` for the endogenous IRF, shape `H × K × Nz`.

# Fields
- `point::Array{T, N}`: point estimates.
- `se::Array{T, N}`: Newey–West HAC standard errors.
- `lower::Array{T, N}`, `upper::Array{T, N}`: confidence-band endpoints.

All four arrays share the same shape. Access these on an [`SPIVResult`](@ref) via
`result.irf_outcome` and `result.irf_endogenous`.
"""
struct IRFBlock{T <: AbstractFloat, N}
    point::Array{T, N}
    se::Array{T, N}
    lower::Array{T, N}
    upper::Array{T, N}

    function IRFBlock{T, N}(
            point::AbstractArray{T, N},
            se::AbstractArray{T, N},
            lower::AbstractArray{T, N},
            upper::AbstractArray{T, N}
    ) where {T <: AbstractFloat, N}
        size(point) == size(se) == size(lower) == size(upper) ||
            throw(DimensionMismatch("IRFBlock components must share the same shape"))
        return new{T, N}(Array(point), Array(se), Array(lower), Array(upper))
    end
end

function IRFBlock(
        point::AbstractArray{T, N},
        se::AbstractArray{T, N},
        lower::AbstractArray{T, N},
        upper::AbstractArray{T, N}
) where {T <: AbstractFloat, N}
    IRFBlock{T, N}(point, se, lower, upper)
end

# NaN-filled stub with a given shape — used by SPIVResult before Phase 4 populates IRFs.
function IRFBlock{T}(dims::Vararg{Int, N}) where {T <: AbstractFloat, N}
    z = fill(T(NaN), dims...)
    return IRFBlock{T, N}(z, copy(z), copy(z), copy(z))
end

# ---------------------------------------------------------------------------
# SPIVResult: the bundle returned by spiv().
# Concrete fields throughout — no Union{T, Nothing}.
# ---------------------------------------------------------------------------

"""
    SPIVResult{T, S}

The bundle returned by [`spiv`](@ref). All fields are concrete (no
`Union{T, Nothing}`). It supports the StatsBase accessors `coef`, `vcov`,
`stderror`, `confint`, `nobs`, `dof_residual`, and `residuals`.

# Fields
- `spec::S`: the variant tag ([`SPIVwithLP`](@ref) or [`SPIVwithVAR`](@ref)).
- `β::Vector{T}`: structural estimate β̂, length `K` (eq. 9).
- `vcov::Matrix{T}`: `K × K` sandwich covariance under strong identification
  (eqs. 18–19).
- `residuals::Matrix{T}`: structural residuals `u_H^⊥`, shape `H × T_eff`.
- `weak_iv::WeakIVDiagnostic{T}`: first-stage weak-IV diagnostic.
- `robust::RobustInference{T}`: AR/KLM robust confidence set.
- `irf_outcome::IRFBlock{T, 2}`: outcome IRF, `H × Nz`.
- `irf_endogenous::IRFBlock{T, 3}`: endogenous IRF, `H × K × Nz`.
- `H, K, Nz, Nx, T_eff::Int`: leads, endogenous regressors, instruments,
  controls, and effective sample size.
"""
struct SPIVResult{T <: AbstractFloat, S <: AbstractSPIVSpec}
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
function SPIVResult(
        spec::S,
        β::AbstractVector{T},
        vcov::AbstractMatrix{T},
        residuals::AbstractMatrix{T};
        H::Int,
        K::Int,
        Nz::Int,
        Nx::Int,
        T_eff::Int = size(residuals, 2)
) where {T <: AbstractFloat, S <: AbstractSPIVSpec}
    length(β) == K || throw(DimensionMismatch("β must have length K = $K"))
    size(vcov) == (K, K) || throw(DimensionMismatch("vcov must be K × K = $K × $K"))
    size(residuals, 1) == H || throw(DimensionMismatch("residuals must have H = $H rows"))
    size(residuals, 2) == T_eff ||
        throw(DimensionMismatch("residuals column count must match T_eff = $T_eff"))
    return SPIVResult{T, S}(
        spec,
        Vector(β),
        Matrix(vcov),
        Matrix(residuals),
        WeakIVDiagnostic{T}(),
        RobustInference{T}(K),
        IRFBlock{T}(H, Nz),
        IRFBlock{T}(H, K, Nz),
        H,
        K,
        Nz,
        Nx,
        T_eff
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

"""
    horizon(r::SPIVResult) -> Int

Number of leads (horizons) `H` used in the estimation.
"""
horizon(r::SPIVResult) = r.H

"""
    weak_iv_test(r::SPIVResult) -> WeakIVDiagnostic

The first-stage weak-instrument diagnostic. See [`WeakIVDiagnostic`](@ref).
"""
weak_iv_test(r::SPIVResult) = r.weak_iv

"""
    robust_inference(r::SPIVResult) -> RobustInference

The weak-identification-robust (AR/KLM) confidence set. See
[`RobustInference`](@ref).
"""
robust_inference(r::SPIVResult) = r.robust

"""
    spec(r::SPIVResult) -> AbstractSPIVSpec

The variant tag the result was produced with ([`SPIVwithLP`](@ref) or
[`SPIVwithVAR`](@ref)).
"""
spec(r::SPIVResult) = r.spec

"""
    irf(r::SPIVResult; response = :outcome) -> IRFBlock

The impulse responses to the instrumented shocks, with HAC standard errors and
normal bands: `response = :outcome` (default) returns the outcome IRF (`H × Nz`),
`response = :endogenous` the endogenous-regressor IRF (`H × K × Nz`). See
[`IRFBlock`](@ref) for the `point`/`se`/`lower`/`upper` fields.
"""
function irf(r::SPIVResult; response::Symbol = :outcome)
    response === :outcome && return r.irf_outcome
    response === :endogenous && return r.irf_endogenous
    throw(ArgumentError("response must be :outcome or :endogenous, got :$response"))
end
