# Core types for SP-IV.
# See docs/technical.md §3 for the math behind the SPIVResult fields.

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
`X_{t-1}` via direct local projections (technical.md §4.1, eq. A.1). This is the
default spec for [`spiv`](@ref).
"""
struct SPIVwithLP <: AbstractSPIVSpec end

"""
    SPIVwithVAR(p = 1) <: AbstractSPIVSpec

VAR variant of SP-IV (forecast-error route). A VAR(`p`) is fit by OLS on the
stacked vector `[Y_t; y_t; z_t]` and the `h`-step forecast errors are formed
from its companion MA representation (technical.md §4.2, eqs. A.2–A.3). The
VAR's own lags replace the LP control-residualisation, so the controls `X`
passed to [`spiv`](@ref) are ignored.

```julia
spiv(y, Y, z, SPIVwithVAR(4); H = 8)   # VAR(4) forecast errors
```
"""
struct SPIVwithVAR <: AbstractSPIVSpec
    p::Int

    function SPIVwithVAR(p::Integer = 1)
        p ≥ 1 || throw(ArgumentError("VAR lag order p must be ≥ 1, got $p"))
        return new(Int(p))
    end
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
- `bounds::Matrix{T}`: `K × 2` per-parameter `(lower, upper)`: the hull of the
  accepted set, with each finite endpoint refined by bisection between the last
  accepted and first rejected grid point. A bound of `-Inf`/`Inf` means the set
  reaches the edge of the search grid — it is unbounded (or extends beyond the grid)
  in that direction; widen `grid` in [`spiv`](@ref) to explore further. `NaN` when no
  grid point is accepted (the set is empty, or narrower than the grid step). For
  `K ≥ 2` the refinement runs along grid lines, so bounds are accurate to the grid
  resolution.
- `confidence_set::Matrix{T}`: the accepted grid points, `N_accepted × K`
  (rows = points, columns = parameters).
- `degrees_freedom::Int`: degrees of freedom of the reference distribution.
- `disconnected::Bool`: `true` when the accepted grid points form more than one
  connected piece; `bounds` then span the gaps (a conservative hull) and the
  pieces can be read off `confidence_set`.
"""
struct RobustInference{T <: AbstractFloat}
    method::Symbol            # :AR or :KLM (or :none for the unpopulated stub)
    critical_value::T
    bounds::Matrix{T}         # K × 2
    confidence_set::Matrix{T} # N_accepted × K
    degrees_freedom::Int
    disconnected::Bool
end

function RobustInference{T}(K::Int) where {T <: AbstractFloat}
    bounds = fill(T(NaN), K, 2)
    cs = Matrix{T}(undef, 0, K)
    return RobustInference{T}(:none, T(NaN), bounds, cs, 0, false)
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

Row `i` is horizon `horizons(r)[i]`. Column `n` is the response to a one-standard-
deviation shock to the `n`-th symmetrically whitened instrument
`(Z⊥Z⊥'/T)^{-1/2} z⊥_t` (eq. 11); with a single instrument, a one-s.d. shock to `z⊥`.

# Fields
- `point::Array{T, N}`: point estimates.
- `se::Array{T, N}`: Newey–West HAC standard errors (delta-method, accounting for
  the estimated whitening matrix).
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
`stderror`, `confint`, `coeftable`, `coefnames`, `nobs`, `dof_residual`, and
`residuals`.

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
- `horizons::Vector{Int}`: the stacked horizons (`0:H-1` by default).
- `level::T`: confidence level shared by `confint`, the robust set, and the IRF
  bands.
- `H, K, Nz, Nx, T_eff::Int`: number of horizons, endogenous regressors,
  instruments, controls (the rank of the control matrix), and effective sample size.
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
    horizons::Vector{Int}
    level::T
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
        T_eff::Int = size(residuals, 2),
        horizons::AbstractVector{<:Integer} = 0:(H - 1),
        level::Real = 0.95
) where {T <: AbstractFloat, S <: AbstractSPIVSpec}
    length(β) == K || throw(DimensionMismatch("β must have length K = $K"))
    size(vcov) == (K, K) || throw(DimensionMismatch("vcov must be K × K = $K × $K"))
    size(residuals, 1) == H || throw(DimensionMismatch("residuals must have H = $H rows"))
    size(residuals, 2) == T_eff ||
        throw(DimensionMismatch("residuals column count must match T_eff = $T_eff"))
    length(horizons) == H ||
        throw(DimensionMismatch("horizons must have length H = $H"))
    return SPIVResult{T, S}(
        spec,
        Vector(β),
        Matrix(vcov),
        Matrix(residuals),
        WeakIVDiagnostic{T}(),
        RobustInference{T}(K),
        IRFBlock{T}(H, Nz),
        IRFBlock{T}(H, K, Nz),
        collect(Int, horizons),
        T(level),
        H,
        K,
        Nz,
        Nx,
        T_eff
    )
end

# ---------------------------------------------------------------------------
# Accessors. StatsBase methods are extended explicitly; SP-IV-specific accessors
# (horizons, weak_iv_test, robust_inference, spec, irf) live in the SystemProjectionsIV namespace.
# ---------------------------------------------------------------------------

StatsBase.coef(r::SPIVResult) = r.β
StatsBase.vcov(r::SPIVResult) = r.vcov
StatsBase.nobs(r::SPIVResult) = r.T_eff
StatsBase.dof_residual(r::SPIVResult) = r.T_eff - r.Nx - r.K
StatsBase.residuals(r::SPIVResult) = r.residuals
StatsBase.stderror(r::SPIVResult) = sqrt.(diag(r.vcov))
StatsBase.coefnames(r::SPIVResult) = ["β[$k]" for k in 1:(r.K)]

"""
    confint(r::SPIVResult; level = r.level) -> Matrix

`K × 2` strong-identification (Wald) confidence intervals `β̂ ± z·se`, using normal
critical values (the asymptotic theory of docs/technical.md §5.3). The default level
is the one passed to [`spiv`](@ref). The Fieldhouse–Mertens MATLAB code uses
Student-`t` quantiles with `dof_residual(r)` degrees of freedom instead; the
difference vanishes with `T_eff`. Under weak instruments use
[`robust_inference`](@ref).
"""
function StatsBase.confint(r::SPIVResult; level::Real = r.level)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    α = 1 - level
    z = norminvcdf(1 - α / 2)
    se = StatsBase.stderror(r)
    return hcat(r.β .- z .* se, r.β .+ z .* se)
end

"""
    coeftable(r::SPIVResult; level = r.level) -> StatsBase.CoefTable

Coefficient table (estimate, std. error, `z`, two-sided normal p-value, Wald
confidence interval) under strong identification.
"""
function StatsBase.coeftable(r::SPIVResult; level::Real = r.level)
    se = StatsBase.stderror(r)
    z = r.β ./ se
    p = 2 .* normccdf.(abs.(z))
    ci = StatsBase.confint(r; level = level)
    pct = @sprintf("%g", 100 * level)
    return StatsBase.CoefTable(
        [r.β, se, z, p, ci[:, 1], ci[:, 2]],
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $pct%", "Upper $pct%"],
        StatsBase.coefnames(r),
        4,
        3
    )
end

"""
    horizons(r::SPIVResult) -> Vector{Int}

The horizons `h` stacked in the estimation (`0:H-1` unless a vector was passed as
`H` to [`spiv`](@ref)).
"""
horizons(r::SPIVResult) = r.horizons

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
