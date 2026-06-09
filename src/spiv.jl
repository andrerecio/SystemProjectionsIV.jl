# SP-IV point estimator — local-projections (LP) and VAR variants.
#
# Implements the closed-form estimator and strong-identification sandwich variance
# from docs/technical.md:
#   - forecast-error construction                         §4.1 (LP, eq. A.1) / §4.2 (VAR, A.2–A.3)
#   - closed-form β̂                                      §3.2 (eq. 9)
#   - feasible sandwich variance                         §5.3–5.4 (eqs. 18–19)
#
# The two variants differ ONLY in step 1 (forecast-error construction, dispatched on the
# spec via `_forecast_errors`); steps 2–5 (estimator, weak-IV, robust inference, IRFs) are
# shared in `_spiv_estimate`. The LP construction lives here; the VAR one in src/var.jl.
#
# Equation-number note: TODO.md refers to β̂ as "eq. 13" and the sandwich as
# "eq. 19". In docs/technical.md, eq. (9) is the GMM closed form actually computed
# here; eq. (13) is the *equivalent* IRF-OLS characterization (cross-checked in the
# tests); eqs. (18)/(19) give the variance.

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Stack H future leads of each column of V (T × m, time in rows) into an
# (H·m) × T_eff matrix, time in columns. For variable k the rows (k-1)H+1 : kH hold
# leads h = 0,…,H-1, i.e. out[(k-1)H + h, t] = V[t + (h-1), k]. Stacking the final
# lead costs the last H-1 rows, so T_eff = T - (H - 1). (docs/technical.md §3.2.)
function _stack_leads(V::AbstractMatrix{T}, H::Int) where {T <: AbstractFloat}
    n, m = size(V)
    T_eff = n - (H - 1)
    out = Matrix{T}(undef, H * m, T_eff)
    for k in 1:m, h in 1:H

        @views out[(k - 1) * H + h, :] .= V[h:(h + T_eff - 1), k]
    end
    return out
end

# Residualize every row of V (vars × T_eff) on the controls in Xc (T_eff × Nx) using
# the FWL annihilator M_X (eq. A.1). With the thin QR factor Q of Xc (orthonormal
# columns spanning the controls), V M_X = V - (V Q) Q'. M_X is never formed.
function _residualize_on(
        V::AbstractMatrix{T}, Q::AbstractMatrix{T}) where {T <: AbstractFloat}
    V .- (V * Q) * Q'
end

# Restriction matrix R = I_K ⊗ vec(I_H) ∈ ℝ^{KH² × K} (docs/technical.md, notation).
# It encodes the constraint that the same β applies at every horizon h = 0,…,H-1.
function _restriction_matrix(::Type{T}, K::Int, H::Int) where {T <: AbstractFloat}
    return kron(Matrix{T}(I, K, K), vec(Matrix{T}(I, H, H)))
end

# ---------------------------------------------------------------------------
# Step 1 — forecast errors (dispatched on the spec). Returns the residualised stacked
# matrices y_perp (H × T_eff), Y_perp (HK × T_eff), Z_perp (Nz × T_eff), the effective
# sample T_eff, and the effective control count Nx used in the dof corrections.
# ---------------------------------------------------------------------------

# LP variant: Jordà local projections of y_{t+h}, Y_{t+h} on z_t, X_{t-1} (eq. A.1).
# `p` is unused (it parameterises the VAR variant only).
function _forecast_errors(
        ::SPIVwithLP,
        yv::AbstractVector{Float64},
        Ym::AbstractMatrix{Float64},
        Xm::AbstractMatrix{Float64},
        Zm::AbstractMatrix{Float64},
        H::Int,
        p::Int
)
    T = length(yv)
    Nx = size(Xm, 2)
    T_eff = T - (H - 1)
    T_eff > 0 || throw(ArgumentError("horizon H = $H leaves no observations (T = $T)"))

    y_H = _stack_leads(reshape(yv, T, 1), H)   # H  × T_eff
    Y_H = _stack_leads(Ym, H)                  # HK × T_eff
    Zc = permutedims(@view Zm[1:T_eff, :])     # Nz × T_eff
    Xc = Xm[1:T_eff, :]                        # T_eff × Nx

    # Thin QR factor of the controls; Q's columns are an orthonormal basis for X.
    Q = Nx == 0 ? Matrix{Float64}(undef, T_eff, 0) : Matrix(qr(Xc).Q)
    y_perp = _residualize_on(y_H, Q)           # H  × T_eff
    Y_perp = _residualize_on(Y_H, Q)           # HK × T_eff
    Z_perp = _residualize_on(Zc, Q)            # Nz × T_eff
    return y_perp, Y_perp, Z_perp, T_eff, Nx
end

# ---------------------------------------------------------------------------
# Steps 2–5 — estimator, variance, weak-IV diagnostic, robust set, IRFs. Shared by both
# variants; operates only on the residualised matrices, so the VAR variant reuses it.
# ---------------------------------------------------------------------------

function _spiv_estimate(
        spec::S,
        y_perp::AbstractMatrix{Float64},
        Y_perp::AbstractMatrix{Float64},
        Z_perp::AbstractMatrix{Float64},
        H::Int,
        K::Int,
        Nz::Int,
        Nx::Int,
        T_eff::Int;
        weak_iv::Symbol,
        α::Real,
        ξ::Real,
        grid_length::Int,
        grid_scale::Real,
        hac::Symbol
) where {S <: AbstractSPIVSpec}
    # --- order condition and degrees of freedom --------------------------
    H * Nz ≥ K || throw(
        ArgumentError(
        "order condition violated: H·Nz = $(H*Nz) < K = $K (docs/technical.md §3.4)"),
    )
    T_eff - Nx - K > 0 || throw(
        ArgumentError(
        "insufficient degrees of freedom: T_eff - Nx - K = $(T_eff - Nx - K) ≤ 0"),
    )
    T_eff - Nx - Nz > 0 || throw(
        ArgumentError(
        "insufficient degrees of freedom for Ŵ₂: T_eff - Nx - Nz = $(T_eff - Nx - Nz) ≤ 0"),
    )

    # --- Gram matrices via P_Z⊥ (§3.2) -----------------------------------
    # Y⊥ P_Z⊥ Y⊥' = (Y⊥ Z⊥')(Z⊥ Z⊥')⁻¹(Z⊥ Y⊥'); the T_eff × T_eff projection is
    # never formed.
    Czz = Symmetric(Z_perp * Z_perp')          # Nz × Nz
    Cyz = Y_perp * Z_perp'                      # HK × Nz
    cyz = y_perp * Z_perp'                      # H  × Nz
    W = Czz \ Cyz'                              # Nz × HK
    YPY = Cyz * W                               # HK × HK  (Y⊥ P_Z⊥ Y⊥')
    yPY = cyz * W                               # H  × HK  (y⊥ P_Z⊥ Y⊥')

    # --- closed-form β̂ (eq. 9) -------------------------------------------
    IH = Matrix{Float64}(I, H, H)
    R = _restriction_matrix(Float64, K, H)     # KH² × K
    A9 = Symmetric(R' * kron(YPY, IH) * R)      # K × K
    β = A9 \ (R' * vec(yPY))                    # K

    # --- structural residuals and Σ̂ (eq. 19) ----------------------------
    û = y_perp - kron(β', IH) * Y_perp         # H × T_eff
    Σ = Symmetric((û * û') ./ (T_eff - Nx - K))  # H × H

    # --- feasible sandwich variance (eqs. 18 + §5.4 plug-in) -------------
    # Var(β̂) = (1/T) A⁻¹ [R'(G ⊗ Σ̂) R] A⁻¹, with G = YPY/T = Θ̂_Y Θ̂_Y' and
    # A = R'(G ⊗ I_H) R = A9 / T.
    A = A9 ./ T_eff
    M = Symmetric(R' * kron(YPY ./ T_eff, Σ) * R)  # K × K
    Ainv = inv(A)
    vcov = Matrix(Symmetric((Ainv * M * Ainv) ./ T_eff))  # K × K, symmetrized

    # --- weak-IV inference (docs/technical.md §6–7) ----------------------
    v_res, W2 = _first_stage_resid_cov(Y_perp, Z_perp, Czz, T_eff, Nx, Nz)
    weak = _weak_iv_diagnostic(YPY, W2, R, H, Nz, ξ, α)
    se = sqrt.(diag(vcov))
    robust = _robust_inference(
        weak_iv,
        β,
        se,
        y_perp,
        Y_perp,
        v_res,
        Z_perp,
        collect(Czz),
        R,
        H,
        K,
        Nz,
        Nx,
        T_eff,
        α,
        grid_length,
        grid_scale
    )

    # --- impulse responses with HAC standard errors (docs/technical.md §3.3) ---
    irf_outcome,
    irf_endogenous = _irf_blocks(
        y_perp, Y_perp, Z_perp, collect(Czz), H, K, Nz, T_eff, α, hac)

    return SPIVResult{Float64, S}(
        spec,
        β,
        vcov,
        û,
        weak,
        robust,
        irf_outcome,
        irf_endogenous,
        H,
        K,
        Nz,
        Nx,
        T_eff
    )
end

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

"""
    spiv(y, Y, X, Z, spec = SPIVwithLP(); H, p = 1, weak_iv = :AR, α = 0.05,
         ξ = 0.10, grid_length = 30, grid_scale = 10, hac = :fixed)

System Projections IV estimator (Lewis & Mertens 2024).

Inputs use the **time-in-rows** convention (`T` periods down the rows):

- `y` — outcome, length-`T` vector.
- `Y` — `T × K` endogenous regressors.
- `X` — `T × Nx` controls (e.g. lagged variables); residualized out via FWL.
- `Z` — `T × Nz` instruments.

`spec` selects the forecast-error construction:

- `SPIVwithLP()` (default) — Jordà local projections of `y_{t+h}, Y_{t+h}` on `z_t` and
  the controls `X_{t-1}` (eq. A.1).
- `SPIVwithVAR()` — a VAR(`p`) on `[Y_t, y_t, z_t]`; forecast errors are built from the
  reduced-form residuals (eqs. A.2–A.3) and the controls `X` are **ignored** (the VAR's
  own lags play that role). `T_eff = T − p − (H − 1)`.

`H` is the projection horizon (number of leads, `h = 0,…,H-1`). Returns an
`SPIVResult{Float64, typeof(spec)}` with the point estimate `β̂` (eq. 9), the
strong-identification sandwich variance (eqs. 18–19), the structural residuals, the
weak-IV diagnostic (Proposition 7), and a robust confidence set.

Keyword arguments:

- `p` — VAR lag order for the `SPIVwithVAR` variant (default 1; ignored for LP).
- `weak_iv` — robust test to invert for the confidence set, `:AR` (default) or `:KLM`.
- `α` — significance level for the weak-IV and robust critical values (default 0.05).
- `ξ` — bias tolerance entering the weak-IV critical value (default 0.10).
- `grid_length` — points per parameter in the confidence-set grid (default 30).
- `grid_scale` — half-width of the grid box in standard errors: each parameter ranges
  over `β̂ ± grid_scale·se` (default 10).
- `hac` — bandwidth rule for the IRF HAC standard errors: `:fixed` (default; lag
  truncation = horizon), `:neweywest` (automatic Newey–West 1994), or `:andrews`
  (automatic Andrews 1991).

The outcome IRF (`H × Nz`) and endogenous IRF (`H × K × Nz`), with per-horizon Newey–West
HAC standard errors and `1 − α` normal bands, are in `result.irf_outcome` /
`result.irf_endogenous` (docs/technical.md §3.3, eq. 11).
"""
function spiv(
        y::AbstractVector{<:Real},
        Y::AbstractMatrix{<:Real},
        X::AbstractMatrix{<:Real},
        Z::AbstractMatrix{<:Real},
        spec::AbstractSPIVSpec = SPIVwithLP();
        H::Int,
        p::Int = 1,
        weak_iv::Symbol = :AR,
        α::Real = 0.05,
        ξ::Real = 0.10,
        grid_length::Int = 30,
        grid_scale::Real = 10,
        hac::Symbol = :fixed
)
    # --- promote to a common float type -----------------------------------
    yv = collect(Float64, y)
    Ym = Matrix{Float64}(Y)
    Xm = Matrix{Float64}(X)
    Zm = Matrix{Float64}(Z)

    T = length(yv)
    K = size(Ym, 2)
    Nz = size(Zm, 2)

    # --- common validation ------------------------------------------------
    H ≥ 1 || throw(ArgumentError("H must be ≥ 1, got $H"))
    p ≥ 1 || throw(ArgumentError("p must be ≥ 1, got $p"))
    hac in (:fixed, :neweywest, :andrews) ||
        throw(ArgumentError("hac must be :fixed, :neweywest, or :andrews, got :$hac"))
    size(Ym, 1) == T ||
        throw(DimensionMismatch("Y must have T = $T rows, got $(size(Ym, 1))"))
    size(Xm, 1) == T ||
        throw(DimensionMismatch("X must have T = $T rows, got $(size(Xm, 1))"))
    size(Zm, 1) == T ||
        throw(DimensionMismatch("Z must have T = $T rows, got $(size(Zm, 1))"))

    # --- step 1 (dispatched) then shared steps 2–5 ------------------------
    y_perp, Y_perp, Z_perp, T_eff, Nx_eff = _forecast_errors(spec, yv, Ym, Xm, Zm, H, p)
    return _spiv_estimate(
        spec,
        y_perp,
        Y_perp,
        Z_perp,
        H,
        K,
        Nz,
        Nx_eff,
        T_eff;
        weak_iv = weak_iv,
        α = α,
        ξ = ξ,
        grid_length = grid_length,
        grid_scale = grid_scale,
        hac = hac
    )
end
