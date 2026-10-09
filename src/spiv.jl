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
# Eq. (9) is the GMM closed form computed here; eq. (13) is the equivalent IRF-OLS form
# used by the Fieldhouse–Mertens MATLAB (docs/ext/fieldhouse_mertens/Make_Table_1.m,
# `run_SPIV_total`), cross-checked in the tests.

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Stack the leads `hs` (e.g. 0:H-1, or a subset such as 3:4:59) of each column of V
# (T × m, time in rows) into an (H·m) × T_eff matrix, time in columns, H = length(hs).
# For variable k the rows (k-1)H+1 : kH hold its leads: out[(k-1)H + i, t] =
# V[t + hs[i], k]. Stacking the furthest lead costs maximum(hs) rows, so
# T_eff = T - maximum(hs). (docs/technical.md §3.2; horizon subsets as in
# docs/ext/fieldhouse_mertens/Make_Table_1.m, `sel_hors`.)
function _stack_leads(
        V::AbstractMatrix{T}, hs::AbstractVector{Int}) where {T <: AbstractFloat}
    n, m = size(V)
    H = length(hs)
    T_eff = n - maximum(hs)
    out = Matrix{T}(undef, H * m, T_eff)
    for k in 1:m, (i, h) in enumerate(hs)

        @views out[(k - 1) * H + i, :] .= V[(h + 1):(h + T_eff), k]
    end
    return out
end
function _stack_leads(V::AbstractMatrix{<:AbstractFloat}, H::Int)
    _stack_leads(V, collect(0:(H - 1)))
end

# Residualize every row of V (vars × T_eff) on the controls in Xc (T_eff × Nx) using
# the FWL annihilator M_X (eq. A.1). With Q an orthonormal basis of the column space of
# the controls, V M_X = V - (V Q) Q'. M_X is never formed.
function _residualize_on(
        V::AbstractMatrix{T}, Q::AbstractMatrix{T}) where {T <: AbstractFloat}
    V .- (V * Q) * Q'
end

# Orthonormal basis of the column space of Xc via a column-pivoted QR, so collinear
# controls (e.g. a constant in X together with `intercept = true`) are handled like
# Stata/R: redundant columns are dropped with a warning and the numerical rank is
# returned as the effective control count used in the dof corrections.
function _control_basis(Xc::AbstractMatrix{T}) where {T <: AbstractFloat}
    n, Nx = size(Xc)
    Nx == 0 && return Matrix{T}(undef, n, 0), 0
    F, r = _scaled_rank_qr(Xc)
    r < Nx &&
        @warn "controls are collinear: dropped $(Nx - r) redundant column(s) " *
              "(e.g. a constant in X while intercept = true)"
    return Matrix(F.Q)[:, 1:r], r
end

# Column-pivoted QR of A with every column scaled to unit norm first, so the numerical
# rank does not depend on the units of the columns (as R's dqrdc2 uses per-column
# norms). Zero columns stay rank-deficient. Returns the factorization and the rank.
function _scaled_rank_qr(A::AbstractMatrix{T}) where {T <: AbstractFloat}
    n, m = size(A)
    s = [norm(c) for c in eachcol(A)]
    F = qr(A ./ ifelse.(s .> 0, s, one(T))', ColumnNorm())
    tol = max(n, m) * eps(T)
    r = count(>(tol), abs.(diag(F.R)))
    return F, r
end

# Instruments must stay linearly independent once the controls are partialled out:
# an instrument spanned by the controls (e.g. a constant with intercept = true) or by
# the other instruments leaves a numerically zero direction in Z⊥ that would be
# whitened into a spurious "shock". Each Z⊥ row is measured relative to its raw norm.
function _check_instruments(Zc::AbstractMatrix{T}, Z_perp::AbstractMatrix{T}) where {T}
    Nz = size(Zc, 1)
    raw = [norm(r) for r in eachrow(Zc)]
    resid = [norm(r) for r in eachrow(Z_perp)]
    spanned = findall(resid .≤ sqrt(eps(T)) .* raw)
    isempty(spanned) || throw(
        ArgumentError(
        "instrument(s) $spanned are (numerically) spanned by the controls — " *
        "e.g. a constant or a control passed as an instrument; remove them"),
    )
    _, r = _scaled_rank_qr(permutedims(Z_perp))
    r == Nz || throw(
        ArgumentError(
        "instruments are collinear after partialling out the controls " *
        "(rank $r < Nz = $Nz); remove the redundant instrument(s)"),
    )
    return nothing
end

# Restriction matrix R = I_K ⊗ vec(I_H) ∈ ℝ^{KH² × K} (docs/technical.md, notation).
# It encodes the constraint that the same β applies at every stacked horizon.
function _restriction_matrix(::Type{T}, K::Int, H::Int) where {T <: AbstractFloat}
    return kron(Matrix{T}(I, K, K), vec(Matrix{T}(I, H, H)))
end

# ---------------------------------------------------------------------------
# Step 1 — forecast errors (dispatched on the spec). Returns the residualised stacked
# matrices y_perp (H × T_eff), Y_perp (HK × T_eff), Z_perp (Nz × T_eff), the effective
# sample T_eff, and the effective control count Nx used in the dof corrections.
# ---------------------------------------------------------------------------

# LP variant: Jordà local projections of y_{t+h}, Y_{t+h} on z_t, X_{t-1} (eq. A.1).
function _forecast_errors(
        ::SPIVwithLP,
        yv::AbstractVector{Float64},
        Ym::AbstractMatrix{Float64},
        Xm::AbstractMatrix{Float64},
        Zm::AbstractMatrix{Float64},
        hs::AbstractVector{Int}
)
    T = length(yv)
    T_eff = T - maximum(hs)
    T_eff > 0 || throw(
        ArgumentError(
        "horizon $(maximum(hs)) leaves no observations (T = $T)"),
    )

    y_H = _stack_leads(reshape(yv, T, 1), hs)  # H  × T_eff
    Y_H = _stack_leads(Ym, hs)                 # HK × T_eff
    Zc = permutedims(@view Zm[1:T_eff, :])     # Nz × T_eff
    Xc = Xm[1:T_eff, :]                        # T_eff × Nx

    Q, Nx = _control_basis(Xc)
    y_perp = _residualize_on(y_H, Q)           # H  × T_eff
    Y_perp = _residualize_on(Y_H, Q)           # HK × T_eff
    Z_perp = _residualize_on(Zc, Q)            # Nz × T_eff
    _check_instruments(Zc, Z_perp)
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
        hs::Vector{Int},
        K::Int,
        Nz::Int,
        Nx::Int,
        T_eff::Int;
        robust::Symbol,
        level::Real,
        tau::Real,
        grid,
        grid_length::Int,
        hac::Symbol
) where {S <: AbstractSPIVSpec}
    H = length(hs)
    α = 1 - level

    # --- order condition and degrees of freedom --------------------------
    H * Nz ≥ K || throw(
        ArgumentError(
        "order condition violated: H·Nz = $(H*Nz) < K = $K (docs/technical.md §3.4)"),
    )
    T_eff - Nx - K > 0 || throw(
        ArgumentError(
        "insufficient degrees of freedom: T_eff - Nx - K = $(T_eff - Nx - K) ≤ 0"),
    )
    # The H × H matrix u⊥ M_Z⊥ u⊥' inverted by the AR/KLM statistics has rank at most
    # T_eff − Nx − Nz, so more stacked horizons than that make it singular.
    H ≤ T_eff - Nx - Nz || throw(
        ArgumentError(
        "too many horizons for the sample: the robust AR/KLM statistics need " *
        "H ≤ T_eff - Nx - Nz = $(T_eff - Nx - Nz), got H = $H; use fewer horizons " *
        "(e.g. a subset such as 0:4:40) or fewer controls"),
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
    weak = _weak_iv_diagnostic(YPY, W2, R, H, Nz, tau, α)
    se = sqrt.(diag(vcov))
    box = _grid_box(grid, β, se)
    robust_set = _robust_inference(
        robust,
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
        box,
        grid_length
    )

    # --- impulse responses with HAC standard errors (docs/technical.md §3.3) ---
    irf_outcome,
    irf_endogenous = _irf_blocks(
        y_perp, Y_perp, Z_perp, collect(Czz), hs, K, Nz, T_eff, α, hac)

    return SPIVResult{Float64, S}(
        spec,
        β,
        vcov,
        û,
        weak,
        robust_set,
        irf_outcome,
        irf_endogenous,
        hs,
        Float64(level),
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

# Promote a vector to a one-column Float64 matrix; copy matrices to Matrix{Float64}.
_as_matrix(A::AbstractVector{<:Real}) = Matrix{Float64}(reshape(A, :, 1))
_as_matrix(A::AbstractMatrix{<:Real}) = Matrix{Float64}(A)

# `H` as an Int means horizons 0:H-1; a vector lists the horizons explicitly.
function _horizons(H::Integer)
    H ≥ 1 ? collect(0:(Int(H) - 1)) : throw(ArgumentError("H must be ≥ 1, got $H"))
end
function _horizons(H::AbstractVector{<:Integer})
    hs = collect(Int, H)
    !isempty(hs) && first(hs) ≥ 0 && all(>(0), diff(hs)) || throw(
        ArgumentError(
        "H must be a positive Int or a non-empty, strictly increasing vector of " *
        "horizons ≥ 0, got $H"),
    )
    return hs
end

# Default grid resolution for the robust set: fine for K = 1 (bisection then refines the
# endpoints), coarser in higher dimensions where cost grows as grid_length^K. Odd, so the
# centre of the default box (β̂) is a grid point.
_default_grid_length(K::Int) = K == 1 ? 201 : K == 2 ? 101 : 31

# Trait: does the spec consume the LP controls (X / intercept / xlags)? The VAR
# variant ignores them — its own lags play that role.
_uses_controls(::SPIVwithLP) = true
_uses_controls(::SPIVwithVAR) = false

# Is row t free of NaN across every input?
_row_ok(t, yv, Ms) = !isnan(yv[t]) && all(M -> all(j -> !isnan(M[t, j]), axes(M, 2)), Ms)

# First and last rows of the usable sample: the NaN-free span across every input once
# the leading padding (lags, lag/lags burn-in) and the trailing padding (leads, e.g. a
# forward moving average built with lag(y, -n)) are dropped. Any other non-finite value
# is a data error, rejected by the caller. Returns (1, 0) when no row is NaN-free.
function _sample_span(yv::AbstractVector{Float64}, Ms::AbstractMatrix{Float64}...)
    first_row = findfirst(t -> _row_ok(t, yv, Ms), eachindex(yv))
    first_row === nothing && return 1, 0
    last_row = findlast(t -> _row_ok(t, yv, Ms), eachindex(yv))
    return first_row, last_row
end

"""
    spiv(y, Y, Z, spec = SPIVwithLP(); H, X = nothing, intercept = true, xlags = 0,
         robust = :AR, level = 0.95, tau = 0.10, grid = 10, grid_length = nothing,
         hac = :fixed)

System Projections IV estimator (Lewis & Mertens 2024).

Inputs use the **time-in-rows** convention (`T` periods down the rows); single
series can be passed as plain vectors:

- `y` — outcome, length-`T` vector.
- `Y` — endogenous regressors, `T × K` matrix (or a length-`T` vector when `K = 1`).
- `Z` — instruments, `T × Nz` matrix (or a length-`T` vector when `Nz = 1`).

```julia
spiv(y, Y, z; H = 8)                       # horizons 0,…,7; intercept-only controls
spiv(y, Y, z; H = 3:4:59)                  # a subset of horizons
spiv(y, Y, z; H = 8, xlags = 4)            # + 4 lags of [y Y z] as controls
spiv(y, Y, z; H = 8, X = lags(w, 4))       # + user-built lagged controls
spiv(y, Y, z; H = 8, robust = :KLM)        # KLM instead of AR robust set
spiv(y, Y, z, SPIVwithVAR(4); H = 8)       # VAR(4) forecast errors
```

`spec` selects the forecast-error construction:

- `SPIVwithLP()` (default) — Jordà local projections of `y_{t+h}, Y_{t+h}` on `z_t` and
  the controls (eq. A.1).
- `SPIVwithVAR(p)` — a VAR(`p`) on `[Y_t, y_t, z_t]`; forecast errors are built from
  the reduced-form residuals (eqs. A.2–A.3) and the controls (`X`, `intercept`, `xlags`)
  are **ignored with a warning** (the VAR's own lags play that role).
  `T_eff = T − p − max(h)`.

`H` sets the stacked horizons: an `Int` means `h = 0,…,H-1`; a vector (e.g.
`3:4:59`, as in Fieldhouse & Mertens) stacks exactly those horizons. Returns an
`SPIVResult{Float64, typeof(spec)}` with the point estimate `β̂` (eq. 9), the
strong-identification sandwich variance (eqs. 18–19), the structural residuals, the
weak-IV diagnostic (Proposition 7), and a robust confidence set.

**Sample adjustment.** Leading and trailing rows containing `NaN` in any input — the
padding that [`lag`](@ref) / [`lags`](@ref) produce for lags and leads — are dropped
from *all* inputs automatically (Stata/R-style), so `X = lags(w, 4)` or a forward
moving average built with `lag(y, -n)` work as-is; the adjusted sample is visible as
`T_eff` in the printed result. Any other non-finite value — `Inf` anywhere, or `NaN`
inside the sample — is a data error and throws an `ArgumentError`.

**Serial correlation.** The strong-identification standard errors use no HAR
correction; this is valid when the controls absorb the serial correlation of `z⊥`
(docs/technical.md §5.4). The default controls are intercept-only, which suits a
serially uncorrelated shock series as instrument; with a persistent instrument add
lagged controls via `xlags` or `X`.

Keyword arguments:

- `X` — additional controls to residualise on (e.g. lagged variables, `X_{t-1}`),
  a `T × Nx` matrix or a length-`T` vector; default none. Collinear columns are
  dropped with a warning.
- `intercept` — prepend a constant column to the controls (default `true`).
- `xlags` — also control for this many lags of `[y Y Z]` (in that column order),
  built and sample-adjusted automatically (default 0). The series are lagged exactly
  as passed: if `y`/`Y` are forward-transformed (e.g. a forward moving average),
  build the controls from the raw series with `X = lags(...)` instead. With
  `xlags = p`, the LP variant conditions on the same information set as
  `SPIVwithVAR(p)`.
- `robust` — weak-IV-robust test inverted for the confidence set: `:AR` (default;
  Anderson–Rubin, `χ²` with `H·Nz` df, fully robust) or `:KLM` (Kleibergen, `χ²` with
  `K` df; more powerful when `H·Nz ≫ K`).
- `level` — confidence level of the Wald intervals, the robust set, and the IRF bands
  (default 0.95). The weak-IV test is run at size `1 − level`.
- `tau` — bias tolerance `τ` of the weak-IV test (Definition 3; default 0.10).
- `grid` — search region for the robust set: a number `c` means `β̂ ± c·se` for each
  parameter (default 10); a tuple `(lo, hi)` applies to every parameter; a `K × 2`
  matrix gives per-parameter `[lo hi]` rows. Use an explicit range when instruments
  are weak — the strong-identification `se` is then unreliable. Bounds that reach the
  edge of the region are reported as `±Inf`.
- `grid_length` — grid points per parameter (default 201 for `K = 1`, 101 for
  `K = 2`, 31 otherwise). Finite bounds are refined by bisection, so this only needs
  to be fine enough to locate the set.
- `hac` — bandwidth rule for the IRF HAC standard errors: `:fixed` (default; lag
  truncation = horizon, a package choice), `:neweywest` (automatic Newey–West 1994),
  or `:andrews` (automatic Andrews 1991).

The outcome IRF (`H × Nz`) and endogenous IRF (`H × K × Nz`), with per-horizon Newey–West
HAC standard errors and `level` normal bands, are in `result.irf_outcome` /
`result.irf_endogenous` (docs/technical.md §3.3, eq. 11), or via [`irf`](@ref).
"""
function spiv(
        y::AbstractVector{<:Real},
        Y::AbstractVecOrMat{<:Real},
        Z::AbstractVecOrMat{<:Real},
        spec::AbstractSPIVSpec = SPIVwithLP();
        H::Union{Integer, AbstractVector{<:Integer}},
        X::Union{Nothing, AbstractVecOrMat{<:Real}} = nothing,
        intercept::Bool = true,
        xlags::Int = 0,
        robust::Symbol = :AR,
        level::Real = 0.95,
        tau::Real = 0.10,
        grid = 10,
        grid_length::Union{Nothing, Int} = nothing,
        hac::Symbol = :fixed
)
    # --- promote to a common float type -----------------------------------
    yv = collect(Float64, y)
    Ym = _as_matrix(Y)
    Zm = _as_matrix(Z)

    T = length(yv)
    K = size(Ym, 2)
    Nz = size(Zm, 2)

    # --- keyword and dimension validation ----------------------------------
    hs = _horizons(H)
    xlags ≥ 0 || throw(ArgumentError("xlags must be ≥ 0, got $xlags"))
    robust in (:AR, :KLM) ||
        throw(ArgumentError("robust must be :AR or :KLM, got :$robust"))
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    0 < tau || throw(ArgumentError("tau must be > 0, got $tau"))
    gl = grid_length === nothing ? _default_grid_length(K) : grid_length
    gl ≥ 2 || throw(ArgumentError("grid_length must be ≥ 2, got $gl"))
    _grid_box(grid, zeros(K), ones(K))         # validate `grid` before estimating
    hac in (:fixed, :neweywest, :andrews) ||
        throw(ArgumentError("hac must be :fixed, :neweywest, or :andrews, got :$hac"))
    size(Ym, 1) == T ||
        throw(DimensionMismatch("Y must have T = $T rows, got $(size(Ym, 1))"))
    size(Zm, 1) == T ||
        throw(DimensionMismatch("Z must have T = $T rows, got $(size(Zm, 1))"))
    Xuser = X === nothing ? Matrix{Float64}(undef, T, 0) : _as_matrix(X)
    size(Xuser, 1) == T ||
        throw(DimensionMismatch("X must have T = $T rows, got $(size(Xuser, 1))"))

    # --- controls: only the LP variant consumes them ------------------------
    if _uses_controls(spec)
        if xlags ≥ 1
            xlags < T ||
                throw(ArgumentError("xlags = $xlags leaves no observations (T = $T)"))
            L = lags(hcat(yv, Ym, Zm), xlags)
            Xuser = size(Xuser, 2) == 0 ? L : hcat(Xuser, L)
        end
    else
        if X !== nothing || !intercept || xlags != 0
            @warn "X, intercept, and xlags are ignored with SPIVwithVAR: the VAR's " *
                  "own lags replace the LP controls"
        end
        Xuser = Matrix{Float64}(undef, T, 0)   # the VAR variant never uses controls
    end

    # --- automatic sample adjustment: drop leading/trailing NaN padding ----
    # (what lag/lags produce for lags and leads). Any other non-finite value — Inf
    # anywhere, NaN inside the sample — is a data error, rejected below.
    first_row, last_row = _sample_span(yv, Ym, Zm, Xuser)
    last_row == 0 && T > 0 &&
        throw(
            ArgumentError(
            "no row is free of NaN across y, Y, Z, and X — no observations left to estimate"),
        )
    if first_row > 1 || last_row < T
        keep = first_row:last_row
        yv = yv[keep]
        Ym = Ym[keep, :]
        Zm = Zm[keep, :]
        Xuser = Xuser[keep, :]
        T = length(keep)
    end
    for (name, arr) in (("y", yv), ("Y", Ym), ("Z", Zm), ("X", Xuser))
        all(isfinite, arr) || throw(
            ArgumentError(
            "$name contains non-finite values (NaN/Inf) inside the sample — only the " *
            "leading/trailing NaN padding of lags and leads is dropped automatically."),
        )
    end

    # --- assemble the controls matrix: [intercept | X] --------------------
    Xm = intercept ? hcat(ones(T), Xuser) : Xuser

    # --- step 1 (dispatched) then shared steps 2–5 ------------------------
    y_perp, Y_perp, Z_perp, T_eff, Nx_eff = _forecast_errors(spec, yv, Ym, Xm, Zm, hs)
    return _spiv_estimate(
        spec,
        y_perp,
        Y_perp,
        Z_perp,
        hs,
        K,
        Nz,
        Nx_eff,
        T_eff;
        robust = robust,
        level = level,
        tau = tau,
        grid = grid,
        grid_length = gl,
        hac = hac
    )
end
