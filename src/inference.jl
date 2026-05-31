# Weak-instrument inference for SP-IV — weak-IV diagnostic and robust confidence sets.
#
# Implements docs/technical.md §6–7 (algorithmic summary §9, steps 4–5):
#   - first-stage residual covariance Ŵ₂                  §9 step 4
#   - weak-IV statistic g and its critical value          §6.6–6.7 (Prop. 7, Thm 1)
#   - Anderson–Rubin statistic                            §7.1 (eq. 20)
#   - Kleibergen KLM statistic                            §7.2 (eq. 21)
#   - grid-search AR/KLM confidence sets
#
# Inputs use the same time-in-columns layout the estimator produces (see spiv.jl):
# y_perp is H × T_eff, Y_perp / v_res are HK × T_eff, Z_perp is Nz × T_eff. Instrument
# projections P_Z⊥ / M_Z⊥ are applied implicitly through Czz = Z⊥ Z⊥' — the
# T_eff × T_eff matrices are never formed: V P_Z⊥ V' = (V Z⊥')(Z⊥ Z⊥')⁻¹(Z⊥ V').

# ---------------------------------------------------------------------------
# First-stage residual covariance Ŵ₂ (docs/technical.md §9 step 4)
# ---------------------------------------------------------------------------

# v̂⊥_H = Ŷ⊥_H − Π̂ Ẑ⊥ are the first-stage residuals (Y⊥ regressed on Z⊥, i.e. Y⊥ M_Z⊥),
# with covariance Ŵ₂ = v̂⊥_H v̂⊥_H' / (T_eff − Nx − Nz). Returns (v_res, Ŵ₂).
function _first_stage_resid_cov(
    Y_perp::AbstractMatrix{T},
    Z_perp::AbstractMatrix{T},
    Czz::AbstractMatrix{T},
    T_eff::Int,
    Nx::Int,
    Nz::Int,
) where {T<:AbstractFloat}
    fitted = (Y_perp * Z_perp') * (Czz \ Z_perp)      # Π̂ Ẑ⊥ = Y⊥ P_Z⊥, HK × T_eff
    v_res = Y_perp .- fitted                           # HK × T_eff
    W2 = (v_res * v_res') ./ (T_eff - Nx - Nz)         # HK × HK
    return v_res, W2
end

# ---------------------------------------------------------------------------
# Weak-IV diagnostic g and critical value (docs/technical.md §6.6–6.7)
# ---------------------------------------------------------------------------

# Test statistic g (Proposition 7, eq. in §6.6) and a conservative critical value
# from the cumulant bounds of Theorem 1 (§6.7), matched to a chi-square reference.
#
# Paper-faithful-bracket note: Proposition 7 writes the statistic as
#   g = N_z⁻¹ mineval{ Φ̂⁻¹ᐟ² (Y⊥ P_Z⊥ Y⊥') Φ̂⁻¹ᐟ² }.
# Φ̂ is K × K while Y⊥ P_Z⊥ Y⊥' is HK × HK, so the inner bracket is the K × K
# concentration-matrix form R'(Y⊥ P_Z⊥ Y⊥' ⊗ I_H)R (Definition 1, §6.3); this is the
# object the asymptotic limit in Prop. 7 is stated for. Implemented as such below.
function _weak_iv_diagnostic(
    YPY::AbstractMatrix{T},
    W2::AbstractMatrix{T},
    R::AbstractMatrix{T},
    H::Int,
    Nz::Int,
    ξ::Real,
    α::Real,
) where {T<:AbstractFloat}
    IH = Matrix{T}(I, H, H)

    Φ = Symmetric(R' * kron(W2, IH) * R)               # Φ̂ = R'(Ŵ₂ ⊗ I_H)R, K × K
    Φ_isqrt = inv(sqrt(Φ))                              # Φ̂⁻¹ᐟ²
    A = Symmetric(R' * kron(YPY, IH) * R)              # R'(Y⊥ P_Z⊥ Y⊥' ⊗ I_H)R, K × K
    g_mat = Symmetric(Φ_isqrt * A * Φ_isqrt)
    g_min = minimum(eigvals(g_mat)) / Nz

    # Critical value: cumulant upper bounds (Theorem 1) matched to a chi-square
    # reference (Patnaik-style three-cumulant approximation), conservative by §6.7.
    ℓ = 1 / ξ
    W2_sqrt = sqrt(Symmetric(W2))                      # Ŵ₂¹ᐟ²
    S = kron(Φ_isqrt, IH) * W2_sqrt                    # KH × HK
    Scal = S * S'                                       # Σ̂ = SS', KH × KH
    m2 = maximum(eigvals(Symmetric(R' * kron(Scal^2, IH) * R)))
    m3 = maximum(eigvals(Symmetric(R' * kron(Scal^3, IH) * R)))

    κ1 = Nz * (1 + ℓ)
    κ2 = 2 * (Nz * m2 + 2 * ℓ * Nz)
    κ3 = 8 * (Nz * m3 + 3 * ℓ * Nz * m2)
    ν = κ2 / κ3
    δ = 8 * κ2 * ν^2
    g_crit = ((chisqinvcdf(δ, 1 - α) - δ) / (4 * ν) + κ1) / Nz

    return WeakIVDiagnostic{T}(T(g_min), T(g_crit), g_min < g_crit)
end

# ---------------------------------------------------------------------------
# Robust statistics evaluated at a candidate b (docs/technical.md §7)
# ---------------------------------------------------------------------------

# Structural residuals under H₀: b = β  ⇒  u⊥_H(b) = y⊥_H − (b' ⊗ I_H) Y⊥_H (H × T_eff).
_resid_at(b, y_perp, Y_perp, H) =
    y_perp .- kron(b', Matrix{eltype(y_perp)}(I, H, H)) * Y_perp

# Anderson–Rubin statistic, eq. (20). d_AR = N_z + N_x.
function _ar_stat(
    b::AbstractVector{T},
    y_perp::AbstractMatrix{T},
    Y_perp::AbstractMatrix{T},
    Z_perp::AbstractMatrix{T},
    Czz::AbstractMatrix{T},
    T_eff::Int,
    d_AR::Int,
) where {T<:AbstractFloat}
    H = size(y_perp, 1)
    u = _resid_at(b, y_perp, Y_perp, H)                # H × T_eff
    uZ = u * Z_perp'                                    # H × Nz
    uPu = uZ * (Czz \ uZ')                              # u P_Z⊥ u', H × H
    uMu = u * u' .- uPu                                 # u M_Z⊥ u', H × H
    return (T_eff - d_AR) * tr(uPu * (uMu \ Matrix{T}(I, H, H)))
end

# Kleibergen KLM statistic, eq. (21). d_K = N_z + N_x. Ported from the reference
# implementation; the implicit P_Z⊥ / M_Z⊥ forms replace the explicit T×T matrices.
function _klm_stat(
    b::AbstractVector{T},
    y_perp::AbstractMatrix{T},
    Y_perp::AbstractMatrix{T},
    v_res::AbstractMatrix{T},
    Z_perp::AbstractMatrix{T},
    Czz::AbstractMatrix{T},
    R::AbstractMatrix{T},
    T_eff::Int,
    d_K::Int,
) where {T<:AbstractFloat}
    H = size(y_perp, 1)
    u = _resid_at(b, y_perp, Y_perp, H)                # H × T_eff

    CzZ = Czz \ Z_perp                                  # (Z⊥Z⊥')⁻¹ Z⊥, Nz × T_eff
    uPz = (u * Z_perp') * CzZ                           # u P_Z⊥, H × T_eff
    u_check = u .- uPz                                  # u M_Z⊥, H × T_eff
    Ξ = u * u' .- (u * Z_perp') * (Czz \ (u * Z_perp')')  # u M_Z⊥ u', H × H

    YPz = (Y_perp * Z_perp') * CzZ                      # Y⊥ P_Z⊥, HK × T_eff
    v_check = v_res .- (v_res * Z_perp') * CzZ          # v⊥ M_Z⊥, HK × T_eff
    Amat = Ξ \ uPz                                      # Ξ⁻¹ u P_Z⊥, H × T_eff
    Y_check = YPz .- (v_check * u_check') * Amat        # Y̌_H, HK × T_eff

    score = Ξ \ (u * Y_check')                          # Ξ⁻¹ u Y̌_H', H × HK
    s = R' * vec(score)                                 # K
    inner = Ξ \ (u * u' * (Ξ \ Matrix{T}(I, H, H)))     # Ξ⁻¹ u u' Ξ⁻¹, H × H
    V = Symmetric(R' * kron(Y_check * Y_check', inner) * R)  # K × K
    return (T_eff - d_K) * dot(s, V \ s)
end

# ---------------------------------------------------------------------------
# Robust confidence set via grid inversion (docs/technical.md §7)
# ---------------------------------------------------------------------------

# Invert the selected test (`:AR` or `:KLM`) over a K-dimensional box centred on β̂:
# each parameter ranges over β̂_k ± grid_scale·se_k with `grid_length` points. A grid
# point is accepted when its statistic is below the χ² critical value (df = H·Nz for
# AR, K for KLM at level 1−α). `bounds` are per-parameter min/max over accepted points.
function _robust_inference(
    method::Symbol,
    β::AbstractVector{T},
    se::AbstractVector{T},
    y_perp::AbstractMatrix{T},
    Y_perp::AbstractMatrix{T},
    v_res::AbstractMatrix{T},
    Z_perp::AbstractMatrix{T},
    Czz::AbstractMatrix{T},
    R::AbstractMatrix{T},
    H::Int,
    K::Int,
    Nz::Int,
    Nx::Int,
    T_eff::Int,
    α::Real,
    grid_length::Int,
    grid_scale::Real,
) where {T<:AbstractFloat}
    d = Nz + Nx
    if method === :AR
        df = H * Nz
        crit = chisqinvcdf(df, 1 - α)
        statf = b -> _ar_stat(b, y_perp, Y_perp, Z_perp, Czz, T_eff, d)
    elseif method === :KLM
        df = K
        crit = chisqinvcdf(df, 1 - α)
        statf = b -> _klm_stat(b, y_perp, Y_perp, v_res, Z_perp, Czz, R, T_eff, d)
    else
        throw(ArgumentError("weak_iv must be :AR or :KLM, got :$method"))
    end

    grids = [
        range(β[k] - grid_scale * se[k], β[k] + grid_scale * se[k]; length = grid_length) for k in 1:K
    ]
    pts = vec(collect(Iterators.product(grids...)))    # Vector of NTuple{K}
    accepted = Vector{Bool}(undef, length(pts))
    Threads.@threads for i in eachindex(pts)
        b = collect(T, pts[i])
        accepted[i] = statf(b) < crit
    end

    idx = findall(accepted)
    cs = Matrix{T}(undef, length(idx), K)
    for (row, i) in enumerate(idx)
        cs[row, :] .= pts[i]
    end

    bounds = fill(T(NaN), K, 2)
    if !isempty(idx)
        for k in 1:K
            col = @view cs[:, k]
            bounds[k, 1] = minimum(col)
            bounds[k, 2] = maximum(col)
        end
    end

    return RobustInference{T}(method, T(crit), bounds, cs, df)
end
