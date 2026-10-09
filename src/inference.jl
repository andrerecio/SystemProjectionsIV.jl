# Weak-instrument inference for SP-IV — weak-IV diagnostic and robust confidence sets.
#
# Implements docs/technical.md §6–7 (algorithmic summary §9, steps 4–5):
#   - first-stage residual covariance Ŵ₂                  §9 step 4
#   - weak-IV statistic g and its critical value          §6.6–6.7 (Prop. 7, Thm 1)
#   - Anderson–Rubin statistic                            §7.1 (eq. 20)
#   - Kleibergen KLM statistic                            §7.2 (eq. 21)
#   - grid-search AR/KLM confidence sets, endpoints refined by bisection (cf. the grid +
#     fzero inversion in docs/ext/fieldhouse_mertens/Make_Table_1.m, `run_SPIV_total`)
#
# Inputs use the same time-in-columns layout the estimator produces (see spiv.jl):
# y_perp is H × T_eff, Y_perp / v_res are HK × T_eff, Z_perp is Nz × T_eff. Instrument
# projections P_Z⊥ / M_Z⊥ are applied implicitly through Czz = Z⊥ Z⊥' — the
# T_eff × T_eff matrices are never formed: V P_Z⊥ V' = (V Z⊥')(Z⊥ Z⊥')⁻¹(Z⊥ V').

# ---------------------------------------------------------------------------
# First-stage residual covariance Ŵ₂ (docs/technical.md §9 step 4)
# ---------------------------------------------------------------------------

# v̂⊥_H = Ŷ⊥_H − Π̂ Ẑ⊥ are the first-stage residuals (Y⊥ regressed on Z⊥, i.e. Y⊥ M_Z⊥),
# with covariance Ŵ₂ = v̂⊥_H v̂⊥_H' / (T_eff − Nx − Nz). Returns (v_res, Ŵ₂). The
# dof-corrected divisor follows the Fieldhouse–Mertens implementation
# (docs/ext/fieldhouse_mertens/Make_Table_1.m, `SigvH`); technical.md §9 writes /T.
function _first_stage_resid_cov(
        Y_perp::AbstractMatrix{T},
        Z_perp::AbstractMatrix{T},
        Czz::AbstractMatrix{T},
        T_eff::Int,
        Nx::Int,
        Nz::Int
) where {T <: AbstractFloat}
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
        tau::Real,
        α::Real
) where {T <: AbstractFloat}
    IH = Matrix{T}(I, H, H)

    Φ = Symmetric(R' * kron(W2, IH) * R)               # Φ̂ = R'(Ŵ₂ ⊗ I_H)R, K × K
    Φ_isqrt = inv(sqrt(Φ))                              # Φ̂⁻¹ᐟ²
    A = Symmetric(R' * kron(YPY, IH) * R)              # R'(Y⊥ P_Z⊥ Y⊥' ⊗ I_H)R, K × K
    g_mat = Symmetric(Φ_isqrt * A * Φ_isqrt)
    g_min = minimum(eigvals(g_mat)) / Nz

    # Critical value: cumulant upper bounds (Theorem 1) matched to a chi-square
    # reference (Patnaik-style three-cumulant approximation), conservative by §6.7.
    ℓ = 1 / tau
    W2_sqrt = sqrt(Symmetric(W2))                      # Ŵ₂¹ᐟ²
    S = kron(Φ_isqrt, IH) * W2_sqrt                    # KH × HK
    Scal = S * S'                                       # Σ̂ = SS', KH × KH
    Scal2 = Scal * Scal                                 # Σ̂², reused for Σ̂³
    m2 = maximum(eigvals(Symmetric(R' * kron(Scal2, IH) * R)))
    m3 = maximum(eigvals(Symmetric(R' * kron(Scal2 * Scal, IH) * R)))

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

# Structural residuals under H₀: b = β  ⇒  u⊥_H(b) = y⊥_H − (b' ⊗ I_H) Y⊥_H (H × T_eff),
# i.e. y⊥_H − Σ_k b_k Y⊥_H[block k] (the Kronecker product is never formed).
function _resid_at(b, y_perp, Y_perp, H)
    u = copy(y_perp)
    for k in eachindex(b)
        @views u .-= b[k] .* Y_perp[((k - 1) * H + 1):(k * H), :]
    end
    return u
end

# Anderson–Rubin statistic, eq. (20). d_AR = N_z + N_x.
function _ar_stat(
        b::AbstractVector{T},
        y_perp::AbstractMatrix{T},
        Y_perp::AbstractMatrix{T},
        Z_perp::AbstractMatrix{T},
        Czz::AbstractMatrix{T},
        T_eff::Int,
        d_AR::Int
) where {T <: AbstractFloat}
    H = size(y_perp, 1)
    u = _resid_at(b, y_perp, Y_perp, H)                # H × T_eff
    uZ = u * Z_perp'                                    # H × Nz
    uPu = uZ * (Czz \ uZ')                              # u P_Z⊥ u', H × H
    uMu = u * u' .- uPu                                 # u M_Z⊥ u', H × H
    return (T_eff - d_AR) * tr(uPu * (uMu \ Matrix{T}(I, H, H)))
end

# Kleibergen KLM statistic, eq. (21). d_K = N_z + N_x. Ported from the reference
# implementation; the implicit P_Z⊥ / M_Z⊥ forms replace the explicit T×T matrices.
# Matches `GetKstat` in docs/ext/fieldhouse_mertens/Make_Table_1.m (with `btrace`).
function _klm_stat(
        b::AbstractVector{T},
        y_perp::AbstractMatrix{T},
        Y_perp::AbstractMatrix{T},
        v_res::AbstractMatrix{T},
        Z_perp::AbstractMatrix{T},
        Czz::AbstractMatrix{T},
        R::AbstractMatrix{T},
        T_eff::Int,
        d_K::Int
) where {T <: AbstractFloat}
    CzZ, YPz, v_check = _klm_precompute(Y_perp, v_res, Z_perp, Czz)
    return _klm_stat(b, y_perp, Y_perp, Z_perp, Czz, CzZ, YPz, v_check, R, T_eff, d_K)
end

# The b-invariant pieces of the KLM statistic, computed once per grid inversion:
# (Z⊥Z⊥')⁻¹Z⊥ (Nz × T_eff), Y⊥ P_Z⊥ (HK × T_eff), and v⊥ M_Z⊥ (HK × T_eff).
function _klm_precompute(
        Y_perp::AbstractMatrix{T},
        v_res::AbstractMatrix{T},
        Z_perp::AbstractMatrix{T},
        Czz::AbstractMatrix{T}
) where {T <: AbstractFloat}
    CzZ = Czz \ Z_perp                                  # (Z⊥Z⊥')⁻¹ Z⊥, Nz × T_eff
    YPz = (Y_perp * Z_perp') * CzZ                      # Y⊥ P_Z⊥, HK × T_eff
    v_check = v_res .- (v_res * Z_perp') * CzZ          # v⊥ M_Z⊥, HK × T_eff
    return CzZ, YPz, v_check
end

function _klm_stat(
        b::AbstractVector{T},
        y_perp::AbstractMatrix{T},
        Y_perp::AbstractMatrix{T},
        Z_perp::AbstractMatrix{T},
        Czz::AbstractMatrix{T},
        CzZ::AbstractMatrix{T},
        YPz::AbstractMatrix{T},
        v_check::AbstractMatrix{T},
        R::AbstractMatrix{T},
        T_eff::Int,
        d_K::Int
) where {T <: AbstractFloat}
    H = size(y_perp, 1)
    u = _resid_at(b, y_perp, Y_perp, H)                # H × T_eff

    uZ = u * Z_perp'                                    # H × Nz
    uPz = uZ * CzZ                                      # u P_Z⊥, H × T_eff
    u_check = u .- uPz                                  # u M_Z⊥, H × T_eff
    Ξ = u * u' .- uZ * (Czz \ uZ')                      # u M_Z⊥ u', H × H

    Amat = Ξ \ uPz                                      # Ξ⁻¹ u P_Z⊥, H × T_eff
    Y_check = YPz .- (v_check * u_check') * Amat        # Y̌_H, HK × T_eff

    score = Ξ \ (u * Y_check')                          # Ξ⁻¹ u Y̌_H', H × HK
    s = R' * vec(score)                                 # K
    inner = Ξ \ (u * u' * (Ξ \ Matrix{T}(I, H, H)))     # Ξ⁻¹ u u' Ξ⁻¹, H × H
    # R'(Y̌Y̌' ⊗ inner)R in block form: V_kl = ⟨block (k,l) of Y̌Y̌', inner⟩.
    YY = Y_check * Y_check'                             # HK × HK
    K = length(b)
    V = Symmetric([dot(view(YY, ((k - 1) * H + 1):(k * H), ((l - 1) * H + 1):(l * H)),
                       inner) for k in 1:K, l in 1:K])  # K × K
    return (T_eff - d_K) * dot(s, V \ s)
end

# ---------------------------------------------------------------------------
# Robust confidence set via grid inversion (docs/technical.md §7)
# ---------------------------------------------------------------------------

# Search box for the robust set, K × 2 rows [lo hi]: a number c means β̂ ± c·se; a
# (lo, hi) tuple applies to every parameter; a K × 2 matrix is used as given.
function _grid_box(c::Real, β::AbstractVector{Float64}, se::AbstractVector{Float64})
    isfinite(c) && c > 0 || throw(ArgumentError("grid must be finite and > 0, got $c"))
    all(s -> isfinite(s) && s > 0, se) || throw(
        ArgumentError(
        "standard errors are not positive and finite, so `grid = c` (β̂ ± c·se) is " *
        "undefined; pass an explicit range, e.g. grid = (lo, hi)"),
    )
    return hcat(β .- c .* se, β .+ c .* se)
end
function _grid_box(
        g::Tuple{Real, Real}, β::AbstractVector{Float64}, se::AbstractVector{Float64})
    _grid_box(repeat(Float64[g[1] g[2]], length(β), 1), β, se)
end
function _grid_box(
        g::AbstractMatrix{<:Real}, β::AbstractVector{Float64}, se::AbstractVector{Float64})
    size(g) == (length(β), 2) || throw(
        ArgumentError("grid matrix must be K × 2 = $(length(β)) × 2, got $(size(g))"),
    )
    all(isfinite, g) && all(g[:, 1] .< g[:, 2]) ||
        throw(ArgumentError("grid rows must be finite with lo < hi, got $g"))
    return Matrix{Float64}(g)
end
function _grid_box(g, β::AbstractVector{Float64}, se::AbstractVector{Float64})
    throw(
        ArgumentError(
        "grid must be a number, a (lo, hi) tuple, or a K × 2 matrix, got $(typeof(g))"),
    )
end

# Invert the selected test (`:AR` or `:KLM`) over the K-dimensional `box` with
# `grid_length` points per parameter. A point is accepted when its statistic is below
# the χ² critical value (df = H·Nz for AR, K for KLM, at level 1−α).
function _robust_inference(
        method::Symbol,
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
        box::Matrix{T},
        grid_length::Int
) where {T <: AbstractFloat}
    d = Nz + Nx
    if method === :AR
        df = H * Nz
        crit = T(chisqinvcdf(df, 1 - α))
        ar = b -> _ar_stat(b, y_perp, Y_perp, Z_perp, Czz, T_eff, d)
        return _invert_grid(ar, method, crit, df, box, grid_length)
    elseif method === :KLM
        df = K
        crit = T(chisqinvcdf(df, 1 - α))
        CzZ, YPz, v_check = _klm_precompute(Y_perp, v_res, Z_perp, Czz)
        klm = b -> _klm_stat(
            b, y_perp, Y_perp, Z_perp, Czz, CzZ, YPz, v_check, R, T_eff, d)
        return _invert_grid(klm, method, crit, df, box, grid_length)
    end
    throw(ArgumentError("robust must be :AR or :KLM, got :$method"))
end

# Grid point i (linear index, first parameter fastest — the Iterators.product order) has
# per-parameter indices c = digits(i - 1; base = n, pad = K) .+ 1.
_grid_coords(i::Int, n::Int, K::Int) = digits(i - 1; base = n, pad = K) .+ 1
function _grid_point(grids::Vector{Vector{T}}, c::Vector{Int}) where {T}
    [grids[k][c[k]] for k in eachindex(c)]
end

# Function barrier: the statistic closure is a concrete type here, so the grid loop is
# type-stable. Bounds are the hull of the accepted points; an endpoint on the edge of the
# box becomes ±Inf (the set extends beyond the grid), any other endpoint is refined by
# bisection against its rejected neighbour (as the fzero step of
# docs/ext/fieldhouse_mertens/Make_Table_1.m, `run_SPIV_total`).
function _invert_grid(
        statf::F, method::Symbol, crit::T, df::Int, box::Matrix{T}, n::Int) where {F, T}
    K = size(box, 1)
    grids = [collect(range(box[k, 1], box[k, 2]; length = n)) for k in 1:K]
    N = n^K
    accepted = Vector{Bool}(undef, N)
    Threads.@threads for i in 1:N
        accepted[i] = statf(_grid_point(grids, _grid_coords(i, n, K))) < crit
    end

    idx = findall(accepted)
    coords = [_grid_coords(i, n, K) for i in idx]
    cs = Matrix{T}(undef, length(idx), K)
    for (row, c) in enumerate(coords)
        cs[row, :] .= _grid_point(grids, c)
    end

    bounds = fill(T(NaN), K, 2)
    if !isempty(idx)
        for k in 1:K
            lo = minimum(c[k] for c in coords)
            hi = maximum(c[k] for c in coords)
            bounds[k, 1] = lo == 1 ? T(-Inf) :
                           _refine_bound(statf, crit, grids, coords, k, lo, -1)
            bounds[k, 2] = hi == n ? T(Inf) :
                           _refine_bound(statf, crit, grids, coords, k, hi, 1)
        end
    end
    disconnected = _n_components(accepted, n, K) > 1
    return RobustInference{T}(method, crit, bounds, cs, df, disconnected)
end

# Refine the bound of parameter k: for every accepted point at the extreme index `ext`
# along axis k, bisect the acceptance boundary between it and its rejected neighbour at
# `ext + dir` (other coordinates fixed); return the most extreme crossing.
function _refine_bound(statf, crit::T, grids::Vector{Vector{T}},
        coords::Vector{Vector{Int}}, k::Int, ext::Int, dir::Int) where {T}
    best = dir < 0 ? T(Inf) : T(-Inf)
    for c in coords
        c[k] == ext || continue
        b = _grid_point(grids, c)
        inside, outside = grids[k][ext], grids[k][ext + dir]
        for _ in 1:50
            mid = (inside + outside) / 2
            b[k] = mid
            if statf(b) < crit
                inside = mid
            else
                outside = mid
            end
        end
        best = dir < 0 ? min(best, inside) : max(best, inside)
    end
    return best
end

# Number of connected pieces of the accepted grid points (neighbours include diagonals,
# so a set is only "disconnected" when separated by a band of rejected points).
function _n_components(accepted::Vector{Bool}, n::Int, K::Int)
    offsets = [digits(m; base = 3, pad = K) .- 1 for m in 0:(3 ^ K - 1)]
    filter!(o -> any(!=(0), o), offsets)
    seen = falses(length(accepted))
    stack = Int[]
    ncomp = 0
    for start in eachindex(accepted)
        (accepted[start] && !seen[start]) || continue
        ncomp += 1
        seen[start] = true
        push!(stack, start)
        while !isempty(stack)
            c = _grid_coords(pop!(stack), n, K)
            for o in offsets
                cn = c .+ o
                all(x -> 1 ≤ x ≤ n, cn) || continue
                j = 1 + sum((cn[k] - 1) * n^(k - 1) for k in 1:K)
                if accepted[j] && !seen[j]
                    seen[j] = true
                    push!(stack, j)
                end
            end
        end
    end
    return ncomp
end
