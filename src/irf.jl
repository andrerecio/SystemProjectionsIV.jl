# Impulse responses for SP-IV with HAC standard errors.
#
# Implements the IRF representation of docs/technical.md §3.3 (eq. 11): the outcome and
# endogenous IRFs are the projections of y⊥_H / Y⊥_H on the whitened instruments
#   Θ̂_y = (y⊥_H Z⊥' / T)(Z⊥ Z⊥' / T)^{-1/2}   ∈ ℝ^{H × Nz}
#   Θ̂_Y = (Y⊥_H Z⊥' / T)(Z⊥ Z⊥' / T)^{-1/2}   ∈ ℝ^{HK × Nz}
# Equivalently, row r of Θ̂ is the coefficient vector of regressing that residualised
# series on the standardised instruments X = (Z⊥Z⊥'/T)^{-1/2} Z⊥ (no intercept). Same
# scaling as `irfsy` / `irfsY` in docs/ext/fieldhouse_mertens/Make_Table_1.m.
#
# Standard errors are a package choice — the paper and the Fieldhouse–Mertens code report
# point IRFs only. They are per-horizon Newey–West HAC sandwiches computed with the
# registered CovarianceMatrices.jl (`aVar` with a `Bartlett` kernel), applied to the
# delta-method influence function of Θ̂, which accounts for the estimated whitening matrix
# S^{-1/2}, S = Z⊥Z⊥'/T (treating S as known under-covers). The `hac` option selects the
# bandwidth rule:
#   :fixed     — lag truncation = horizon h. `Bartlett(h+1)` reproduces the Newey–West
#                weights 1 − j/(h+1) for lags j = 1,…,h (h = 0 ⇒ Γ₀ only, i.e. White/HC0).
#   :neweywest — data-driven Newey–West (1994) automatic bandwidth, `Bartlett{NeweyWest}`.
#   :andrews   — data-driven Andrews (1991) automatic bandwidth, `Bartlett{Andrews}`.
# Inputs follow the estimator's time-in-columns layout (y_perp is H × T_eff, etc.).

# Kernel for the requested bandwidth rule. A fresh kernel is built per call because the
# automatic estimators store the selected bandwidth in the kernel object.
function _hac_kernel(hac::Symbol, bw::Int)
    hac === :fixed && return Bartlett(bw)
    hac === :neweywest && return Bartlett{NeweyWest}()
    hac === :andrews && return Bartlett{Andrews}()
    throw(ArgumentError("hac must be :fixed, :neweywest, or :andrews, got :$hac"))
end

# HAC standard errors of one IRF row θ (coefficients of regressing `yrow` on the
# standardised instruments X, T_eff × Nz). With S = QΛQ' and w_t = Q'x_t, the influence
# function of θ̂ is (row form, linearising S^{-1/2} via the Sylvester equation of S^{1/2})
#   ψ_t' = e_t x_t' + θ' Q [(w_t w_t' − I) ∘ C] Q',   C_ij = √λ_j / (√λ_i + √λ_j),
# with e the regression residual; for Nz = 1 this is e_t x_t + θ(x_t² − 1)/2. The HAC
# covariance is the sandwich (X'X)⁻¹ · meat · (X'X)⁻¹ with `meat` the unscaled kernel sum
# of ψ (X'X = T·I by construction).
function _irf_row_se(
        X::AbstractMatrix{T},
        Wq::AbstractMatrix{T},
        Q::AbstractMatrix{T},
        C::AbstractMatrix{T},
        yrow::AbstractVector{T},
        θ::AbstractVector{T},
        XtXinv::AbstractMatrix{T},
        bw::Int,
        hac::Symbol
) where {T <: AbstractFloat}
    e = yrow .- X * θ
    a = Q' * θ                                         # θ in the eigenbasis of S
    G = ((Wq .* a') * C) .* Wq .- (a .* diag(C))'      # T_eff × Nz, rows a'[(w w' − I)∘C]
    ψ = X .* e .+ G * Q'                               # T_eff × Nz influence function
    meat = aVar(_hac_kernel(hac, bw), ψ; demean = false, scale = false)
    V = XtXinv * meat * XtXinv
    return sqrt.(diag(V))
end

# Outcome (H × Nz) and endogenous (H × K × Nz) IRF blocks with HAC bands at level 1 − α.
# Row i is horizon hs[i].
function _irf_blocks(
        y_perp::AbstractMatrix{T},
        Y_perp::AbstractMatrix{T},
        Z_perp::AbstractMatrix{T},
        Czz::AbstractMatrix{T},
        hs::AbstractVector{Int},
        K::Int,
        Nz::Int,
        T_eff::Int,
        α::Real,
        hac::Symbol
) where {T <: AbstractFloat}
    H = length(hs)
    λ, Q = eigen(Symmetric(Czz ./ T_eff))              # S = Z⊥Z⊥'/T = QΛQ'
    sλ = sqrt.(λ)
    iZ = Q * (Q' ./ sλ)                                 # S^{-1/2}, Nz × Nz
    C = [sλ[j] / (sλ[i] + sλ[j]) for i in 1:Nz, j in 1:Nz]
    X = permutedims(iZ * Z_perp)                       # standardised instruments, T_eff × Nz
    Wq = X * Q                                          # rows w_t' = (Q'x_t)'
    XtXinv = inv(Symmetric(X' * X))                    # (= I_Nz / T_eff by construction)
    z = norminvcdf(1 - α / 2)

    Θy = (y_perp * Z_perp' ./ T_eff) * iZ              # H × Nz
    ΘY = (Y_perp * Z_perp' ./ T_eff) * iZ              # HK × Nz

    # Outcome block: row i is horizon hs[i] ⇒ :fixed HAC bandwidth hs[i] + 1.
    se_y = similar(Θy)
    for i in 1:H
        se_y[i, :] .= _irf_row_se(
            X, Wq, Q, C, (@view y_perp[i, :]), Θy[i, :], XtXinv, hs[i] + 1, hac)
    end
    outcome = IRFBlock(Θy, se_y, Θy .- z .* se_y, Θy .+ z .* se_y)

    # Endogenous block reshaped to H × K × Nz: row (k-1)H+i ↔ variable k, horizon hs[i].
    pt = Array{T, 3}(undef, H, K, Nz)
    se = Array{T, 3}(undef, H, K, Nz)
    for k in 1:K, i in 1:H

        r = (k - 1) * H + i
        θ = ΘY[r, :]
        pt[i, k, :] .= θ
        se[i, k, :] .= _irf_row_se(
            X, Wq, Q, C, (@view Y_perp[r, :]), θ, XtXinv, hs[i] + 1, hac)
    end
    endogenous = IRFBlock(pt, se, pt .- z .* se, pt .+ z .* se)

    return outcome, endogenous
end
