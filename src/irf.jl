# Impulse responses for SP-IV with HAC standard errors.
#
# Implements the IRF representation of docs/technical.md §3.3 (eq. 11): the outcome and
# endogenous IRFs are the projections of y⊥_H / Y⊥_H on the whitened instruments
#   Θ̂_y = (y⊥_H Z⊥' / T)(Z⊥ Z⊥' / T)^{-1/2}   ∈ ℝ^{H × Nz}
#   Θ̂_Y = (Y⊥_H Z⊥' / T)(Z⊥ Z⊥' / T)^{-1/2}   ∈ ℝ^{HK × Nz}
# Equivalently, row r of Θ̂ is the coefficient vector of regressing that residualised
# series on the standardised instruments Z_std = (Z⊥Z⊥'/T)^{-1/2} Z⊥ (no intercept).
#
# Per-horizon Newey–West HAC standard errors use the registered CovarianceMatrices.jl
# (`aVar` with a `Bartlett` kernel) as the long-run-variance engine. The `hac` option
# selects the bandwidth rule:
#   :fixed     — lag truncation = horizon (the SP-IV / paper default). At horizon h
#                (h = 0,…,H-1) `Bartlett(h+1)` reproduces the Newey–West weights
#                1 − j/(h+1) for lags j = 1,…,h (h = 0 ⇒ Γ₀ only, i.e. White/HC0).
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

# HAC standard errors of one IRF row: coefficients `b` from regressing `yrow` on the
# standardised instruments `X` (T_eff × Nz). The HAC coefficient covariance is the
# sandwich (X'X)⁻¹ · meat · (X'X)⁻¹ with `meat` the unscaled kernel sum of the moment
# matrix X .* e (e the regression residual) — docs/technical.md §3.3 / §5.4.
function _irf_row_se(
        X::AbstractMatrix{T},
        yrow::AbstractVector{T},
        b::AbstractVector{T},
        XtXinv::AbstractMatrix{T},
        bw::Int,
        hac::Symbol
) where {T <: AbstractFloat}
    e = yrow .- X * b
    mm = X .* e                                        # T_eff × Nz moment matrix
    meat = aVar(_hac_kernel(hac, bw), mm; demean = false, scale = false)
    V = XtXinv * meat * XtXinv
    return sqrt.(diag(V))
end

# Outcome (H × Nz) and endogenous (H × K × Nz) IRF blocks with HAC bands at level 1 − α.
function _irf_blocks(
        y_perp::AbstractMatrix{T},
        Y_perp::AbstractMatrix{T},
        Z_perp::AbstractMatrix{T},
        Czz::AbstractMatrix{T},
        H::Int,
        K::Int,
        Nz::Int,
        T_eff::Int,
        α::Real,
        hac::Symbol
) where {T <: AbstractFloat}
    iZ = inv(sqrt(Symmetric(Czz ./ T_eff)))            # (Z⊥Z⊥'/T)^{-1/2}, Nz × Nz
    X = permutedims(iZ * Z_perp)                       # standardised instruments, T_eff × Nz
    XtXinv = inv(Symmetric(X' * X))                    # (= I_Nz / T_eff by construction)
    z = norminvcdf(1 - α / 2)

    Θy = (y_perp * Z_perp' ./ T_eff) * iZ              # H × Nz
    ΘY = (Y_perp * Z_perp' ./ T_eff) * iZ              # HK × Nz

    # Outcome block: row h is horizon h-1 ⇒ HAC bandwidth h (lag truncation h-1).
    se_y = similar(Θy)
    for h in 1:H
        se_y[h, :] .= _irf_row_se(X, vec(y_perp[h, :]), Θy[h, :], XtXinv, h, hac)
    end
    outcome = IRFBlock(Θy, se_y, Θy .- z .* se_y, Θy .+ z .* se_y)

    # Endogenous block reshaped to H × K × Nz: row (k-1)H+h ↔ variable k, horizon h-1.
    pt = Array{T, 3}(undef, H, K, Nz)
    se = Array{T, 3}(undef, H, K, Nz)
    for k in 1:K, h in 1:H

        r = (k - 1) * H + h
        b = ΘY[r, :]
        pt[h, k, :] .= b
        se[h, k, :] .= _irf_row_se(X, vec(Y_perp[r, :]), b, XtXinv, h, hac)
    end
    endogenous = IRFBlock(pt, se, pt .- z .* se, pt .+ z .* se)

    return outcome, endogenous
end
