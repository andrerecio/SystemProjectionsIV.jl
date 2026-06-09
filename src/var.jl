# VAR-variant forecast-error construction (docs/technical.md §4.2, eqs. A.2–A.3).
#
# Reimplemented (cited) from gragusa MacroEconometricsTools.jl
# (`docs/ext/gragusa/MacroEconometricsTools.jl/src/var/estimation.jl` and
# `var/identification.jl::compute_ma_matrices`), reusing `create_lags` / `companion_form`
# from src/utilities.jl. Only the forecast-error construction (step 1) is VAR-specific;
# the resulting y⊥/Y⊥/Z⊥ matrices have the same shapes as the LP variant and feed the
# shared `_spiv_estimate`.

# Fit a VAR(p) on the stacked observables W = [Y | y | z] (T × m, m = K+1+Nz), build the
# h-step forecast errors X̂⊥_t(h) = Σ_{j=0}^h Â^{h-j} ê_{t+j} (eq. A.3), and select rows:
#   ŷ⊥_H  (H × T_eff)   — the y-row across horizons
#   Ŷ⊥_H  (HK × T_eff)  — the K Y-rows across horizons (variable-major, matching _stack_leads)
#   Ẑ⊥    (Nz × T_eff)  — the z-rows of the one-step residual ê (= X̂⊥_t(0))
# T_eff = (T − p) − (H − 1). Nx_eff = 1 + m·p (VAR regressors per equation) is the
# effective control count used in the dof corrections.
function _var_forecast_errors(
        yv::AbstractVector{Float64},
        Ym::AbstractMatrix{Float64},
        Zm::AbstractMatrix{Float64},
        H::Int,
        p::Int
)
    T = length(yv)
    K = size(Ym, 2)
    Nz = size(Zm, 2)
    m = K + 1 + Nz
    W = hcat(Ym, yv, Zm)                       # T × m, columns [Y(1:K) | y(K+1) | z]
    y_idx = K + 1
    z_idx = (K + 2):(K + 1 + Nz)

    # --- VAR(p) by OLS (eq. A.2) -----------------------------------------
    lagged = create_lags(W, p)                 # T × (1 + m·p); first p rows hold NaN lags
    valid = (p + 1):T
    Xd = lagged[valid, :]                       # (T-p) × (1 + m·p)
    Wv = W[valid, :]                            # (T-p) × m
    A = Xd \ Wv                                 # (1 + m·p) × m  (intercept row 1, then lags)
    ê = Wv .- Xd * A                            # (T-p) × m  reduced-form residuals
    # Lag-coefficient array A_l[i, a] = coef of variable a at lag l in equation i.
    lags3 = reshape(permutedims(A[2:end, :]), m, m, p)
    F = companion_form(lags3)                   # (m·p) × (m·p) companion matrix

    # --- MA coefficients Φ_k = (F^k)[1:m, 1:m], k = 0,…,H-1 (Φ_0 = I) ----
    Φ = Vector{Matrix{Float64}}(undef, H)
    Φ[1] = Matrix{Float64}(I, m, m)
    if H > 1
        Fpow = copy(F)                          # F^1
        for k in 2:H
            Φ[k] = Fpow[1:m, 1:m]
            Fpow = Fpow * F
        end
    end

    # --- forecast errors (eq. A.3) and row selection ---------------------
    Te = size(ê, 1)                             # T - p
    T_eff = Te - (H - 1)
    T_eff > 0 || throw(
        ArgumentError(
        "VAR horizon/lags leave no observations: T - p - (H-1) = $T_eff ≤ 0 (T=$T, p=$p, H=$H)"),
    )

    y_perp = Matrix{Float64}(undef, H, T_eff)
    Y_perp = Matrix{Float64}(undef, H * K, T_eff)
    Z_perp = Matrix{Float64}(undef, Nz, T_eff)
    xh = Vector{Float64}(undef, m)
    tmp = Vector{Float64}(undef, m)
    for c in 1:T_eff
        Z_perp[:, c] .= @view ê[c, z_idx]       # h = 0 ⇒ X̂⊥(c,0) = ê_c
        for h in 0:(H - 1)
            fill!(xh, 0.0)
            for j in 0:h
                mul!(tmp, Φ[h - j + 1], @view ê[c + j, :])
                xh .+= tmp
            end
            y_perp[h + 1, c] = xh[y_idx]
            for k in 1:K
                Y_perp[(k - 1) * H + (h + 1), c] = xh[k]
            end
        end
    end

    return y_perp, Y_perp, Z_perp, T_eff, 1 + m * p
end

# VAR variant of step 1 (dispatched). `X` (controls) is ignored — the VAR lags subsume it.
function _forecast_errors(
        ::SPIVwithVAR,
        yv::AbstractVector{Float64},
        Ym::AbstractMatrix{Float64},
        Xm::AbstractMatrix{Float64},
        Zm::AbstractMatrix{Float64},
        H::Int,
        p::Int
)
    return _var_forecast_errors(yv, Ym, Zm, H, p)
end
