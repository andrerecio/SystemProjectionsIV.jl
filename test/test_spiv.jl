# Phase 2 tests: the SP-IV point estimator (LP variant).
# References are to docs/technical.md.

using Test
using SystemProjectionsIV
using StatsBase: coef, vcov, residuals, stderror, nobs, dof_residual
using LinearAlgebra: I, Symmetric
using Statistics: mean
using Random

# Residualize-on-constant == demean across time; mirrors the estimator's FWL step
# when X is a single intercept column.
_demean(M) = M .- mean(M, dims = 2)

# Top-level (non-closure) wrapper so @inferred sees a plain generic function.
_runspiv(y, Y, Z, h) = spiv(y, Y, Z; H = h)

@testset "spiv (LP variant)" begin

    # -------------------------------------------------------------------
    # 1. Recovery of a known β on a synthetic IV DGP.
    #    y_t = β Y_t + u_t with Y endogenous (Cov(Y,u) ≠ 0) but instrument
    #    z_t = ε_t exogenous. Θ_y(h) = β Θ_Y(h) at every horizon ⇒ β̂ → β.
    # -------------------------------------------------------------------
    @testset "known-β recovery" begin
        Random.seed!(20240530)
        T = 4000
        H = 4
        β_true = 0.7
        a = [1.0, 0.6, 0.3, 0.15]        # IRF of Y to the shock, horizons 0..H-1
        ε = randn(T)
        u = 0.3 .* randn(T)
        ν = 0.2 .* randn(T)
        ρ = 0.8                          # endogeneity: Y loads on the structural error

        Yv = zeros(T)
        for t in 1:T
            s = 0.0
            for j in 0:(H - 1)
                t - j ≥ 1 && (s += a[j + 1] * ε[t - j])
            end
            Yv[t] = s + ρ * u[t] + ν[t]
        end
        yv = β_true .* Yv .+ u

        res = spiv(yv, Yv, ε; H = H)

        @test coef(res)[1] ≈ β_true atol = 0.05
        @test size(residuals(res)) == (H, T - (H - 1))
        @test size(vcov(res)) == (1, 1)
        @test stderror(res)[1] > 0
    end

    # -------------------------------------------------------------------
    # 2. Key-insight equivalence: the closed form (eq. 9) equals the
    #    stacked-IRF OLS (eq. 13) to machine precision. Overidentified
    #    case (K=2, Nz=2, H=3) to exercise the block machinery.
    # -------------------------------------------------------------------
    @testset "eq. (9) ≡ eq. (13)" begin
        Random.seed!(7)
        T = 300
        H = 3
        K = 2
        Nz = 2
        Yv = randn(T, K)
        Zv = randn(T, Nz)
        yv = randn(T)

        res = spiv(yv, Yv, Zv; H = H)

        # Independent eq. (13) path on the same demeaned, lead-stacked data.
        y_H = _demean(SystemProjectionsIV._stack_leads(reshape(yv, T, 1), H))  # H × T_eff
        Y_H = _demean(SystemProjectionsIV._stack_leads(Yv, H))                 # HK × T_eff
        T_eff = T - (H - 1)
        Z_p = _demean(permutedims(Zv[1:T_eff, :]))                            # Nz × T_eff

        Cisq = inv(sqrt(Symmetric(Z_p * Z_p')))      # (Z⊥ Z⊥')^{-1/2}
        S = Cisq * Z_p                                # Nz × T_eff, orthonormal rows
        ΘY = Y_H * S'                                 # HK × Nz
        Θy = y_H * S'                                 # H  × Nz

        # Reshape to HNz × K and HNz × 1 (any consistent (h,n) row order works).
        SY = zeros(H * Nz, K)
        Sy = zeros(H * Nz)
        r = 0
        for h in 1:H, n in 1:Nz

            r += 1
            for k in 1:K
                SY[r, k] = ΘY[(k - 1) * H + h, n]
            end
            Sy[r] = Θy[h, n]
        end
        β13 = (SY' * SY) \ (SY' * Sy)

        @test coef(res) ≈ β13 rtol = 1e-9
    end

    # -------------------------------------------------------------------
    # 3. Sandwich variance matches an independent hand computation on a
    #    tiny just-identified problem (H=1, K=1, Nz=1). Reduces to the
    #    Wald IV estimator with σ̂² / (Y⊥ P_Z Y⊥').
    # -------------------------------------------------------------------
    @testset "hand-computed sandwich" begin
        Random.seed!(99)
        T = 200
        Yv = randn(T)
        z = 0.8 .* Yv .+ randn(T)        # relevant instrument
        yv = 0.5 .* Yv .+ randn(T)

        res = spiv(yv, Yv, z; H = 1)

        yd = yv .- mean(yv)
        Yd = Yv .- mean(Yv)
        zd = z .- mean(z)
        β_hand = sum(yd .* zd) / sum(Yd .* zd)
        û = yd .- β_hand .* Yd
        Σ_hand = sum(abs2, û) / (T - 1 - 1)          # T_eff - Nx - K, Nx = K = 1
        YPY = sum(Yd .* zd)^2 / sum(abs2, zd)
        v_hand = Σ_hand / YPY

        @test coef(res)[1] ≈ β_hand
        @test vcov(res)[1, 1] ≈ v_hand
        @test dof_residual(res) == T - 1 - 1
    end

    # -------------------------------------------------------------------
    # 4. Input validation.
    # -------------------------------------------------------------------
    @testset "validation" begin
        T = 50
        # order condition H·Nz = 1 < K = 3
        @test_throws ArgumentError spiv(randn(T), randn(T, 3), randn(T, 1); H = 1)
        # row-count mismatch
        @test_throws DimensionMismatch spiv(randn(T), randn(T - 1, 1), randn(T, 1); H = 2)
        # H too large for the sample
        @test_throws ArgumentError spiv(randn(5), randn(5, 1), randn(5, 1); H = 6)
        # insufficient degrees of freedom (intercept + 2 controls ⇒ Nx = 3)
        @test_throws ArgumentError spiv(
            randn(4), randn(4, 1), randn(4, 1); H = 1, X = randn(4, 2))
        # interior non-finite values (beyond the auto-dropped burn-in) are rejected
        y_hole = randn(T)
        y_hole[25] = NaN
        @test_throws ArgumentError spiv(y_hole, randn(T, 1), randn(T, 1); H = 2)
        X_hole = randn(T, 1)
        X_hole[30, 1] = Inf
        @test_throws ArgumentError spiv(
            randn(T), randn(T, 1), randn(T, 1); H = 2, X = X_hole)
        # only the NaN sentinel marks burn-in: a leading Inf is a data error
        y_inf = randn(T)
        y_inf[1] = Inf
        @test_throws ArgumentError spiv(y_inf, randn(T, 1), randn(T, 1); H = 2)
        # nothing left after the burn-in
        @test_throws ArgumentError spiv(
            fill(NaN, T), randn(T, 1), randn(T, 1); H = 2)
        # trim leaves too few observations for the horizon
        @test_throws ArgumentError spiv(
            randn(6), randn(6, 1), randn(6, 1); H = 4, X = lags(randn(6), 3))
        # xlags validation
        @test_throws ArgumentError spiv(
            randn(T), randn(T, 1), randn(T, 1); H = 2, xlags = -1)
        @test_throws ArgumentError spiv(
            randn(T), randn(T, 1), randn(T, 1); H = 2, xlags = T)
    end

    # -------------------------------------------------------------------
    # 5. Type stability and untouched NaN stubs for later phases.
    # -------------------------------------------------------------------
    @testset "type stability & stubs" begin
        Random.seed!(1)
        T = 200
        H = 3
        K = 1
        Nz = 1
        yv = randn(T)
        Yv = randn(T, K)
        Zv = randn(T, Nz)

        res = @inferred _runspiv(yv, Yv, Zv, H)
        @test res isa SPIVResult{Float64, SPIVwithLP}

        # Phases 3–4 fill the weak-IV / robust / IRF blocks.
        wk = weak_iv_test(res)
        @test isfinite(wk.g_min)
        @test robust_inference(res).method == :AR
        @test all(isfinite, res.irf_outcome.point)
        @test all(isfinite, res.irf_endogenous.point)
        @test nobs(res) == T - (H - 1)
        @test spec(res) === SPIVwithLP()
    end

    # -------------------------------------------------------------------
    # 6. Input ergonomics: vector promotion, the intercept default, and the
    #    X keyword are exact re-parameterizations of the matrix interface.
    # -------------------------------------------------------------------
    @testset "input ergonomics" begin
        Random.seed!(42)
        T = 200
        H = 3
        yv = randn(T)
        Yv = randn(T)
        z = randn(T)
        W = randn(T, 2)                   # extra controls

        # Vectors and one-column matrices are interchangeable (bit-identical).
        r_vec = spiv(yv, Yv, z; H = H)
        r_mat = spiv(yv, reshape(Yv, T, 1), reshape(z, T, 1); H = H)
        @test coef(r_vec) == coef(r_mat)
        @test vcov(r_vec) == vcov(r_mat)

        # Default intercept ≡ passing the constant column explicitly.
        r_explicit = spiv(yv, Yv, z; H = H, intercept = false, X = ones(T, 1))
        @test coef(r_vec) == coef(r_explicit)
        @test vcov(r_vec) == vcov(r_explicit)
        @test r_vec.Nx == 1

        # X is appended after the intercept; Nx counts both.
        r_ctrl = spiv(yv, Yv, z; H = H, X = W)
        @test r_ctrl.Nx == 3
        r_ctrl2 = spiv(yv, Yv, z; H = H, intercept = false, X = hcat(ones(T), W))
        @test coef(r_ctrl) == coef(r_ctrl2)

        # intercept = false with no X gives the no-controls path.
        r_none = spiv(yv, Yv, z; H = H, intercept = false)
        @test r_none.Nx == 0
        @test isfinite(coef(r_none)[1])
    end

    # -------------------------------------------------------------------
    # 7. Automatic sample adjustment: leading non-finite rows (the lag/lags
    #    burn-in) are dropped from every input, bit-identically to trimming
    #    by hand.
    # -------------------------------------------------------------------
    @testset "auto-trim burn-in" begin
        Random.seed!(11)
        T = 300
        H = 3
        p = 4
        yv = randn(T)
        Yv = randn(T)
        z = randn(T)
        keep = (p + 1):T

        # X = lags(...) passed untrimmed ≡ the manual keep-dance (bit-identical).
        r_auto = spiv(yv, Yv, z; H = H, X = lags(Yv, p))
        r_hand = spiv(yv[keep], Yv[keep], z[keep]; H = H, X = lags(Yv, p)[keep, :])
        @test coef(r_auto) == coef(r_hand)
        @test vcov(r_auto) == vcov(r_hand)
        @test r_auto.irf_outcome.point == r_hand.irf_outcome.point
        @test nobs(r_auto) == (T - p) - (H - 1)

        # The burn-in can sit in y or Z themselves, not only in X.
        y_pad = copy(yv)
        y_pad[1:2] .= NaN
        r_ypad = spiv(y_pad, Yv, z; H = H)
        r_ycut = spiv(yv[3:end], Yv[3:end], z[3:end]; H = H)
        @test coef(r_ypad) == coef(r_ycut)

        # Ragged burn-in: the longest one wins (X has p bad rows, y has 2).
        r_rag = spiv(y_pad, Yv, z; H = H, X = lags(Yv, p))
        @test coef(r_rag) == coef(r_hand)
        @test nobs(r_rag) == (T - p) - (H - 1)

        # Vector X input with burn-in also works.
        r_vecX = spiv(yv, Yv, z; H = H, X = lag(Yv, 1))
        r_vecXh = spiv(yv[2:end], Yv[2:end], z[2:end]; H = H, X = lag(Yv, 1)[2:end])
        @test coef(r_vecX) == coef(r_vecXh)
    end

    # -------------------------------------------------------------------
    # 8. xlags: one keyword builds the lagged-[y Y Z] controls and adjusts
    #    the sample — the SPIVwithVAR(p) information set for the LP variant.
    # -------------------------------------------------------------------
    @testset "xlags" begin
        Random.seed!(12)
        T = 400
        H = 3
        p = 2
        yv = randn(T)
        Yv = randn(T, 2)
        z = randn(T)
        keep = (p + 1):T

        # xlags ≡ hand-built lags of [y Y Z] ≡ the fully manual call (bit-identical).
        r_x = spiv(yv, Yv, z; H = H, xlags = p)
        r_built = spiv(yv, Yv, z; H = H, X = lags(hcat(yv, Yv, z), p))
        r_hand = spiv(
            yv[keep], Yv[keep, :], z[keep]; H = H,
            X = lags(hcat(yv, Yv, z), p)[keep, :])
        @test coef(r_x) == coef(r_built)
        @test vcov(r_x) == vcov(r_built)
        @test coef(r_x) == coef(r_hand)

        # Nx = intercept + (K + 1 + Nz) · xlags.
        K, Nz = size(Yv, 2), 1
        @test r_x.Nx == 1 + (K + 1 + Nz) * p
        @test nobs(r_x) == (T - p) - (H - 1)

        # Combines with user controls (user X first, lag block appended).
        W = randn(T, 1)
        r_both = spiv(yv, Yv, z; H = H, X = W, xlags = p)
        r_both2 = spiv(yv, Yv, z; H = H, X = hcat(W, lags(hcat(yv, Yv, z), p)))
        @test coef(r_both) == coef(r_both2)
        @test r_both.Nx == 1 + 1 + (K + 1 + Nz) * p

        # xlags = 0 is the default (off).
        r_off = spiv(yv, Yv, z; H = H, xlags = 0)
        r_def = spiv(yv, Yv, z; H = H)
        @test coef(r_off) == coef(r_def)

        # LP with xlags = p is comparable to SPIVwithVAR(p) on a persistent DGP.
        Random.seed!(13)
        Tc = 4000
        εc = randn(Tc)
        uc = 0.3 .* randn(Tc)
        Yc = zeros(Tc)
        for t in 1:Tc, j in 0:3

            t - j ≥ 1 && (Yc[t] += 0.8^j * εc[t - j])
        end
        Yc .+= 0.7 .* uc .+ 0.2 .* randn(Tc)
        yc = 0.5 .* Yc .+ uc
        r_lp = spiv(yc, Yc, εc; H = 4, xlags = 1)
        r_var = spiv(yc, Yc, εc, SPIVwithVAR(); H = 4, p = 1)
        @test coef(r_lp)[1] ≈ coef(r_var)[1] atol = 0.1
    end

    # -------------------------------------------------------------------
    # 9. lag / lags data helpers (exported for building the X controls).
    # -------------------------------------------------------------------
    @testset "lag / lags helpers" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        @test isequal(lag(x, 1), [NaN, 1.0, 2.0, 3.0, 4.0])
        @test lag(x, 0) == x
        @test isequal(lag(x, -1), [2.0, 3.0, 4.0, 5.0, NaN])

        L = lags(x, 2)
        @test size(L) == (5, 2)
        @test isequal(L[:, 1], lag(x, 1))
        @test isequal(L[:, 2], lag(x, 2))
        @test all(isfinite, L[3:end, :])

        # Matrix input: lag-major column order [lag(X,1) lag(X,2) ...].
        M = [1.0 10.0; 2.0 20.0; 3.0 30.0; 4.0 40.0]
        LM = lags(M, 2)
        @test size(LM) == (4, 4)
        @test isequal(LM[:, 1:2], lag(M, 1))
        @test isequal(LM[:, 3:4], lag(M, 2))

        @test_throws ArgumentError lags(x, 0)

        # Round-trip into spiv with trimmed burn-in rows.
        Random.seed!(8)
        T = 300
        yv = randn(T)
        Yv = randn(T)
        z = randn(T)
        p = 2
        keep = (p + 1):T
        r = spiv(yv[keep], Yv[keep], z[keep]; H = 2, X = lags(yv, p)[keep, :])
        @test r.Nx == 1 + p
    end
end
