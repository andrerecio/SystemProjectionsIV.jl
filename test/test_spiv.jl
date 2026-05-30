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
_runspiv(y, Y, X, Z, h) = spiv(y, Y, X, Z; H = h)

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

        res = spiv(yv, reshape(Yv, T, 1), ones(T, 1), reshape(ε, T, 1); H = H)

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

        res = spiv(yv, Yv, ones(T, 1), Zv; H = H)

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

        res = spiv(yv, reshape(Yv, T, 1), ones(T, 1), reshape(z, T, 1); H = 1)

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
        @test_throws ArgumentError spiv(randn(T), randn(T, 3), ones(T, 1), randn(T, 1); H = 1)
        # row-count mismatch
        @test_throws DimensionMismatch spiv(randn(T), randn(T - 1, 1), ones(T, 1), randn(T, 1); H = 2)
        # H too large for the sample
        @test_throws ArgumentError spiv(randn(5), randn(5, 1), ones(5, 1), randn(5, 1); H = 6)
        # insufficient degrees of freedom
        @test_throws ArgumentError spiv(randn(4), randn(4, 1), ones(4, 3), randn(4, 1); H = 1)
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
        Xv = ones(T, 1)

        res = @inferred _runspiv(yv, Yv, Xv, Zv, H)
        @test res isa SPIVResult{Float64, SPIVwithLP}

        # Phase 2 fills β/vcov/residuals; inference & IRF blocks stay NaN stubs.
        wk = weak_iv_test(res)
        @test isnan(wk.g_min) && wk.is_weak == false
        @test robust_inference(res).method == :none
        @test all(isnan, res.irf_outcome.point)
        @test all(isnan, res.irf_endogenous.point)
        @test nobs(res) == T - (H - 1)
        @test spec(res) === SPIVwithLP()
    end
end
