using Test
using SystemProjectionsIV
using StatsBase

@testset "types" begin
    @testset "spec hierarchy" begin
        @test SPIVwithLP <: AbstractSPIVSpec
        @test SPIVwithVAR <: AbstractSPIVSpec
        @test SPIVwithLP() isa AbstractSPIVSpec
        @test SPIVwithVAR() isa AbstractSPIVSpec
    end

    @testset "ForecastErrors" begin
        H, K, Nz, T_eff = 5, 2, 3, 100
        y_perp = randn(H, T_eff)
        Y_perp = randn(H * K, T_eff)
        Z_perp = randn(Nz, T_eff)

        fe = ForecastErrors(y_perp, Y_perp, Z_perp; H = H, K = K, Nz = Nz)

        @test fe isa ForecastErrors{Float64}
        @test size(fe.y_perp) == (H, T_eff)
        @test size(fe.Y_perp) == (H * K, T_eff)
        @test size(fe.Z_perp) == (Nz, T_eff)
        @test (fe.H, fe.K, fe.Nz) == (H, K, Nz)

        # Dimension validation
        @test_throws DimensionMismatch ForecastErrors(
            randn(H + 1, T_eff),
            Y_perp,
            Z_perp;
            H = H,
            K = K,
            Nz = Nz
        )
        @test_throws DimensionMismatch ForecastErrors(
            y_perp,
            randn(H * K, T_eff + 1),
            Z_perp;
            H = H,
            K = K,
            Nz = Nz
        )
        @test_throws DimensionMismatch ForecastErrors(
            y_perp,
            Y_perp,
            randn(Nz + 1, T_eff);
            H = H,
            K = K,
            Nz = Nz
        )
    end

    @testset "WeakIVDiagnostic stub" begin
        d = WeakIVDiagnostic{Float64}()
        @test d isa WeakIVDiagnostic{Float64}
        @test isnan(d.g_min)
        @test isnan(d.critical_value)
        @test d.is_weak === false
    end

    @testset "RobustInference stub" begin
        K = 3
        r = RobustInference{Float64}(K)
        @test r isa RobustInference{Float64}
        @test r.method === :none
        @test size(r.bounds) == (K, 2)
        @test all(isnan, r.bounds)
        @test size(r.confidence_set) == (0, K)
        @test r.degrees_freedom == 0
    end

    @testset "IRFBlock" begin
        # Direct construction (N = 2)
        H, Nz = 6, 2
        b = IRFBlock(randn(H, Nz), randn(H, Nz), randn(H, Nz), randn(H, Nz))
        @test b isa IRFBlock{Float64, 2}
        @test size(b.point) == (H, Nz)

        # Shape mismatch
        @test_throws DimensionMismatch IRFBlock(
            randn(H, Nz),
            randn(H + 1, Nz),
            randn(H, Nz),
            randn(H, Nz)
        )

        # NaN-filled stub (N = 3)
        K = 4
        b3 = IRFBlock{Float64}(H, K, Nz)
        @test b3 isa IRFBlock{Float64, 3}
        @test size(b3.point) == (H, K, Nz)
        @test all(isnan, b3.point)
        @test all(isnan, b3.se)
        @test all(isnan, b3.lower)
        @test all(isnan, b3.upper)
    end

    @testset "SPIVResult stub constructor" begin
        H, K, Nz, Nx, T_eff = 5, 2, 3, 4, 80
        β = randn(K)
        V = let A = randn(K, K);
            A * A'
        end          # PSD covariance
        u = randn(H, T_eff)

        r = SPIVResult(SPIVwithLP(), β, V, u; H = H, K = K, Nz = Nz, Nx = Nx, T_eff = T_eff)

        @test r isa SPIVResult{Float64, SPIVwithLP}
        @test r.β == β
        @test r.vcov == V
        @test r.residuals == u
        @test (r.H, r.K, r.Nz, r.Nx, r.T_eff) == (H, K, Nz, Nx, T_eff)
        @test r.weak_iv isa WeakIVDiagnostic{Float64}
        @test r.robust isa RobustInference{Float64}
        @test r.irf_outcome isa IRFBlock{Float64, 2}
        @test r.irf_endogenous isa IRFBlock{Float64, 3}
        @test size(r.irf_outcome.point) == (H, Nz)
        @test size(r.irf_endogenous.point) == (H, K, Nz)
        @test all(isnan, r.irf_outcome.point)
        @test all(isnan, r.irf_endogenous.point)

        # Dispatch picks up the spec type parameter
        r_var = SPIVResult(
            SPIVwithVAR(),
            β,
            V,
            u;
            H = H,
            K = K,
            Nz = Nz,
            Nx = Nx,
            T_eff = T_eff
        )
        @test r_var isa SPIVResult{Float64, SPIVwithVAR}

        # Dimension validation
        @test_throws DimensionMismatch SPIVResult(
            SPIVwithLP(),
            randn(K + 1),
            V,
            u;
            H = H,
            K = K,
            Nz = Nz,
            Nx = Nx,
            T_eff = T_eff
        )
        @test_throws DimensionMismatch SPIVResult(
            SPIVwithLP(),
            β,
            randn(K, K + 1),
            u;
            H = H,
            K = K,
            Nz = Nz,
            Nx = Nx,
            T_eff = T_eff
        )
        @test_throws DimensionMismatch SPIVResult(
            SPIVwithLP(),
            β,
            V,
            randn(H + 1, T_eff);
            H = H,
            K = K,
            Nz = Nz,
            Nx = Nx,
            T_eff = T_eff
        )
    end

    @testset "accessors return correct values" begin
        H, K, Nz, Nx, T_eff = 5, 2, 3, 4, 80
        β = randn(K)
        V = let A = randn(K, K);
            A * A'
        end
        u = randn(H, T_eff)
        r = SPIVResult(SPIVwithLP(), β, V, u; H = H, K = K, Nz = Nz, Nx = Nx, T_eff = T_eff)

        @test coef(r) == β
        @test vcov(r) == V
        @test nobs(r) == T_eff
        @test dof_residual(r) == T_eff - Nx - K
        @test residuals(r) == u
        @test stderror(r) ≈ sqrt.([V[i, i] for i in 1:K])
        @test horizon(r) == H
        @test weak_iv_test(r) === r.weak_iv
        @test robust_inference(r) === r.robust
        @test spec(r) === r.spec
        @test irf(r) === r.irf_outcome
        @test irf(r; response = :outcome) === r.irf_outcome
        @test irf(r; response = :endogenous) === r.irf_endogenous
        @test_throws ArgumentError irf(r; response = :bogus)

        # Strong-IV normal CI: width = 2 · z_{0.975} · SE
        ci = confint(r; level = 0.95)
        @test size(ci) == (K, 2)
        widths = ci[:, 2] .- ci[:, 1]
        z975 = 1.959963984540054   # Φ⁻¹(0.975)
        @test widths ≈ 2 .* z975 .* stderror(r) rtol = 1e-8
        # Centred on β
        @test (ci[:, 1] .+ ci[:, 2]) ./ 2 ≈ β rtol = 1e-10
    end

    @testset "accessor type stability" begin
        H, K, Nz, Nx, T_eff = 5, 2, 3, 4, 80
        β = randn(K)
        V = let A = randn(K, K);
            A * A'
        end
        u = randn(H, T_eff)
        r = SPIVResult(SPIVwithLP(), β, V, u; H = H, K = K, Nz = Nz, Nx = Nx, T_eff = T_eff)

        @test @inferred(coef(r)) === r.β
        @test @inferred(vcov(r)) === r.vcov
        @test @inferred(nobs(r)) == T_eff
        @test @inferred(dof_residual(r)) == T_eff - Nx - K
        @test @inferred(residuals(r)) === r.residuals
        @test @inferred(stderror(r)) isa Vector{Float64}
        @test @inferred(horizon(r)) == H
        @test @inferred(weak_iv_test(r)) === r.weak_iv
        @test @inferred(robust_inference(r)) === r.robust
        @test @inferred(spec(r)) === r.spec
        @test @inferred(confint(r; level = 0.95)) isa Matrix{Float64}
    end
end
