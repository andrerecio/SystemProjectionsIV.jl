# Phase 4 tests: impulse responses and per-horizon HAC standard errors.
# References are to docs/technical.md §3.3 (eq. 11).

using Test
using SystemProjectionsIV
using SystemProjectionsIV: _stack_leads, _irf_blocks
using StatsBase: coef
using LinearAlgebra: I, Symmetric, diag
using Statistics: mean
using Random

_dmi(M) = M .- mean(M, dims = 2)

# Residualize-on-constant copies of the matrices spiv() builds internally (Nx = 1).
function _resid_irf(y, Y, Z, H)
    T = length(y)
    T_eff = T - (H - 1)
    y_H = _dmi(_stack_leads(reshape(y, T, 1), H))
    Y_H = _dmi(_stack_leads(Y, H))
    Z_p = _dmi(permutedims(Z[1:T_eff, :]))
    Czz = Symmetric(Z_p * Z_p')
    return y_H, Y_H, Z_p, Czz, T_eff
end

# Explicit Newey–West Bartlett sandwich for one IRF row (lag truncation bw-1):
# meat = Γ₀ + Σ_{j=1}^{bw-1} (1 − j/bw)(Γⱼ + Γⱼ'), V = (X'X)⁻¹ meat (X'X)⁻¹.
function _hac_explicit(X, yrow, b, bw)
    e = yrow .- X * b
    mm = X .* e
    meat = mm' * mm
    for j in 1:(bw - 1)
        Γ = mm[1:(end - j), :]' * mm[(1 + j):end, :]
        meat .+= (1 - j / bw) .* (Γ .+ Γ')
    end
    XtXinv = inv(Symmetric(X' * X))
    return sqrt.(diag(XtXinv * meat * XtXinv))
end

@testset "irf (IRFs + HAC SEs)" begin

    # -------------------------------------------------------------------
    # 1. IRF point estimates equal the independent eq. (11) computation,
    #    and the stacked-IRF OLS reproduces β̂ (eq. 13). Overidentified
    #    case K=2, Nz=2, H=3 to exercise the H×K×Nz reshaping.
    # -------------------------------------------------------------------
    @testset "IRF points ≡ eq. (11) / stacked-IRF" begin
        Random.seed!(7)
        T = 300
        H = 3
        K = 2
        Nz = 2
        Yv = randn(T, K)
        Zv = randn(T, Nz)
        yv = randn(T)

        res = spiv(yv, Yv, ones(T, 1), Zv; H = H)

        y_H, Y_H, Z_p, Czz, T_eff = _resid_irf(yv, Yv, Zv, H)
        iZ = inv(sqrt(Symmetric(Czz ./ T_eff)))
        Θy = (y_H * Z_p' ./ T_eff) * iZ                     # H × Nz
        ΘY = (Y_H * Z_p' ./ T_eff) * iZ                     # HK × Nz

        @test res.irf_outcome.point ≈ Θy rtol = 1e-10
        for k in 1:K, h in 1:H, n in 1:Nz
            @test res.irf_endogenous.point[h, k, n] ≈ ΘY[(k - 1) * H + h, n] rtol = 1e-10
        end

        # Stacked-IRF OLS of Θy on ΘY returns β̂ (eq. 13).
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
        @test (SY' * SY) \ (SY' * Sy) ≈ coef(res) rtol = 1e-9
    end

    # -------------------------------------------------------------------
    # 2. HAC SEs equal an explicit Bartlett sandwich at every horizon
    #    (validates the CovarianceMatrices call + the Bartlett(h) ↔ lag
    #    h-1 mapping). Horizon 0 (bw = 1) reduces to the White/HC0 SE.
    # -------------------------------------------------------------------
    @testset "HAC ≡ explicit Bartlett sandwich" begin
        Random.seed!(99)
        T = 250
        H = 4
        Yv = randn(T, 1)
        Zv = reshape(0.7 .* vec(Yv) .+ randn(T), T, 1)
        yv = 0.5 .* vec(Yv) .+ randn(T)

        res = spiv(yv, Yv, ones(T, 1), Zv; H = H)

        y_H, Y_H, Z_p, Czz, T_eff = _resid_irf(yv, Yv, Zv, H)
        iZ = inv(sqrt(Symmetric(Czz ./ T_eff)))
        X = permutedims(iZ * Z_p)                           # T_eff × Nz

        for h in 1:H
            bo = res.irf_outcome.point[h, :]
            @test res.irf_outcome.se[h, :] ≈ _hac_explicit(X, vec(y_H[h, :]), bo, h) rtol = 1e-8
            be = res.irf_endogenous.point[h, 1, :]
            @test res.irf_endogenous.se[h, 1, :] ≈ _hac_explicit(X, vec(Y_H[h, :]), be, h) rtol = 1e-8
        end

        # Horizon 0 ⇒ White/HC0 (Γ₀-only) standard error.
        b0 = res.irf_outcome.point[1, :]
        e0 = vec(y_H[1, :]) .- X * b0
        mm0 = X .* e0
        XtXinv = inv(Symmetric(X' * X))
        se_hc0 = sqrt.(diag(XtXinv * (mm0' * mm0) * XtXinv))
        @test res.irf_outcome.se[1, :] ≈ se_hc0 rtol = 1e-10
    end

    # -------------------------------------------------------------------
    # 3. Shapes, positive SEs, bands bracket the point.
    # -------------------------------------------------------------------
    @testset "bands and shapes" begin
        Random.seed!(20240530)
        T = 400
        H = 4
        Yv = randn(T, 1)
        Zv = reshape(0.8 .* vec(Yv) .+ randn(T), T, 1)
        yv = 0.6 .* vec(Yv) .+ randn(T)

        res = spiv(yv, Yv, ones(T, 1), Zv; H = H)
        io = res.irf_outcome
        ie = res.irf_endogenous

        @test size(io.point) == (H, 1)
        @test size(ie.point) == (H, 1, 1)
        @test all(io.se .> 0) && all(ie.se .> 0)
        @test all(io.lower .< io.point .< io.upper)
        @test all(ie.lower .< ie.point .< ie.upper)
        @test all(isfinite, io.point) && all(isfinite, ie.point)
    end

    # -------------------------------------------------------------------
    # 4. Type stability of the IRF kernel.
    # -------------------------------------------------------------------
    @testset "type stability" begin
        Random.seed!(5)
        T = 200
        H = 3
        Yv = randn(T, 1)
        Zv = reshape(0.6 .* vec(Yv) .+ randn(T), T, 1)
        yv = 0.4 .* vec(Yv) .+ randn(T)

        y_H, Y_H, Z_p, Czz, T_eff = _resid_irf(yv, Yv, Zv, H)
        out = @inferred _irf_blocks(
            y_H, Y_H, Z_p, Matrix(Czz), H, 1, 1, T_eff, 0.05, :fixed)
        @test out isa Tuple{
            SystemProjectionsIV.IRFBlock{Float64, 2},
            SystemProjectionsIV.IRFBlock{Float64, 3}
        }
    end
end
