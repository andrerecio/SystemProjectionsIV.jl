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

# Explicit Newey–West Bartlett sum of an influence-function matrix ψ (T × Nz) with
# lag truncation bw-1: Γ₀ + Σ_{j=1}^{bw-1} (1 − j/bw)(Γⱼ + Γⱼ').
function _bartlett_meat(ψ, bw)
    meat = ψ' * ψ
    for j in 1:(bw - 1)
        Γ = ψ[1:(end - j), :]' * ψ[(1 + j):end, :]
        meat .+= (1 - j / bw) .* (Γ .+ Γ')
    end
    return meat
end

# Fréchet derivative of S ↦ S^{-1/2} in direction E: with L solving the Sylvester
# equation S^{1/2} L + L S^{1/2} = E (solved via its Kronecker form),
# D[E] = −S^{-1/2} L S^{-1/2}. Independent of src/irf.jl's eigenbasis formula.
function _dinvsqrt(S, E)
    n = size(S, 1)
    Sh = sqrt(Symmetric(S))
    In = Matrix{Float64}(I, n, n)
    L = reshape((kron(In, Sh) + kron(Sh', In)) \ vec(E), n, n)
    Sih = inv(Sh)
    return -Sih * L * Sih
end

# Delta-method HAC SE of one IRF row θ̂ = (y Z⊥'/T) S^{-1/2}, S = Z⊥Z⊥'/T (time in
# columns; `yrow` has length T). Influence function of observation t:
#   ψ_t = (y_t z_t' − y Z⊥'/T) S^{-1/2} + (y Z⊥'/T) · D[z_t z_t' − S],
# and V = (1/T²) · Bartlett-sum(ψ) with lag truncation bw − 1.
function _irf_se_delta(Z_p, yrow, bw)
    Nz, T = size(Z_p)
    S = (Z_p * Z_p') ./ T
    Sih = inv(sqrt(Symmetric(S)))
    m = (yrow' * Z_p') ./ T                               # 1 × Nz
    ψ = zeros(T, Nz)
    for t in 1:T
        zt = Z_p[:, t]
        ψ[t, :] = vec((yrow[t] .* zt' .- m) * Sih .+ m * _dinvsqrt(S, zt * zt' .- S))
    end
    return sqrt.(diag(_bartlett_meat(ψ, bw))) ./ T
end

# Newey–West SE that treats the whitening S^{-1/2} as known (the pre-fix formula):
# regression scores x_t e_t only. Used to show the delta-method term matters.
function _irf_se_naive(X, yrow, b, bw)
    e = yrow .- X * b
    XtXinv = inv(Symmetric(X' * X))
    return sqrt.(diag(XtXinv * _bartlett_meat(X .* e, bw) * XtXinv))
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

        res = spiv(yv, Yv, Zv; H = H)

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
    # 2. HAC SEs equal an explicit delta-method Bartlett sandwich at every
    #    horizon: validates the influence function (which accounts for the
    #    estimated whitening S^{-1/2}), the CovarianceMatrices call, and the
    #    :fixed Bartlett(h+1) ↔ lag-h mapping. Nz = 1, then the overidentified
    #    Nz = 2, K = 2 case with correlated instruments.
    # -------------------------------------------------------------------
    @testset "HAC ≡ explicit delta-method sandwich" begin
        Random.seed!(99)
        T = 250
        H = 4
        Yv = randn(T, 1)
        Zv = reshape(0.7 .* vec(Yv) .+ randn(T), T, 1)
        yv = 0.5 .* vec(Yv) .+ randn(T)

        res = spiv(yv, Yv, Zv; H = H)
        y_H, Y_H, Z_p, Czz, T_eff = _resid_irf(yv, Yv, Zv, H)

        for h in 1:H                                    # row h ↔ horizon h-1 ⇒ bw = h
            @test res.irf_outcome.se[h, :] ≈ _irf_se_delta(Z_p, vec(y_H[h, :]), h) rtol = 1e-8
            @test res.irf_endogenous.se[h, 1, :] ≈ _irf_se_delta(Z_p, vec(Y_H[h, :]), h) rtol = 1e-8
        end

        # Nz = 1 closed form at horizon 0 (Γ₀ only): ψ_t = e_t x_t + θ (x_t² − 1)/2,
        # x the standardised instrument.
        x = vec(Z_p) ./ sqrt(sum(abs2, Z_p) / T_eff)
        θ = res.irf_outcome.point[1, 1]
        e = vec(y_H[1, :]) .- θ .* x
        ψ = e .* x .+ θ .* (x .^ 2 .- 1) ./ 2
        @test res.irf_outcome.se[1, 1] ≈ sqrt(sum(abs2, ψ)) / T_eff rtol = 1e-10

        # Overidentified: K = 2 endogenous, Nz = 2 correlated, non-centred instruments.
        Random.seed!(314)
        T2, H2, K2 = 300, 3, 2
        Z2 = randn(T2, 2) * [1.0 0.6; 0.0 0.8] .+ 0.3
        Y2 = Z2 * [0.8 0.2; -0.3 0.9] .+ randn(T2, K2)
        y2 = Y2 * [0.5, -0.4] .+ randn(T2)
        r2 = spiv(y2, Y2, Z2; H = H2)
        y_H2, Y_H2, Z_p2, _, _ = _resid_irf(y2, Y2, Z2, H2)
        for h in 1:H2
            @test r2.irf_outcome.se[h, :] ≈ _irf_se_delta(Z_p2, vec(y_H2[h, :]), h) rtol = 1e-8
            for k in 1:K2
                @test r2.irf_endogenous.se[h, k, :] ≈
                      _irf_se_delta(Z_p2, vec(Y_H2[(k - 1) * H2 + h, :]), h) rtol = 1e-8
            end
        end
        # Bands are point ± z·se at the result's level.
        z = 1.959963984540054
        @test r2.irf_outcome.lower ≈ r2.irf_outcome.point .- z .* r2.irf_outcome.se
        @test r2.irf_endogenous.upper ≈ r2.irf_endogenous.point .+ z .* r2.irf_endogenous.se
        r90 = spiv(y2, Y2, Z2; H = H2, level = 0.9)
        z90 = 1.6448536269514722
        @test r90.irf_outcome.se == r2.irf_outcome.se
        @test r90.irf_outcome.upper ≈ r90.irf_outcome.point .+ z90 .* r90.irf_outcome.se
    end

    # -------------------------------------------------------------------
    # 2b. Horizon subsets: IRF rows equal the matching rows of the contiguous
    #     result with the same max horizon (same T_eff), including the HAC SEs —
    #     so the :fixed bandwidth uses the true horizon hs[i] + 1, not i.
    # -------------------------------------------------------------------
    @testset "horizon subset IRFs" begin
        Random.seed!(17)
        T = 300
        Yv = randn(T, 2)
        Zv = randn(T, 2)
        yv = randn(T)
        hs = [1, 4, 6]
        rs = spiv(yv, Yv, Zv; H = hs)
        rc = spiv(yv, Yv, Zv; H = 7)
        rows = hs .+ 1
        @test rs.irf_outcome.point ≈ rc.irf_outcome.point[rows, :] rtol = 1e-10
        @test rs.irf_outcome.se ≈ rc.irf_outcome.se[rows, :] rtol = 1e-10
        @test rs.irf_endogenous.point ≈ rc.irf_endogenous.point[rows, :, :] rtol = 1e-10
        @test rs.irf_endogenous.se ≈ rc.irf_endogenous.se[rows, :, :] rtol = 1e-10

        y_H, Y_H, Z_p, _, _ = _resid_irf(yv, Yv, Zv, 7)
        for (i, h) in enumerate(hs)
            @test rs.irf_outcome.se[i, :] ≈ _irf_se_delta(Z_p, vec(y_H[h + 1, :]), h + 1) rtol = 1e-8
        end
    end

    # -------------------------------------------------------------------
    # 2c. Monte Carlo: with a strong first stage and Nz = 1, the delta-method
    #     IRF bands have near-nominal coverage of the true endogenous IRF,
    #     while the SE that treats S^{-1/2} as known under-covers (≈ 74%).
    #     iid design: z ~ N(0,1), Y = z + 0.5v, so Θ_Y(0) = 1 per 1-s.d. shock.
    # -------------------------------------------------------------------
    @testset "IRF band coverage (Monte Carlo)" begin
        rng = Xoshiro(2024)
        T, reps, a = 200, 2000, 1.0
        cover = 0
        cover_naive = 0
        for _ in 1:reps
            z = randn(rng, T)
            Y = a .* z .+ 0.5 .* randn(rng, T)
            y = 0.5 .* Y .+ randn(rng, T)
            ie = spiv(y, Y, z; H = 1, grid_length = 2).irf_endogenous
            cover += ie.lower[1, 1, 1] ≤ a ≤ ie.upper[1, 1, 1]
            zp = z .- mean(z)
            X = reshape(zp ./ sqrt(mean(abs2, zp)), T, 1)
            se_n = _irf_se_naive(X, Y .- mean(Y), [ie.point[1, 1, 1]], 1)[1]
            cover_naive += abs(ie.point[1, 1, 1] - a) ≤ 1.959963984540054 * se_n
        end
        @test cover / reps ≈ 0.95 atol = 0.02
        @test cover_naive / reps < 0.85
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

        res = spiv(yv, Yv, Zv; H = H)
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
            y_H, Y_H, Z_p, Matrix(Czz), collect(0:(H - 1)), 1, 1, T_eff, 0.05, :fixed)
        @test out isa Tuple{
            SystemProjectionsIV.IRFBlock{Float64, 2},
            SystemProjectionsIV.IRFBlock{Float64, 3}
        }
    end
end
