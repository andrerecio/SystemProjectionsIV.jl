# Phase 3 tests: weak-IV diagnostic and AR/KLM robust confidence sets.
# References are to docs/technical.md §6–7.

using Test
using SystemProjectionsIV
using SystemProjectionsIV:
                           _stack_leads,
                           _restriction_matrix,
                           _first_stage_resid_cov,
                           _weak_iv_diagnostic,
                           _ar_stat,
                           _klm_stat
using StatsBase: coef
using LinearAlgebra: I, Symmetric, tr
using Statistics: mean
using StatsFuns: chisqinvcdf
using Random

_dm(M) = M .- mean(M, dims = 2)

# Residualize-on-constant copies of the matrices spiv() builds internally (Nx = 1).
function _residualized(y, Y, Z, H)
    T = length(y)
    T_eff = T - (H - 1)
    y_H = _dm(_stack_leads(reshape(y, T, 1), H))     # H × T_eff
    Y_H = _dm(_stack_leads(Y, H))                     # HK × T_eff
    Z_p = _dm(permutedims(Z[1:T_eff, :]))             # Nz × T_eff
    Czz = Symmetric(Z_p * Z_p')
    return y_H, Y_H, Z_p, Czz, T_eff
end

# Synthetic IV DGP with a known β (mirrors test_spiv.jl).
function _dgp(T, H, β, seed; relevance = 1.0, ρ = 0.8)
    Random.seed!(seed)
    a = relevance .* [0.8^h for h in 0:(H - 1)]
    ε = randn(T)
    u = 0.3 .* randn(T)
    ν = 0.2 .* randn(T)
    Yv = zeros(T)
    for t in 1:T
        s = 0.0
        for j in 0:(H - 1)
            t - j ≥ 1 && (s += a[j + 1] * ε[t - j])
        end
        Yv[t] = s + ρ * u[t] + ν[t]
    end
    yv = β .* Yv .+ u
    return yv, reshape(Yv, T, 1), reshape(ε, T, 1)
end

# Explicit (T_eff × T_eff) projection ports of eqs. (20)/(21), mirroring the
# reference C++; used to cross-check the implicit forms in src/inference.jl.
function _ar_explicit(b, y_H, Y_H, Z_p, T_eff, d_AR)
    H = size(y_H, 1)
    Pz = Z_p' * (Symmetric(Z_p * Z_p') \ Z_p)
    Mz = Matrix{Float64}(I, T_eff, T_eff) - Pz
    u = y_H - kron(b', Matrix{Float64}(I, H, H)) * Y_H
    uPu = u * Pz * u'
    uMu = u * Mz * u'
    return (T_eff - d_AR) * tr(uPu * inv(uMu))
end

function _klm_explicit(b, y_H, Y_H, v_res, Z_p, R, T_eff, d_K)
    H = size(y_H, 1)
    Pz = Z_p' * (Symmetric(Z_p * Z_p') \ Z_p)
    Mz = Matrix{Float64}(I, T_eff, T_eff) - Pz
    u = y_H - kron(b', Matrix{Float64}(I, H, H)) * Y_H
    u_check = u * Mz
    v_check = v_res * Mz
    Ξ = u * Mz * u'
    A = (u_check * u_check') \ (u * Pz)
    Y_check = Y_H * Pz - v_check * u_check' * A
    s = R' * vec(Ξ \ (u * Y_check'))
    inner = Ξ \ (u * u' * (Ξ \ Matrix{Float64}(I, H, H)))
    V = R' * kron(Y_check * Y_check', inner) * R
    return (T_eff - d_K) * (s' * (V \ s))
end

@testset "inference (weak-IV / AR / KLM)" begin

    # -------------------------------------------------------------------
    # 1. g_min reduces to the first-stage F in the just-identified scalar
    #    case (K = Nz = H = 1): g = (Y⊥ P_Z⊥ Y⊥') / σ̂²_v = F.
    # -------------------------------------------------------------------
    @testset "g_min ↔ first-stage F" begin
        Random.seed!(11)
        T = 300
        H = 1
        Yv = randn(T, 1)
        z = 0.7 .* vec(Yv) .+ randn(T)
        yv = 0.5 .* vec(Yv) .+ randn(T)

        res = spiv(yv, Yv, ones(T, 1), reshape(z, T, 1); H = H)

        y_H, Y_H, Z_p, Czz, T_eff = _residualized(yv, Yv, reshape(z, T, 1), H)
        v_res, _ = _first_stage_resid_cov(Y_H, Z_p, Czz, T_eff, 1, 1)
        YPY = (Y_H * Z_p') * (Czz \ (Z_p * Y_H'))
        σ2v = (v_res * v_res')[1] / (T_eff - 1 - 1)     # T_eff − Nx − Nz
        F = YPY[1] / σ2v

        @test weak_iv_test(res).g_min ≈ F rtol = 1e-10
        @test weak_iv_test(res).g_min > 0
    end

    # -------------------------------------------------------------------
    # 2. AR and KLM implicit forms equal an explicit-projection computation
    #    (eqs. 20/21) on a small overidentified-in-H case.
    # -------------------------------------------------------------------
    @testset "AR/KLM ≡ explicit projection" begin
        Random.seed!(202)
        T = 60
        H = 2
        Yv = randn(T, 1)
        Zv = reshape(0.6 .* vec(Yv) .+ randn(T), T, 1)
        yv = 0.4 .* vec(Yv) .+ randn(T)

        y_H, Y_H, Z_p, Czz, T_eff = _residualized(yv, Yv, Zv, H)
        v_res, _ = _first_stage_resid_cov(Y_H, Z_p, Czz, T_eff, 1, 1)
        R = _restriction_matrix(Float64, 1, H)
        d = 1 + 1                                        # Nz + Nx

        for b0 in (-0.3, 0.0, 0.4, 1.1)
            b = [b0]
            @test _ar_stat(b, y_H, Y_H, Z_p, Czz, T_eff, d) ≈
                  _ar_explicit(b, y_H, Y_H, Z_p, T_eff, d) rtol = 1e-9
            @test _klm_stat(b, y_H, Y_H, v_res, Z_p, Czz, R, T_eff, d) ≈
                  _klm_explicit(b, y_H, Y_H, v_res, Z_p, R, T_eff, d) rtol = 1e-9
        end
    end

    # -------------------------------------------------------------------
    # 3. AR confidence set has ≈ nominal coverage. Over Monte Carlo reps,
    #    AR(β_true) should fall below χ²_{H·Nz} at the (1−α) quantile with
    #    frequency ≈ 1−α.
    # -------------------------------------------------------------------
    @testset "AR coverage" begin
        α = 0.05
        H = 2
        Nz = 1
        T = 200
        β = 0.5
        reps = 1000
        crit = chisqinvcdf(H * Nz, 1 - α)
        d_AR = Nz + 1                                    # Nz + Nx (intercept)

        covered = 0
        for r in 1:reps
            yv, Yv, Zv = _dgp(T, H, β, 1000 + r; relevance = 0.7)
            y_H, Y_H, Z_p, Czz, T_eff = _residualized(yv, Yv, Zv, H)
            covered += _ar_stat([β], y_H, Y_H, Z_p, Czz, T_eff, d_AR) < crit
        end
        @test covered / reps ≈ (1 - α) atol = 0.03
    end

    # -------------------------------------------------------------------
    # 4. Populated result: strong instruments ⇒ not weak; CS brackets β̂.
    # -------------------------------------------------------------------
    @testset "populated result" begin
        yv, Yv, Zv = _dgp(400, 4, 0.7, 20240530)
        res = spiv(yv, Yv, ones(400, 1), Zv; H = 4)

        wk = weak_iv_test(res)
        @test isfinite(wk.g_min) && wk.g_min ≥ 0
        @test isfinite(wk.critical_value)
        @test wk.is_weak == false                        # strong instruments

        rb = robust_inference(res)
        @test rb.method == :AR
        @test rb.degrees_freedom == 4 * 1                # H · Nz
        @test size(rb.confidence_set, 1) > 0
        @test rb.bounds[1, 1] ≤ coef(res)[1] ≤ rb.bounds[1, 2]

        # KLM path on the same data.
        res_klm = spiv(yv, Yv, ones(400, 1), Zv; H = 4, weak_iv = :KLM)
        rbk = robust_inference(res_klm)
        @test rbk.method == :KLM
        @test rbk.degrees_freedom == 1                   # K
        @test rbk.bounds[1, 1] ≤ coef(res_klm)[1] ≤ rbk.bounds[1, 2]
    end

    # -------------------------------------------------------------------
    # 5. Type stability of the internal kernels.
    # -------------------------------------------------------------------
    @testset "type stability" begin
        Random.seed!(5)
        T = 150
        H = 3
        Yv = randn(T, 1)
        Zv = reshape(0.6 .* vec(Yv) .+ randn(T), T, 1)
        yv = 0.4 .* vec(Yv) .+ randn(T)

        y_H, Y_H, Z_p, Czz, T_eff = _residualized(yv, Yv, Zv, H)
        v_res, W2 = _first_stage_resid_cov(Y_H, Z_p, Czz, T_eff, 1, 1)
        R = _restriction_matrix(Float64, 1, H)
        YPY = (Y_H * Z_p') * (Czz \ (Z_p * Y_H'))

        @test @inferred(_ar_stat([0.4], y_H, Y_H, Z_p, Czz, T_eff, 2)) isa Float64
        @test @inferred(_klm_stat([0.4], y_H, Y_H, v_res, Z_p, Czz, R, T_eff, 2)) isa
              Float64
        @test @inferred(_weak_iv_diagnostic(YPY, Matrix(W2), R, H, 1, 0.10, 0.05)) isa
              WeakIVDiagnostic{Float64}
    end
end
