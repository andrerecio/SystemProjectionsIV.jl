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
using StatsBase: coef, stderror
using LinearAlgebra: I, Symmetric, tr, eigvals
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

# Scalar AR / KLM statistics b ↦ stat([b]) on the residualised data (intercept only).
function _stat_fns(y, Y, Z, H)
    y_H, Y_H, Z_p, Czz, T_eff = _residualized(y, Y, Z, H)
    Nz = size(Z_p, 1)
    v_res, _ = _first_stage_resid_cov(Y_H, Z_p, Czz, T_eff, 1, Nz)
    R = _restriction_matrix(Float64, 1, H)
    d = Nz + 1
    ar = b -> _ar_stat([b], y_H, Y_H, Z_p, Czz, T_eff, d)
    klm = b -> _klm_stat([b], y_H, Y_H, v_res, Z_p, Czz, R, T_eff, d)
    return ar, klm
end

# Endpoints of the accepted set {b : f(b) < crit} located on a fine grid over [lo, hi],
# each refined by linear interpolation of f − crit between the last accepted and the
# first rejected grid point. Assumes the set is interior to [lo, hi].
function _fine_bounds(f, crit, lo, hi; n = 20001)
    g = range(lo, hi; length = n)
    s = [f(b) for b in g] .- crit
    acc = findall(<(0), s)
    i, j = first(acc), last(acc)
    l = g[i - 1] + (g[i] - g[i - 1]) * (-s[i - 1]) / (s[i] - s[i - 1])
    u = g[j] + (g[j + 1] - g[j]) * (-s[j]) / (s[j + 1] - s[j])
    return l, u
end

# Port of the Fieldhouse–Mertens MATLAB K statistic, `GetKstat` in
# docs/ext/fieldhouse_mertens/Make_Table_1.m with `btrace` from
# docs/ext/fieldhouse_mertens/auxiliary_files/btrace.m. MATLAB A/B = A B'(B B')⁻¹.
_mrdiv(A, B) = (A * B') / (B * B')
function _btrace(Y, H)
    NR, NC = size(Y, 1) ÷ H, size(Y, 2) ÷ H
    return [tr(Y[(1 + (r - 1) * H):(r * H), (1 + (c - 1) * H):(c * H)])
            for r in 1:NR, c in 1:NC]
end
function _getkstat(uH, Z, YH, Nx, H, K)
    Nz, T = size(Z)
    IH = Matrix{Float64}(I, H, H)
    R = kron(Matrix{Float64}(I, K, K), vec(IH))
    PYH = _mrdiv(YH, Z) * Z
    vH = YH - PYH
    PuH = _mrdiv(uH, Z) * Z
    MuH = uH - PuH
    MvH = vH - _mrdiv(vH, Z) * Z
    V = MuH * MuH'
    iV = IH / V
    tYhat = PYH - _mrdiv(MvH, MuH) * PuH
    Sigg = R' * kron(tYhat * tYhat', iV * uH * uH' * iV) * R / (T - Nz - Nx)
    iSigg = Matrix{Float64}(I, K, K) / Sigg
    bt = _btrace(iV * uH * tYhat', H)
    return (bt * iSigg * bt')[1]
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

        res = spiv(yv, Yv, z; H = H)

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
        res = spiv(yv, Yv, Zv; H = 4)

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
        res_klm = spiv(yv, Yv, Zv; H = 4, robust = :KLM)
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

    # -------------------------------------------------------------------
    # 6. Robust-set endpoints: the bisection-refined bounds agree with the
    #    crossings located on a very fine grid (K = 1, AR and KLM); β̂ is a
    #    grid point under the default (odd) grid_length.
    # -------------------------------------------------------------------
    @testset "robust bounds ≡ fine grid" begin
        H = 4
        yv, Yv, Zv = _dgp(400, H, 0.7, 20240530)
        ar, klm = _stat_fns(yv, Yv, Zv, H)
        for (method, f) in ((:AR, ar), (:KLM, klm))
            res = spiv(yv, Yv, Zv; H = H, robust = method)
            rb = robust_inference(res)
            β̂, se = coef(res)[1], stderror(res)[1]
            @test rb.method === method
            @test rb.disconnected === false
            @test all(isfinite, rb.bounds)
            l, u = _fine_bounds(f, rb.critical_value, β̂ - 10se, β̂ + 10se)
            @test rb.bounds[1, 1] ≈ l atol = 1e-6
            @test rb.bounds[1, 2] ≈ u atol = 1e-6
            # the refined endpoints sit on the acceptance boundary
            @test f(rb.bounds[1, 1]) ≈ rb.critical_value rtol = 1e-6
            @test f(rb.bounds[1, 2]) ≈ rb.critical_value rtol = 1e-6
            # default grid_length = 201 over β̂ ± 10·se ⇒ step 0.1·se, β̂ on the grid
            cs = vec(rb.confidence_set)
            @test any(c -> isapprox(c, β̂; atol = 1e-12 * max(1, abs(β̂))), cs)
            @test all(d -> isapprox(d, 0.1se; rtol = 1e-8), diff(sort(cs)))
            # bounds are at most one grid step outside the accepted grid points
            @test 0 ≤ minimum(cs) - rb.bounds[1, 1] < 0.1se
            @test 0 ≤ rb.bounds[1, 2] - maximum(cs) < 0.1se
        end
        @test robust_inference(spiv(yv, Yv, Zv; H = H)).degrees_freedom == H
    end

    # -------------------------------------------------------------------
    # 7. Search grid options: (lo, hi) tuple, K × 2 matrix, grid_length; edges
    #    reached by the accepted set are reported as ±Inf.
    # -------------------------------------------------------------------
    @testset "grid options and ±Inf" begin
        H = 4
        yv, Yv, Zv = _dgp(400, H, 0.7, 20240530)
        r_def = spiv(yv, Yv, Zv; H = H)
        b_def = robust_inference(r_def).bounds

        # an explicit range containing the set gives the same refined bounds
        r_tup = spiv(yv, Yv, Zv; H = H, grid = (0.0, 1.5), grid_length = 151)
        rb_t = robust_inference(r_tup)
        @test rb_t.bounds ≈ b_def atol = 1e-8
        @test all(0.0 .≤ rb_t.confidence_set .≤ 1.5)
        @test all(d -> isapprox(d, 0.01; rtol = 1e-8), diff(vec(rb_t.confidence_set)))
        # a K × 2 matrix is the same box as the tuple (bit-identical)
        r_mat = spiv(yv, Yv, Zv; H = H, grid = [0.0 1.5], grid_length = 151)
        @test robust_inference(r_mat).bounds == rb_t.bounds
        @test robust_inference(r_mat).confidence_set == rb_t.confidence_set

        # a box cutting through the set ⇒ the cut side is ±Inf, the other is refined
        lo_in = (b_def[1, 1] + coef(r_def)[1]) / 2
        r_cut = spiv(yv, Yv, Zv; H = H, grid = (lo_in, 2.0))
        @test robust_inference(r_cut).bounds[1, 1] == -Inf
        @test robust_inference(r_cut).bounds[1, 2] ≈ b_def[1, 2] atol = 1e-8
        r_cut2 = spiv(yv, Yv, Zv; H = H, grid = (-1.0, (b_def[1, 2] + coef(r_def)[1]) / 2))
        @test robust_inference(r_cut2).bounds[1, 1] ≈ b_def[1, 1] atol = 1e-8
        @test robust_inference(r_cut2).bounds[1, 2] == Inf

        # irrelevant instrument: the AR set is unbounded ⇒ [-Inf, Inf]
        yw, Yw, Zw = _dgp(250, H, 0.7, 1; relevance = 0.0)
        r_weak = spiv(yw, Yw, Zw; H = H)
        @test weak_iv_test(r_weak).is_weak
        @test robust_inference(r_weak).bounds == [-Inf Inf]
        r_weak2 = spiv(yw, Yw, Zw; H = H, grid = (-50.0, 50.0))
        @test robust_inference(r_weak2).bounds == [-Inf Inf]

        # K = 2 with a per-parameter K × 2 box
        Random.seed!(77)
        T = 300
        Z2 = randn(T, 2)
        Y2 = Z2 * [1.0 0.3; -0.2 0.9] .+ randn(T, 2)
        y2 = Y2 * [0.5, -0.4] .+ randn(T)
        box = [0.0 1.0; -1.0 0.2]
        r2 = spiv(y2, Y2, Z2; H = 3, grid = box, grid_length = 41)
        rb2 = robust_inference(r2)
        @test size(rb2.confidence_set, 2) == 2
        @test all(box[1, 1] .≤ rb2.confidence_set[:, 1] .≤ box[1, 2])
        @test all(box[2, 1] .≤ rb2.confidence_set[:, 2] .≤ box[2, 2])
        @test all(isfinite, rb2.bounds)
        @test all(rb2.bounds[:, 1] .< coef(r2) .< rb2.bounds[:, 2])
        # default K = 2 grid: 101 points per parameter centred on β̂
        rb2d = robust_inference(spiv(y2, Y2, Z2; H = 3))
        @test any(i -> rb2d.confidence_set[i, :] ≈ coef(r2), axes(rb2d.confidence_set, 1))
    end

    # -------------------------------------------------------------------
    # 8. Disconnected KLM set (weak-ish design, relevance 0.15): flagged, the
    #    bounds are the hull, and a rejected point lies inside the hull.
    # -------------------------------------------------------------------
    @testset "disconnected KLM set" begin
        H = 4
        yv, Yv, Zv = _dgp(250, H, 0.7, 924; relevance = 0.15)
        res = spiv(yv, Yv, Zv; H = H, robust = :KLM)
        rb = robust_inference(res)
        @test rb.disconnected
        @test all(isfinite, rb.bounds)
        cs = sort(vec(rb.confidence_set))
        @test rb.bounds[1, 1] ≤ cs[1] && cs[end] ≤ rb.bounds[1, 2]
        step = 0.1 * stderror(res)[1]
        gaps = findall(>(1.5step), diff(cs))
        @test !isempty(gaps)
        _, klm = _stat_fns(yv, Yv, Zv, H)
        mid = (cs[gaps[1]] + cs[gaps[1] + 1]) / 2
        @test klm(mid) > rb.critical_value               # a hole inside the hull
        # the AR set on the same data is a single interval
        @test !robust_inference(spiv(yv, Yv, Zv; H = H)).disconnected
    end

    # -------------------------------------------------------------------
    # 9. Invalid inference options are rejected with an ArgumentError.
    # -------------------------------------------------------------------
    @testset "option validation" begin
        yv, Yv, Zv = _dgp(200, 2, 0.7, 3)
        @test_throws ArgumentError spiv(yv, Yv, Zv; H = 2, robust = :bogus)
        @test_throws ArgumentError spiv(yv, Yv, Zv; H = 2, robust = :none)
        for lv in (0, 1, 1.5, -0.1)
            @test_throws ArgumentError spiv(yv, Yv, Zv; H = 2, level = lv)
        end
        @test_throws ArgumentError spiv(yv, Yv, Zv; H = 2, tau = 0)
        @test_throws ArgumentError spiv(yv, Yv, Zv; H = 2, grid_length = 1)
        @test_throws ArgumentError spiv(yv, Yv, Zv; H = 2, grid = 0)
        @test_throws ArgumentError spiv(yv, Yv, Zv; H = 2, grid = -3)
        @test_throws ArgumentError spiv(yv, Yv, Zv; H = 2, grid = (1.0, 0.0))
        @test_throws ArgumentError spiv(yv, Yv, Zv; H = 2, grid = (-Inf, 1.0))
        @test_throws ArgumentError spiv(yv, Yv, Zv; H = 2, grid = "wide")
        @test_throws ArgumentError spiv(yv, Yv, Zv; H = 2, grid = [0.0 1.0; 0.0 1.0])
        @test_throws ArgumentError spiv(yv, Yv, Zv; H = 2, grid = Inf)
        @test_throws MethodError spiv(yv, Yv, Zv; H = 2, weak_iv = :AR)   # removed kwarg

        # invalid levels on the post-estimation accessors
        res = spiv(yv, Yv, Zv; H = 2)
        for lv in (0, 1, -0.5, 1.5)
            @test_throws ArgumentError confint(res; level = lv)
            @test_throws ArgumentError coeftable(res; level = lv)
        end
    end

    # -------------------------------------------------------------------
    # 9b. Ill-posed designs are rejected rather than producing garbage robust sets:
    #     more stacked horizons than T_eff − Nx − Nz (singular u M_Z u'), and
    #     instruments spanned by the controls or collinear with each other.
    # -------------------------------------------------------------------
    @testset "ill-posed designs" begin
        yv, Yv, Zv = _dgp(80, 2, 0.7, 3)
        # H = 15 stacked horizons but T_eff − Nx − Nz = 20 − 13 − 1 = 6
        @test_throws ArgumentError spiv(yv, Yv, Zv; H = 0:4:59, xlags = 4)
        @test spiv(yv, Yv, Zv; H = 0:4:47, xlags = 4) isa SPIVResult

        yv, Yv, Zv = _dgp(300, 3, 0.7, 1)
        z = vec(Zv)
        @test_throws ArgumentError spiv(yv, Yv, hcat(z, ones(300)); H = 3)  # constant
        @test_throws ArgumentError spiv(yv, Yv, z; H = 3, X = z)            # z in X
        @test_throws ArgumentError spiv(yv, Yv, hcat(z, 2 .* z); H = 3)     # collinear Z
    end

    # Column scaling of the controls must not change the rank decision or β̂.
    @testset "collinearity check is scale-invariant" begin
        yv, Yv, Zv = _dgp(300, 3, 0.7, 5)
        Random.seed!(11)
        w1 = 3 .+ cumsum(0.01 .* randn(300))
        w2 = randn(300)
        r1 = spiv(yv, Yv, Zv; H = 3, X = hcat(w1, w2))
        r2 = @test_logs spiv(yv, Yv, Zv; H = 3, X = hcat(1e9 .* w1, 1e-4 .* w2))
        @test r2.Nx == r1.Nx == 3
        @test coef(r2) ≈ coef(r1) rtol = 1e-8
    end

    # -------------------------------------------------------------------
    # 10. K = 2, Nz = 2, Nx = 3 against the Fieldhouse–Mertens MATLAB: the KLM
    #     statistic vs `GetKstat` + `btrace`, and g_min vs the `gmin` lines of
    #     `run_SPIV_total` (docs/ext/fieldhouse_mertens/Make_Table_1.m), with the
    #     controls partialled out MATLAB-style (yHo = yH − (yH/X)X).
    # -------------------------------------------------------------------
    @testset "KLM / g_min ≡ Fieldhouse–Mertens MATLAB (K = 2)" begin
        Random.seed!(2718)
        T, H, K, Nz = 300, 3, 2, 2
        W = randn(T, 2)                                   # extra controls ⇒ Nx = 3
        Zv = randn(T, Nz) .+ 0.2 .* W[:, 1]
        Yv = Zv * [0.9 0.3; -0.2 0.8] .+ 0.5 .* W .+ randn(T, K)
        yv = Yv * [0.6, -0.3] .+ 0.4 .* W[:, 2] .+ randn(T)

        res = spiv(yv, Yv, Zv; H = H, X = W, robust = :KLM)
        @test res.Nx == 3

        T_eff = T - (H - 1)
        yH = _stack_leads(reshape(yv, T, 1), H)
        YH = _stack_leads(Yv, H)
        Zt = permutedims(Zv[1:T_eff, :])
        Xc = vcat(permutedims(W[1:T_eff, :]), ones(1, T_eff))  # [XL; ones], MATLAB order
        Nx = size(Xc, 1)
        yHo = yH - _mrdiv(yH, Xc) * Xc
        YHo = YH - _mrdiv(YH, Xc) * Xc
        Zo = Zt - _mrdiv(Zt, Xc) * Xc
        IH = Matrix{Float64}(I, H, H)
        R = kron(Matrix{Float64}(I, K, K), vec(IH))

        # g_min (MATLAB `gmin`)
        PYHo = _mrdiv(YHo, Zo) * Zo
        vH = YHo - PYHo
        SigvH = vH * vH' / (size(vH, 2) - Nz - Nx)
        Omega = Symmetric(R' * kron(SigvH, IH) * R)
        Om_ih = inv(sqrt(Omega))
        gmin = minimum(eigvals(Symmetric(
            Om_ih * R' * kron(PYHo * PYHo', IH) * R * Om_ih / Nz)))
        @test weak_iv_test(res).g_min ≈ gmin rtol = 1e-9

        # KLM statistic at several candidate b (MATLAB `GetKstat`)
        Czz = Symmetric(Zo * Zo')
        for b in ([0.6, -0.3], [0.0, 0.0], [1.2, 0.4], coef(res))
            uH = yHo - kron(b', IH) * YHo
            @test _klm_stat(b, yHo, YHo, vH, Zo, Czz, R, T_eff, Nz + Nx) ≈
                  _getkstat(uH, Zo, YHo, Nx, H, K) rtol = 1e-8
        end

        # the reported K = 2 set: accepted points pass the MATLAB statistic
        rb = robust_inference(res)
        @test rb.degrees_freedom == K
        @test rb.critical_value ≈ chisqinvcdf(K, 0.95)
        cs = rb.confidence_set
        @test size(cs, 1) > 0
        for i in round.(Int, range(1, size(cs, 1); length = 5))
            b = cs[i, :]
            @test _getkstat(yHo - kron(b', IH) * YHo, Zo, YHo, Nx, H, K) < rb.critical_value
        end
    end
end
