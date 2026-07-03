# Phase 5b tests: the SPIVwithVAR variant (forecast-error route, docs/technical.md §4.2).

using Test
using SystemProjectionsIV
using SystemProjectionsIV: _var_forecast_errors, create_lags
using StatsBase: coef
using Random

# VAR(1)-generated DGP: z is an exogenous shock; Y is endogenous (correlated with the
# structural error u); y = β·Y + u. SP-IV identifies β via the response to z.
function _dgp_var(T, β, seed)
    Random.seed!(seed)
    z = randn(T)
    u = randn(T)
    Y = zeros(T)
    for t in 2:T
        Y[t] = 0.5 * Y[t - 1] + z[t] + 0.6 * u[t]
    end
    y = β .* Y .+ u
    return y, reshape(Y, T, 1), reshape(z, T, 1)
end

# Top-level wrapper so @inferred sees a plain generic function.
_runvar(y, Y, Z, h, p) = spiv(y, Y, Z, SPIVwithVAR(); H = h, p = p)

@testset "spiv (VAR variant)" begin

    # -------------------------------------------------------------------
    # 1. Dispatch, return type, and shapes. T_eff = T − p − (H − 1).
    # -------------------------------------------------------------------
    @testset "dispatch and shapes" begin
        T, β, H, p = 800, 0.5, 4, 1
        y, Y, Z = _dgp_var(T, β, 1)
        res = spiv(y, Y, Z, SPIVwithVAR(); H = H, p = p)

        @test res isa SPIVResult{Float64, SPIVwithVAR}
        @test spec(res) === SPIVwithVAR()
        T_eff = T - p - (H - 1)
        @test size(residuals(res)) == (H, T_eff)
        @test size(res.irf_outcome.point) == (H, 1)
        @test size(res.irf_endogenous.point) == (H, 1, 1)
        @test nobs(res) == T_eff
    end

    # -------------------------------------------------------------------
    # 2. β recovery on a VAR(1) DGP, and comparability with the LP variant
    #    on the same data (the Phase-5 exit criterion).
    # -------------------------------------------------------------------
    @testset "β recovery and LP≈VAR" begin
        T, β = 4000, 0.7
        y, Y, Z = _dgp_var(T, β, 42)

        rvar = spiv(y, Y, Z, SPIVwithVAR(); H = 4, p = 1)
        rlp = spiv(y, Y, Z; H = 4)

        @test coef(rvar)[1] ≈ β atol = 0.05
        @test coef(rvar)[1] ≈ coef(rlp)[1] atol = 0.05
        # higher lag order also runs and recovers β
        rvar2 = spiv(y, Y, Z, SPIVwithVAR(); H = 4, p = 2)
        @test coef(rvar2)[1] ≈ β atol = 0.05
    end

    # -------------------------------------------------------------------
    # 3. Row selection: at h = 0 the forecast error equals the one-step VAR
    #    residual ê, so Z_perp and the h=0 rows of y_perp come from ê.
    # -------------------------------------------------------------------
    @testset "row selection ≡ VAR residuals" begin
        T, H, p = 500, 3, 1
        y, Y, Z = _dgp_var(T, 0.6, 7)
        K, Nz = size(Y, 2), size(Z, 2)

        yp, Yp, Zp, T_eff, Nx_eff = _var_forecast_errors(vec(y), Y, Z, H, p)

        # Independently fit the VAR(p) and form residuals ê.
        W = hcat(Y, y, Z)
        lagged = create_lags(W, p)
        valid = (p + 1):T
        Xd = lagged[valid, :]
        Wv = W[valid, :]
        ê = Wv .- Xd * (Xd \ Wv)
        y_idx = K + 1
        z_idx = (K + 2):(K + 1 + Nz)

        @test T_eff == (T - p) - (H - 1)
        @test Nx_eff == 1 + (K + 1 + Nz) * p
        @test Zp ≈ permutedims(ê[1:T_eff, z_idx]) rtol = 1e-10
        @test yp[1, :] ≈ ê[1:T_eff, y_idx] rtol = 1e-10      # h = 0 ⇒ X̂⊥(·,0) = ê
        @test Yp[1, :] ≈ ê[1:T_eff, 1] rtol = 1e-10          # K=1, variable-1 horizon-0 row
    end

    # -------------------------------------------------------------------
    # 4. LP-only kwargs are ignored with a single warning; the result is
    #    bit-identical to the clean call. Leading burn-in rows in y/Y/Z are
    #    still trimmed for the VAR variant.
    # -------------------------------------------------------------------
    @testset "ignored kwargs warn" begin
        T, H, p = 500, 3, 1
        y, Y, Z = _dgp_var(T, 0.6, 21)

        r_clean = @test_logs spiv(y, Y, Z, SPIVwithVAR(); H = H, p = p)

        warn_msg = r"ignored with SPIVwithVAR"
        r_X = @test_logs (:warn, warn_msg) spiv(
            y, Y, Z, SPIVwithVAR(); H = H, p = p, X = randn(T, 2))
        @test_logs (:warn, warn_msg) spiv(
            y, Y, Z, SPIVwithVAR(); H = H, p = p, intercept = false)
        @test_logs (:warn, warn_msg) spiv(y, Y, Z, SPIVwithVAR(); H = H, p = p, xlags = 2)
        # all three at once still produce a single warning
        r_all = @test_logs (:warn, warn_msg) spiv(
            y, Y, Z, SPIVwithVAR(); H = H, p = p,
            X = randn(T, 2), intercept = false, xlags = 2)
        @test coef(r_X) == coef(r_clean)
        @test coef(r_all) == coef(r_clean)

        # NaN-padded X is ignored wholesale — it neither errors nor trims.
        r_nanX = @test_logs (:warn, warn_msg) spiv(
            y, Y, Z, SPIVwithVAR(); H = H, p = p, X = lags(vec(y), 4))
        @test coef(r_nanX) == coef(r_clean)
        @test nobs(r_nanX) == nobs(r_clean)

        # Burn-in in the observables themselves is trimmed like the LP variant.
        y_pad = copy(y)
        y_pad[1:2] .= NaN
        r_pad = spiv(y_pad, Y, Z, SPIVwithVAR(); H = H, p = p)
        r_cut = spiv(y[3:end], Y[3:end, :], Z[3:end, :], SPIVwithVAR(); H = H, p = p)
        @test coef(r_pad) == coef(r_cut)
        @test nobs(r_pad) == (T - 2) - p - (H - 1)
    end

    # -------------------------------------------------------------------
    # 5. Inference / IRF blocks are populated and finite; type stability.
    # -------------------------------------------------------------------
    @testset "populated blocks & type stability" begin
        T = 1000
        y, Y, Z = _dgp_var(T, 0.5, 3)
        res = @inferred _runvar(y, Y, Z, 3, 1)
        @test res isa SPIVResult{Float64, SPIVwithVAR}

        wk = weak_iv_test(res)
        @test isfinite(wk.g_min) && wk.g_min ≥ 0
        @test robust_inference(res).method == :AR
        @test all(isfinite, res.irf_outcome.point) && all(isfinite, res.irf_outcome.se)
        @test all(isfinite, res.irf_endogenous.point)

        @test @inferred(_var_forecast_errors(vec(y), Y, Z, 3, 1)) isa
              Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Int, Int}
    end
end
