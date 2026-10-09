# Public API surface: StatsBase re-exports, coeftable / coefnames, horizons, and the
# single `level` shared by confint, coeftable, the robust set, and the IRF bands.

using Test
using SystemProjectionsIV
using Random
using LinearAlgebra: diag
using StatsFuns: normccdf, chisqinvcdf

# A fresh module with only `using SystemProjectionsIV`: the StatsBase accessors must
# be reachable without `using StatsBase`.
module ReexportCheck
using SystemProjectionsIV
using Random: Xoshiro, randn

function run()
    rng = Xoshiro(3)
    T = 200
    z = randn(rng, T)
    Y = z .+ randn(rng, T)
    y = 0.5 .* Y .+ randn(rng, T)
    r = spiv(y, Y, z; H = 3)
    return (
        coef = coef(r), vcov = vcov(r), stderror = stderror(r), confint = confint(r),
        coeftable = coeftable(r), coefnames = coefnames(r), nobs = nobs(r),
        dof_residual = dof_residual(r), residuals = residuals(r), r = r)
end
end

@testset "public API" begin
    @testset "StatsBase re-exports in a fresh module" begin
        out = ReexportCheck.run()
        r = out.r
        @test out.coef == r.β
        @test out.vcov == r.vcov
        @test out.stderror ≈ sqrt.(diag(r.vcov))
        @test size(out.confint) == (1, 2)
        @test out.coefnames == ["β[1]"]
        @test out.nobs == r.T_eff
        @test out.dof_residual == r.T_eff - r.Nx - r.K
        @test out.residuals === r.residuals
        for f in (:coef, :vcov, :stderror, :confint, :coeftable, :coefnames, :nobs,
            :dof_residual, :residuals, :horizons)
            @test isdefined(ReexportCheck, f)
        end
    end

    Random.seed!(404)
    T = 300
    Z = randn(T, 2)
    Y = Z * [1.0 0.2; 0.1 0.8] .+ randn(T, 2)
    y = Y * [0.5, -0.3] .+ randn(T)

    @testset "coeftable / coefnames" begin
        r = spiv(y, Y, Z; H = 3, grid_length = 21)
        @test coefnames(r) == ["β[1]", "β[2]"]
        ct = coeftable(r)
        @test ct.rownms == coefnames(r)
        @test ct.colnms ==
              ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"]
        @test ct.cols[1] == coef(r)
        @test ct.cols[2] == stderror(r)
        @test ct.cols[3] ≈ coef(r) ./ stderror(r)
        @test ct.cols[4] ≈ 2 .* normccdf.(abs.(coef(r) ./ stderror(r)))
        @test hcat(ct.cols[5], ct.cols[6]) == confint(r)
        @test ct.pvalcol == 4 && ct.teststatcol == 3
        s = sprint(show, ct)
        @test occursin("β[2]", s) && occursin("Lower 95%", s)
        @test coeftable(r; level = 0.9).colnms[5] == "Lower 90%"
    end

    @testset "horizons" begin
        r = spiv(y, Y, Z; H = 4, grid_length = 21)
        @test horizons(r) == [0, 1, 2, 3]
        @test horizons(r) isa Vector{Int}
        @test horizons(spiv(y, Y, Z; H = 2:3:8, grid_length = 21)) == [2, 5, 8]
    end

    @testset "level consistency" begin
        r95 = spiv(y, Y, Z; H = 3, grid_length = 41)
        r90 = spiv(y, Y, Z; H = 3, level = 0.9, grid_length = 41)
        @test r95.level == 0.95
        @test r90.level == 0.9
        @test coef(r90) == coef(r95)
        @test vcov(r90) == vcov(r95)

        # confint / coeftable default to the stored level
        @test confint(r90) == confint(r90; level = 0.9)
        @test confint(r90) ≈ confint(r95; level = 0.9)
        z90 = 1.6448536269514722
        @test confint(r90) ≈ hcat(coef(r90) .- z90 .* stderror(r90),
            coef(r90) .+ z90 .* stderror(r90))
        @test coeftable(r90).colnms[5] == "Lower 90%"

        # robust set at the same level: χ² critical value and a nested set
        rb90, rb95 = robust_inference(r90), robust_inference(r95)
        @test rb90.critical_value ≈ chisqinvcdf(3 * 2, 0.9)
        @test size(rb90.confidence_set, 1) < size(rb95.confidence_set, 1)

        # IRF bands at the same level (identical SEs)
        @test r90.irf_outcome.se == r95.irf_outcome.se
        @test r90.irf_outcome.upper ≈ r90.irf_outcome.point .+ z90 .* r90.irf_outcome.se

        # weak-IV test is run at size 1 − level (only the critical value moves)
        @test weak_iv_test(r90).g_min == weak_iv_test(r95).g_min
        @test weak_iv_test(r90).critical_value < weak_iv_test(r95).critical_value

        # show prints the stored level
        p = sprint((io, x) -> show(io, MIME("text/plain"), x), r90)
        @test occursin("90% CI", p)
        @test occursin("Robust (AR, 90%", p)
    end
end
