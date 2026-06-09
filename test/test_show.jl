# Phase 5a tests: automatic-bandwidth HAC option, Base.show, IRF plot recipe,
# and the README quick-start.

using Test
using SystemProjectionsIV
using StatsBase: coef, confint
using Random
using RecipesBase

# Allow apply_recipe to run without a Plots backend (standard recipe-testing shim).
RecipesBase.is_key_supported(::Symbol) = true

# Strong-IV synthetic design with a known β.
function _dgp_show(T, H, β, seed)
    Random.seed!(seed)
    ε = randn(T)
    u = 0.3 .* randn(T)
    Y = zeros(T)
    for t in 1:T, j in 0:(H - 1)

        t - j ≥ 1 && (Y[t] += 0.8^j * ε[t - j])
    end
    Y .+= 0.7 .* u .+ 0.2 .* randn(T)
    y = β .* Y .+ u
    return y, reshape(Y, T, 1), reshape(ε, T, 1)
end

@testset "show / recipe / auto-HAC" begin

    # -------------------------------------------------------------------
    # 1. Automatic-bandwidth HAC: :fixed reproduces the Phase-4 result;
    #    :neweywest / :andrews run, give finite positive SEs, and differ
    #    from :fixed at longer horizons.
    # -------------------------------------------------------------------
    @testset "auto-bandwidth HAC" begin
        y, Y, Z = _dgp_show(400, 4, 0.7, 20240530)
        rf = spiv(y, Y, Z; H = 4)                  # default :fixed
        rf2 = spiv(y, Y, Z; H = 4, hac = :fixed)
        rnw = spiv(y, Y, Z; H = 4, hac = :neweywest)
        ra = spiv(y, Y, Z; H = 4, hac = :andrews)

        @test rf.irf_outcome.se == rf2.irf_outcome.se            # default == :fixed
        for r in (rnw, ra)
            @test all(isfinite, r.irf_outcome.se) && all(r.irf_outcome.se .> 0)
            @test all(isfinite, r.irf_endogenous.se) && all(r.irf_endogenous.se .> 0)
            @test all(r.irf_outcome.lower .< r.irf_outcome.point .< r.irf_outcome.upper)
        end
        # Automatic selection departs from the fixed lag = horizon rule somewhere.
        @test rnw.irf_outcome.se != rf.irf_outcome.se

        @test_throws ArgumentError spiv(y, Y, Z; H = 4, hac = :bogus)
    end

    # -------------------------------------------------------------------
    # 2. Base.show: compact and MIME"text/plain" forms.
    # -------------------------------------------------------------------
    @testset "show" begin
        y, Y, Z = _dgp_show(400, 4, 0.7, 1)
        res = spiv(y, Y, Z; H = 4)

        compact = sprint(show, res)
        @test occursin("SPIVResult", compact)
        @test occursin("H=4", compact)

        pretty = sprint((io, x) -> show(io, MIME("text/plain"), x), res)
        @test occursin("System Projections IV", pretty)
        @test occursin("local projections", pretty)
        @test occursin("Weak-IV", pretty)
        @test occursin("β[1", pretty)
        @test occursin("Robust (AR", pretty)
    end

    # -------------------------------------------------------------------
    # 3. Plot recipe yields series for both responses (no Plots backend).
    # -------------------------------------------------------------------
    @testset "plot recipe" begin
        y, Y, Z = _dgp_show(300, 3, 0.5, 2)
        res = spiv(y, Y, Z; H = 3)

        so = RecipesBase.apply_recipe(Dict{Symbol, Any}(), res)
        se = RecipesBase.apply_recipe(Dict{Symbol, Any}(:response => :endogenous), res)
        @test length(so) ≥ 1
        @test length(se) ≥ 1
        # the point-IRF series carries the horizon axis 0:H-1
        @test any(s -> s.args[1] == 0:2, so)
        @test_throws ArgumentError RecipesBase.apply_recipe(
            Dict{Symbol, Any}(:response => :bogus),
            res
        )
    end

    # -------------------------------------------------------------------
    # 4. README quick-start runs end-to-end and recovers β.
    # -------------------------------------------------------------------
    @testset "README quick-start" begin
        Random.seed!(1234)
        T, H, β = 400, 8, 0.5
        ε = randn(T)
        u = 0.3 .* randn(T)
        Y = zeros(T)
        for t in 1:T, j in 0:3

            t - j ≥ 1 && (Y[t] += 0.8^j * ε[t - j])
        end
        Y .+= 0.7 .* u .+ 0.2 .* randn(T)
        y = β .* Y .+ u

        result = spiv(y, Y, ε; H = H, weak_iv = :AR)

        @test isfinite(coef(result)[1])
        @test coef(result)[1] ≈ β atol = 0.1
        @test size(confint(result)) == (1, 2)
        @test isfinite(weak_iv_test(result).g_min)
        @test robust_inference(result).method == :AR
        @test size(result.irf_outcome.point) == (H, 1)
    end
end
