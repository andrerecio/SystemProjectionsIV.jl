using Test
using Aqua
using SystemProjectionsIV

@testset "SystemProjectionsIV.jl" begin
    @testset "smoke" begin
        @test isdefined(Main, :SystemProjectionsIV)
    end

    @testset "Aqua" begin
        # Skip compat check for gragusa git deps tracked on `main` via [sources];
        # pinning compat would break CI whenever upstream bumps their version.
        Aqua.test_all(
            SystemProjectionsIV;
            deps_compat = (ignore = [:LocalProjections, :MacroEconometricTools, :Regress],),
        )
    end

    include("test_types.jl")
    include("test_spiv.jl")
end
