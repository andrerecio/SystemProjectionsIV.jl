using Test
using SystemProjectionsIV

@testset "SystemProjectionsIV.jl" begin
    @testset "smoke" begin
        @test isdefined(Main, :SystemProjectionsIV)
    end

    include("test_types.jl")
    include("test_spiv.jl")
end
