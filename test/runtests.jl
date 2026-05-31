using Test
using Aqua
using SystemProjectionsIV

@testset "SystemProjectionsIV.jl" begin
    @testset "smoke" begin
        @test isdefined(Main, :SystemProjectionsIV)
    end

    @testset "Aqua" begin
        Aqua.test_all(SystemProjectionsIV)
    end

    include("test_types.jl")
    include("test_spiv.jl")
    include("test_inference.jl")
end
