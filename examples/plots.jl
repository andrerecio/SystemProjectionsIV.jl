# examples/plots.jl
#
# Plotting demo for SP-IV impulse responses. Plots.jl is NOT a dependency of the
# package — the package only ships a lightweight RecipesBase recipe (src/recipes.jl).
# This script runs in the separate `examples/` environment, which DOES depend on
# Plots, so it activates the recipe without touching the package's own deps.
#
# Run with:
#     julia --project=examples examples/plots.jl
#
# It writes two PNGs next to this script.

using SystemProjectionsIV
using Plots
using Random

# Known-β DGP with a decaying IRF (same shape as examples/demo.jl section 1).
function make_dgp(; T, H, β_true, a, ρ, seed = 20240530)
    Random.seed!(seed)
    ε = randn(T)
    u = 0.3 .* randn(T)
    ν = 0.2 .* randn(T)
    Yv = zeros(T)
    for t in 1:T
        s = 0.0
        for j in 0:(length(a) - 1)
            t - j ≥ 1 && (s += a[j + 1] * ε[t - j])
        end
        Yv[t] = s + ρ * u[t] + ν[t]
    end
    yv = β_true .* Yv .+ u
    return yv, Yv, ε
end

y, Y, Z = make_dgp(; T = 4000, H = 8, β_true = 0.7, a = [1.0, 0.6, 0.3, 0.15], ρ = 0.8)
res = spiv(y, Y, Z; H = 8)

here = @__DIR__

# Outcome IRF (default). The recipe draws point IRF + shaded band + dotted zero line.
p1 = plot(res)
savefig(p1, joinpath(here, "irf_outcome.png"))

# Endogenous IRF — pass response = :endogenous to the recipe.
p2 = plot(res; response = :endogenous)
savefig(p2, joinpath(here, "irf_endogenous.png"))

println("Wrote:")
println("  ", joinpath(here, "irf_outcome.png"))
println("  ", joinpath(here, "irf_endogenous.png"))
