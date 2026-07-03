# Plots.jl recipe for SP-IV impulse responses (lightweight: RecipesBase only, no Plots
# hard dependency). Modelled on docs/ext/gragusa/MacroEconometricsTools.jl/src/plots_recipes.jl.
#
# Usage (with Plots loaded):  plot(result)                      # outcome IRF
#                             plot(result; response = :endogenous)
# Each panel shows the point IRF, a shaded band (lower–upper), and a dotted zero line.

@recipe function f(r::SPIVResult; response = :outcome)
    blk = irf(r; response = response)   # validates `response`, selects the IRF block
    h = 0:(r.H - 1)

    if response === :outcome
        layout := (1, r.Nz)
        legend := false
        for n in 1:r.Nz
            @series begin
                subplot := n
                title := "outcome ← shock $n"
                seriestype := :hline
                linestyle := :dot
                linecolor := :gray
                [0.0]
            end
            @series begin
                subplot := n
                ribbon := (
                    blk.point[:, n] .- blk.lower[:, n], blk.upper[:, n] .- blk.point[:, n])
                fillalpha --> 0.2
                xguide --> "horizon"
                h, blk.point[:, n]
            end
        end
    else
        layout := (r.K, r.Nz)
        legend := false
        idx = 0
        for k in 1:r.K, n in 1:r.Nz

            idx += 1
            @series begin
                subplot := idx
                seriestype := :hline
                linestyle := :dot
                linecolor := :gray
                [0.0]
            end
            @series begin
                subplot := idx
                title := "Y$k ← shock $n"
                ribbon := (
                    blk.point[:, k, n] .- blk.lower[:, k, n],
                    blk.upper[:, k, n] .- blk.point[:, k, n]
                )
                fillalpha --> 0.2
                xguide --> "horizon"
                h, blk.point[:, k, n]
            end
        end
    end
end
