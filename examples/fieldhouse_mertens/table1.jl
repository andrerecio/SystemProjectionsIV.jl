# examples/fieldhouse_mertens/table1.jl
#
# Replicates rows 1–5 of Table 1 in Fieldhouse & Mertens, "The Returns to Government
# R&D: Evidence from U.S. Appropriations Shocks": SP-IV estimates of the elasticity of
# TFP with respect to government R&D capital, with 95% KLM confidence intervals.
# Run from the package environment:
#
#     julia --project examples/fieldhouse_mertens/table1.jl
#
# The specification follows `run_SPIV_total` in
# docs/ext/fieldhouse_mertens/Make_Table_1.m; the data are the columns of their
# DATAQ dataset that these rows use (see README.md in this folder).

using SystemProjectionsIV
using Printf

# ---------------------------------------------------------------------------
# Data: a numeric CSV with a date column; empty cells are stored as NaN.
# ---------------------------------------------------------------------------

function read_data(path)
    lines = readlines(path)
    names = split(lines[1], ',')
    vals = permutedims(hcat((parse.(Float64, split(l, ',')[2:end])
    for l in lines[2:end])...))
    return Dict(String(n) => vals[:, j] for (j, n) in enumerate(names[2:end]))
end

data = read_data(joinpath(@__DIR__, "table1_data.csv"))   # 1947Q1–2021Q4

# Outcomes are 4-quarter averages ending at t+h, (x[t+h-3] + … + x[t+h]) / 4.
ma4(x) = [t ≥ 4 ? (x[t - 3] + x[t - 2] + x[t - 1] + x[t]) / 4 : NaN for t in eachindex(x)]

# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

const HORIZONS = 3:4:59    # every 4th quarter from h = 3
const P = 4                # lags of the controls

function controls(cum)
    [cum, "ln_util", "ln_tfp_util", "ln_rK_PRD_fernald",
        "R_EW_COMB_TECH", "ln_rK_GRD", "Military_news_scaled"]
end

# The KLM set can be disconnected. F–M report the connected piece around their CUE
# estimate; here we take the piece around β̂, which is the same piece for rows 1–5.
# Locate it on their 201-point grid over `box`, then re-run between the rejected grid
# points on either side, where `spiv` refines the bounds by bisection. A piece that
# reaches the edge of `box` gets an infinite bound (F–M report the edge itself).
function estimate(tfp, z, cum, contemporaneous, box)
    y = ma4(data[tfp])
    Y = ma4(data["ln_rK_GRD"])
    X = hcat((data[c] for c in contemporaneous)...,
        lags(hcat((data[c] for c in [tfp; controls(cum)])...), P))
    fit(grid) = spiv(y, Y, data[z]; H = HORIZONS, X = X, robust = :KLM, grid = grid)

    r = fit(box)
    g = collect(range(box...; length = 201))
    accepted = falses(201)
    for b in robust_inference(r).confidence_set
        accepted[argmin(abs.(g .- b))] = true
    end
    lo = hi = argmin(abs.(g .- coef(r)[1]))
    while lo > 1 && accepted[lo - 1]
        lo -= 1
    end
    while hi < 201 && accepted[hi + 1]
        hi += 1
    end
    ci = robust_inference(fit((g[max(lo - 1, 1)], g[min(hi + 1, 201)]))).bounds
    return coef(r)[1], ci[1, 1], ci[1, 2]
end

# ---------------------------------------------------------------------------
# Table 1, rows 1–5 (K = 1). Columns: TFP with moderate, low, and high
# infrastructure adjustment; F–M's published values in brackets.
# ---------------------------------------------------------------------------

rows = [
    (1, "Exo ND", "z_NDEF", "ln_cum_rALL_NDEF", String[], (-1, 2),
        [(0.11, 0.09, 0.15), (0.11, 0.09, 0.15), (0.10, 0.08, 0.13)]),
    (2, "Exo ND, No NASA", "z_NDEF_exspace", "ln_cum_rALL_NDEF", ["z_SPACE"], (-1, 2),
        [(0.11, 0.08, 0.20), (0.12, 0.08, 0.21), (0.10, 0.07, 0.19)]),
    (3, "All ND", "z_allNDEF", "ln_cum_rALL_NDEF", String[], (-2, 2),
        [(0.10, 0.09, 0.14), (0.11, 0.09, 0.15), (0.09, 0.07, 0.13)]),
    (4, "Exo D", "z_NSEC", "ln_cum_rALL_NSEC", String[], (-2, 2),
        [(-0.13, -1.20, 0.04), nothing, nothing]),
    (5, "All D", "z_allNSEC", "ln_cum_rALL_NSEC", String[], (-2, 2),
        [(-0.11, -1.11, 0.05), nothing, nothing])
]
tfps = ["ln_tfp_util_adj2", "ln_tfp_util_adj1", "ln_tfp_util_adj3"]

cell(b, lo, hi) = @sprintf("%5.2f (%5.2f, %5.2f)", b, lo, hi)

println("Table 1, rows 1–5: SP-IV elasticities, 95% KLM intervals [published]\n")
@printf("%-3s %-16s  %-21s  %-21s  %-21s\n", "", "Instrument", "Moderate", "Low", "High")
for (row, label, z, cum, contemporaneous, box, published) in rows
    @printf("[%d] %-16s", row, label)
    for (tfp, pub) in zip(tfps, published)
        pub === nothing && continue
        @printf("  %s", cell(estimate(tfp, z, cum, contemporaneous, box)...))
    end
    @printf("\n    %-16s", "")
    for pub in published
        pub === nothing || @printf(" [%s]", cell(pub...))
    end
    println()
end
