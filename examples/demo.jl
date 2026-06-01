# examples/demo.jl
#
# Full-API walkthrough of the SP-IV estimator (Lewis & Mertens 2024).
# Text-only; run from the package environment:
#
#     julia --project examples/demo.jl
#
# It drives the whole public surface area on synthetic data with a KNOWN β, so
# every section can be sanity-checked against the truth. Nothing here is part of
# the package — it is a demonstration / exploration script.

using SystemProjectionsIV
using StatsBase: coef, vcov, stderror, confint, nobs, dof_residual, residuals
using Random
using Printf

# ---------------------------------------------------------------------------
# Small printing helpers (local to this script).
# ---------------------------------------------------------------------------

section(title) = (println(); println("="^72); println("  ", title); println("="^72))

function print_coef_table(res; truth = nothing)
    β = coef(res)
    se = stderror(res)
    ci = confint(res)                       # K × 2, 95%
    @printf("  %-6s %10s %10s %9s   %-20s\n", "param", "estimate", "std.err.", "z",
        "95% CI")
    for k in eachindex(β)
        z = β[k] / se[k]
        @printf("  β[%d]   %10.4f %10.4f %9.2f   [%7.4f, %7.4f]\n",
            k, β[k], se[k], z, ci[k, 1], ci[k, 2])
    end
    if truth !== nothing
        @printf("  (true β = %s)\n", string(truth))
    end
    @printf("  T_eff = %d, dof_residual = %d\n", nobs(res), dof_residual(res))
end

function print_weak_iv(res)
    w = weak_iv_test(res)
    verdict = w.is_weak ? "WEAK (g_min < critical value)" : "strong"
    @printf("  g_min = %.4f, critical value = %.4f  ⇒  instruments %s\n",
        w.g_min, w.critical_value, verdict)
end

function print_robust(res)
    r = robust_inference(res)
    @printf("  method = %s, critical value = %.4f, df = %d, accepted points = %d\n",
        r.method, r.critical_value, r.degrees_freedom, size(r.confidence_set, 1))
    for k in 1:size(r.bounds, 1)
        @printf("  β[%d] robust set ∈ [%7.4f, %7.4f]\n", k, r.bounds[k, 1], r.bounds[k, 2])
    end
end

function print_outcome_irf(res)
    irf = res.irf_outcome                   # H × Nz
    H, Nz = size(irf.point)
    println("  outcome IRF (H × Nz = $H × $Nz):  point [lower, upper] (se)")
    for n in 1:Nz
        println("   instrument $n:")
        for h in 1:H
            @printf("    h=%d  %8.4f  [%7.4f, %7.4f]  (se %.4f)\n",
                h - 1, irf.point[h, n], irf.lower[h, n], irf.upper[h, n], irf.se[h, n])
        end
    end
end

function print_endogenous_irf(res)
    irf = res.irf_endogenous                # H × K × Nz
    H, K, Nz = size(irf.point)
    println("  endogenous IRF (H × K × Nz = $H × $K × $Nz):  point (se)")
    for n in 1:Nz, k in 1:K
        println("   instrument $n → endogenous Y[$k]:")
        for h in 1:H
            @printf("    h=%d  %8.4f  (se %.4f)\n", h - 1, irf.point[h, k, n], irf.se[h, k, n])
        end
    end
end

# ---------------------------------------------------------------------------
# 1. Synthetic DGP with a known β and a known IRF decay.
#    y_t = β Y_t + u_t ; Y endogenous (loads on u) ; instrument z = ε exogenous.
#    Θ_y(h) = β Θ_Y(h) at every horizon ⇒ β̂ → β.  (mirrors test/test_spiv.jl)
# ---------------------------------------------------------------------------

function make_dgp(; T, H, β_true, a, ρ, instrument_loading = 1.0, seed = 20240530)
    Random.seed!(seed)
    ε = randn(T)                            # exogenous instrument / shock
    u = 0.3 .* randn(T)                     # structural error
    ν = 0.2 .* randn(T)                     # extra Y noise

    Yv = zeros(T)
    for t in 1:T
        s = 0.0
        for j in 0:(length(a) - 1)
            t - j ≥ 1 && (s += instrument_loading * a[j + 1] * ε[t - j])
        end
        Yv[t] = s + ρ * u[t] + ν[t]         # ρ ≠ 0 ⇒ endogeneity
    end
    yv = β_true .* Yv .+ u
    return yv, reshape(Yv, T, 1), ones(T, 1), reshape(ε, T, 1)
end

const T = 4000
const H = 4
const β_true = 0.7
const a = [1.0, 0.6, 0.3, 0.15]            # IRF of Y to the shock, horizons 0..H-1
const ρ = 0.8

y, Y, X, Z = make_dgp(; T = T, H = H, β_true = β_true, a = a, ρ = ρ)

println("SP-IV full-API walkthrough")
@printf("DGP: T=%d, H=%d, true β=%.2f, IRF decay a=%s, endogeneity ρ=%.2f\n",
    T, H, β_true, string(a), ρ)
println("Inputs: y (T,), Y (T×1), X=intercept (T×1), Z=instrument (T×1)")

# ---------------------------------------------------------------------------
# 2. LP variant (default) — strong-identification estimates + summary table.
# ---------------------------------------------------------------------------

section("2. LP variant — spiv(y, Y, X, Z; H)")
res_lp = spiv(y, Y, X, Z; H = H)
print_coef_table(res_lp; truth = β_true)
println()
println("  Built-in summary table (show):")
show(res_lp)
println()

# ---------------------------------------------------------------------------
# 3. VAR variant — same data, compare β̂ to LP (Phase 5b: should be comparable).
# ---------------------------------------------------------------------------

section("3. VAR variant — spiv(y, Y, X, Z, SPIVwithVAR(); H, p)")
res_var = spiv(y, Y, X, Z, SPIVwithVAR(); H = H, p = 1)
print_coef_table(res_var; truth = β_true)
@printf("\n  LP  β̂ = %.4f\n  VAR β̂ = %.4f   (difference %.4f)\n",
    coef(res_lp)[1], coef(res_var)[1], abs(coef(res_lp)[1] - coef(res_var)[1]))

# ---------------------------------------------------------------------------
# 4. Weak-IV diagnostic — strong design vs a deliberately weak one.
# ---------------------------------------------------------------------------

section("4. Weak-IV diagnostic — weak_iv_test(res)")
println(" Strong-instrument design (instrument_loading = 1.0):")
print_weak_iv(res_lp)

println("\n Weak-instrument design (instrument_loading = 0.02):")
yw, Yw, Xw, Zw = make_dgp(; T = T, H = H, β_true = β_true, a = a, ρ = ρ,
    instrument_loading = 0.02, seed = 11)
res_weak = spiv(yw, Yw, Xw, Zw; H = H)
print_weak_iv(res_weak)

# ---------------------------------------------------------------------------
# 5. Robust inference — AR and KLM confidence sets via grid inversion.
# ---------------------------------------------------------------------------

section("5. Robust inference — robust_inference(res), AR vs KLM")
println(" Anderson–Rubin (weak_iv = :AR):")
res_ar = spiv(y, Y, X, Z; H = H, weak_iv = :AR)
print_robust(res_ar)

println("\n Kleibergen LM (weak_iv = :KLM):")
res_klm = spiv(y, Y, X, Z; H = H, weak_iv = :KLM)
print_robust(res_klm)

# ---------------------------------------------------------------------------
# 6. Impulse responses — outcome (H×Nz) and endogenous (H×K×Nz) with HAC bands.
# ---------------------------------------------------------------------------

section("6. Impulse responses — res.irf_outcome / res.irf_endogenous")
print_outcome_irf(res_lp)
println()
print_endogenous_irf(res_lp)

# ---------------------------------------------------------------------------
# 7. Overidentified, multi-parameter case (K=2, Nz=2) to show output shapes.
# ---------------------------------------------------------------------------

section("7. Overidentified case — K=2, Nz=2 (relevant instruments)")
Random.seed!(7)
let Tg = 1000, Hg = 3, Kg = 2, Nzg = 2
    β1, β2 = 0.5, -0.3
    decay = [1.0, 0.5, 0.25]              # response of each Y to a shock, horizons 0..2
    Zg = randn(Tg, Nzg)                  # exogenous instruments
    u = 0.3 .* randn(Tg)                 # structural error
    L1 = [1.0 0.4]                        # Y[1] loads on both instruments
    L2 = [0.3 1.0]                        # Y[2] loads on both instruments
    Yg = zeros(Tg, Kg)
    for t in 1:Tg
        s1 = 0.0
        s2 = 0.0
        for j in 0:(length(decay) - 1)
            if t - j ≥ 1
                s1 += decay[j + 1] * (L1[1] * Zg[t - j, 1] + L1[2] * Zg[t - j, 2])
                s2 += decay[j + 1] * (L2[1] * Zg[t - j, 1] + L2[2] * Zg[t - j, 2])
            end
        end
        Yg[t, 1] = s1 + 0.8 * u[t] + 0.2 * randn()   # endogeneity via u
        Yg[t, 2] = s2 + 0.8 * u[t] + 0.2 * randn()
    end
    yg = β1 .* Yg[:, 1] .+ β2 .* Yg[:, 2] .+ u
    res_oi = spiv(yg, Yg, ones(Tg, 1), Zg; H = Hg)
    println("  (true β = [$β1, $β2])")
    print_coef_table(res_oi)
    @printf("\n  outcome IRF shape    = %s\n", string(size(res_oi.irf_outcome.point)))
    @printf("  endogenous IRF shape = %s\n", string(size(res_oi.irf_endogenous.point)))
    print_robust(res_oi)
end

section("Done")
println("All sections ran. See coefficients, diagnostics, and IRF tables above.")
