# Pretty-printing for SPIVResult: a compact one-liner and a REPL summary table.
# Layout modelled on the gragusa result show methods (docs/ext/gragusa/LocalProjections.jl);
# hand-formatted with Printf to avoid a PrettyTables dependency.

_variant_name(::SPIVwithLP) = "local projections (LP)"
_variant_name(::SPIVwithVAR) = "vector autoregression (VAR)"

function Base.show(io::IO, r::SPIVResult)
    print(
        io,
        "SPIVResult(",
        _variant_name(r.spec),
        ", H=",
        r.H,
        ", K=",
        r.K,
        ", Nz=",
        r.Nz,
        ", T_eff=",
        r.T_eff,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", r::SPIVResult)
    β = coef(r)
    se = stderror(r)
    ci = confint(r)
    wk = weak_iv_test(r)
    rb = robust_inference(r)

    println(io, "System Projections IV — ", _variant_name(r.spec))
    println(
        io,
        "  horizons H=",
        r.H,
        ", K=",
        r.K,
        " endogenous, Nz=",
        r.Nz,
        " instruments, Nx=",
        r.Nx,
        " controls, T_eff=",
        r.T_eff,
    )
    println(io)
    println(io, "Coefficients (strong-identification sandwich):")
    @printf(
        io,
        "  %-8s %10s %10s %9s   %s\n",
        "param",
        "estimate",
        "std.err.",
        "z",
        "95% CI"
    )
    for k in 1:r.K
        zk = se[k] > 0 ? β[k] / se[k] : NaN
        @printf(
            io,
            "  β[%-3d] %12.4f %10.4f %9.2f   [%.4f, %.4f]\n",
            k,
            β[k],
            se[k],
            zk,
            ci[k, 1],
            ci[k, 2],
        )
    end
    println(io)
    verdict = wk.is_weak ? "weak" : "strong"
    @printf(
        io,
        "Weak-IV (g): %.3f   critical value: %.3f   ⇒ instruments %s\n",
        wk.g_min,
        wk.critical_value,
        verdict,
    )
    if rb.method === :none || isempty(rb.confidence_set)
        @printf(io, "Robust (%s): empty confidence set\n", String(rb.method))
    else
        print(io, "Robust (", String(rb.method), ", df=", rb.degrees_freedom, "):")
        for k in 1:r.K
            @printf(io, " β[%d]∈[%.4f, %.4f]", k, rb.bounds[k, 1], rb.bounds[k, 2])
        end
        println(io)
    end
    print(
        io,
        "IRFs: outcome ",
        size(r.irf_outcome.point),
        ", endogenous ",
        size(r.irf_endogenous.point),
        " (point/se/lower/upper in result.irf_*)",
    )
end
