# Pretty-printing for SPIVResult: a compact one-liner and a REPL summary table.
# Layout modelled on the gragusa result show methods (docs/ext/gragusa/LocalProjections.jl);
# hand-formatted with Printf to avoid a PrettyTables dependency.

_variant_name(::SPIVwithLP) = "local projections (LP)"
_variant_name(s::SPIVwithVAR) = "vector autoregression (VAR($(s.p)))"

# Horizons as printed: a compact range when they are evenly spaced.
function _horizons_str(hs::Vector{Int})
    length(hs) == 1 && return string(hs[1])
    step = hs[2] - hs[1]
    all(==(step), diff(hs)) || return join(hs, ",")
    return step == 1 ? "$(hs[1]):$(hs[end])" : "$(hs[1]):$(step):$(hs[end])"
end
_pct(level) = @sprintf("%g%%", 100 * level)

function Base.show(io::IO, r::SPIVResult)
    print(
        io,
        "SPIVResult(",
        _variant_name(r.spec),
        ", h=",
        _horizons_str(r.horizons),
        ", K=",
        r.K,
        ", Nz=",
        r.Nz,
        ", T_eff=",
        r.T_eff,
        ")"
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
        "  horizons h=",
        _horizons_str(r.horizons),
        " (H=",
        r.H,
        "), K=",
        r.K,
        " endogenous, Nz=",
        r.Nz,
        " instruments, Nx=",
        r.Nx,
        " controls, T_eff=",
        r.T_eff
    )
    println(io)
    println(io, "Coefficients (strong-identification sandwich):")
    @printf(io,
        "  %-8s %10s %10s %9s   %s\n",
        "param",
        "estimate",
        "std.err.",
        "z",
        _pct(r.level) * " CI")
    for k in 1:r.K
        zk = se[k] > 0 ? β[k] / se[k] : NaN
        @printf(io,
            "  %-8s %10.4f %10.4f %9.2f   [%.4f, %.4f]\n",
            "β[$k]",
            β[k],
            se[k],
            zk,
            ci[k, 1],
            ci[k, 2],)
    end
    println(io)
    verdict = wk.is_weak ? "weak" : "strong"
    @printf(io,
        "Weak-IV (g): %.3f   critical value: %.3f   ⇒ instruments %s\n",
        wk.g_min,
        wk.critical_value,
        verdict,)
    if rb.method === :none || isempty(rb.confidence_set)
        @printf(io,
            "Robust (%s): no accepted grid point (empty set, or narrower than the grid step)\n",
            String(rb.method))
    else
        print(io, "Robust (", String(rb.method), ", ", _pct(r.level), ", df=",
            rb.degrees_freedom, "):")
        for k in 1:r.K
            @printf(io, " β[%d]∈[%.4f, %.4f]", k, rb.bounds[k, 1], rb.bounds[k, 2])
        end
        println(io)
        any(isinf, rb.bounds) &&
            println(io,
                "  note: ±Inf — the set reaches the edge of the search grid " *
                "(unbounded or wider than the grid); widen `grid` to explore")
        rb.disconnected &&
            println(io,
                "  note: the set is disconnected; bounds are its hull " *
                "(pieces in robust_inference(r).confidence_set)")
    end
    print(
        io,
        "IRFs: outcome ",
        size(r.irf_outcome.point),
        ", endogenous ",
        size(r.irf_endogenous.point),
        " (point/se/lower/upper in result.irf_*)"
    )
end

"""
    summary([io::IO,] r::SPIVResult)

Print the full estimation summary — coefficient table, weak-IV verdict, robust
confidence set, and IRF shapes — like R's `summary()`. Identical to the REPL
display of `r`; returns `nothing`.
"""
function Base.summary(io::IO, r::SPIVResult)
    show(io, MIME("text/plain"), r)
    println(io)
    return nothing
end
Base.summary(r::SPIVResult) = summary(stdout, r)
