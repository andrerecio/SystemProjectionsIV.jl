# TODO — SystemProjectionsIV.jl

Phased roadmap to take the package from skeleton to a tagged `v0.1.0`. Phases are sequential: do not start phase N+1 until phase N's **Exit criteria** are met. Within a phase, tick checkboxes as deliverables land.

References throughout this file point at `docs/technical.md` (Lewis & Mertens 2024 math spec) and `docs/technical_julia.md` (design philosophy).

---

## Phase 0 — Package skeleton ✅

Goal: a valid, installable, testable Julia package with the planned dependencies wired up.

- [x] Create `Project.toml` with `name = "SystemProjectionsIV"`, a fresh UUID, `version = "0.0.1"`, `[deps]` entries for `MacroEconometricTools`, `LocalProjections`, `Regress`, and `[sources]` URLs pointing at the gragusa github repos. `[compat] julia = "1.11"` (bumped from 1.10 because the `[sources]` block is a Julia 1.11+ feature; the upstream Project.toml files also use it).
- [x] Create the module file `src/SystemProjectionsIV.jl` that `using`s the three upstream packages. `src/utilities.jl` is intentionally **not** included yet — it copy-adapts utilities from MacroEconometricTools, so including it now would clash with the upstream definitions. The audit in Phase 5 decides what (if anything) to keep.
- [x] Create `test/runtests.jl` with a minimal `using Test, SystemProjectionsIV` smoke test.
- [x] Add `.github/workflows/CI.yml` running `Pkg.test()` on Julia `1.11` and `nightly` (Ubuntu).
- [x] Verify upstream packages resolve and load (`Pkg.instantiate()` + `using SystemProjectionsIV` succeed locally).

**Exit criteria:** `using SystemProjectionsIV` succeeds in a fresh REPL ✅; `Pkg.test()` runs green ✅ (1 pass / 0 fail); CI passes on a no-op PR (pending push).

---

## Phase 1 — Core types ✅

Goal: nail down the data structures `spiv()` will produce, before writing the estimator.

- [x] `abstract type AbstractSPIVSpec` with empty subtypes `SPIVwithLP` and `SPIVwithVAR` for dispatch.
- [x] `struct ForecastErrors{T<:AbstractFloat}` holding `y_perp` (H × T_eff), `Y_perp` (HK × T_eff), `Z_perp` (Nz × T_eff) plus dimensions; constructor validates shapes.
- [x] Auxiliary types `WeakIVDiagnostic{T}` (g_min, critical value, is_weak), `RobustInference{T}` (method, critical value, K×2 bounds, accepted-points matrix, df), `IRFBlock{T, N}` (point/se/lower/upper Arrays parameterised on N — N=2 for outcome, N=3 for endogenous). Strong-IV CI table is *derived* via `confint`, not stored.
- [x] `struct SPIVResult{T<:AbstractFloat, S<:AbstractSPIVSpec}` bundling spec, β, vcov, residuals, weak-IV diagnostic, robust-inference output, both IRF blocks, and dimensions (H, K, Nz, Nx, T_eff).
- [x] Accessor methods extending `StatsBase`: `coef`, `vcov`, `nobs`, `dof_residual`, `residuals`, `stderror`, `confint`. SP-IV-specific accessors: `horizon`, `weak_iv_test`, `robust_inference`, `spec`.
- [x] Constructors enforce concrete fields — no `Union{T, Nothing}`. Stub constructor `SPIVResult(spec, β, vcov, residuals; H, K, Nz, Nx, T_eff)` fills inference / IRF fields with NaN content for Phase 2 to use before Phases 3–4 populate them.
- [x] Tests: 73 assertions across spec dispatch, dimension validation, NaN stubs, accessor round-trips, and `@inferred` type-stability checks.

**Exit criteria:** all accessors round-trip ✅; type stability verified for every accessor via `@inferred` ✅; `Pkg.test()` green (73 pass, 0 fail) ✅.

---

## Phase 2 — SP-IV point estimator (LP variant) ✅

Goal: closed-form β̂ and strong-IV sandwich variance for `SPIVwithLP`.

Implemented in `src/spiv.jl`. Inputs use the **time-in-rows** (T×·) convention.

- [x] `spiv(y, Y, X, Z; H, weak_iv = :AR, ξ = 0.10, grid_length = 30)` returns an `SPIVResult{Float64, SPIVwithLP}`. (`weak_iv`/`ξ`/`grid_length` are accepted to fix the public signature but unused until Phase 3.)
- [x] Input validation: dimension checks; identification condition `H · Nz ≥ K`; sufficient degrees of freedom (`T_eff − Nx − K > 0`).
- [x] Build stacked future paths `y_H` (`H × T_eff`) and `Y_H` (`HK × T_eff`); `T_eff = T − (H−1)` — see `docs/technical.md` §3.2.
- [x] Residualise `y_H`, `Y_H`, `Z` on `X` via the FWL projection `M_X` (eq. A.1). **Done with direct QR-based linear algebra** (`V − (V Q) Qᵀ`), not `LocalProjections.jl`: the latter is DataFrame/formula-based and exports no matrix residualiser, which would clash with the matrix-valued `spiv` signature. Revisit in the Phase 5 utilities audit.
- [x] Form `P_Z⊥` implicitly and the Gram matrices `Y⊥ P_Z⊥ Y⊥'`, `y⊥ P_Z⊥ Y⊥'` via `(·Z⊥')(Z⊥Z⊥')⁻¹(Z⊥·')` (the T_eff×T_eff projection is never materialised).
- [x] Restriction matrix `R = I_K ⊗ vec(I_H)`; solve `β̂ = [R'(YPY ⊗ I_H) R]⁻¹ R' vec(yPY)` — this is **eq. (9)** (TODO previously cited eq. 13; eq. 13 is the equivalent IRF-OLS form).
- [x] Sandwich variance per eqs. (18)–(19): `Var(β̂) = T_eff⁻¹ A⁻¹ [R'(G⊗Σ̂)R] A⁻¹`, `G = YPY/T_eff`. Strong-IV CI table derived via `confint`.
- [x] Tests (`test/test_spiv.jl`):
  - On a fixed synthetic DGP with known β, the point estimate recovers β within Monte Carlo tolerance.
  - The closed-form β̂ equals the stacked-IRF OLS regression `(Θ̂_Y' Θ̂_Y)⁻¹ Θ̂_Y' Θ̂_y` to machine precision (paper's "key insight" equivalence).
  - Strong-IV SEs match an independent hand-computed sandwich on a tiny problem.
- [x] Added `Random` + `Statistics` to `Project.toml` `[extras]`/`[targets].test` — the synthetic-DGP tests need them and `Pkg.test()` runs in a sandbox that only exposes declared test deps.

**Exit criteria:** all three tests pass ✅; `Pkg.test()` green (92 pass, 0 fail) ✅. `summary`/`show` deferred to Phase 5 (the blocks it would print are still NaN stubs); Phase-2 coherence is verified through the `coef`/`vcov`/`stderror`/`confint` accessors instead.

---

## Phase 3 — Weak-IV inference

Goal: weak-IV diagnostic plus robust AR and KLM confidence sets.

- [ ] First-stage residual covariance `Ŵ_2`; weak-IV statistic `g_min = N_z⁻¹ · mineval{Φ̂⁻¹/² (Y⊥ P_Z⊥ Y⊥') Φ̂⁻¹/²}` where `Φ̂ = R'(Ŵ_2 ⊗ I_H) R` (Proposition 7).
- [ ] Imhof-bounded critical value matching the first three cumulants of the bias-tolerance reference distribution; expose `ξ` as a kwarg (default 0.10).
- [ ] Anderson–Rubin statistic (eq. 20) with `χ²_{H·N_z}` critical value.
- [ ] Kleibergen LM statistic (eq. 21) with `χ²_K` critical value.
- [ ] Grid-search confidence sets over a K-dimensional box around β̂ (default `grid_length = 30`); parallelise with `Threads.@threads`. Extract per-parameter min/max bounds.
- [ ] Record AR vs KLM selection (kwarg `weak_iv`) in the result; default to `:AR` since it dominates for `K > 2`.
- [ ] Tests:
  - AR confidence set has correct empirical coverage (≈ nominal) over a Monte Carlo of the weak-instrument DGP from the paper.
  - AR and KLM statistics agree with their textbook formulas on a hand-computed small case (K = Nz = 1, H = 1).
  - `g_min` reduces to the first-stage F statistic in the just-identified scalar case.

**Exit criteria:** all tests pass; `summary(result)` reports both the weak-IV diagnostic and the robust confidence set.

---

## Phase 4 — IRFs and HAC standard errors

Goal: outcome and endogenous IRFs with HAC bands populated in `SPIVResult`.

- [ ] Compute outcome IRF (`H × Nz`) and endogenous IRF (`H · K × Nz`) from the residualised projections.
- [ ] Newey–West HAC SEs per horizon `h` with lag truncation `h - 1`. Prefer `Regress.jl`'s HAC implementation if it exposes one; otherwise implement directly.
- [ ] Populate `result.irf.outcome` and `result.irf.endogenous` with point, SE, 95% lower / upper bounds.
- [ ] Tests:
  - HAC SE reduces to OLS SE when residuals are iid (closed-form check on a designed input).
  - IRF point estimates equal those implied by the stacked-IRF representation from Phase 2.
  - Bands have positive width and bracket the point estimate.

**Exit criteria:** all tests pass; the README quick-start runs end-to-end and prints a non-trivial IRF table.

---

## Phase 5 — VAR variant, polish, v0.1.0

Goal: ship.

- [ ] `SPIVwithVAR` variant: compute forecast errors `X̂_t^⊥(h) = Σ_{j=0}^h Â^{h-j} ê_{t+j}` using `MacroEconometricTools.jl`'s VAR estimator (appendix A of `docs/technical.md`).
- [ ] Dispatch: `spiv(y, Y, X, Z, ::SPIVwithVAR; ...)` returns an `SPIVResult` with the same shape as the LP variant.
- [ ] `Base.show(io, ::SPIVResult)` and a `summary(::SPIVResult)` method printing a clean table.
- [ ] Plotting recipe for IRFs (Plots.jl recipe via `RecipesBase`, to avoid pulling Plots into the main dependency tree).
- [ ] Audit `src/utilities.jl`: remove any function now supplied by `MacroEconometricTools.jl`; keep what isn't.
- [ ] Doctests in the README quick-start and the `spiv` docstring.
- [ ] Tag `v0.1.0`.

**Exit criteria:** README quick-start runs as a doctest; both `SPIVwithLP` and `SPIVwithVAR` produce comparable estimates on the same data (within sampling tolerance); CI green; version bumped and tagged.

---

## Out of scope (for v0.1.0)

The following appear in `docs/technical_julia.md` because that document describes a broader project. They are **not** targeted here:

- Bayesian VAR estimation, Minnesota / Normal–Wishart priors.
- Sign restrictions, Cholesky-only identification, set-identified IRFs.
- Generic constraint system (zero, fixed, block-exogeneity constraints on VAR coefficients).
- Wild / standard / block bootstrap inference (SP-IV uses analytical sandwich + AR/KLM, not bootstrap).

Revisit only if a concrete user need emerges.
