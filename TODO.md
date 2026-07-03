# TODO — SystemProjectionsIV.jl

Phased roadmap to take the package from skeleton to a tagged `v0.1.0`. Phases are sequential: do not start phase N+1 until phase N's **Exit criteria** are met. Within a phase, tick checkboxes as deliverables land.

References throughout this file point at `docs/technical.md` (Lewis & Mertens 2024 math spec) and `docs/technical_julia.md` (design philosophy).

---

## Dependency policy change (2026-05-31)

Dropped the three gragusa git dependencies (`MacroEconometricTools`, `LocalProjections`, `Regress`) and the `[sources]` block. Their full sources are now vendored as reference material under `docs/ext/gragusa/`; we reimplement only the functions SP-IV needs directly in `src/`, taking inspiration from gragusa's OLS/IV style — **with a source citation on any adapted code** (see `CLAUDE.md` → Dependencies). HAC standard errors will come from the registered `CovarianceMatrices.jl`, added in Phase 4. Each estimator is reimplemented in the phase that first needs it, not up front.

Surgery step (today):

- [x] Remove the three git deps and `[sources]` from `Project.toml`; runtime deps are now `LinearAlgebra`, `StatsBase`, `StatsFuns`.
- [x] Drop the three `using` lines from `src/SystemProjectionsIV.jl`.
- [x] Trim `src/utilities.jl` to the working lag/matrix/companion utilities + a gragusa citation header; delete the dead `VARModel`-dependent functions (no upstream type to dispatch on). File stays un-`include`d until Phase 5.
- [x] Simplify the Aqua call to `Aqua.test_all(SystemProjectionsIV)` (no more git deps to ignore).
- [x] Update `CLAUDE.md` (Dependencies, layout, Tooling & CI) and this file.

**Exit criteria:** `using SystemProjectionsIV` loads with no git fetches; `Pkg.test()` green (existing 92 tests + Aqua, including stale-deps/compat checks); no reference to the three git deps remains outside `docs/ext/` and intended citation comments.

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

## Phase 3 — Weak-IV inference ✅

Goal: weak-IV diagnostic plus robust AR and KLM confidence sets.

Implemented in `src/inference.jl`; wired into `spiv()` (populates `weak_iv` and `robust`).

- [x] First-stage residual covariance `Ŵ_2`; weak-IV statistic `g_min = N_z⁻¹ · mineval{Φ̂⁻¹/² (Y⊥ P_Z⊥ Y⊥') Φ̂⁻¹/²}` where `Φ̂ = R'(Ŵ_2 ⊗ I_H) R` (Proposition 7). The K×K bracket is implemented as the concentration-matrix form `R'(Y⊥ P_Z⊥ Y⊥' ⊗ I_H) R` (Definition 1, §6.3) — the paper's notation is dimensionally abbreviated; see the source comment in `_weak_iv_diagnostic`.
- [x] Critical value from the Theorem 1 cumulant upper bounds (κ₁ = H·N_z(1+1/ξ), κ₂, κ₃) matched to a chi-square reference (Patnaik three-cumulant approximation, conservative per §6.7). `ξ` is a kwarg (default 0.10); significance `α` a kwarg (default 0.05). NOTE: this is the moment-matched bound, **not** the full Nagar `B(W)` / Imhof approximating-distribution search — the in-repo docs defer the `h` expansion to an Online Appendix (deferred; see Phase notes).
- [x] Anderson–Rubin statistic (eq. 20) with `χ²_{H·N_z}` critical value; `d_AR = N_z + N_x`.
- [x] Kleibergen LM statistic (eq. 21) with `χ²_K` critical value; `d_K = N_z + N_x`.
- [x] Grid-search confidence sets over a K-dimensional box centred on β̂ (`β̂ ± grid_scale·se`, `grid_scale = 10`, `grid_length = 30` points per parameter); parallelised with `Threads.@threads`. Per-parameter min/max bounds extracted.
- [x] Record AR vs KLM selection (kwarg `weak_iv`, default `:AR`) in the result.
- [x] Tests (`test/test_inference.jl`):
  - AR confidence set has ≈ nominal coverage over a Monte Carlo of the IV DGP.
  - AR and KLM statistics match an explicit-projection computation of eqs. (20)/(21) (overidentified-in-H case); `g_min` reduces to the first-stage F in the just-identified scalar case (K = Nz = H = 1).
  - Type stability of the kernels via `@inferred`.

**Exit criteria:** all tests pass ✅ (`Pkg.test()` green, 127 pass / 0 fail). `summary(result)` printing is deferred to Phase 5 (the diagnostic and confidence set are exposed via the `weak_iv_test` / `robust_inference` accessors).

---

## Phase 4 — IRFs and HAC standard errors ✅

Goal: outcome and endogenous IRFs with HAC bands populated in `SPIVResult`.

Implemented in `src/irf.jl`; wired into `spiv()` (populates `irf_outcome` / `irf_endogenous`).

- [x] Compute outcome IRF (`H × Nz`) and endogenous IRF (`H × K × Nz`) from the residualised projections via eq. (11): `Θ̂ = (·⊥ Z⊥' / T)(Z⊥Z⊥'/T)^{-1/2}` (docs/technical.md §3.3).
- [x] Newey–West HAC SEs per horizon `h` with lag truncation `h - 1`, using the registered `CovarianceMatrices.jl` (`aVar` + fixed-bandwidth `Bartlett(h)`, which equals Newey–West at lag `h-1`; horizon 0 ⇒ Γ₀-only White/HC0). `CovarianceMatrices` added to `Project.toml` `[deps]` + `[compat]`. Per-row sandwich `(X'X)⁻¹ · meat · (X'X)⁻¹` on the standardised instruments.
- [x] Populate `result.irf_outcome` and `result.irf_endogenous` with point, SE, and `1 − α` normal bands (`z = norminvcdf(1 − α/2)`).
- [x] Tests (`test/test_irf.jl`):
  - HAC SE matches an explicit Bartlett sandwich at every horizon; horizon 0 reduces to the White/HC0 SE.
  - IRF point estimates equal the independent eq. (11) computation and reproduce β̂ through the stacked-IRF OLS (eq. 13).
  - Bands have positive width and bracket the point estimate; type stability via `@inferred`.

**Exit criteria:** all tests pass ✅ (`Pkg.test()` green). `summary`/`show` printing and the README quick-start doctest are deferred to Phase 5 (IRFs are exposed via the `result.irf_outcome` / `result.irf_endogenous` fields).

---

## Phase 5a — polish + v0.1.0 ✅

Goal: ship the LP variant with the polish layer.

- [x] Automatic-bandwidth HAC option for the IRFs: `hac` kwarg on `spiv` — `:fixed`
  (default; lag truncation = horizon, the paper rule), `:neweywest` (automatic
  Newey–West 1994), `:andrews` (automatic Andrews 1991), all via `CovarianceMatrices.jl`
  (`Bartlett{NeweyWest}` / `{Andrews}`). Implemented in `src/irf.jl` (`_hac_kernel`).
- [x] `Base.show(io, ::SPIVResult)` (compact) and `Base.show(io, ::MIME"text/plain", …)`
  (a clean Printf table: coefficients with SE/z/CI, weak-IV verdict, robust bounds, IRF
  shapes) — `src/show.jl`. No PrettyTables dependency.
- [x] IRF plotting recipe via `RecipesBase` (`@recipe`, `response = :outcome`/`:endogenous`,
  point + ribbon bands), no `Plots` hard dependency — `src/recipes.jl`.
- [x] README quick-start rewritten as a runnable, test-verified example; `version = "0.1.0"`.
- [x] Tests (`test/test_show.jl`): auto-HAC (`:fixed` ≡ default; `:neweywest`/`:andrews`
  finite & differing), `show` content, recipe series via `apply_recipe`, README run.

**Exit criteria:** `Pkg.test()` green ✅. Creating the actual `v0.1.0` git tag/release is
left to the user (like commits/pushes). Full Documenter doctest site deferred.

---

## Phase 5b — VAR variant (forecast-error route) ✅

Goal: the `SPIVwithVAR` variant. **Scope (confirmed):** route A — the forecast-error
construction feeding the shared estimator/inference/IRF. The VAR-IRF restriction (route B)
is deferred (see below).

Implemented in `src/var.jl`; `src/spiv.jl` refactored so step 1 (forecast errors) is
dispatched on the spec and steps 2–5 are shared (`_forecast_errors` + `_spiv_estimate`).

- [x] `SPIVwithVAR` variant: fit a VAR(p) by OLS on the stacked vector `[Y_t; y_t; z_t]` (p
  a kwarg, default 1; the VAR's own lags replace the LP control-residualization, so the
  controls `X` are ignored), build forecast errors `X̂_t^⊥(h) = Σ_{j=0}^h Â^{h-j} ê_{t+j}`
  (eq. A.3) via companion MA powers, and select rows for `ŷ⊥/Ŷ⊥/Ẑ⊥`. Reimplemented (cited)
  from `docs/ext/gragusa/MacroEconometricsTools.jl`, reusing `src/utilities.jl`
  (`create_lags`, `companion_form`). `T_eff = T − p − (H − 1)`; dof control count
  `Nx_eff = 1 + m·p` (m = K+1+Nz).
- [x] Dispatch: `spiv(y, Y, X, Z, SPIVwithVAR(); H, p, ...)` returns an
  `SPIVResult{Float64, SPIVwithVAR}` with the same shape as the LP variant; the LP path is
  unchanged (`spec` defaults to `SPIVwithLP()`).
- [x] `src/utilities.jl` is now `include`d and consumed by the VAR estimator.
- [x] Tests (`test/test_var.jl`): dispatch/shapes, β recovery on a VAR(1) DGP,
  LP≈VAR comparability on shared data, row-selection ≡ VAR one-step residuals, populated
  inference/IRF blocks, `@inferred`.

**Exit criteria:** both `SPIVwithLP` and `SPIVwithVAR` produce comparable estimates on the
same data ✅; `Pkg.test()` green (203 pass) ✅.

### Phase 5c — API ergonomics ✅

Breaking signature change while the package is unregistered, prompted by an
ergonomics review: every call site in the repo passed `X = ones(T, 1)` and most
needed `reshape` for univariate series.

- [x] `spiv(y, Y, Z, spec; H, X = nothing, intercept = true, ...)` — controls move to the
  `X` keyword; an intercept is prepended by default (`intercept = false` opts out);
  `Y`/`Z`/`X` accept vectors (promoted to one-column matrices).
- [x] Non-finite inputs rejected with a helpful `ArgumentError` (NaN-padded lags must be
  trimmed before estimating).
- [x] Export `lag` and new `lags(x, p)` (gragusa-mirrored data helpers) for building
  lagged controls; documented in `docs/src/api.md`.
- [x] All call sites, README, docs pages, and examples updated; equivalence tests
  (vector ≡ matrix input, default intercept ≡ explicit constant column).

**Exit criteria:** default-intercept path bit-identical to the old `X = ones(T, 1)`
results ✅; `Pkg.test()` green ✅; docs build clean ✅.

### Phase 5d — sample handling + accessors (v0.2.0) ✅

Second ergonomics pass for the applied-economist audience, all additive:

- [x] Automatic sample adjustment (Stata/R-style): `spiv` drops the leading rows that
  contain `NaN` in any input (the burn-in from `lag`/`lags`), so `X = lags(Y, p)`
  works without the manual `keep = (p+1):T` dance. Any other non-finite value — `Inf`
  anywhere, `NaN` after the burn-in — remains a hard `ArgumentError` (data errors,
  not lag padding). Trim is silent; the adjusted sample shows up as `T_eff`.
- [x] `xlags` kwarg: `spiv(y, Y, Z; H, xlags = p)` appends `lags(hcat(y, Y, Z), p)` to
  the controls (auto-trimmed) — the same conditioning information set as
  `SPIVwithVAR()` with lag order `p`.
- [x] `Base.summary(::SPIVResult)` prints the full coefficient/diagnostics table (R
  muscle-memory; deliberate bend of the "brief string" contract on our own type) and
  exported `irf(r; response = :outcome/:endogenous)` returns the `IRFBlock`.
- [x] `@warn` when `X` / `intercept = false` / `xlags` are passed with `SPIVwithVAR`
  (previously silently ignored; NaN-padded `X` under VAR used to *error*, now
  warns and runs — intentional behavior change).
- [x] Tests: auto-trim ≡ manual trim and `xlags` ≡ hand-built lags bit-identical;
  `@test_logs` for the VAR warnings; `summary`/`irf` round-trips; README/docs/examples
  updated.

**Exit criteria:** equivalence tests bit-identical ✅; `Pkg.test()` green ✅; docs build
clean ✅.

### Phase 5b' — VAR-IRF restriction (route B, deferred / experimental)

Not implemented; under-specified in-repo and unverifiable. Would additionally:
- replace the Gram matrices in eq. (9) with `Θ̂^VAR = Â^h D̂` outer products (§4.2) — needs
  a chosen identification scheme `D̂` (Cholesky / external-IV; the docs give none);
- apply the §7.3 VAR-restricted AR/KLM substitutions, with the **FE-only AR** (the AR test
  is not robust under VAR-IRF restrictions; KLM is) and an undocumented Delta-method AR
  variance.

---

## Out of scope (for v0.1.0)

The following appear in `docs/technical_julia.md` because that document describes a broader project. They are **not** targeted here:

- Bayesian VAR estimation, Minnesota / Normal–Wishart priors.
- Sign restrictions, Cholesky-only identification, set-identified IRFs.
- Generic constraint system (zero, fixed, block-exogeneity constraints on VAR coefficients).
- Wild / standard / block bootstrap inference (SP-IV uses analytical sandwich + AR/KLM, not bootstrap).

Revisit only if a concrete user need emerges.
