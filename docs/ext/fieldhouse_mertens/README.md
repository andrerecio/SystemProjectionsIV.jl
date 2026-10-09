# Fieldhouse & Mertens — SP-IV MATLAB reference code

SP-IV estimation scripts extracted from the replication package of

> Fieldhouse, Andrew J., and Karel Mertens. "The Returns to Government R&D:
> Evidence from U.S. Appropriations Shocks."

Redistributed under the package's MIT license (`LICENSE.txt`). Only the scripts that
**estimate SP-IV** are kept; data construction, source data, LP/IRF figures, and
plotting helpers are omitted, so these files do not run standalone (they expect the
`DATAQ` timetable built by the original `MASTER.m`). They are **reference material**
for `src/`, like `docs/ext/gragusa/`.

| File | SP-IV content |
|---|---|
| `Make_Table_1.m` | Canonical implementation: `run_SPIV_total` (K=1, 1-D KLM grid + `fzero`), `run_SPIV_decomp` (K=2, 2-D KLM grid + interpolation + flood fill), `GetARObj` (CUE objective), `GetKstat` (KLM statistic), `gmin` first-stage test, Wald CIs |
| `Make_Table_2.m` | Rate-of-return variant (cumulated differences), K=1 and K=2 |
| `Make_Table_E1.m` | Wald-only inference |
| `Make_Table_E2.m` | Constant-elasticity decomposition (K=2) |
| `Make_Table_E3.m` | Total (G+P) R&D capital, K=1 and K=2 |
| `Make_Figure_10.m` | Leave-window-out subsample SP-IV (Wald + KLM) |
| `Make_Figure_E1.m` | Illustration: outcome IRF vs β̂ × endogenous IRF |
| `Make_Figure_E2.m`, `Make_Figure_E3.m` | 2-D KLM confidence-set heatmaps |
| `Make_Figure_E4.m`, `Make_Figure_E5.m` | Robustness re-estimations |
| `auxiliary_files/btrace.m` | Block trace used by `GetKstat` |
| `auxiliary_files/nancumsum.m` | NaN-aware cumulative sum used to build `YH` |
