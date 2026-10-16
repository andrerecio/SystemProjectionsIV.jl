# Fieldhouse & Mertens, Table 1: data

`table1.jl` replicates rows 1–5 of Table 1 in

> Fieldhouse, Andrew J., and Karel Mertens. "The Returns to Government R&D:
> Evidence from U.S. Appropriations Shocks."

with `spiv`. `table1_data.csv` holds the 17 series those rows use, taken from the quarterly
dataset (`DATAQ`, 1947Q1–2021Q4) that `MASTER.m` builds in their replication package. Only
these columns are redistributed; the full package is not part of this repository. Values
are copied unchanged; empty cells are written as `NaN`.

| Column | Description |
|---|---|
| `date` | First day of the quarter |
| `ln_tfp_util_adj1`, `_adj2`, `_adj3` | Log utilization-adjusted TFP, net of government non-R&D capital (low, moderate, high adjustment) |
| `ln_rK_GRD` | Log real government R&D capital |
| `z_NDEF`, `z_allNDEF` | Exogenous / all nondefense R&D appropriations shocks, scaled by lagged nondefense R&D capital |
| `z_NDEF_exspace`, `z_SPACE` | `z_NDEF` excluding NASA, and the NASA shocks |
| `z_NSEC`, `z_allNSEC` | Exogenous / all national-security R&D appropriations shocks, scaled by lagged defense R&D capital |
| `ln_cum_rALL_NDEF`, `ln_cum_rALL_NSEC` | Log cumulated nondefense / national-security shocks |
| `ln_util`, `ln_tfp_util` | Log utilization and log utilization-adjusted TFP |
| `ln_rK_PRD_fernald` | Log real private R&D capital |
| `R_EW_COMB_TECH` | Equal-weighted combined tech stock returns |
| `Military_news_scaled` | Scaled military spending news |

## Sources and license

The series are derived (by Fieldhouse & Mertens's construction scripts) from:

- BEA fixed-asset and NIPA tables and NSF/NCSES R&D data (capital stocks, shock scaling);
- Fieldhouse & Mertens's narrative R&D appropriations shocks (`z_*`, `ln_cum_rALL_*`);
- John Fernald's quarterly utilization-adjusted TFP data, Federal Reserve Bank of San
  Francisco (`ln_tfp_util*`, `ln_util`, `ln_rK_PRD_fernald`);
- the Kenneth French Data Library (`R_EW_COMB_TECH`);
- Ramey & Zubairy's replication data (`Military_news_scaled`).

The replication package's license is reproduced in `LICENSE.txt`: MIT for the code, and a
data notice under which the source data remain subject to their providers' terms. Please
cite the paper, and the original data sources above, when using these data.
