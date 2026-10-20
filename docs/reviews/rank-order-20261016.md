# Rank-order test (non-circular): does the score order forward excess?

Date: 2026-10-16. Pre-registered in `docs/strategy-charter.md` promotion
rule item 1. Method: `data/backtest_assumptions/cohorts_*.csv` (8 decisions,
5,688 obs from the corrected `sweep` harness), restrict to
`entry_eligible==True` (n=2,825), per-decision score terciles (scores are
not comparable across dates — actor counts vary), top-vs-bottom tercile
mean 90d SPY-relative excess, difference with clustered SE (cluster =
decision, 8 clusters, bar |t|>=2).

Grouped sample report: 5,688 source rows (2,825 eligible rows), 8 distinct
decision-date clusters; distinct decision-date × member groups: UNKNOWN from
the published summary (the cohort CSVs are not present in this checkout).
The row/episode totals are not substitutes for independent groups.

## Result: FAIL. The score does not rank-order forward excess.

| decision | top n | bot n | top mean | bot mean | top−bot |
|----------|------:|------:|---------:|---------:|--------:|
| 2023-01-15 | 153 | 178 | −3.46 | −6.15 | +2.70 |
| 2023-07-15 | 102 | 214 | −2.81 | −5.59 | +2.78 |
| 2024-01-15 | 169 | 198 | −5.02 | −3.72 | −1.30 |
| 2024-07-15 | 139 | 175 | −1.04 | −0.90 | −0.14 |
| 2025-01-15 | 164 | 205 | −2.53 | −0.89 | −1.64 |
| 2025-07-15 | 104 | 228 | −1.05 | −1.39 | +0.33 |
| 2026-01-15 | 109 | 221 | −3.25 | −1.32 | −1.93 |
| 2026-07-01 |  36 |  40 | −7.72 | −4.12 | −3.60 |

Pooled top−bot: **−0.35pp, clustered SE 0.79, t=−0.44**, 95% CI [−1.94, +1.24].
This is a difference of means across 8 decision dates; the 5,688 observation
rows do not provide 5,688 independent units.
Signs flip across decisions (positive twice early, negative thereafter) —
no stable direction, let alone significance.

Level result: EVERY top-tercile bucket trails SPY (−1.04 to −7.72), as do
all bottom buckets. The eligible universe as a whole averages −2.56pp vs
SPY. The filter does not clear the market.

## Caveats (do not rescue the result)

1. Harness weights lack the production 0.5× skill discount, so terciles
   are on harness scores, not production scores. Bound: 14.5% of eligible
   rows involve a skilled actor (the only rows whose order could shift).
   The gap to significance (t=−0.44 vs bar 2, unstable signs) is far too
   large for a 14.5%-row reweighting to close.
2. Entry is next-close, not the traded limit-fill rule (charter item 2).
   That moves levels, not rank order — it cannot manufacture ordering
   from a null contrast.
3. 8 decision-date clusters: the displayed SE/CI use these clusters; distinct
   date×member groups remain UNKNOWN. The CI excludes anything above +1.24pp
   under this estimator and clustering scheme.

## Consequence

Charter promotion rule item 1: TESTED → FAIL. Sizing stays flat $3k;
paper remains the defensible posture.

### Superseded/corrected interpretation (2026-10-20)

The previous consequence sentence called the eligible/ineligible contrast a
“loss-cutting rule.” That interpretation is withdrawn. The separate
eligible/ineligible result is a difference of means: eligible −2.55999pp
versus ineligible −0.52216pp (difference −2.03783pp, reported clustered
t=−4.65; `data/backtest_assumptions/summary.csv:5`). It is not a loss
reduction and does not establish a causal filter effect.
