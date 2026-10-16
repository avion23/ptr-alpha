# First-principles review: the strategy, not the plumbing (2026-10-16)

Skill: first-principles-review, phases 0–6 in order. Subagents were
unavailable (DeepSeek no credits, GLM quota to 05:52 UTC, GPT types banned
by owner rule), so all phases executed directly. Plumbing (as-of integrity,
determinism, receipts) passed its own audit; this review asks whether the
machine is worth running.

## Phase 0 — Requirements

See `docs/strategy-charter.md` (R6 + table, promotion rule). Headline
gaps found: no objective function existed (fixed by the charter); skill
null (R7 FAIL); costs unmodeled (R8 PARTIAL after 2026-10-16 sensitivity);
dividend avoidance operator-only (R9 FAIL).

## Phase 1 — What is predicted

Four conflated quantities: (a) P(insider informed) — the priors gesture
here; (b) expected forward excess — what the backtest measures;
(c) entry quality — what the 1.05 gate enforces; (d) public
post-disclosure drift — what the skill label actually measures
(`actors.py:132`: entry = next session after *disclosure*). The score mixes
(a) with (c), is validated against (b), labeled by (d). Fundamentally the
system is a ranker over entry-quality-weighted recent activity and must be
specified as that, not as skill-weighted alpha.

## Phase 2 — Logic defects (all verified in code)

1. Production weights are 5-valued constants ({0.52, 0.26, 0.60, 0.30, 0});
   `compute_actor_weights` (`actors.py:373`) always calls `compute_weight`
   with n_matured=0, so Beta-binomial shrinkage never executes and
   `age_weight` had zero callers (removed 2026-10-16). The score is an
   actor-count ranker wearing skill clothing.
2. Backtest/production weight mismatch (harness omits the 0.5× discount;
   14.5% of eligible rows affected — bounded, not fatal).
3. Skill label (post-disclosure drift) ≠ traded outcome (proximity to the
   insider print). No line connects them.
4. Correction 2026-10-16: the fill gap overclaimed — for eligible rows the
   next close already satisfies the limit, so a day-limit order fills with
   certainty on closes-only data (small-size assumption). Remaining: spread
   only, closed by sensitivity (item 6). RESOLVED, charter updated.

## Phase 3 — Closed form

Score(T) = 1.5^C(T) · Σ_{a∈A(T)} w_a · d_a (`setups.py:46–302`): C(T)∈{0,1}
watchlist-13F corroboration; w_a five-valued; d_a = 1 with
position-evidence else 0.5^{lag/90}; A(T) requires valid entry_ref,
current ≤ 1.05·ref, quote ≤5d, w·d ≥ 0.05. Rank iff score ≥ 1.0, or
(> 0.25 with ≥2 actors or corroboration).
Findings: position-evidence flag doubles contribution on a
data-availability bit (economics-free); per-actor row choice (max decay,
alphabetical tiebreak) rarely matters post-gate; discontinuities at 1.05×,
0.05, 0.25, 1.0 make 6.76-vs-5.72 mostly actor-count order; 1.5× on the
whole sum is an unjustified interaction (second-order).

## Phases 4+5 — No compute problem

Unscoped run 3.7 min (38s events, 52s batched holdings, trivial scoring).
One O(N)-Python pass over 25k purchases (`pipeline.py:396`) is acceptable
at weekly cadence. Replay/receipts reuse the production kernel. No
restructure warranted; the data problem is coverage, not throughput.

## Phase 6 — Evidence

- Skill nulls (t=0.81, 0.55, 8 clusters) are informative, not underpowered.
- Eligible −2.56 vs ineligible −0.52 (t=−4.65) validates the ENTRY FILTER,
  not skill (confounded with value/liquidity/attention by construction).
- Rank-order test 2026-10-16 (`docs/reviews/rank-order-20261016.md`):
  among-eligible top−bot −0.35pp, t=−0.44 — the score does not order
  returns. Every top-tercile bucket trails SPY.
- Costs: 0–50bps round-trip moves eligible −2.56→−3.06pp; verdicts
  unchanged; tercile contrast identical.

## Verdict

No demonstrated alpha; the filter cuts losses; the score does not order
returns. Promotion rule unmet — flat $3k ceiling, paper defensible.
Ranked follow-ups: charter (done), rank-order test (done, FAIL),
dead math (half done — shrinkage unwired pending revalidation),
limit-fill protocol (dissolved by proof), ex-div gate (open — needs a
dividend feed), costs (closed by sensitivity).
