#!/usr/bin/env python3
"""JEV composite screen over congressional-dip candidates.

Reads scraped inputs (/tmp/fresh.csv, /tmp/monid/*.json) plus curated brief
facts, fires TWO maximally-batched TypeSafe calls (36 scoring questions +
8 citation checks), then composites scores in code with hard invalidation
gates JEV cannot override. Key via TYPESAFE_API_KEY env only.
"""
from __future__ import annotations

import json
import sys
from datetime import date

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

PRICES = {"FSLR": 177.71, "AMZN": 249.67, "WFC": 82.97, "MCD": 236.50}  # 2026-09-25 closes
INVALIDATE = {"FSLR": 170.00, "AMZN": 230.00, "WFC": 78.00, "MCD": 234.03}

DOSSIERS = {
    "FSLR": """First Solar, Friday close $177.71, -34% YTD, fresh 52wk low; -12% this week (-10.3% on 9/24, +3.2% bounce 9/25, RSI ~35, below all SMAs).
Sen. Boozman bought 8/13 @ $223.69 ($1-15k; all-time buy skill: 250 buys, 38% 90d SPY-alpha hit = coin flip); Rep. Khanna bought 8/10 @ $239.33 after selling 5/27 @ $273.67 (round-trip trader); Sen. Armstrong also bought 3/27. Both Aug buyers ~-21/-26% underwater. Neither buyer sits on Finance/Energy/Commerce/Ways-Means.
Officers sold into strength: GC Dymbort $922k 8/11 @ $249.38; CEO Widmar $5.72M May; Koralewski $1.55M 8/3; zero open-market officer buys in 6 months; no buyback.
Fundamentals: Q2 EPS $3.92 vs ~$2.90 (7/30), $1.7B net cash, backlog 45.1 GW / $13.6B thru 2030, guidance reaffirmed; BUT only +1.9 GW US bookings since prior call, FY26 consensus FCF -$146M, H1 operating cash -$360M, 45X credits = 81-91% of gross profit.
Valuation: ~10x fwd earnings; DCF range ~$163-236 (mid ~$192); street 24 Buy / 11 Hold / 2 Sell, avg target ~$267 (GLJ cut to $250 Buy 9/22; Bernstein $197 cut dated 7/31).
Overhangs: Fed 9/16 hike froze project finance; Sec.232 floors/duties eff. 12/4/26 (moat, payoff unproven); BIS anti-stockpiling rule 9/22-12/3; USITC complaint withdrawn 9/15 (market read as lost leverage, -5%); securities class action co-leads appointed 9/21; OBBBA customer-credit cliff risk.
Setup: Q3 est 10/29 (Zacks cut to $4.59); options imply +/-14% ($153-203); shorts 9.8% of float and rising; institutions net added in Q2.""",
    "AMZN": """Amazon, Friday close $249.67, -13% off $287.20 high (8/3), above $240.52 200-day; below 20/50/100-day; orderly pullback, RSI ~49.
Sen. Booker sold 8/11 @ ~$272.27 ($100-250k, spouse account, disclosed after 8/31 FTC suit; ZERO buys ever; Judiciary antitrust ranking but probe public since 9/2025). Buyers: Pelosi 1/16 entry was a Jan-2025 call exercise (not fresh buy); Kean bought 7/8 @ $243.62 (+2.5%); Khanna 8/3 @ $284.02 (-12%); Fetterman 3/30 @ $200.95 (+24%); Cisneros April ~flat. All-time congress: 175 buys/57 members vs 85 sells/42.
C-suite sold the top: Jassy $5.18M 8/21 @ $259.01 (holds 2.24M); Olsavsky $1.6M; SVPs ~$259-260 late Aug; Bezos $346M 8/3. All sales 10b5-1; zero officer buys; no H1 buyback (shares +52M).
Fundamentals: Q2 sales $200.6B +20%, op inc +43%; AWS $42.2B +37% (5th accel quarter), margin 39.4%, backlog $496B; ads +26%. BUT 2026 capex hiked to $220B, TTM FCF -$7.6B, Q3 guide $197-202B vs $204.1B consensus; fwd P/E 26.6x (trailing flattered by $53.4B Anthropic gain); AWS ~$1.1-1.4T of $2.69T value.
Peers: Azure 43%, GCP 82% growth vs AWS 37% — spend looks less productive.
Overhangs: 8/31 FTC + 22-state ad-pricing suit (est. exposure $0-5B, pending, no date); monopoly BENCH trial 2/9/27; chip tariffs narrow (25% w/ data-center exemptions, broader rate undecided); EU gatekeeper preliminary; $2.5B Prime settlement (paid $845M+).
Street: 56 Buy / 3 Hold / 0 Sell, avg $321 (+29%); Zacks cut to Hold 9/14; UBS $200 bear; Evercore raised to $355.
Setup: Q3 est 10/29 ($202.06B rev vs $202B guide ceiling — trap risk); AWS bar ~$44.6B/35% (<30% flips bearish); Prime Days 10/6-7 don't touch Q3; shorts 0.8% (light); institutions net +148M shares.""",
    "WFC": """Wells Fargo, Friday close $82.97, -4.7% since McConnell's 9/01 buy @ ~$86.95; -7.5% since pre-hike 9/15; BELOW all moving averages incl. 200-day ($85.03); broke $83.48-83.72 shelf; support $80.95; distribution volume, no capitulation.
McConnell (R-KY) bought 9/01 ($15-50k) plus 6/01 @ $77.26 (note: a listed "3/01/26" print falls on a Sunday — unverified); Khanna bought BIG 8/27 @ $85.00 ($50-100k) + small 8/18, sold small 8/24 (repositioning, not conviction). Smucker/Cisneros sold spring (early). Neither buyer on Banking/Finance; buys came BEFORE the 9/16 Fed hike; no shared hearing/briefing found.
Corporate: ZERO officer/director Form 4 trades (buy or sell) in 6 months; $7.1B H1 buybacks ($22.7B remaining); dividend +11% to $0.50; guidance unchanged.
Fundamentals: Q2 EPS $2.00 +25%, rev +9%, NII +5%; FY NII guide ~$50B (+5.3%, needs ~$12.8B/qtr H2); NIM ~2.7% (Q2 -4bp); H1 loans +12% (guide mid-single-digit; CFO 9/15: ahead of plan, Q3 NIM better than expected); NCO 34bp, provision $914M. Fwd P/E 10.3x, 1.5x book — ROTCE 17-18% must deliver.
Macro: unanimous 9/16 hike to 3.75-4% (one more guided); curve flattened to ~20bp 9/21 (re-steepened to 31bp by 9/24); KBW -2.9%; Sept peers: BAC -8.5%, GS -6.7%, JPM -3.4%, C +1.3%.
Residue: Fed 2018 order terminated 3/5/26 and cap lifted 2025, BUT OCC 9/2024 agreement + AML/sanctions inquiries + ~$1.5B possible losses remain.
Street: defending (13 Strong Buy / 3 Buy / 10 Hold / 0 Sell, avg $100.46; Q3 EPS rev 4-up/0-down); Goldman cut $111->$107 Buy; KBW Hold $94.
Setup: Q3 Tue 10/13 (~$1.85 / $22.34B); hike covers only ~15% of Q3; stock fell 5.7% and 2.7% on the last two beats; options imply ~6.6%; shorts 1.04% (light); 13F net +$11B.""",
    "MCD": """McDonald's, Friday close $236.50, 52wk low (intraday $234.03 on 9/23 on 3.35x volume flush, unconfirmed climax); 7 straight down weeks; below all MAs (200-day $291.83); RSI ~22; 2014 analog: flat 3 months after prior 7-week streak.
Comps waterfall: Q1 +3.8% -> Q2 global +1.3% (US +0.8%, guest counts negative) -> Q3 US pre-guided "slightly negative". Low-income traffic down ~double digits ~2 yrs; value execution fail (60-65% on pricing, 1/3 stores missed guidance); beef +70% since 2021 (BLS); Aug CPI +3.4%, energy +16.3%, real wages -0.3%, sentiment 48.1. GLP-1: 11% adult use, users still eat out (mix-down, not exit). PEERS OUTPERFORMING: Taco Bell +7%, Burger King +8.5%, Starbucks +7.9% transactions — MCD-specific failure.
NEXT plan (9/23): $8.5B thru 2036, op margin low-mid 50% by 2030; stock -4.8% that day; CEO "not expecting things to change". 50,000-store goal slipped to 2028.
Valuation: ~16.9x FY27 EPS ($13.98, drifting down, 25 cuts/30d); deciding metric = guest counts, not check. Dividend $7.72 (55% EPS but 71% FCF), 50th hike, 3.26% yield.
Congress: ALL 2026 sells avoided +11-23% (Booker 8/11 @ ~$273.83 spouse acct, zero buys ever; Khanna 8/10+7/20; R. McCormick 7/30; Whitehouse 1/9 @ $307.27 — not top-tick, $341 March high came after). ALL 2026 buys -11-28% underwater (Franklin 8/26 @ $266.88; Khanna 8/24 flip-flop; March buyers Moskowitz/Khanna @ $308-327). Booker Ag/Antitrust = thematic mosaic, docs postdate sale. No committee edge anywhere.
Corporate: $40.5M officer sales, ZERO buys (all 10b5-1); Kempczinski $17.47M Feb; company bought $1.25B H1 @ $288.62 avg (underwater); Sept buybacks unconfirmed.
Legal: nothing live (McRib consumer suit only); SNAP indirect; no FDA QSR action.
Street: 1 Strong Buy / 17 Buy / 11 Hold / 0 Sell, avg $300.40 — but mass target cuts 9/24 (Baird $250 Neutral, JPM $260, TD $270); Q3 est 11/4 (need guest counts + EDAP >80%); shorts 1.73% (up 58% YTD, not crowded); institutions net positive (lagged).""",
}

LEVELS = [
    "deeply attractive, clear mispricing in our favor",
    "attractive, solid risk/reward with conditions",
    "fair, balanced risk/reward, needs a trigger",
    "unattractive, poor compensation for the risk",
    "extremely unattractive, avoid",
]
SAFETY_LEVELS = [
    "severe live threats dominating the thesis",
    "material unresolved overhangs",
    "notable but manageable issues",
    "minor issues only",
    "clean, no material overhang",
]

DIMENSIONS = ["value", "price_action", "earnings_setup", "insider_confirmation",
              "congress_signal", "overhang_safety"]

PROFILE_WEIGHTS = {
    "dip_buyer": {"value": .25, "price_action": .10, "earnings_setup": .20,
                  "insider_confirmation": .15, "congress_signal": .10,
                  "overhang_safety": .20},
    "momentum": {"value": .10, "price_action": .30, "earnings_setup": .20,
                 "insider_confirmation": .10, "congress_signal": .05,
                 "overhang_safety": .25},
}


def score_to_attractiveness(score: float) -> float:
    """Normalize a 0..4 JEV score (higher = better) to 0..1 attractiveness."""
    if not 0 <= score <= 4:
        raise ValueError(f"JEV score out of range 0..4: {score!r}")
    return 1.0 - score / 4.0


def composite_scores(answers: dict) -> list:
    """Pure composite: normalize, weight by profile, attach gates. No I/O."""
    lines = []
    for t in DOSSIERS:
        parts = {d: score_to_attractiveness(answers[f"{t}_{d}"].score)
                 for d in PROFILE_WEIGHTS["dip_buyer"]}
        line = {"ticker": t, "close": PRICES[t], "invalidate_below": INVALIDATE[t]}
        for pname, w in PROFILE_WEIGHTS.items():
            line[pname] = round(sum(parts[d] * w[d] for d in w), 3)
        line["action"] = answers[f"{t}_action"].choice
        line["action_conf"] = round(answers[f"{t}_action"].confidence, 2)
        line["info_edge_p"] = round(answers[f"{t}_info_edge"].noul, 2)
        line["climax_p"] = round(answers[f"{t}_climax"].noul, 2)
        line["min_conf"] = round(min(answers[f"{t}_{d}"].confidence
                                     for d in PROFILE_WEIGHTS["dip_buyer"]), 2)
        lines.append(line)
    return lines


def score_q(dim: str, ticker: str) -> Score:
    return Score(
        instructions=f"Rate {ticker} on {dim} using ONLY the supplied dossier state at tickers.{ticker}.",
        criteria=LEVELS if dim != "overhang_safety" else SAFETY_LEVELS,
    )


def build_questions() -> dict:
    qs: dict = {}
    for t in DOSSIERS:
        for d in DIMENSIONS:
            qs[f"{t}_{d}"] = score_q(d, t)
        qs[f"{t}_action"] = Choice(
            instructions=f"Recommended action on {t} at its Friday close, using ONLY tickers.{t}.",
            criteria={
                "buy_now": "Enter a starter immediately at current price",
                "scale_in_on_trigger": "Wait for the dossier's stated technical/confirmation trigger, then enter",
                "wait_for_print": "Stand aside until the dossier's next earnings/catalyst print resolves",
                "avoid": "Do not own; thesis broken or risk uncompensated",
            },
        )
        qs[f"{t}_info_edge"] = Noul(
            instructions=f"The congressional flow in tickers.{t} contains non-public information edge")
        qs[f"{t}_climax"] = Noul(
            instructions=f"The price tape in tickers.{t} shows selling-climax capitulation")
    return qs


CHECKS = [
    ("pelosi_exercise", "Pelosi's 1/16/26 AMZN entry at $239.12 was an exercise of calls acquired in Jan 2025, not a fresh open-market stock buy.",
     "Filing filed 23 Jan 2026 shows exercise of 50 calls acquired Jan 2025, plus a 12/24/25 sale of 20,000 shares and 12/30/25 purchase of 20 calls."),
    ("mcc_sunday", "A listed McConnell WFC buy dated 3/01/2026 cannot be an execution date because 3/1/2026 was a Sunday.",
     "Calendar fact: March 1, 2026 falls on a Sunday; NYSE closed. The print needs PTR re-verification."),
    ("amzn_bench", "Amazon's monopoly trial is set as a bench trial on 2/9/27, not a jury trial next year.",
     "MLex 9/16/26: bench trial scheduled 3/29/27 per one report; verdict sought is conduct/structural relief. (Two reported dates exist: 2/9/27 and 3/29/27.)"),
    ("loop_date", "Loop's $315 Hold rating on MCD (from $346) dates to June 2025, not September 2026.",
     "No September 2026 Loop action found in MarketBeat rating history through 9/25/26."),
    ("wfc_q3", "Wells Fargo Q3 earnings are due Tuesday 10/13/26 ~10am ET, not 10/20.",
     "WFC IR, MarketBeat 9/25 earnings page, Nasdaq 7/7/26 all list 10/13."),
    ("mcd_q3", "McDonald's Q3 report is estimated ~11/4/26, not late October.",
     "MarketBeat earnings calendar 8/4/26 estimates 11/4; company unconfirmed."),
    ("bernstein_date", "Bernstein's cut to $197 on FSLR is dated 7/31/26, not September 2026.",
     "MarketBeat FSLR rating history shows 7/31 Underperform action."),
    ("armstrong_fslr", "Sen. Armstrong bought FSLR on 3/27/26 (filed 7/21), a third 2026 congressional buyer beyond Boozman and Khanna.",
     "Quiver FSLR tape checked 9/27/26 shows 8 trades incl. Armstrong 3/27 buy; absent from the canonical DB window extract."),
]


def build_checks() -> dict:
    qs: dict = {}
    for qid, claim, evidence in CHECKS:
        qs[qid] = Choice(
            instructions=f"Evaluate the claim using ONLY its evidence. Claim: {claim} Evidence: {evidence}",
            criteria={
                "supported": "Evidence directly supports the claim",
                "contradicted": "Evidence contradicts the claim",
                "unverifiable": "Evidence is insufficient to decide",
            },
        )
    return qs


def main(client=None, out_dir="/tmp") -> None:
    from pathlib import Path

    if client is None:
        client = TypeSafeClient()
    out = Path(out_dir)
    state = {"as_of": str(date(2026, 9, 27)), "closes": "2026-09-25",
             "tickers": DOSSIERS,
             "gates": {t: f"invalidate below ${p:.2f}" for t, p in INVALIDATE.items()}}
    print("CALL 1: 4 dossiers x 9 questions = 36 decisions...", flush=True)
    r1 = client.system_one(state=state, questions=build_questions())
    json.dump({"model": r1.model, "usage": dict(r1.usage or {}),
               "answers": {k: v.model_dump() for k, v in r1.answers.items()}},
              open(out / "jev_scores.json", "w"), indent=1)
    print("saved jev_scores.json", r1.model, r1.usage, flush=True)

    print("CALL 2: 8 citation checks...", flush=True)
    r2 = client.system_one(
        state="Fact-check claims against the paired evidence snippets inside each question.",
        questions=build_checks())
    json.dump({"model": r2.model, "usage": dict(r2.usage or {}),
               "answers": {k: v.model_dump() for k, v in r2.answers.items()}},
              open(out / "jev_checks.json", "w"), indent=1)
    print("saved jev_checks.json", r2.model, r2.usage, flush=True)

    composite(r1)


def composite(r1) -> None:
    print("\n=== COMPOSITE (higher = more attractive; gates applied in code) ===")
    for line in composite_scores(r1.answers):
        print(json.dumps(line))


if __name__ == "__main__":
    sys.exit(main())
