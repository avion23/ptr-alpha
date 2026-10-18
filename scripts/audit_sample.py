#!/usr/bin/env python3
"""Pixel-audit tooling for staged GLM OCR records (FP review 2026-10-18).

Implements the acceptance measurement the FP ruling requires before any
canonical promotion: a stratified random sample of staged documents,
rendered back to pixels for independent owner review, and an exact
one-sided binomial bound on the material-error rate against an
owner-approved budget (illustrative <1% at 95% confidence).

Finding 9 fail-closed semantics (report can never manufacture acceptance):

* Every sampled doc requires an explicit verdict. For a doc without a
  sticky failure, missing/invalid/None verdicts make it UNRESOLVED; with any
  unresolved sampled doc the report state is ``incomplete`` and
  ``budget_met`` is ``None`` (it can never default to clean).
* A ``fail`` is STICKY for the population assessment: verdict records
  APPEND to ``audit-results.json`` (history) and the report uses the
  WORST verdict per doc. A later pass marks the doc repaired
  (``repaired_count``) but the original failure stays counted in both
  numerator and denominator.
* The report is computed over a FROZEN sample lot: ``sample`` records
  doc_ids + per-doc staged-record sha256 + model + prompt_sha256 +
  parser_version in ``audit-sample-<seed>.json`` at sample time. The
  report verifies each pass-only doc's CURRENT staged-record sha against
  the frozen sha; mismatch -> UNRESOLVED ("recipe/staged-record changed
  after sampling"). Sticky failures remain resolved regardless of later
  staged-record drift. The sha excludes audit-mutable fields so that
  stamping a verdict never looks like drift.
* Omission detection is optional. When ``--ground-truth-dir`` is supplied,
  each passing doc needs a ground-truth file and its normalized row
  MULTISET (asset, type, tx_date, notif_date, amount) must match the staged
  rows; otherwise that doc is UNRESOLVED or a critical failure. Without the
  option the report explicitly certifies pixels only, not row completeness.
  Promotion consumes staged rows, so any requested omission check must run
  BEFORE ``promote``; a passing doc whose staging record is gone is
  UNRESOLVED.
* Completion is distinct from budget acceptance. A report is ``complete``
  only when a valid, non-empty frozen lot is available, every lot doc has a
  resolving verdict, and every passing doc has current frozen-record
  evidence (plus row-multiset evidence when ground truth was requested).
  A sticky fail is resolved evidence even if it was later re-recorded; a
  later pass does not erase it. Verdicts outside the frozen lot do not block
  completion or enter the current-lot binomial bound, but their sticky
  failures remain in a separate all-history population tally. ``budget_met``
  is always a boolean for a complete report and means the critical-only
  current-lot exact upper bound is <= ``--max-rate``; it is ``None`` only
  for an incomplete report.
* Stratification is reported honestly: per-stratum counts/rates plus a
  pooled rate weighted by staged population counts per year (frozen at
  sample time). Without population weights the pooled rate falls back
  to an explicitly-labelled unweighted-stratum mean.
* Owner-approved error budget (2026-10-18): transaction-critical fields
  (transaction_type, transaction_date, ticker/asset identity,
  amount_letter) must be error-free; asset-description TEXT
  normalization differences (repeated issuer wording dropped, minor
  spelling) are COSMETIC — flagged and tracked, not blocking.
  ``record --severity cosmetic`` classifies a fail as cosmetic; the
  report's ``budget_met`` counts CRITICAL fails only (the one-sided
  binomial bound applies to the critical rate) while cosmetic fails
  stay visible (count + docs). Entries recorded before ``--severity``
  existed carry no severity and count CRITICAL (conservative); the
  report prints a note so the operator can re-record reclassified
  items. Critical dominates: an explicit critical fail is never
  downgraded by a later cosmetic re-record.

Subcommands:

* ``sample  --per-year N --seed S``
    Stratified random sample (``random.Random(S)``) of staged successfully
    parsed records -> frozen-lot file ``audit-sample-<seed>.json`` plus a
    review bundle ``audit-review/`` holding the rendered page PNGs and a
    copy of each staging JSON. Deterministic given the seed and staging
    population.
* ``record  --doc-id ID --verdict pass|fail [--severity critical|cosmetic] --notes "..."`
    APPEND to the ``audit-results.json`` history and stamp the staging
    record's ``audit`` field (the field ``glm_stage.py promote
    --require-audit-pass`` gates on). A later pass never erases an
    earlier fail from the population assessment. ``--severity``
    (default critical) classifies a fail: cosmetic fails are tracked
    separately and do not block promotion (owner-approved budget).
* ``report  [--max-rate 0.01] [--ground-truth-dir DIR] [--sample-file P]``
    Material-error accounting over the frozen lot with the one-sided
    exact 95% binomial upper bound. Exit codes: 0 complete + budget met,
    1 complete + budget NOT met, 2 INCOMPLETE audit (refuses: the full-lot
    budget cannot be certified). Writes ``audit-report-state.json`` next to the
    results file with ``state`` in {"complete", "incomplete"} and
    ``budget_met`` in {true, false, null} -- the loose file contract
    ``promote --require-audit-pass`` should consult and refuse when
    ``state != "complete"`` or ``budget_met is not True``. A complete report
    may still miss the owner budget; that is a completed measurement, not an
    incomplete audit.

Every report begins with the ruling caveat:

    LLM-assisted audit — silver standard; not a substitute for human
    pixel verification
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scripts.gemini_ocr_common import pdf_sha256  # noqa: E402,F401  (re-exported)

AUDIT_RESULTS_NAME = "audit-results.json"
REPORT_STATE_NAME = "audit-report-state.json"
REPORT_STATE_SCHEMA = "audit-report-state-v2"
SAMPLE_SCHEMA = "audit-sample-lot-v2"
RENDER_DPI = 150
DEFAULT_MAX_RATE = 0.01
DEFAULT_CONFIDENCE = 0.95

# Fields the audit workflow itself writes into a staging record. They are
# excluded from the frozen staged-record sha so that recording a verdict is
# never misread as recipe drift.
AUDIT_MUTABLE_FIELDS = frozenset(
    {"audit", "audit_severity", "audit_notes", "audited_at_utc"}
)

# Owner-approved error budget (2026-10-18): transaction-critical fields
# (transaction_type, transaction_date, ticker/asset identity, amount_letter)
# must be error-free; asset-description TEXT normalization differences are
# COSMETIC — tracked, not blocking. ``budget_met`` counts critical fails only.
VALID_SEVERITIES = ("critical", "cosmetic")

CAVEAT = (
    "LLM-assisted audit — silver standard; not a substitute for human "
    "pixel verification"
)
PROMOTE_ORDER_NOTE = (
    "NOTE: Promotion consumes staged rows; if you request omission checks, "
    "run this report BEFORE `promote`."
)


def staging_root(data_dir: str | Path = "data") -> Path:
    return Path(data_dir) / "staging" / "glm-backlog"


def utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _atomic_write_json(path: Path, payload) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------
# binomial bound
# --------------------------------------------------------------------------


def _binomial_cdf_at_most(n: int, f: int, p: float) -> float:
    """P(X <= f) for X ~ Binomial(n, p)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if f >= n else 0.0
    total = 0.0
    for k in range(f + 1):
        total += math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))
    return min(total, 1.0)


def _exact_upper_bound(f: int, n: int, confidence: float) -> float:
    """Smallest p with P(X <= f | n, p) <= 1 - confidence (Clopper-Pearson)."""
    alpha = 1.0 - confidence
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _binomial_cdf_at_most(n, f, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def binomial_upper_bound(f: int, n: int, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """One-sided exact ``confidence`` upper bound on the error rate.

    scipy when available, else exact computation via ``math.comb`` bisection.
    For f=0 this is ``1 - (1-confidence)^(1/n)`` (0.997% at n=299, 95%).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    f = max(f, 0)
    if f >= n:
        return 1.0
    try:
        from scipy.stats import beta

        return float(beta.ppf(confidence, f + 1, n - f))
    except Exception:
        return _exact_upper_bound(f, n, confidence)


# --------------------------------------------------------------------------
# frozen-lot hashing
# --------------------------------------------------------------------------


def stable_record_view(record: dict) -> dict:
    """Record content minus audit-mutable fields (the frozen evidence)."""
    return {
        key: value
        for key, value in record.items()
        if key not in AUDIT_MUTABLE_FIELDS
    }


def staged_record_sha256(record: dict) -> str:
    """sha256 of the canonical JSON of the audit-stable record view."""
    canonical = json.dumps(
        stable_record_view(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# sample
# --------------------------------------------------------------------------


def load_staging_records(data_dir: str | Path = "data") -> list[tuple[int, dict]]:
    """(year, record) pairs for every readable staging JSON, sorted."""
    root = staging_root(data_dir)
    records = []
    for year_dir in sorted(root.glob("[0-9]" * 4), key=lambda p: p.name):
        for path in sorted(year_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            records.append((int(year_dir.name), record))
    return records


def _sampling_frame(records: list[tuple[int, dict]]) -> dict[int, list[str]]:
    """doc_ids per year eligible for sampling (parsed success or no_txs)."""
    by_year: dict[int, list[str]] = {}
    for year, record in records:
        if record.get("parse_status") not in ("success", "no_txs"):
            continue
        by_year.setdefault(year, []).append(str(record["doc_id"]))
    return by_year


def build_sample(
    records: list[tuple[int, dict]], per_year: int, seed: int
) -> list[dict]:
    """Stratified random sample of successfully parsed staged records.

    Deterministic: fresh ``random.Random(seed)``, doc ids sorted within each
    year, years visited in ascending order.

    Asset-review oversampling (interface-audit item 3): records stamped
    ``asset_review: true`` by the staging client (likely asset-text
    normalization damage: missing-ticker, substring patterns) are drawn
    FIRST within their stratum — all of them when they fit the quota,
    ``rng.sample`` when they overflow it — and only the leftover slots are
    filled from the unflagged docs. The lot stays stratified and
    seed-deterministic; flagged docs get guaranteed representation so the
    audit sees the risky docs instead of hoping to.
    """
    by_year = _sampling_frame(records)
    flagged = {
        str(record["doc_id"])
        for _, record in records
        if record.get("asset_review")
    }
    rng = random.Random(seed)
    sample = []
    for year in sorted(by_year):
        doc_ids = sorted(set(by_year[year]))
        flag_here = [d for d in doc_ids if d in flagged]
        plain = [d for d in doc_ids if d not in flagged]
        chosen: list[str] = []
        if flag_here:
            chosen.extend(rng.sample(flag_here, min(per_year, len(flag_here))))
        slots = per_year - len(chosen)
        if slots > 0 and plain:
            chosen.extend(rng.sample(plain, min(slots, len(plain))))
        sample.extend(
            {"doc_id": doc_id, "year": year} for doc_id in sorted(chosen)
        )
    return sample


def freeze_lot(
    records: list[tuple[int, dict]], sample: list[dict]
) -> list[dict]:
    """Freeze per-doc evidence at sample time (FP review finding 9, item 3).

    Each lot entry records the doc_id, its year, the staged-record sha256
    (over the audit-stable record view), and the extraction recipe fields
    (model, prompt_sha256, parser_version). The report later refuses docs
    whose current staged record no longer matches this fingerprint.
    """
    record_by_id: dict[str, tuple[int, dict]] = {}
    for year, record in records:
        record_by_id.setdefault(str(record["doc_id"]), (year, record))
    lot = []
    for item in sample:
        doc_id = str(item["doc_id"])
        found = record_by_id.get(doc_id)
        if found is None:
            lot.append(
                {
                    "doc_id": doc_id,
                    "year": int(item["year"]),
                    "staged_record_sha256": None,
                    "model": None,
                    "prompt_sha256": None,
                    "parser_version": None,
                    "asset_review": None,
                }
            )
            continue
        _, record = found
        lot.append(
            {
                "doc_id": doc_id,
                "year": int(item["year"]),
                "staged_record_sha256": staged_record_sha256(record),
                "model": record.get("model"),
                "prompt_sha256": record.get("prompt_sha256"),
                "parser_version": record.get("parser_version"),
                "asset_review": bool(record.get("asset_review")),
            }
        )
    return lot


def write_sample_file(
    records: list[tuple[int, dict]],
    sample: list[dict],
    *,
    seed: int,
    per_year: int,
    out_path: Path,
) -> dict:
    """Write the frozen-lot sample file (schema audit-sample-lot-v2)."""
    frame_counts = Counter(year for year in _sampling_frame(records))
    envelope = {
        "schema": SAMPLE_SCHEMA,
        "seed": int(seed),
        "per_year": int(per_year),
        "created_at_utc": utc_now_iso(),
        "frame_counts_by_year": {
            str(year): frame_counts[year] for year in sorted(frame_counts)
        },
        "lot": freeze_lot(records, sample),
    }
    _atomic_write_json(out_path, envelope)
    return envelope


def render_pdf_pages(pdf_path: str | Path, out_dir: Path, dpi: int = RENDER_DPI) -> list[Path]:
    """Render pages to out_dir/page-N.png via pdftoppm -r <dpi> -png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    result = subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", str(pdf_path), str(prefix)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "pdftoppm failed: " + (result.stderr.strip() or f"exit {result.returncode}")
        )
    pages = []
    for png in out_dir.glob("page-*.png"):
        pages.append((int(png.stem.rsplit("-", 1)[-1]), png))
    pages.sort()
    if not pages:
        raise RuntimeError("pdftoppm produced no page images")
    return [png for _, png in pages]


def emit_review_bundle(
    sample: list[dict],
    data_dir: str | Path = "data",
    *,
    render=True,
) -> list[str]:
    """Write audit-review/<year>/<doc_id>/{page-*.png, record.json}."""
    problems = []
    bundle_root = staging_root(data_dir) / "audit-review"
    for item in sample:
        year, doc_id = int(item["year"]), str(item["doc_id"])
        record_path = staging_root(data_dir) / str(year) / f"{doc_id}.json"
        doc_dir = bundle_root / str(year) / doc_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
        doc_dir.mkdir(parents=True)
        if record_path.exists():
            shutil.copy2(record_path, doc_dir / "record.json")
        else:
            problems.append(f"{doc_id}: staging record missing")
        pdf_path = Path(data_dir) / str(year) / "pdfs" / f"{doc_id}.pdf"
        if not render:
            continue
        try:
            render_pdf_pages(pdf_path, doc_dir / "pages")
        except Exception as exc:
            problems.append(f"{doc_id}: {exc}")
    return problems


def cmd_sample(args) -> int:
    records = load_staging_records(args.data_dir)
    sample = build_sample(records, args.per_year, args.seed)
    root = staging_root(args.data_dir)
    out_path = root / f"audit-sample-{args.seed}.json"
    envelope = write_sample_file(
        records,
        sample,
        seed=args.seed,
        per_year=args.per_year,
        out_path=out_path,
    )
    problems = (
        emit_review_bundle(sample, args.data_dir)
        if not args.no_bundle
        else []
    )
    flagged_in_lot = sum(
        1
        for item in envelope["lot"]
        if item.get("asset_review")
    )
    print(
        f"Sampled {len(sample)} docs -> {out_path} "
        f"(frozen lot: staged-record sha256 + model/prompt/parser at sample "
        f"time; the report refuses docs whose record changed after sampling)"
    )
    if flagged_in_lot:
        print(
            f"  asset_review docs drawn first: {flagged_in_lot} "
            "(flagged at staging: likely asset-text normalization damage)"
        )
    for item in envelope["lot"]:
        print(f"  {item['year']}/{item['doc_id']} sha={item['staged_record_sha256'][:12]}")
    for problem in problems:
        print(f"  bundle problem: {problem}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------


def record_verdict(
    data_dir: str | Path,
    doc_id: str,
    verdict: str,
    notes: str = "",
    severity: str = "critical",
) -> dict:
    """APPEND an audit result to the history and stamp the staging record.

    The results file is an append-only history: a later pass never erases
    an earlier fail (the report takes the WORST verdict per doc). The
    entry binds the verdict to the current staged record's frozen sha so
    drift introduced after the review is detectable, AND to the record's
    extraction identity (``pdf_sha256 + prompt_sha256 + parsed_at_utc``,
    finding 6): ``glm_stage promote`` requires a pass entry whose identity
    still matches the live record, so a re-stage after the audit forces a
    re-audit instead of inheriting the stale verdict.

    ``severity`` (default ``critical``) classifies a FAIL per the
    owner-approved error budget: ``cosmetic`` = asset-description TEXT
    normalization issues only, tracked separately and NOT blocking
    promotion. It is stored on the entry (and stamped as
    ``audit_severity``) so the report can split the budget math.
    """
    if verdict not in ("pass", "fail"):
        raise ValueError("verdict must be 'pass' or 'fail'")
    if severity not in VALID_SEVERITIES:
        raise ValueError("severity must be 'critical' or 'cosmetic'")
    root = staging_root(data_dir)

    found_year = None
    stable_sha = None
    identity = {field: None for field in ("pdf_sha256", "prompt_sha256", "parsed_at_utc")}
    for year_dir in sorted(root.glob("[0-9]" * 4)):
        path = year_dir / f"{doc_id}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            stable_sha = staged_record_sha256(record)
            identity = {
                "pdf_sha256": record.get("pdf_sha256"),
                "prompt_sha256": record.get("prompt_sha256"),
                "parsed_at_utc": record.get("parsed_at_utc"),
            }
            record["audit"] = verdict
            record["audit_severity"] = severity
            record["audit_notes"] = notes
            _atomic_write_json(path, record)
            found_year = int(year_dir.name)
            break
    audited_at = utc_now_iso()
    entry = {
        "doc_id": str(doc_id),
        "verdict": verdict,
        "severity": severity,
        "notes": notes,
        "audited_at_utc": audited_at,
        "staged_record_sha256": stable_sha,
        "pdf_sha256": identity["pdf_sha256"],
        "prompt_sha256": identity["prompt_sha256"],
        "parsed_at_utc": identity["parsed_at_utc"],
    }
    results_path = root / AUDIT_RESULTS_NAME
    results = []
    if results_path.exists():
        try:
            loaded = json.loads(results_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                results = loaded
        except (OSError, json.JSONDecodeError):
            results = []
    results.append(entry)
    _atomic_write_json(results_path, results)

    return {"entry": entry, "staging_year": found_year}


def cmd_record(args) -> int:
    try:
        result = record_verdict(
            args.data_dir, args.doc_id, args.verdict, args.notes, args.severity
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    entry = result["entry"]
    where = (
        f"staging record {result['staging_year']}/{entry['doc_id']}.json stamped"
        if result["staging_year"] is not None
        else "no staging record found (results file updated only)"
    )
    print(
        f"Recorded {entry['verdict']} "
        f"(severity: {entry['severity']}) for {entry['doc_id']} ({where})"
    )
    if entry["verdict"] == "fail" and entry["severity"] == "cosmetic":
        print(
            "Cosmetic fail: tracked separately; does NOT block promotion "
            "(owner-approved error budget)."
        )
    print(
        "History is append-only: an earlier fail remains counted for the "
        "population even after a later pass (doc marked repaired)."
    )
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def load_audit_results(data_dir: str | Path = "data") -> list[dict]:
    path = staging_root(data_dir) / AUDIT_RESULTS_NAME
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return loaded if isinstance(loaded, list) else []


def load_sample_lot(
    data_dir: str | Path = "data", path: str | Path | None = None
) -> tuple[dict | None, str | None]:
    """Load the frozen sample lot (newest audit-sample-*.json by default).

    Returns (envelope, None) or (None, refusal reason).
    """
    root = staging_root(data_dir)
    if path is None:
        candidates = sorted(root.glob("audit-sample-*.json"))
        if not candidates:
            return None, "no frozen sample lot (run `sample` first)"
        path = candidates[-1]
    path = Path(path)
    if not path.exists():
        return None, f"sample file not found: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable sample file {path}: {exc}"
    if not (
        isinstance(payload, dict)
        and payload.get("schema") == SAMPLE_SCHEMA
        and isinstance(payload.get("lot"), list)
    ):
        return None, (
            f"{path.name} predates frozen-lot format ({SAMPLE_SCHEMA}); "
            "re-run `sample`"
        )
    return payload, None


def load_promoted_doc_ids(data_dir: str | Path = "data") -> set[str]:
    path = staging_root(data_dir) / "promoted.jsonl"
    if not path.exists():
        return set()
    promoted = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            promoted.add(str(json.loads(line)["doc_id"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return promoted


def load_ground_truth_rows(ground_truth_dir: str | Path, doc_id: str) -> list | None:
    """Rows from <ground-truth-dir>/<doc_id>.json (list or {"rows": [...]})."""
    path = Path(ground_truth_dir) / f"{doc_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("rows", []))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"unsupported ground-truth format in {path}")


def _normalize_row(row) -> tuple[str, str, str, str, str]:
    """Normalized (asset, type, tx_date, notif_date, amount) tuple."""
    if isinstance(row, dict):
        values = [
            row.get("asset"),
            row.get("type"),
            row.get("tx_date", row.get("date")),
            row.get("notif_date"),
            row.get("amount", row.get("amount_letter")),
        ]
    elif isinstance(row, (list, tuple)):
        padded = list(row) + [None] * (5 - len(row))
        values = padded[:5]
    else:
        values = [row, None, None, None, None]
    return tuple(
        "" if value is None else str(value).strip().casefold() for value in values
    )


def _verdict_of(entry) -> str | None:
    """Explicit 'pass'/'fail' verdict, else None (missing/invalid/None)."""
    if not isinstance(entry, dict):
        return None
    verdict = entry.get("verdict")
    if isinstance(verdict, str) and verdict.lower() in ("pass", "fail"):
        return verdict.lower()
    return None


def _severity_of(entry) -> str | None:
    """Explicit 'critical'/'cosmetic' severity, else None (legacy entry).

    Entries recorded before ``record --severity`` existed carry no
    severity field; they resolve conservatively (critical) in
    :func:`_doc_fail_severity` unless a later explicit classification
    exists for the doc.
    """
    if not isinstance(entry, dict):
        return None
    severity = entry.get("severity")
    if isinstance(severity, str) and severity.lower() in VALID_SEVERITIES:
        return severity.lower()
    return None


def _doc_fail_severity(fail_entries: list) -> tuple[str, bool]:
    """(worst severity, legacy-unclassified?) across a doc's fail entries.

    Owner-approved error budget: critical dominates — a doc with any
    explicitly-critical fail entry stays critical, so re-recording can
    never launder a critical fail into a cosmetic one. Entries with no
    explicit severity count as critical ONLY when no fail entry for the
    doc carries an explicit severity (the conservative default); a later
    ``record --severity cosmetic`` reclassification then governs — that
    is the operator's call, never this tool's.
    """
    explicit = [
        severity
        for entry in fail_entries
        if (severity := _severity_of(entry)) is not None
    ]
    if any(severity == "critical" for severity in explicit):
        return "critical", False
    if "cosmetic" in explicit:
        return "cosmetic", False
    return "critical", True


def _weighted_pooled_rate(
    strata: list[dict], frame_counts: dict, rate_key: str = "rate"
) -> tuple[float, str]:
    """Population-weighted pooled rate; weights = staged counts per year.

    ``rate_key`` selects which per-stratum rate is pooled (the report
    pools the CRITICAL rate — the budget-relevant one).
    """
    weights = {}
    for key, count in frame_counts.items():
        try:
            weights[int(key)] = float(count)
        except (TypeError, ValueError):
            continue
    numerator = 0.0
    denominator = 0.0
    for stratum in strata:
        if stratum.get(rate_key) is None:
            continue
        weight = weights.get(int(stratum["year"]))
        if weight is None:
            continue
        numerator += weight * stratum[rate_key]
        denominator += weight
    if denominator <= 0.0:
        rates = [s[rate_key] for s in strata if s.get(rate_key) is not None]
        if not rates:
            return float("nan"), (
                "unavailable (no resolved strata and no population weights)"
            )
        return sum(rates) / len(rates), (
            "unweighted-stratum mean (population weights unavailable)"
        )
    return numerator / denominator, (
        "population-weighted (stratum weights = staged counts per year, "
        "frozen at sample time)"
    )


def _completion_guidance(reason: str) -> str:
    """One actionable repair for each condition that prevents completion."""
    reason = reason.lower()
    if "no verdict recorded" in reason or "invalid/missing verdict" in reason:
        return (
            "record an explicit verdict with audit_sample.py record --doc-id "
            "<id> --verdict pass|fail"
        )
    if "recipe/staged-record changed" in reason:
        return "restore the sampled record or re-freeze and re-audit the changed record"
    if "staging record missing" in reason:
        return "run report before promote; restore/re-stage this record and re-audit it"
    if "no ground-truth file" in reason:
        return (
            "add <ground-truth-dir>/<id>.json, or omit --ground-truth-dir for an "
            "explicit pixel-only certification"
        )
    if "staged rows unavailable" in reason:
        return (
            "restore staged rows and report before promote, or omit "
            "--ground-truth-dir for pixel-only certification"
        )
    if "current staged records not supplied" in reason:
        return "rerun report with the live staged records available"
    if "frozen sample lot" in reason or "frozen lot" in reason or "sample file" in reason:
        return "create/replace the lot with audit_sample.py sample --per-year N --seed S"
    return "resolve this condition, then rerun audit_sample.py report; do not promote"


def build_report(
    results: list[dict],
    staged_rows_by_doc: dict[str, list],
    promoted_doc_ids: set[str],
    *,
    ground_truth_dir: str | Path | None = None,
    max_rate: float = DEFAULT_MAX_RATE,
    confidence: float = DEFAULT_CONFIDENCE,
    lot: list[dict] | None = None,
    staged_records_by_doc: dict[str, dict] | None = None,
    frame_counts_by_year: dict | None = None,
) -> dict:
    """Fail-closed material-error accounting over the audit history.

    Verdict semantics (finding 9): the audit history is append-only and
    the WORST verdict per doc wins. Any recorded ``fail`` is STICKY — a
    later pass marks the doc repaired (``repaired_count``) but the
    failure stays in numerator and denominator; a sticky fail is resolved
    evidence. Missing/invalid/None verdicts make a doc UNRESOLVED. A
    pass-only doc with a frozen lot must still have a current staged record
    whose sha matches the frozen sha. If ``ground_truth_dir`` is supplied,
    the ground-truth file and staged rows must also be available and their
    normalized row MULTISETS (asset, type, tx_date, notif_date, amount)
    equal. Ground truth is optional: without it, passes are pixel-only.
    Promotion is not required for a pass to resolve, but a promoted doc
    whose staging record is gone is UNRESOLVED.

    Owner-approved error budget: ``budget_met``, ``rate`` and
    ``upper_bound`` are computed over CRITICAL fails only
    (transaction-critical fields: transaction_type, transaction_date,
    ticker/asset identity, amount_letter). Cosmetic fails (asset-
    description TEXT normalization differences, recorded with
    ``--severity cosmetic``) stay in ``failures``/``failure_count`` and
    are reported via ``cosmetic_failures``/``cosmetic_failure_count``
    but never block promotion. Omission/invention findings are critical.
    Fail entries recorded before ``--severity`` existed count critical
    (conservative) unless an explicit reclassification exists for the
    doc; such docs are listed in ``legacy_unclassified_failures`` so the
    operator can re-record them. Critical dominates: an explicit
    critical fail is never downgraded by a later cosmetic re-record.

    Completion contract: with a frozen lot, ``state`` is "complete" iff the
    lot is non-empty and every lot doc resolves. A sticky fail resolves a
    doc without further evidence checks. A pass-only doc resolves only when
    its current staging record matches the frozen sha; if ground truth was
    requested, its ground-truth file and staged rows must also be available
    and their normalized row multisets must match. Missing/invalid verdicts,
    missing staged records, sha drift, requested-but-missing ground truth,
    unavailable row data, malformed/empty lots all make the state
    "incomplete" and are named in ``completion_reasons``. Passed-but-
    unpromoted docs are resolved; the report is intended to run before
    promotion. Verdicts outside the lot do not block completion and are
    excluded from the current-lot rate/bound, but sticky outside-lot fails
    remain visible in the additive all-history ``population_*`` tally.
    The statistical budget is intentionally per frozen lot: mixing prior
    out-of-sample failures into this lot's numerator while retaining its
    denominator would not be a valid bound for the frozen-lot estimand.

    ``budget_met`` is None only when incomplete; for every complete report
    it is a real bool, ``upper_bound <= max_rate`` (or False when there are
    no resolved docs to measure).
    """
    history: dict[str, list[dict]] = {}
    for entry in results or []:
        if isinstance(entry, dict) and "doc_id" in entry:
            history.setdefault(str(entry["doc_id"]), []).append(entry)

    lot_issues: list[str] = []
    if lot is not None:
        lot_by_id = {}
        for index, item in enumerate(lot):
            if not isinstance(item, dict) or not str(item.get("doc_id") or "").strip():
                lot_issues.append(
                    f"malformed frozen lot entry at index {index} (missing doc_id)"
                )
                continue
            doc_id = str(item["doc_id"])
            if doc_id in lot_by_id:
                lot_issues.append(f"duplicate doc_id {doc_id} in frozen lot")
                continue
            try:
                int(item["year"])
            except (KeyError, TypeError, ValueError):
                lot_issues.append(
                    f"malformed frozen lot entry for {doc_id} (invalid year)"
                )
                item = {**item, "year": 0}
            lot_by_id[doc_id] = item
        population = sorted(lot_by_id)
        if not population:
            lot_issues.append("frozen sample lot contains no valid documents")
    else:
        lot_by_id = {}
        population = sorted(history)
    outside_lot = sorted(set(history) - set(lot_by_id)) if lot is not None else []

    failures: list[tuple[str, str]] = []
    failure_records: list[tuple[str, str, str]] = []  # (doc_id, reason, severity)
    legacy_unclassified: list[str] = []
    unresolved: list[tuple[str, str]] = []
    clean: list[str] = []
    passed_unpromoted: list[str] = []
    repaired: list[str] = []
    omission_details: dict[str, dict] = {}
    unresolved_attempts = 0

    for doc_id in population:
        entries = history.get(doc_id, [])
        verdicts = [_verdict_of(entry) for entry in entries]
        unresolved_attempts += sum(1 for verdict in verdicts if verdict is None)

        # 1) STICKY failures first: once recorded, a fail is never erased
        #    (neither by a later pass nor by record drift afterwards).
        if "fail" in verdicts:
            fail_entries = [
                entry for entry, verdict in zip(entries, verdicts) if verdict == "fail"
            ]
            severity, legacy = _doc_fail_severity(fail_entries)
            failure_records.append((doc_id, "verdict:fail", severity))
            if legacy:
                legacy_unclassified.append(doc_id)
            last_fail = max(i for i, v in enumerate(verdicts) if v == "fail")
            if any(v == "pass" for v in verdicts[last_fail + 1 :]):
                repaired.append(doc_id)
            continue
        # 2) explicit-verdict requirement.
        if not entries:
            unresolved.append((doc_id, "no verdict recorded"))
            continue
        if any(verdict is None for verdict in verdicts):
            unresolved.append((doc_id, "invalid/missing verdict in audit history"))
            continue
        # 3) frozen-lot evidence binding (only still relevant for docs that
        #    have not already failed).
        if lot_by_id:
            current = (staged_records_by_doc or {}).get(doc_id)
            frozen_sha = lot_by_id[doc_id].get("staged_record_sha256")
            if staged_records_by_doc is None:
                unresolved.append(
                    (doc_id, "current staged records not supplied for frozen-sha check")
                )
                continue
            if current is None:
                unresolved.append(
                    (
                        doc_id,
                        "staging record missing after sampling (promotion consumes "
                        "staged rows; omission checks must run BEFORE promote)",
                    )
                )
                continue
            if frozen_sha != staged_record_sha256(current):
                unresolved.append(
                    (doc_id, "recipe/staged-record changed after sampling")
                )
                continue
        # 4) omission detection: row MULTISETS, only with ground truth.
        if ground_truth_dir is not None:
            gt_rows = load_ground_truth_rows(ground_truth_dir, doc_id)
            staged_rows = staged_rows_by_doc.get(doc_id)
            if gt_rows is None:
                unresolved.append((doc_id, "no ground-truth file for omission check"))
                continue
            if staged_rows is None:
                unresolved.append((doc_id, "staged rows unavailable for omission check"))
                continue
            gt_counts = Counter(_normalize_row(row) for row in gt_rows)
            staged_counts = Counter(_normalize_row(row) for row in staged_rows)
            missing = gt_counts - staged_counts
            invented = staged_counts - gt_counts
            if missing or invented:
                detail = {
                    "missing": [list(t) for t in sorted(missing.elements())],
                    "invented": [list(t) for t in sorted(invented.elements())],
                }
                omission_details[doc_id] = detail
                if missing and len(gt_rows) > len(staged_rows):
                    reason = (
                        f"omission: ground-truth rows {len(gt_rows)} > "
                        f"staged rows {len(staged_rows)}"
                    )
                elif missing:
                    reason = (
                        "omission: staged rows missing ground-truth rows "
                        f"{detail['missing']}"
                    )
                else:
                    reason = (
                        "invention: staged rows absent from ground truth "
                        f"{detail['invented']}"
                    )
                # Omission/invention findings are transaction-critical:
                # auto-downgrading a row diff to cosmetic is never this
                # tool's call (severity comes from the operator's record).
                failure_records.append((doc_id, reason, "critical"))
                continue
        if doc_id in promoted_doc_ids:
            clean.append(doc_id)
        else:
            passed_unpromoted.append(doc_id)

    # Owner-approved error budget: the binomial math below counts
    # CRITICAL fails only; cosmetic fails are tracked, not blocking.
    failures = [(doc_id, reason) for doc_id, reason, _ in failure_records]
    critical_failures = [
        (doc_id, reason) for doc_id, reason, s in failure_records if s == "critical"
    ]
    cosmetic_failures = [
        (doc_id, reason) for doc_id, reason, s in failure_records if s == "cosmetic"
    ]
    outside_lot_critical_failures: list[tuple[str, str]] = []
    outside_lot_cosmetic_failures: list[tuple[str, str]] = []
    outside_lot_passes: list[str] = []
    outside_lot_unresolved: list[str] = []
    for doc_id in outside_lot:
        entries = history[doc_id]
        verdicts = [_verdict_of(entry) for entry in entries]
        if "fail" in verdicts:
            fail_entries = [
                entry for entry, verdict in zip(entries, verdicts) if verdict == "fail"
            ]
            severity, _ = _doc_fail_severity(fail_entries)
            target = (
                outside_lot_critical_failures
                if severity == "critical"
                else outside_lot_cosmetic_failures
            )
            target.append((doc_id, "verdict:fail"))
        elif verdicts and all(verdict == "pass" for verdict in verdicts):
            outside_lot_passes.append(doc_id)
        else:
            outside_lot_unresolved.append(doc_id)
    population_critical_failures = critical_failures + outside_lot_critical_failures
    population_cosmetic_failures = cosmetic_failures + outside_lot_cosmetic_failures
    population_failures = population_critical_failures + population_cosmetic_failures
    failed_ids = {doc_id for doc_id, _ in failures}
    critical_ids = {doc_id for doc_id, _ in critical_failures}
    unresolved_ids = {doc_id for doc_id, _ in unresolved}
    audited = len(failures) + len(clean) + len(passed_unpromoted)
    failure_count = len(failures)
    critical_failure_count = len(critical_failures)
    rate = (critical_failure_count / audited) if audited else float("nan")
    upper = (
        binomial_upper_bound(critical_failure_count, audited, confidence)
        if audited
        else float("nan")
    )

    completion_reasons = list(lot_issues)
    completion_reasons.extend(
        f"{doc_id}: {reason}" for doc_id, reason in unresolved
    )
    complete = not completion_reasons
    if not complete:
        budget_met = None
    elif audited:
        budget_met = bool(upper <= max_rate)
    else:
        budget_met = False

    # Stratification honesty: per-stratum accounting + population-weighted
    # pooling (weights = staged counts per year, frozen at sample time).
    strata: list[dict] = []
    pooled_rate = float("nan")
    pooling_method = None
    if lot is not None:
        per_year: dict[int, list[str]] = {}
        for doc_id in population:
            per_year.setdefault(int(lot_by_id[doc_id].get("year") or 0), []).append(doc_id)
        for year in sorted(per_year):
            docs = per_year[year]
            stratum_failures = sum(1 for d in docs if d in failed_ids)
            stratum_critical = sum(1 for d in docs if d in critical_ids)
            stratum_unresolved = sum(1 for d in docs if d in unresolved_ids)
            stratum_clean = sum(1 for d in docs if d in clean)
            stratum_unpromoted = sum(1 for d in docs if d in passed_unpromoted)
            stratum_resolved = stratum_failures + stratum_clean + stratum_unpromoted
            strata.append(
                {
                    "year": year,
                    "docs": len(docs),
                    "resolved": stratum_resolved,
                    "failures": stratum_failures,
                    "critical_failures": stratum_critical,
                    "unresolved": stratum_unresolved,
                    "clean": stratum_clean,
                    "passed_unpromoted": stratum_unpromoted,
                    "rate": (
                        (stratum_failures / stratum_resolved)
                        if stratum_resolved
                        else None
                    ),
                    "critical_rate": (
                        (stratum_critical / stratum_resolved)
                        if stratum_resolved
                        else None
                    ),
                }
            )
        if strata:
            if frame_counts_by_year:
                pooled_rate, pooling_method = _weighted_pooled_rate(
                    strata, frame_counts_by_year, rate_key="critical_rate"
                )
            else:
                rates = [
                    s["critical_rate"] for s in strata if s["critical_rate"] is not None
                ]
                if rates:
                    pooled_rate = sum(rates) / len(rates)
                pooling_method = (
                    "unweighted-stratum mean (population weights unavailable)"
                )

    return {
        "state": "complete" if complete else "incomplete",
        "completion_reasons": completion_reasons,
        "completion_guidance": [
            _completion_guidance(reason) for reason in completion_reasons
        ],
        "caveat": CAVEAT,
        "lot_size": len(population),
        "audited": audited,
        "failures": failures,
        "failure_count": failure_count,
        "critical_failures": critical_failures,
        "critical_failure_count": critical_failure_count,
        "cosmetic_failures": cosmetic_failures,
        "cosmetic_failure_count": len(cosmetic_failures),
        "population_failures": population_failures,
        "population_failure_count": len(population_failures),
        "population_critical_failures": population_critical_failures,
        "population_critical_failure_count": len(population_critical_failures),
        "population_cosmetic_failures": population_cosmetic_failures,
        "population_cosmetic_failure_count": len(population_cosmetic_failures),
        "legacy_unclassified_failures": legacy_unclassified,
        "clean": clean,
        "passed_unpromoted": passed_unpromoted,
        "unresolved": unresolved,
        "unresolved_count": len(unresolved),
        "unresolved_attempts": unresolved_attempts,
        "repaired": repaired,
        "repaired_count": len(repaired),
        "outside_lot": outside_lot,
        "outside_lot_critical_failures": outside_lot_critical_failures,
        "outside_lot_critical_failure_count": len(outside_lot_critical_failures),
        "outside_lot_cosmetic_failures": outside_lot_cosmetic_failures,
        "outside_lot_cosmetic_failure_count": len(outside_lot_cosmetic_failures),
        "outside_lot_passes": outside_lot_passes,
        "outside_lot_unresolved": outside_lot_unresolved,
        "omission_check_status": (
            "configured" if ground_truth_dir is not None else "not_run"
        ),
        "omission_details": omission_details,
        "rate": rate,
        "upper_bound": upper,
        "confidence": confidence,
        "max_rate": max_rate,
        "budget_met": budget_met,
        "strata": strata,
        "pooled_rate": pooled_rate,
        "pooling_method": pooling_method,
    }


def _num_or_none(value: float) -> float | None:
    """JSON-safe number (NaN/inf -> None)."""
    if value is None:
        return None
    try:
        if math.isnan(value) or math.isinf(value):
            return None
    except TypeError:
        return None
    return value


def _incomplete_state_payload(
    reason: str,
    *,
    max_rate: float,
    sample_file: str | None,
    omission_check_status: str = "not_run",
) -> dict:
    return {
        "schema": REPORT_STATE_SCHEMA,
        "state": "incomplete",
        "reason": reason,
        "completion_reasons": [reason],
        "completion_guidance": [_completion_guidance(reason)],
        "budget_met": None,
        "lot_size": 0,
        "audited": 0,
        "failure_count": 0,
        "critical_failures": [],
        "critical_failure_count": 0,
        "cosmetic_failures": [],
        "cosmetic_failure_count": 0,
        "population_critical_failures": [],
        "population_critical_failure_count": 0,
        "population_cosmetic_failures": [],
        "population_cosmetic_failure_count": 0,
        "population_failure_count": 0,
        "unresolved_count": 0,
        "unresolved": [],
        "repaired_count": 0,
        "rate": None,
        "upper_bound": None,
        "confidence": DEFAULT_CONFIDENCE,
        "max_rate": max_rate,
        "pooled_rate": None,
        "pooling_method": None,
        "sample_file": sample_file,
        "outside_lot": [],
        "outside_lot_critical_failures": [],
        "outside_lot_critical_failure_count": 0,
        "outside_lot_cosmetic_failures": [],
        "outside_lot_cosmetic_failure_count": 0,
        "outside_lot_passes": [],
        "outside_lot_unresolved": [],
        "omission_check_status": omission_check_status,
        "caveat": CAVEAT,
        "generated_at_utc": utc_now_iso(),
    }


def cmd_report(args) -> int:
    root = staging_root(args.data_dir)
    envelope, lot_error = load_sample_lot(args.data_dir, args.sample_file)

    if envelope is None:
        state_payload = _incomplete_state_payload(
            lot_error,
            max_rate=args.max_rate,
            sample_file=None,
            omission_check_status=(
                "configured" if args.ground_truth_dir is not None else "not_run"
            ),
        )
        _atomic_write_json(root / REPORT_STATE_NAME, state_payload)
        print(CAVEAT)
        print(f"INCOMPLETE AUDIT: {lot_error}")
        print(f"HOW TO FIX: {_completion_guidance(lot_error)}")
        print("budget_met cannot be computed; canonical promotion must not proceed.")
        return 2

    sample_file_name = Path(args.sample_file).name if args.sample_file else (
        sorted((root).glob("audit-sample-*.json"))[-1].name
    )
    records = load_staging_records(args.data_dir)
    staged_records_by_doc = {str(record["doc_id"]): record for _, record in records}
    staged_rows_by_doc = {
        doc_id: list(record.get("rows") or [])
        for doc_id, record in staged_records_by_doc.items()
    }
    promoted = load_promoted_doc_ids(args.data_dir)
    report = build_report(
        load_audit_results(args.data_dir),
        staged_rows_by_doc,
        promoted,
        ground_truth_dir=args.ground_truth_dir,
        max_rate=args.max_rate,
        lot=envelope["lot"],
        staged_records_by_doc=staged_records_by_doc,
        frame_counts_by_year=envelope.get("frame_counts_by_year"),
    )

    pct = lambda value: f"{value * 100:.4g}%"  # noqa: E731

    print(report["caveat"])
    print(
        f"Sample lot: {sample_file_name} "
        f"(seed {envelope.get('seed')}, {report['lot_size']} docs, frozen at "
        f"{envelope.get('created_at_utc')})"
    )
    print(PROMOTE_ORDER_NOTE)
    if args.ground_truth_dir is None:
        print(
            "Omission check not run (--ground-truth-dir not provided); "
            "this is a pixel-only certification, not a row-completeness check. "
            "Use --ground-truth-dir to add row-multiset omission/invention checking."
        )
    print(f"Audited (resolved) docs:   {report['audited']} of lot {report['lot_size']}")
    rate_text = pct(report["rate"]) if report["audited"] else "n/a"
    print(
        f"Critical fails: {report['critical_failure_count']} (rate {rate_text}) "
        f"| cosmetic fails: {report['cosmetic_failure_count']}"
    )
    print(
        "  (owner budget: transaction-critical fields must be error-free; "
        "asset-description text diffs are cosmetic)"
    )
    print(
        f"  (failures are sticky; repaired after fail: {report['repaired_count']})"
    )
    for doc_id, reason in report["critical_failures"]:
        print(f"  FAIL {doc_id}: {reason}")
        if doc_id in report["repaired"]:
            print(
                f"       {doc_id}: a later pass was recorded (doc repaired) but "
                "the original failure remains counted"
            )
    for doc_id, reason in report["cosmetic_failures"]:
        print(f"  COSMETIC-FAIL {doc_id}: {reason} (tracked; not blocking)")
        if doc_id in report["repaired"]:
            print(
                f"       {doc_id}: a later pass was recorded (doc repaired) but "
                "the original failure remains counted"
            )
    if report["legacy_unclassified_failures"]:
        print(
            f"NOTE: {len(report['legacy_unclassified_failures'])} fail(s) predate "
            "--severity and carry no explicit severity; counted CRITICAL "
            "(conservative). If a fail is asset-description-text only "
            "(cosmetic), re-record it: record --doc-id <id> --verdict fail "
            "--severity cosmetic"
        )
        for doc_id in report["legacy_unclassified_failures"]:
            print(f"  LEGACY-UNCLASSIFIED {doc_id}")
    print(f"Clean (pass+promoted):     {len(report['clean'])}")
    print(f"Passed but not promoted:   {len(report['passed_unpromoted'])}")
    if report["state"] != "complete":
        reason_text = "; ".join(report["completion_reasons"])
        prefix = (
            f"{report['unresolved_count']} unresolved sampled docs"
            if report["unresolved_count"]
            else reason_text
        )
        print(
            f"INCOMPLETE AUDIT: {prefix}; budget_met cannot be computed and "
            "the full-lot budget cannot be certified "
            "(fail-closed, never clean)"
        )
        for reason, guidance in zip(
            report["completion_reasons"], report["completion_guidance"]
        ):
            print(f"  INCOMPLETE-REASON {reason}; HOW TO FIX: {guidance}")
    if report["unresolved"]:
        for doc_id, reason in report["unresolved"]:
            print(f"  UNRESOLVED {doc_id}: {reason}")
    if report["outside_lot"]:
        print(
            f"WARN: {len(report['outside_lot'])} OUTSIDE-LOT doc(s) have historical "
            "verdicts; they do not block current-lot completion. Population sticky "
            f"tally: {report['population_critical_failure_count']} critical and "
            f"{report['population_cosmetic_failure_count']} cosmetic failure(s) "
            "across history, including the outside-lot contribution of "
            f"{report['outside_lot_critical_failure_count']} critical and "
            f"{report['outside_lot_cosmetic_failure_count']} cosmetic. Current-lot "
            f"rate/bound use n={report['audited']} and exclude outside-lot docs."
        )
        outside_critical = {
            doc_id for doc_id, _ in report["outside_lot_critical_failures"]
        }
        outside_cosmetic = {
            doc_id for doc_id, _ in report["outside_lot_cosmetic_failures"]
        }
        outside_passes = set(report["outside_lot_passes"])
        for doc_id in report["outside_lot"]:
            if doc_id in outside_critical:
                status = "fail (critical; remains in population tally)"
            elif doc_id in outside_cosmetic:
                status = "fail (cosmetic; remains in population tally)"
            elif doc_id in outside_passes:
                status = "pass-only"
            else:
                status = "unresolved/invalid history (outside this lot)"
            print(f"  OUTSIDE-LOT {doc_id}: {status}")
    for stratum in report["strata"]:
        rate_text = (
            pct(stratum["rate"]) if stratum["rate"] is not None else "n/a"
        )
        print(
            f"  stratum {stratum['year']}: docs {stratum['docs']} "
            f"resolved {stratum['resolved']} failures {stratum['failures']} "
            f"(critical {stratum['critical_failures']}) "
            f"unresolved {stratum['unresolved']} rate {rate_text}"
        )
    if report["strata"]:
        pooled = report["pooled_rate"]
        pooled_text = pct(pooled) if not math.isnan(pooled) else "n/a"
        print(f"Pooled critical rate [{report['pooling_method']}]: {pooled_text}")
    if report["state"] == "complete" and report["audited"]:
        print(f"Critical-error rate:       {pct(report['rate'])}")
        print(
            f"One-sided exact {report['confidence']:.0%} upper bound "
            f"(critical): {pct(report['upper_bound'])}"
        )
        met = report["budget_met"]
        print(
            f"Owner budget (critical-only exact upper bound <= "
            f"{pct(report['max_rate'])} at {report['confidence']:.0%} confidence): "
            f"{'MET' if met else 'NOT MET'}"
        )
        if met and report["cosmetic_failure_count"]:
            print(
                f"Cosmetic fails: {report['cosmetic_failure_count']} — "
                "tracked, do not block promotion (owner-approved budget)."
            )
        if not met:
            print(
                "Canonical promotion of the audited population is not "
                "justified by this audit; sample more documents or fix "
                "failures before promoting."
            )
    elif report["state"] == "complete":
        print("No audit results yet; record verdicts first.")

    if report["state"] != "complete":
        reason = "; ".join(report["completion_reasons"])
        state_payload = {
            "schema": REPORT_STATE_SCHEMA,
            "state": "incomplete",
            "reason": reason,
            "completion_reasons": report["completion_reasons"],
            "completion_guidance": report["completion_guidance"],
            "budget_met": None,
            "lot_size": report["lot_size"],
            "audited": report["audited"],
            "failure_count": report["failure_count"],
            "critical_failures": [
                [d, r] for d, r in report["critical_failures"]
            ],
            "critical_failure_count": report["critical_failure_count"],
            "cosmetic_failures": [
                [d, r] for d, r in report["cosmetic_failures"]
            ],
            "cosmetic_failure_count": report["cosmetic_failure_count"],
            "population_critical_failures": [
                [d, r] for d, r in report["population_critical_failures"]
            ],
            "population_critical_failure_count": report[
                "population_critical_failure_count"
            ],
            "population_cosmetic_failures": [
                [d, r] for d, r in report["population_cosmetic_failures"]
            ],
            "population_cosmetic_failure_count": report[
                "population_cosmetic_failure_count"
            ],
            "population_failure_count": report["population_failure_count"],
            "unresolved_count": report["unresolved_count"],
            "unresolved": [[d, r] for d, r in report["unresolved"]],
            "repaired_count": report["repaired_count"],
            "rate": _num_or_none(report["rate"]),
            "upper_bound": _num_or_none(report["upper_bound"]),
            "confidence": report["confidence"],
            "max_rate": report["max_rate"],
            "pooled_rate": _num_or_none(report["pooled_rate"]),
            "pooling_method": report["pooling_method"],
            "sample_file": sample_file_name,
            "outside_lot": report["outside_lot"],
            "outside_lot_critical_failures": [
                [d, r] for d, r in report["outside_lot_critical_failures"]
            ],
            "outside_lot_critical_failure_count": report[
                "outside_lot_critical_failure_count"
            ],
            "outside_lot_cosmetic_failures": [
                [d, r] for d, r in report["outside_lot_cosmetic_failures"]
            ],
            "outside_lot_cosmetic_failure_count": report[
                "outside_lot_cosmetic_failure_count"
            ],
            "outside_lot_passes": report["outside_lot_passes"],
            "outside_lot_unresolved": report["outside_lot_unresolved"],
            "omission_check_status": report["omission_check_status"],
            "caveat": CAVEAT,
            "generated_at_utc": utc_now_iso(),
        }
    else:
        state_payload = {
            "schema": REPORT_STATE_SCHEMA,
            "state": "complete",
            "reason": None,
            "completion_reasons": [],
            "completion_guidance": [],
            "budget_met": report["budget_met"],
            "lot_size": report["lot_size"],
            "audited": report["audited"],
            "failure_count": report["failure_count"],
            "critical_failures": [
                [d, r] for d, r in report["critical_failures"]
            ],
            "critical_failure_count": report["critical_failure_count"],
            "cosmetic_failures": [
                [d, r] for d, r in report["cosmetic_failures"]
            ],
            "cosmetic_failure_count": report["cosmetic_failure_count"],
            "population_critical_failures": [
                [d, r] for d, r in report["population_critical_failures"]
            ],
            "population_critical_failure_count": report[
                "population_critical_failure_count"
            ],
            "population_cosmetic_failures": [
                [d, r] for d, r in report["population_cosmetic_failures"]
            ],
            "population_cosmetic_failure_count": report[
                "population_cosmetic_failure_count"
            ],
            "population_failure_count": report["population_failure_count"],
            "unresolved_count": 0,
            "unresolved": [],
            "repaired_count": report["repaired_count"],
            "rate": _num_or_none(report["rate"]),
            "upper_bound": _num_or_none(report["upper_bound"]),
            "confidence": report["confidence"],
            "max_rate": report["max_rate"],
            "pooled_rate": _num_or_none(report["pooled_rate"]),
            "pooling_method": report["pooling_method"],
            "sample_file": sample_file_name,
            "outside_lot": report["outside_lot"],
            "outside_lot_critical_failures": [
                [d, r] for d, r in report["outside_lot_critical_failures"]
            ],
            "outside_lot_critical_failure_count": report[
                "outside_lot_critical_failure_count"
            ],
            "outside_lot_cosmetic_failures": [
                [d, r] for d, r in report["outside_lot_cosmetic_failures"]
            ],
            "outside_lot_cosmetic_failure_count": report[
                "outside_lot_cosmetic_failure_count"
            ],
            "outside_lot_passes": report["outside_lot_passes"],
            "outside_lot_unresolved": report["outside_lot_unresolved"],
            "omission_check_status": report["omission_check_status"],
            "caveat": CAVEAT,
            "generated_at_utc": utc_now_iso(),
        }
    _atomic_write_json(root / REPORT_STATE_NAME, state_payload)

    if report["state"] != "complete":
        return 2
    return 0 if report["budget_met"] else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_sample.py",
        description=(
            "Pixel audit for staged GLM OCR records: frozen stratified "
            "sampling, review bundle, append-only verdict history, and the "
            "fail-closed exact binomial material-error budget check (FP "
            "review 2026-10-18, finding 9)."
        ),
    )
    parser.add_argument("--data-dir", default="data", help="data directory (default: data)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_sample = subparsers.add_parser(
        "sample", help="stratified random sample + frozen lot + pixel review bundle"
    )
    p_sample.add_argument("--per-year", type=int, default=25)
    p_sample.add_argument("--seed", type=int, default=20261018)
    p_sample.add_argument(
        "--no-bundle", action="store_true", help="skip rendering the review bundle"
    )

    p_record = subparsers.add_parser("record", help="record one doc's audit verdict")
    p_record.add_argument("--doc-id", required=True)
    p_record.add_argument("--verdict", choices=["pass", "fail"], required=True)
    p_record.add_argument(
        "--severity",
        choices=list(VALID_SEVERITIES),
        default="critical",
        help=(
            "fail classification per the owner-approved error budget "
            "(default: critical). cosmetic = asset-description TEXT "
            "normalization issues only; tracked, not blocking promotion."
        ),
    )
    p_record.add_argument("--notes", default="")

    p_report = subparsers.add_parser(
        "report",
        help="critical-error rate + binomial budget check (fail-closed; "
        "cosmetic fails tracked, not blocking)",
    )
    p_report.add_argument(
        "--max-rate",
        type=float,
        default=DEFAULT_MAX_RATE,
        help="owner budget on the material-error rate (default 0.01)",
    )
    p_report.add_argument(
        "--ground-truth-dir",
        default=None,
        help="dir of <doc_id>.json ground-truth row files; omissions/inventions "
        "(row-multiset differences) count as failures; pass-verdict docs "
        "without ground truth count as UNRESOLVED",
    )
    p_report.add_argument(
        "--sample-file",
        default=None,
        help="frozen-lot sample file (default: newest audit-sample-*.json)",
    )

    args = parser.parse_args(argv)
    if args.command == "sample":
        return cmd_sample(args)
    if args.command == "record":
        return cmd_record(args)
    if args.command == "report":
        return cmd_report(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
