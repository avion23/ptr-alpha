#!/usr/bin/env python3
"""Pixel-audit tooling for staged GLM OCR records (FP review 2026-10-18).

Implements the acceptance measurement the FP ruling requires before any
canonical promotion: a stratified random sample of staged documents,
rendered back to pixels for independent owner review, and an exact
one-sided binomial bound on the material-error rate against an
owner-approved budget (illustrative <1% at 95% confidence).

Subcommands:

* ``sample  --per-year N --seed S``
    Stratified random sample (``random.Random(S)``) of staged successfully
    parsed records -> ``audit-sample-<seed>.json`` plus a review bundle
    ``audit-review/`` holding the rendered page PNGs and a copy of each
    staging JSON. Deterministic given the seed and staging population.
* ``record  --doc-id ID --verdict pass|fail --notes "..."`
    Append to ``audit-results.json`` and stamp the staging record's
    ``audit`` field (the field ``glm_stage.py promote
    --require-audit-pass`` gates on).
* ``report  [--max-rate 0.01] [--ground-truth-dir DIR]``
    Material-error rate (fails/audited; omissions — ground-truth rows >
    staged rows — count as failures) with the one-sided exact 95% binomial
    upper bound, and whether the owner budget is met. Only pass-verdict,
    promoted, omission-free docs count as clean.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scripts.gemini_ocr_common import pdf_sha256  # noqa: E402

AUDIT_RESULTS_NAME = "audit-results.json"
RENDER_DPI = 150
DEFAULT_MAX_RATE = 0.01
DEFAULT_CONFIDENCE = 0.95


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


def build_sample(
    records: list[tuple[int, dict]], per_year: int, seed: int
) -> list[dict]:
    """Stratified random sample of successfully parsed staged records.

    Deterministic: fresh ``random.Random(seed)``, doc ids sorted within each
    year, years visited in ascending order.
    """
    by_year: dict[int, list[str]] = {}
    for year, record in records:
        if record.get("parse_status") not in ("success", "no_txs"):
            continue
        by_year.setdefault(year, []).append(str(record["doc_id"]))
    rng = random.Random(seed)
    sample = []
    for year in sorted(by_year):
        doc_ids = sorted(set(by_year[year]))
        chosen = rng.sample(doc_ids, min(per_year, len(doc_ids)))
        sample.extend({"doc_id": doc_id, "year": year} for doc_id in sorted(chosen))
    return sample


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
    _atomic_write_json(out_path, sample)
    problems = (
        emit_review_bundle(sample, args.data_dir)
        if not args.no_bundle
        else []
    )
    print(f"Sampled {len(sample)} docs -> {out_path}")
    for item in sample:
        print(f"  {item['year']}/{item['doc_id']}")
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
) -> dict:
    """Append an audit result and stamp the staging record's audit field."""
    if verdict not in ("pass", "fail"):
        raise ValueError("verdict must be 'pass' or 'fail'")
    root = staging_root(data_dir)
    entry = {
        "doc_id": str(doc_id),
        "verdict": verdict,
        "notes": notes,
        "audited_at_utc": utc_now_iso(),
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

    found = None
    for year_dir in sorted(root.glob("[0-9]" * 4)):
        path = year_dir / f"{doc_id}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            record["audit"] = verdict
            record["audit_notes"] = notes
            record["audited_at_utc"] = entry["audited_at_utc"]
            _atomic_write_json(path, record)
            found = int(year_dir.name)
            break
    return {"entry": entry, "staging_year": found}


def cmd_record(args) -> int:
    try:
        result = record_verdict(args.data_dir, args.doc_id, args.verdict, args.notes)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    entry = result["entry"]
    where = (
        f"staging record {result['staging_year']}/{entry['doc_id']}.json stamped"
        if result["staging_year"] is not None
        else "no staging record found (results file updated only)"
    )
    print(f"Recorded {entry['verdict']} for {entry['doc_id']} ({where})")
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


def build_report(
    results: list[dict],
    staged_rows_by_doc: dict[str, list],
    promoted_doc_ids: set[str],
    *,
    ground_truth_dir: str | Path | None = None,
    max_rate: float = DEFAULT_MAX_RATE,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    """Material-error accounting over audit results (latest verdict per doc).

    A doc is a material failure when its verdict is ``fail`` OR (ground
    truth supplied) its ground-truth row count exceeds the staged row count
    (an omission). Only pass-verdict, promoted, omission-free docs count as
    clean; the denominator is every audited doc.
    """
    latest: dict[str, dict] = {}
    for entry in results:
        if not isinstance(entry, dict) or "doc_id" not in entry:
            continue
        latest[str(entry["doc_id"])] = entry  # last entry wins

    failures = []  # (doc_id, reason)
    clean = []
    passed_unpromoted = []
    for doc_id in sorted(latest):
        entry = latest[doc_id]
        verdict = str(entry.get("verdict", "")).lower()
        if verdict == "fail":
            failures.append((doc_id, "verdict:fail"))
            continue
        if ground_truth_dir is not None:
            gt_rows = load_ground_truth_rows(ground_truth_dir, doc_id)
            staged_rows = staged_rows_by_doc.get(doc_id)
            if gt_rows is not None and staged_rows is not None:
                if len(gt_rows) > len(staged_rows):
                    failures.append(
                        (
                            doc_id,
                            f"omission: ground-truth rows {len(gt_rows)} > "
                            f"staged rows {len(staged_rows)}",
                        )
                    )
                    continue
        if doc_id in promoted_doc_ids:
            clean.append(doc_id)
        else:
            passed_unpromoted.append(doc_id)

    n = len(latest)
    f = len(failures)
    upper = binomial_upper_bound(f, n, confidence) if n else float("nan")
    return {
        "audited": n,
        "failures": failures,
        "failure_count": f,
        "clean": clean,
        "passed_unpromoted": passed_unpromoted,
        "rate": (f / n) if n else float("nan"),
        "upper_bound": upper,
        "confidence": confidence,
        "max_rate": max_rate,
        "budget_met": bool(n) and upper <= max_rate,
    }


def cmd_report(args) -> int:
    results = load_audit_results(args.data_dir)
    staged_rows_by_doc = {
        str(record["doc_id"]): list(record.get("rows", []))
        for _, record in load_staging_records(args.data_dir)
    }
    promoted = load_promoted_doc_ids(args.data_dir)
    report = build_report(
        results,
        staged_rows_by_doc,
        promoted,
        ground_truth_dir=args.ground_truth_dir,
        max_rate=args.max_rate,
    )
    pct = lambda value: f"{value * 100:.4g}%"  # noqa: E731
    print(f"Audited docs:              {report['audited']}")
    print(f"Material failures:         {report['failure_count']}")
    for doc_id, reason in report["failures"]:
        print(f"  FAIL {doc_id}: {reason}")
    print(f"Clean (pass+promoted):     {len(report['clean'])}")
    print(f"Passed but not promoted:   {len(report['passed_unpromoted'])}")
    if report["audited"]:
        print(f"Material-error rate:       {pct(report['rate'])}")
        print(
            f"One-sided exact {report['confidence']:.0%} upper bound: "
            f"{pct(report['upper_bound'])}"
        )
        met = report["budget_met"]
        print(
            f"Owner budget (material-error rate < {pct(report['max_rate'])} "
            f"at {report['confidence']:.0%} confidence): "
            f"{'MET' if met else 'NOT MET'}"
        )
        if not met:
            print(
                "Canonical promotion of the audited population is not "
                "justified by this audit; sample more documents or fix "
                "failures before promoting."
            )
    else:
        print("No audit results yet; record verdicts first.")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_sample.py",
        description=(
            "Pixel audit for staged GLM OCR records: stratified sampling, "
            "review bundle, verdict recording, and the exact binomial "
            "material-error budget check (FP review 2026-10-18)."
        ),
    )
    parser.add_argument("--data-dir", default="data", help="data directory (default: data)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_sample = subparsers.add_parser(
        "sample", help="stratified random sample + pixel review bundle"
    )
    p_sample.add_argument("--per-year", type=int, default=25)
    p_sample.add_argument("--seed", type=int, default=20261018)
    p_sample.add_argument(
        "--no-bundle", action="store_true", help="skip rendering the review bundle"
    )

    p_record = subparsers.add_parser("record", help="record one doc's audit verdict")
    p_record.add_argument("--doc-id", required=True)
    p_record.add_argument("--verdict", choices=["pass", "fail"], required=True)
    p_record.add_argument("--notes", default="")

    p_report = subparsers.add_parser(
        "report", help="material-error rate + binomial budget check"
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
        help="dir of <doc_id>.json ground-truth row files; omissions (more "
        "ground-truth rows than staged rows) count as failures",
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
