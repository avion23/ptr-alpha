"""Fail-closed audit report semantics (FP review 2026-10-18, finding 9).

Reproduces the Astra probes as regression tests:

* 299 null verdicts must NOT yield ``budget_met=True`` -- the report
  refuses (state ``incomplete``, ``budget_met None``).
* fail-then-pass must NOT reduce the failure count (sticky failures;
  repaired tracked separately).
* staged-record sha drift after sampling -> UNRESOLVED.
* omission detection compares row MULTISETS, not counts.
* population-weighted pooling math on a known small case.
* determinism of sampling, freezing, and reporting.
* owner error budget: cosmetic fails (asset-description text) are
  tracked but never block promotion; ``budget_met`` counts critical
  fails only; legacy entries without severity count critical.

Every case runs in temporary directories; ``data/congress.duckdb`` is
never opened and no model calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_sample


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_record(doc_id: str, year: int = 2024, rows: list | None = None) -> dict:
    return {
        "doc_id": doc_id,
        "year": year,
        "pdf_sha256": f"sha-{doc_id}",
        "model": "glm-5.3-flash",
        "prompt_sha256": "prompt-abc",
        "parser_version": "v1-glm53-vision",
        "parsed_at_utc": "2026-10-18T00:00:00Z",
        "pages": 1,
        "member": "Jane Doe",
        "rows": rows if rows is not None else [],
        "raw_output": None,
        "parse_status": "success",
        "error": None,
        "audit": None,
    }


def write_record(data_dir: Path, record: dict) -> Path:
    path = (
        audit_sample.staging_root(data_dir)
        / str(record["year"])
        / f"{record['doc_id']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_envelope(data_dir: Path, seed: int = 1, frame_counts: dict | None = None) -> dict:
    records = audit_sample.load_staging_records(data_dir)
    sample = [
        {"doc_id": str(record["doc_id"]), "year": int(record["year"])}
        for _, record in records
    ]
    envelope = {
        "schema": audit_sample.SAMPLE_SCHEMA,
        "seed": seed,
        "per_year": len(sample),
        "created_at_utc": "2026-10-18T00:00:00Z",
        "frame_counts_by_year": frame_counts or {},
        "lot": audit_sample.freeze_lot(records, sample),
    }
    out = audit_sample.staging_root(data_dir) / f"audit-sample-{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
    return envelope


def pass_entry(doc_id: str, at: str = "t1") -> dict:
    return {"doc_id": doc_id, "verdict": "pass", "notes": "", "audited_at_utc": at}


def fail_entry(doc_id: str, at: str = "t1", notes: str = "", severity: str | None = None) -> dict:
    """A fail entry; ``severity=None`` simulates a legacy (pre---severity) entry."""
    entry = {"doc_id": doc_id, "verdict": "fail", "notes": notes, "audited_at_utc": at}
    if severity is not None:
        entry["severity"] = severity
    return entry


@pytest.fixture
def data_dir(tmp_path) -> Path:
    return tmp_path / "data"


# --------------------------------------------------------------------------
# Astra probe 1: null/missing verdicts must refuse, never count as clean
# --------------------------------------------------------------------------


def test_299_null_verdicts_refuse_report():
    results = [
        {"doc_id": f"d{i}", "verdict": None, "notes": "", "audited_at_utc": "t"}
        for i in range(299)
    ]
    report = audit_sample.build_report(results, {}, set(), ground_truth_dir=None)
    assert report["state"] == "incomplete"
    assert report["unresolved_count"] == 299
    assert report["budget_met"] is None          # can never default to clean
    assert report["failure_count"] == 0
    assert report["clean"] == []
    assert all(
        reason == "invalid/missing verdict in audit history"
        for _, reason in report["unresolved"]
    )


def test_lot_without_verdicts_refuses_report():
    lot = [{"doc_id": f"d{i}", "year": 2024, "staged_record_sha256": "x"} for i in range(299)]
    report = audit_sample.build_report(
        [],
        {},
        set(),
        lot=lot,
        staged_records_by_doc={f"d{i}": make_record(f"d{i}") for i in range(299)},
    )
    assert report["state"] == "incomplete"
    assert report["unresolved_count"] == 299
    assert report["budget_met"] is None
    assert {reason for _, reason in report["unresolved"]} == {"no verdict recorded"}


def test_invalid_verdict_string_refuses():
    report = audit_sample.build_report(
        [{"doc_id": "a", "verdict": "seems-fine", "audited_at_utc": "t"}],
        {},
        set(),
    )
    assert report["state"] == "incomplete"
    assert report["unresolved_count"] == 1
    assert report["budget_met"] is None


# --------------------------------------------------------------------------
# Astra probe 2: fail-then-pass stays a failure (repaired tracked separately)
# --------------------------------------------------------------------------


def test_fail_then_pass_is_sticky_failure_and_tracked_repaired():
    results = [fail_entry("x", "t1"), pass_entry("x", "t2")]
    report = audit_sample.build_report(results, {}, promoted_doc_ids={"x"})
    assert report["failure_count"] == 1
    assert ("x", "verdict:fail") in report["failures"]
    assert report["repaired"] == ["x"]
    assert report["repaired_count"] == 1
    assert report["audited"] == 1  # repaired doc stays in the denominator
    assert report["clean"] == []   # a repaired doc is never "clean"
    assert report["budget_met"] is False  # 1 failure in 1 -> bound 1.0 > 1%


def test_pass_then_fail_is_also_a_failure():
    results = [pass_entry("x", "t1"), fail_entry("x", "t2")]
    report = audit_sample.build_report(results, {}, promoted_doc_ids={"x"})
    assert report["failure_count"] == 1
    assert ("x", "verdict:fail") in report["failures"]
    assert report["repaired_count"] == 0


def test_record_command_appends_history_and_report_uses_worst_verdict(data_dir):
    write_record(data_dir, make_record("9001", rows=[["X", "Purchase", "01/15/24", None, "A"]]))
    audit_sample.record_verdict(data_dir, "9001", "fail", "invented row")
    audit_sample.record_verdict(data_dir, "9001", "pass", "re-checked, matches")
    results = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-results.json").read_text()
    )
    assert [entry["verdict"] for entry in results] == ["fail", "pass"]  # append-only

    records = audit_sample.load_staging_records(data_dir)
    lot = audit_sample.freeze_lot(records, [{"doc_id": "9001", "year": 2024}])
    current = dict(records[0][1])
    report = audit_sample.build_report(
        results, {"9001": current.get("rows", [])}, {"9001"}, lot=lot,
        staged_records_by_doc={"9001": current},
    )
    assert report["failure_count"] == 1
    assert report["repaired_count"] == 1
    assert report["budget_met"] is False


# --------------------------------------------------------------------------
# frozen lot: sha drift -> unresolved
# --------------------------------------------------------------------------


def test_sha_drift_after_sampling_is_unresolved():
    record = make_record("s1", rows=[["X", "Purchase", "01/15/24", None, "A"]])
    records = [(2024, record)]
    lot = audit_sample.freeze_lot(records, [{"doc_id": "s1", "year": 2024}])
    assert lot[0]["staged_record_sha256"] == audit_sample.staged_record_sha256(record)

    drifted = dict(record, model="some-other-model")  # recipe change after sampling
    report = audit_sample.build_report(
        [pass_entry("s1")],
        {"s1": drifted["rows"]},
        {"s1"},
        lot=lot,
        staged_records_by_doc={"s1": drifted},
    )
    assert report["state"] == "incomplete"
    assert report["unresolved"] == [("s1", "recipe/staged-record changed after sampling")]
    assert report["budget_met"] is None


def test_row_change_after_sampling_is_unresolved():
    record = make_record("s2", rows=[["X", "Purchase", "01/15/24", None, "A"]])
    lot = audit_sample.freeze_lot([(2024, record)], [{"doc_id": "s2", "year": 2024}])
    restaged = dict(record, rows=[["X", "Purchase", "01/16/24", None, "A"]])
    report = audit_sample.build_report(
        [pass_entry("s2")], {"s2": restaged["rows"]}, {"s2"},
        lot=lot, staged_records_by_doc={"s2": restaged},
    )
    assert ("s2", "recipe/staged-record changed after sampling") in report["unresolved"]


def test_verdict_stamping_does_not_look_like_drift():
    """Stamping audit fields (record workflow) must keep the doc resolvable."""
    record = make_record("s3", rows=[])
    lot = audit_sample.freeze_lot([(2024, record)], [{"doc_id": "s3", "year": 2024}])
    stamped = dict(record, audit="pass", audit_notes="ok", audited_at_utc="t9")
    report = audit_sample.build_report(
        [pass_entry("s3")], {"s3": []}, {"s3"},
        lot=lot, staged_records_by_doc={"s3": stamped},
    )
    assert report["state"] == "complete"
    assert report["clean"] == ["s3"]
    assert report["unresolved_count"] == 0


def test_sticky_failure_survives_sha_drift():
    """A recorded fail is never demoted to unresolved (count can't shrink)."""
    record = make_record("s4", rows=[])
    lot = audit_sample.freeze_lot([(2024, record)], [{"doc_id": "s4", "year": 2024}])
    drifted = dict(record, rows=[["Y", "Sale", "02/02/24", None, "B"]])
    report = audit_sample.build_report(
        [fail_entry("s4")], {"s4": drifted["rows"]}, set(),
        lot=lot, staged_records_by_doc={"s4": drifted},
    )
    assert report["failure_count"] == 1
    assert ("s4", "verdict:fail") in report["failures"]


def test_promoted_before_omission_check_is_unresolved():
    """Promotion consumes staged rows: no record left -> no clean pass."""
    lot = [{"doc_id": "gone", "year": 2024, "staged_record_sha256": "x"}]
    report = audit_sample.build_report(
        [pass_entry("gone")], {}, {"gone"}, lot=lot, staged_records_by_doc={}
    )
    assert report["state"] == "incomplete"
    assert report["unresolved_count"] == 1
    assert "staging record missing" in report["unresolved"][0][1]
    assert report["budget_met"] is None


def test_lot_entry_freezes_recipe_fields():
    record = make_record("s5")
    lot = audit_sample.freeze_lot([(2024, record)], [{"doc_id": "s5", "year": 2024}])
    assert lot[0]["model"] == "glm-5.3-flash"
    assert lot[0]["prompt_sha256"] == "prompt-abc"
    assert lot[0]["parser_version"] == "v1-glm53-vision"
    assert lot[0]["staged_record_sha256"] == audit_sample.staged_record_sha256(record)


# --------------------------------------------------------------------------
# omission detection: row multisets, not counts
# --------------------------------------------------------------------------


def test_multiset_equal_count_content_diff_is_omission():
    gt_rows = [
        ["A", "Purchase", "01/15/24", None, "A"],
        ["B", "Sale", "02/02/24", None, "B"],
    ]
    staged_rows = [
        ["A", "Purchase", "01/15/24", None, "A"],
        ["B", "Purchase", "02/02/24", None, "B"],  # type flipped; counts equal
    ]
    # Ground truth must come from a file; write it.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m1.json").write_text(json.dumps({"rows": gt_rows}))
        report = audit_sample.build_report(
            [pass_entry("m1")], {"m1": staged_rows}, {"m1"},
            ground_truth_dir=td,
        )
    assert report["failure_count"] == 1
    reason = report["failures"][0][1]
    assert reason.startswith("omission: staged rows missing ground-truth rows")
    assert report["omission_details"]["m1"]["missing"] == [["b", "sale", "02/02/24", "", "b"]]
    assert report["budget_met"] is False


def test_multiset_order_insensitive_match_is_clean():
    staged = [
        ["A", "Purchase", "01/15/24", None, "A"],
        ["B", "Sale", "02/02/24", None, "B"],
    ]
    gt = list(reversed(staged))  # same multiset, different order
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m2.json").write_text(json.dumps({"rows": gt}))
        report = audit_sample.build_report(
            [pass_entry("m2")], {"m2": staged}, {"m2"}, ground_truth_dir=td
        )
    assert report["state"] == "complete"
    assert report["clean"] == ["m2"]
    assert report["budget_met"] is False  # 0 failures but n=1: bound 95% > 1%


def test_invented_rows_are_material_failures():
    staged = [["A", "Purchase", "01/15/24", None, "A"], ["Z", "Sale", "03/03/24", None, "C"]]
    gt = [["A", "Purchase", "01/15/24", None, "A"]]
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m3.json").write_text(json.dumps({"rows": gt}))
        report = audit_sample.build_report(
            [pass_entry("m3")], {"m3": staged}, {"m3"}, ground_truth_dir=td
        )
    assert report["failure_count"] == 1
    assert report["failures"][0][1].startswith("invention: staged rows absent from ground truth")


def test_count_omission_keeps_legacy_reason_string():
    """gt 2 rows vs staged 1 row keeps the human-readable count message."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m4.json").write_text(
            json.dumps(
                {
                    "rows": [
                        ["A", "Purchase", "01/15/24", None, "A"],
                        ["A2", "Sale", "01/16/24", None, "A"],
                    ]
                }
            )
        )
        report = audit_sample.build_report(
            [pass_entry("m4")],
            {"m4": [["A", "Purchase", "01/15/24", None, "A"]]},
            {"m4"},
            ground_truth_dir=td,
        )
    assert ("m4", "omission: ground-truth rows 2 > staged rows 1") in report["failures"]


def test_pass_without_ground_truth_is_unresolved_not_clean():
    import tempfile

    with tempfile.TemporaryDirectory() as td:  # empty ground-truth dir
        report = audit_sample.build_report(
            [pass_entry("m5")], {"m5": []}, {"m5"}, ground_truth_dir=td
        )
    assert report["state"] == "incomplete"
    assert report["unresolved"] == [("m5", "no ground-truth file for omission check")]
    assert report["budget_met"] is None


def test_fail_verdict_with_no_ground_truth_stays_failure():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        report = audit_sample.build_report(
            [fail_entry("m6")], {"m6": []}, set(), ground_truth_dir=td
        )
    assert report["failure_count"] == 1
    assert ("m6", "verdict:fail") in report["failures"]


# --------------------------------------------------------------------------
# stratification honesty: population-weighted pooling
# --------------------------------------------------------------------------


def _pooling_fixture():
    records = []
    for i in range(10):
        records.append((2023, make_record(f"t{i}", year=2023)))
    for i in range(2):
        records.append((2024, make_record(f"u{i}", year=2024)))
    results = [pass_entry(f"t{i}") for i in range(10)]
    results += [fail_entry("u0"), pass_entry("u1")]
    promoted = {f"t{i}" for i in range(10)} | {"u1"}
    return records, results, promoted


def test_population_weighted_pooling_known_case():
    records, results, promoted = _pooling_fixture()
    lot = audit_sample.freeze_lot(records, [
        {"doc_id": record["doc_id"], "year": year} for year, record in records
    ])
    staged_by_doc = {record["doc_id"]: record["rows"] for _, record in records}
    report = audit_sample.build_report(
        results, staged_by_doc, promoted,
        lot=lot,
        staged_records_by_doc={record["doc_id"]: record for _, record in records},
        frame_counts_by_year={"2023": 90, "2024": 10},
    )
    assert report["state"] == "complete"
    strata = {s["year"]: s for s in report["strata"]}
    assert strata[2023]["docs"] == 10 and strata[2023]["failures"] == 0
    assert strata[2024]["docs"] == 2 and strata[2024]["failures"] == 1
    assert strata[2023]["rate"] == 0.0
    assert strata[2024]["rate"] == pytest.approx(0.5)
    # (90 * 0.0 + 10 * 0.5) / 100 = 0.05 -- NOT the unweighted mean 0.25.
    assert report["pooled_rate"] == pytest.approx(0.05)
    assert report["pooling_method"].startswith("population-weighted")
    assert "staged counts per year" in report["pooling_method"]


def test_unweighted_fallback_is_labelled():
    records, results, promoted = _pooling_fixture()
    lot = audit_sample.freeze_lot(records, [
        {"doc_id": record["doc_id"], "year": year} for year, record in records
    ])
    report = audit_sample.build_report(
        results,
        {record["doc_id"]: record["rows"] for _, record in records},
        promoted,
        lot=lot,
        staged_records_by_doc={record["doc_id"]: record for _, record in records},
        frame_counts_by_year=None,
    )
    assert report["pooled_rate"] == pytest.approx(0.25)  # mean(0.0, 0.5)
    assert "unweighted-stratum mean" in report["pooling_method"]


# --------------------------------------------------------------------------
# outside-lot verdicts and sample-file contract
# --------------------------------------------------------------------------


def test_verdict_outside_frozen_lot_refuses():
    lot = [{"doc_id": "a", "year": 2024, "staged_record_sha256": "x"}]
    report = audit_sample.build_report(
        [pass_entry("a"), fail_entry("z")],
        {}, set(),
        lot=lot,
        staged_records_by_doc={"a": make_record("a")},
    )
    assert report["state"] == "incomplete"
    assert report["outside_lot"] == ["z"]
    assert report["budget_met"] is None


def test_load_sample_lot_refuses_legacy_bare_list(data_dir):
    root = audit_sample.staging_root(data_dir)
    root.mkdir(parents=True)
    (root / "audit-sample-7.json").write_text(json.dumps([{"doc_id": "a", "year": 2024}]))
    envelope, error = audit_sample.load_sample_lot(data_dir)
    assert envelope is None
    assert "frozen-lot format" in error


def test_load_sample_lot_picks_newest(data_dir):
    write_record(data_dir, make_record("a"))
    write_envelope(data_dir, seed=1)
    envelope, error = audit_sample.load_sample_lot(data_dir)
    assert error is None
    assert envelope["schema"] == audit_sample.SAMPLE_SCHEMA
    assert [entry["doc_id"] for entry in envelope["lot"]] == ["a"]


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_report_is_deterministic():
    records, results, promoted = _pooling_fixture()
    lot = audit_sample.freeze_lot(records, [
        {"doc_id": record["doc_id"], "year": year} for year, record in records
    ])
    kwargs = dict(
        lot=lot,
        staged_records_by_doc={record["doc_id"]: record for _, record in records},
        frame_counts_by_year={"2023": 90, "2024": 10},
    )
    one = audit_sample.build_report(results, {r["doc_id"]: r["rows"] for _, r in records}, promoted, **kwargs)
    two = audit_sample.build_report(results, {r["doc_id"]: r["rows"] for _, r in records}, promoted, **kwargs)
    assert one == two


def test_freeze_lot_is_deterministic(data_dir):
    write_record(data_dir, make_record("d1", year=2023))
    write_record(data_dir, make_record("d2", year=2024))
    records = audit_sample.load_staging_records(data_dir)
    sample = audit_sample.build_sample(records, per_year=1, seed=7)
    assert audit_sample.freeze_lot(records, sample) == audit_sample.freeze_lot(records, sample)
    assert audit_sample.build_sample(records, per_year=1, seed=7) == audit_sample.build_sample(
        records, per_year=1, seed=7
    )


# --------------------------------------------------------------------------
# CLI: refusal exit codes + audit-report-state.json contract
# --------------------------------------------------------------------------


def test_cli_report_first_line_is_caveat_and_budget_not_met_exits_1(data_dir, capsys):
    record = make_record("p1", rows=[["X", "Purchase", "01/15/24", None, "A"]])
    write_record(data_dir, record)
    write_envelope(data_dir, seed=3)
    root = audit_sample.staging_root(data_dir)
    (root / "promoted.jsonl").write_text('{"doc_id": "p1", "inserted": 1}\n')
    gt = data_dir / "gt"
    gt.mkdir()
    (gt / "p1.json").write_text(json.dumps({"rows": record["rows"]}))
    # The reviewer records an explicit pass (stamps the staging record;
    # frozen sha is unaffected by audit-mutable fields).
    audit_sample.record_verdict(data_dir, "p1", "pass", "matches pixels")

    rc = audit_sample.main(
        ["--data-dir", str(data_dir), "report", "--ground-truth-dir", str(gt)]
    )
    out = capsys.readouterr().out
    assert rc == 1  # complete, but 0/1 failures -> 95% upper bound 95% > 1%
    assert out.splitlines()[0] == audit_sample.CAVEAT
    assert "Promotion consumes staged rows" in out or "BEFORE `promote`" in out
    assert "NOT MET" in out
    state = json.loads((root / "audit-report-state.json").read_text())
    assert state["state"] == "complete"
    assert state["budget_met"] is False


def test_cli_report_null_verdicts_refuse_exit_2(data_dir, capsys):
    for i in range(3):
        write_record(data_dir, make_record(f"n{i}"))
    write_envelope(data_dir, seed=4)
    root = audit_sample.staging_root(data_dir)
    # Astra's probe: null verdicts in the results file.
    (root / "audit-results.json").write_text(
        json.dumps(
            [
                {"doc_id": f"n{i}", "verdict": None, "notes": "", "audited_at_utc": "t"}
                for i in range(3)
            ]
        )
    )
    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "INCOMPLETE AUDIT: 3 unresolved" in out
    assert "budget_met cannot be computed" in out
    state = json.loads((root / "audit-report-state.json").read_text())
    assert state["state"] == "incomplete"
    assert state["budget_met"] is None


def test_cli_report_without_sample_lot_refuses(data_dir, capsys):
    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "no frozen sample lot" in out
    state = json.loads((audit_sample.staging_root(data_dir) / "audit-report-state.json").read_text())
    assert state["state"] == "incomplete"
    assert state["budget_met"] is None


def test_cli_report_sha_drift_refuses(data_dir, capsys):
    record = make_record("p9", rows=[["X", "Purchase", "01/15/24", None, "A"]])
    write_record(data_dir, record)
    write_envelope(data_dir, seed=5)
    # Recipe change after sampling: re-stage with different rows.
    write_record(data_dir, dict(record, rows=[["X", "Purchase", "01/16/24", None, "A"]]))
    root = audit_sample.staging_root(data_dir)
    (root / "audit-results.json").write_text(json.dumps([pass_entry("p9")]))
    (root / "promoted.jsonl").write_text('{"doc_id": "p9"}\n')

    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "recipe/staged-record changed after sampling" in out
    state = json.loads((root / "audit-report-state.json").read_text())
    assert state["state"] == "incomplete"
    assert state["budget_met"] is None
    assert state["unresolved_count"] == 1


# --------------------------------------------------------------------------
# finding 6: verdicts bind to the audited record's extraction identity
# --------------------------------------------------------------------------


def test_record_verdict_binds_record_identity(data_dir):
    record = make_record("id1", rows=[["X", "Purchase", "01/15/24", None, "A"]])
    write_record(data_dir, record)
    audit_sample.record_verdict(data_dir, "id1", "pass", "matches pixels")
    (entry,) = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-results.json").read_text()
    )
    # glm_stage.promote requires this triple to still match the live record.
    assert entry["pdf_sha256"] == record["pdf_sha256"]
    assert entry["prompt_sha256"] == record["prompt_sha256"]
    assert entry["parsed_at_utc"] == record["parsed_at_utc"]


def test_verdict_without_record_leaves_identity_unbound(data_dir):
    entry = audit_sample.record_verdict(data_dir, "ghost", "fail", "no record")[
        "entry"
    ]
    assert entry["staged_record_sha256"] is None
    assert entry["pdf_sha256"] is None
    assert entry["prompt_sha256"] is None
    assert entry["parsed_at_utc"] is None


def test_identity_bound_entries_keep_report_state_semantics(data_dir):
    """The finding-9 contract must keep passing with identity-bound entries."""
    record = make_record("id2", rows=[["X", "Purchase", "01/15/24", None, "A"]])
    write_record(data_dir, record)
    records = audit_sample.load_staging_records(data_dir)
    lot = audit_sample.freeze_lot(records, [{"doc_id": "id2", "year": 2024}])
    audit_sample.record_verdict(data_dir, "id2", "pass", "ok")
    records_after = audit_sample.load_staging_records(data_dir)
    report = audit_sample.build_report(
        audit_sample.load_audit_results(data_dir),
        {r["doc_id"]: r.get("rows", []) for _, r in records_after},
        {"id2"},
        lot=lot,
        staged_records_by_doc={r["doc_id"]: r for _, r in records_after},
    )
    assert report["state"] == "complete"
    assert report["clean"] == ["id2"]
    assert report["unresolved_count"] == 0


# --------------------------------------------------------------------------
# owner error budget: transaction-critical vs cosmetic fails
# --------------------------------------------------------------------------


def test_record_severity_defaults_critical_and_validates(data_dir):
    write_record(data_dir, make_record("r1"))
    write_record(data_dir, make_record("r2"))
    audit_sample.record_verdict(data_dir, "r1", "fail", "wrong tx_date")  # default
    audit_sample.record_verdict(
        data_dir, "r2", "fail", "issuer wording dropped", severity="cosmetic"
    )
    entries = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-results.json").read_text()
    )
    assert entries[0]["severity"] == "critical"  # backward-compatible default
    assert entries[1]["severity"] == "cosmetic"
    with pytest.raises(ValueError):
        audit_sample.record_verdict(data_dir, "r1", "fail", severity="minor")

    # The severity stamp is audit-mutable: it must never read as recipe drift.
    stamped = json.loads(
        (audit_sample.staging_root(data_dir) / "2024" / "r2.json").read_text()
    )
    assert stamped["audit_severity"] == "cosmetic"


def test_severity_stamp_does_not_break_frozen_sha():
    record = make_record("r3")
    lot = audit_sample.freeze_lot([(2024, record)], [{"doc_id": "r3", "year": 2024}])
    stamped = dict(
        record, audit="fail", audit_severity="cosmetic", audit_notes="x", audited_at_utc="t"
    )
    report = audit_sample.build_report(
        [fail_entry("r3", severity="cosmetic")],
        {"r3": []},
        set(),
        lot=lot,
        staged_records_by_doc={"r3": stamped},
    )
    assert report["state"] == "complete"  # stamping is not drift
    assert report["cosmetic_failure_count"] == 1


def test_cosmetic_only_fail_meets_budget_with_zero_critical_rate():
    """299 docs, 1 cosmetic fail: critical rate 0 -> bound ~0.997% <= 1%."""
    results = [fail_entry("c1", severity="cosmetic")]
    results += [pass_entry(f"p{i}") for i in range(298)]
    promoted = {f"p{i}" for i in range(298)}
    report = audit_sample.build_report(results, {}, promoted)
    assert report["state"] == "complete"
    assert report["failure_count"] == 1  # still a recorded (sticky) failure
    assert ("c1", "verdict:fail") in report["failures"]
    assert report["critical_failure_count"] == 0
    assert report["cosmetic_failures"] == [("c1", "verdict:fail")]
    assert report["cosmetic_failure_count"] == 1
    assert report["rate"] == 0.0  # budget math is critical-only
    assert report["upper_bound"] <= 0.01
    assert report["budget_met"] is True  # cosmetic fails do NOT block


def test_mixed_severities_budget_reflects_critical_only():
    """Same n, same total fail count; only the severity flips the budget."""

    def cohort(severity):
        results = [fail_entry("x1", severity=severity)]
        results += [pass_entry(f"p{i}") for i in range(399)]
        return results, {f"p{i}" for i in range(399)}

    results, promoted = cohort("critical")
    critical_report = audit_sample.build_report(results, {}, promoted)
    assert critical_report["failure_count"] == 1
    assert critical_report["critical_failure_count"] == 1
    assert critical_report["cosmetic_failure_count"] == 0
    assert critical_report["rate"] == pytest.approx(1 / 400)
    assert critical_report["budget_met"] is False  # bound(1, 400) ~ 1.3% > 1%

    results, promoted = cohort("cosmetic")
    cosmetic_report = audit_sample.build_report(results, {}, promoted)
    assert cosmetic_report["failure_count"] == 1  # total unchanged
    assert cosmetic_report["critical_failure_count"] == 0
    assert cosmetic_report["cosmetic_failure_count"] == 1
    assert cosmetic_report["rate"] == 0.0
    assert cosmetic_report["budget_met"] is True  # bound(0, 400) ~ 0.75% <= 1%


def test_legacy_fail_without_severity_counts_critical_conservatively():
    report = audit_sample.build_report([fail_entry("L1")], {}, set())
    assert report["critical_failure_count"] == 1
    assert report["cosmetic_failure_count"] == 0
    assert report["legacy_unclassified_failures"] == ["L1"]
    assert report["budget_met"] is False


def test_reclassification_via_rerecord_with_severity_cosmetic():
    """Operator re-records a legacy fail as cosmetic: the explicit call governs."""
    results = [fail_entry("L2"), fail_entry("L2", at="t2", severity="cosmetic")]
    report = audit_sample.build_report(results, {}, set())
    assert report["failures"] == [("L2", "verdict:fail")]  # still sticky
    assert report["critical_failure_count"] == 0
    assert report["cosmetic_failure_count"] == 1
    assert report["legacy_unclassified_failures"] == []
    assert report["repaired_count"] == 0  # fail -> fail is not a repair


def test_critical_fail_is_never_downgraded_by_later_cosmetic_rerecord():
    results = [
        fail_entry("L3", severity="critical"),
        fail_entry("L3", at="t2", severity="cosmetic"),
    ]
    report = audit_sample.build_report(results, {}, set())
    assert report["critical_failure_count"] == 1
    assert report["cosmetic_failure_count"] == 0
    assert report["budget_met"] is False


def test_omission_failures_count_critical():
    """Row-multiset mismatches are transaction-critical; no auto-downgrade."""
    import tempfile

    staged = [["A", "Purchase", "01/15/24", None, "A"]]
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "o1.json").write_text(
            json.dumps(
                {
                    "rows": [
                        ["A", "Purchase", "01/15/24", None, "A"],
                        ["A2", "Sale", "01/16/24", None, "A"],
                    ]
                }
            )
        )
        report = audit_sample.build_report(
            [pass_entry("o1")], {"o1": staged}, {"o1"}, ground_truth_dir=td
        )
    assert report["failures"] == [("o1", "omission: ground-truth rows 2 > staged rows 1")]
    assert report["critical_failure_count"] == 1
    assert report["cosmetic_failure_count"] == 0


def test_cli_report_prints_critical_cosmetic_split(data_dir, capsys):
    for i in range(3):
        write_record(data_dir, make_record(f"m{i}"))
    write_envelope(data_dir, seed=13)
    root = audit_sample.staging_root(data_dir)
    (root / "audit-results.json").write_text(
        json.dumps(
            [
                {
                    "doc_id": "m0",
                    "verdict": "fail",
                    "severity": "critical",
                    "notes": "wrong tx type",
                    "audited_at_utc": "t",
                },
                {
                    "doc_id": "m1",
                    "verdict": "fail",
                    "severity": "cosmetic",
                    "notes": "issuer wording dropped",
                    "audited_at_utc": "t",
                },
                {"doc_id": "m2", "verdict": "pass", "notes": "", "audited_at_utc": "t"},
            ]
        )
    )
    (root / "promoted.jsonl").write_text('{"doc_id": "m2"}\n')

    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    assert rc == 1  # 1 critical fail in 3 -> bound > 1%
    assert "Critical fails: 1 (rate" in out
    assert "| cosmetic fails: 1" in out
    assert "FAIL m0: verdict:fail" in out
    assert "COSMETIC-FAIL m1" in out
    assert "tracked; not blocking" in out
    assert "NOT MET" in out


def test_cli_report_legacy_fail_prints_reclassify_note_and_counts_critical(data_dir, capsys):
    write_record(data_dir, make_record("L9"))
    write_envelope(data_dir, seed=9)
    root = audit_sample.staging_root(data_dir)
    # Legacy entry: written before --severity existed (no severity field).
    (root / "audit-results.json").write_text(json.dumps([fail_entry("L9")]))

    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    assert rc == 1  # conservative: 1 critical fail in 1 -> NOT MET
    assert "counted CRITICAL" in out
    assert "--severity cosmetic" in out
    assert "LEGACY-UNCLASSIFIED L9" in out
    state = json.loads((root / "audit-report-state.json").read_text())
    assert state["critical_failure_count"] == 1
    assert state["cosmetic_failure_count"] == 0


def test_state_file_gains_critical_failures_and_keeps_v2_contract(data_dir, capsys):
    write_record(data_dir, make_record("S1"))
    write_envelope(data_dir, seed=11)
    audit_sample.record_verdict(
        data_dir, "S1", "fail", "asset description spacing", severity="cosmetic"
    )

    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    assert rc == 1  # complete; 0 critical in 1 -> bound 95% > 1%
    state = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-report-state.json").read_text()
    )
    # Unchanged audit-report-state-v2 contract keys:
    for key in (
        "schema", "state", "reason", "budget_met", "lot_size", "audited",
        "failure_count", "unresolved_count", "unresolved", "repaired_count",
        "rate", "upper_bound", "confidence", "max_rate", "pooled_rate",
        "pooling_method", "sample_file", "caveat", "generated_at_utc",
    ):
        assert key in state, key
    assert state["schema"] == "audit-report-state-v2"  # UNCHANGED
    assert state["state"] == "complete"
    assert state["budget_met"] is False
    assert state["rate"] == 0.0          # budget math is critical-only
    assert state["upper_bound"] > 0.01
    assert state["failure_count"] == 1   # total failures keep their meaning
    # New keys alongside the existing fields:
    assert state["critical_failures"] == []
    assert state["critical_failure_count"] == 0
    assert state["cosmetic_failures"] == [["S1", "verdict:fail"]]
    assert state["cosmetic_failure_count"] == 1


def test_incomplete_state_payload_carries_severity_keys(data_dir, capsys):
    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])  # no lot
    assert rc == 2
    state = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-report-state.json").read_text()
    )
    assert state["schema"] == "audit-report-state-v2"
    assert state["critical_failures"] == []
    assert state["critical_failure_count"] == 0
    assert state["cosmetic_failures"] == []
    assert state["cosmetic_failure_count"] == 0
    assert state["budget_met"] is None


def test_cli_record_accepts_severity_flag(data_dir, capsys):
    write_record(data_dir, make_record("c9"))
    rc = audit_sample.main(
        [
            "--data-dir", str(data_dir), "record",
            "--doc-id", "c9", "--verdict", "fail",
            "--severity", "cosmetic", "--notes", "text-only",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "severity: cosmetic" in out
    assert "does NOT block promotion" in out
    (entry,) = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-results.json").read_text()
    )
    assert entry["severity"] == "cosmetic"
