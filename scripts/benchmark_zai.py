"""Bounded, sequential reasoning experiment; never publishes pipeline results.

Run only after the operator has authorized live calls and drained other users
of the endpoint. Each invocation accepts 1–3 frozen ResultBundle JSON paths.
The new output directory is private; raw responses and parsed fields stay there.
Only the safe summary is printed. There are no retries or admission receipts.
Compare two efforts (MAX/LOW by default), or run one follow-up effort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzer.ocr import common
from analyzer.ocr.workers import load_result
from analyzer.vision_engines import (
    VisionBConfig,
    VisionBConfigError,
    classify_visionb_error,
    read_document_with_vision_b,
    request_max_tokens,
)
from scripts.glm_stage import RENDER_DPI, rasterize_page_data_urls

REQUEST_DEADLINE = 210


def _write(path: Path, value) -> None:
    # Parent creates a mode-0700 directory; exclusive files prevent overwrites.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)


def _config() -> VisionBConfig:
    config = VisionBConfig.from_env()
    if config is None or config.model.lower() != "glm-5.3-flash":
        raise VisionBConfigError("benchmark requires configured glm-5.3-flash")
    return config


def _summary(directory: Path, effort: str) -> dict:
    source = json.loads((directory / "source.json").read_text())
    return {
        "doc_id": source["doc_id"],
        "artifact_sha256": source["artifact_sha256"],
        "model": _config().model,
        "params": {"reasoning_effort": effort, "temperature": 0,
                   "max_tokens": request_max_tokens(source["page_count"], model=_config().model,
                                                    reasoning_effort=effort)},
        "dpi": RENDER_DPI, "page_count": source["page_count"],
        "render_elapsed_seconds": source.get("render_elapsed_seconds"),
        "finish_reason": None, "usage": None, "cached_tokens": None,
        "parsed_count": None, "failure_type": None,
    }


def _case(directory: Path, effort: str) -> int:
    """Private child: one request, shared renderer/parser/client, no retries."""
    summary = _summary(directory, effort)
    start = time.monotonic()
    try:
        config = _config()
        summary["model"] = config.model
        images = json.loads((directory / "images.json").read_text())
        raw_path = directory / f"{effort}-response.json"

        def retain(body: str) -> None:
            with raw_path.open("x", encoding="utf-8") as stream:
                stream.write(body)

        text = read_document_with_vision_b(
            images, config, reasoning_effort=effort, response_sink=retain,
        )
        parsed = common.parse_gemini_output(
            text, expected_page_count=summary["page_count"],
        )
        _write(directory / f"{effort}-parsed.json", {
            "member": parsed.member, "rows": list(parsed.transactions),
            "raw_row_count": parsed.raw_row_count,
        })
        summary["parsed_count"] = len(parsed.transactions)
    except Exception as exc:  # noqa: BLE001 -- private evidence boundary; never expose output
        # Exception text may include model output or HTTP bodies: never log it.
        summary["failure_type"] = classify_visionb_error(exc)
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - start, 3)
        raw_path = directory / f"{effort}-response.json"
        if raw_path.exists():
            try:
                body = json.loads(raw_path.read_text())
                reason = body.get("choices", [{}])[0].get("finish_reason")
                summary["finish_reason"] = reason if reason in (
                    None, "stop", "length", "tool_calls", "sensitive", "content_filter",
                    "model_context_window_exceeded", "network_error",
                ) else "unknown"
                usage = body.get("usage") or {}
                # Allowlist token counters: never echo arbitrary provider fields.
                summary["usage"] = {
                    key: usage[key] for key in
                    ("prompt_tokens", "completion_tokens", "total_tokens")
                    if type(usage.get(key)) is int
                }
                for key in ("completion_tokens_details", "prompt_tokens_details"):
                    details = usage.get(key) or {}
                    summary["usage"][key] = {
                        name: details[name] for name in ("reasoning_tokens", "cached_tokens")
                        if type(details.get(name)) is int
                    }
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                summary["cached_tokens"] = cached if type(cached) is int else None
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                summary["failure_type"] = summary["failure_type"] or "malformed_evidence"
        _write(directory / f"{effort}-summary.json", summary)
    return int(summary["failure_type"] is not None)


def _run_case(directory: Path, effort: str, *, timeout: float = REQUEST_DEADLINE) -> dict:
    if not 0 < timeout <= REQUEST_DEADLINE:
        raise ValueError("request deadline must be within 210 seconds")
    start = time.monotonic()
    try:
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--_case", str(directory), "--_effort", effort],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for the client child on timeout.
        # Provider-side cancellation is unknown; later cases may be affected.
        return {**_summary(directory, effort), "failure_type": "deadline",
                "censored": True, "deadline_seconds": timeout,
                "total_elapsed_seconds": round(time.monotonic() - start, 3)}
    path = directory / f"{effort}-summary.json"
    if not path.exists():
        return {**_summary(directory, effort), "failure_type": "child_exit",
                "child_returncode": child.returncode}
    summary = json.loads(path.read_text())
    if child.returncode and not summary["failure_type"]:
        summary["failure_type"] = "child_exit"
    summary["total_elapsed_seconds"] = round(time.monotonic() - start, 3)
    summary["censored"] = summary["failure_type"] == "truncated"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", type=Path, default=[])
    parser.add_argument("--effort", action="append", choices=("max", "high", "low"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, help="new directory under data/pipeline/benchmarks/zai-effort")
    parser.add_argument("--_case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_effort", choices=("max", "high", "low"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    if args._case is not None:
        if args._effort is None:
            parser.error("private child requires effort")
        return _case(args._case, args._effort)
    if not 1 <= len(args.result) <= 3 or args.output is None:
        parser.error("require 1–3 --result paths and a new --output directory")
    efforts = args.effort or ["max", "low"]
    if len(efforts) > 2 or len(set(efforts)) != len(efforts):
        parser.error("require one or two distinct efforts; at most six requests")
    _config()  # Validate before rendering; no network call.
    root = args.data_dir.resolve()
    base = root / "pipeline/benchmarks/zai-effort"
    output = args.output.resolve()
    if not output.is_relative_to(base) or output == base:
        parser.error("output must be beneath data-dir/pipeline/benchmarks/zai-effort")
    # Validate and freeze ALL sources before spending any request. No DB opens.
    sources = []
    for path in args.result:
        result = load_result(path)
        work = result.work
        artifact = Path(work.artifact_path)
        if not artifact.is_absolute():
            artifact = root.parent / artifact
        content = artifact.read_bytes()
        if hashlib.sha256(content).hexdigest() != work.artifact_sha256:
            raise ValueError("source artifact hash mismatch")
        if work.doc_id in [item[0].doc_id for item in sources]:
            raise ValueError("duplicate source document")
        sources.append((work, content))
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    cases = []
    for index, (work, content) in enumerate(sources):
        directory = output / work.doc_id
        directory.mkdir(mode=0o700)
        pdf = directory / "source.pdf"
        pdf.write_bytes(content)
        if common.pdf_page_count(pdf) != work.page_count:
            raise ValueError("source page count mismatch")
        render_start = time.monotonic()
        images = rasterize_page_data_urls(pdf)
        if len(images) != work.page_count:
            raise ValueError("rendered page count mismatch")
        _write(directory / "images.json", images)
        _write(directory / "source.json", {
            "doc_id": work.doc_id, "artifact_sha256": work.artifact_sha256,
            "page_count": work.page_count,
            "render_elapsed_seconds": round(time.monotonic() - render_start, 3),
        })
        order = efforts if index % 2 == 0 else list(reversed(efforts))
        cases.extend((directory, effort) for effort in order)
    summaries = []
    for directory, effort in cases:  # At most six requests, strictly sequential.
        summary = _run_case(directory, effort)
        summary.setdefault("doc_id", directory.name)
        summary.setdefault("params", {"reasoning_effort": effort})
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        if summary["failure_type"] in ("quota", "auth", "config", "permanent", "child_exit"):
            break  # Never retry; deadline/truncation remain censored comparisons.
    _write(output / "summary.json", {"cases": summaries, "planned_requests": len(cases),
                                    "complete": len(summaries) == len(cases)
                                    and all(s["failure_type"] is None for s in summaries)})
    return int(len(summaries) != len(cases) or any(s["failure_type"] for s in summaries))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI must not print private source data
        # No traceback: source paths/headers/model excerpts may contain PII.
        print(json.dumps({"failure_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1)
