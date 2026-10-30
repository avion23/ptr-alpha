"""Offline, bounded evidence for the old House ruled checkbox form only.

Not a transaction parser or approval: every detected row band is retained. OCR
asset/date candidates do not certify identity, chronology, or row completeness.
Amount indices refer to this page's source headings, NEVER a range lookup table.
Usage: python scripts/house_grid_evidence.py PDF NEW_OUTPUT_DIR [--read-dates]
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date
from itertools import pairwise
from pathlib import Path

os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from pdf2image.exceptions import PDFPageCountError, PDFPopplerTimeoutError

from analyzer.parsing.ocr_parser import _date_pattern, _orient_image


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_crop(gray, box, directory, name):
    x0, y0, x1, y1 = map(int, box)
    path = directory / f"{name}.png"
    if not cv2.imwrite(str(path), gray[y0:y1, x0:x1]):
        raise OSError(f"cannot save {path}")
    return {"path": path.name, "bbox": [x0, y0, x1, y1], "sha256": digest(path)}


def ocr(image, psm=6):
    try:
        data = pytesseract.image_to_data(image, config=f"--psm {psm}",
                                        output_type=pytesseract.Output.DICT, timeout=20)
    except RuntimeError as exc:
        return {"text": "", "min_confidence": 0, "error": str(exc)}
    words = [(t, float(c)) for t, c in zip(data["text"], data["conf"]) if t.strip()]
    return {"text": " ".join(t for t, _ in words),
            "min_confidence": min((c for _, c in words), default=0)}


def centers(mask):
    indices = np.flatnonzero(mask)
    return [int(np.median(g)) for g in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1) if len(g)]


def grid(gray):
    """Find lines, not transactions. Reject ambiguous/missing geometry."""
    h, w = gray.shape
    if min(h, w) < 200:
        return gray, np.zeros_like(gray), {"candidates": [], "reason": "unsupported_page_dimensions"}
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    lines = cv2.HoughLinesP(ink, 1, np.pi / 1800, w // 4,
                            minLineLength=w // 3, maxLineGap=20)
    angles = [] if lines is None else [float(np.degrees(np.arctan2(y1-y0, x1-x0)))
        for x0, y0, x1, y1 in lines.reshape(-1, 4) if abs(y1-y0) < .05 * abs(x1-x0)]
    angle = float(np.median(angles)) if angles else 0.0
    affine = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    gray = cv2.warpAffine(gray, affine, (w, h), borderValue=255)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, w // 6), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((h // 12, 1), np.uint8))
    contours = cv2.findContours(cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8)),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    candidates = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < .65 * w or ch < .15 * h:
            continue
        vx = cv2.dilate(vertical[y:y+ch, x:x+cw], np.ones((1, 9), np.uint8))
        hy = cv2.dilate(horizontal[y:y+ch, x:x+cw], np.ones((9, 1), np.uint8))
        xs = [x + p for p in centers((vx > 0).sum(axis=0) > .5 * ch)]
        ys = [y + p for p in centers((hy > 0).sum(axis=1) > .65 * cw)]
        if not (16 <= len(xs) <= 18 and 6 <= len(ys) <= 15):
            continue
        widths, heights = np.diff(xs), np.diff(ys)
        narrow, row = float(np.median(widths[2:5])), float(np.median(heights[2:]))
        if not (widths[1] > 4*narrow and .5*narrow < widths[0] < 2.5*narrow
                and np.all((widths[2:5] > .75*narrow) & (widths[2:5] < 1.3*narrow))
                and np.all((widths[5:7] > 1.5*narrow) & (widths[5:7] < 3.5*narrow))
                and np.all((widths[7:] > .75*narrow) & (widths[7:] < 2*narrow))
                and np.all(heights[:2] > 1.5*row)
                and np.all((heights[2:] > .5*row) & (heights[2:] < 1.5*row))):
            continue
        candidates.append({"xs": xs, "ys": ys, "bbox": [x, y, x+cw, y+ch],
                           "amount_complete": len(xs) == 18 and x+cw-xs[-1] < 12})
    clean = ink & ~cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8))
    return gray, clean, {"deskew_degrees": angle, "affine_from_upright": affine.tolist(),
                         "candidates": candidates}


def ink_observation(clean, box):
    x0, y0, x1, y1 = box
    # Fixed 200-DPI/2200px renderer; inset excludes ruling, not whole blank rows.
    x0, y0, x1, y1 = x0+7, y0+7, x1-7, y1-7
    cell = clean[y0:y1, x0:x1]
    _, _, stats, _ = cv2.connectedComponentsWithStats(cell, 8)
    components = [{"bbox": [int(x+x0), int(y+y0), int(x+x0+w), int(y+y0+h)],
                   "pixels": int(area)} for x, y, w, h, area in stats[1:] if area >= 8]
    pixels = int(np.count_nonzero(cell))
    state = "blank" if pixels <= 4 else "unknown"
    if len(components) == 1:
        a, b, c, d = components[0]["bbox"]
        if c-a >= 6 and d-b >= 8 and components[0]["pixels"] >= 20 and components[0]["pixels"] >= .85*pixels:
            state = "ink"
    return {"state": state, "interior_bbox": [x0, y0, x1, y1],
            "ink_pixels": pixels, "components": components}


def selection(cells):
    marked = [i for i, c in enumerate(cells) if c["state"] == "ink"]
    return marked[0] if len(marked) == 1 and all(
        c["state"] == "blank" for i, c in enumerate(cells) if i != marked[0]) else None


def date_candidate(reads):
    text = reads[0]["text"]
    if not (all(v["min_confidence"] >= 80 and v["text"] == text for v in reads)
            and re.fullmatch(_date_pattern(), text)):
        return None
    try:
        month, day, year = text.split("/")
        date(int(year) + (2000 if len(year) == 2 else 0), int(month), int(day))
    except ValueError:
        return None
    return text


def analyze_page(gray, directory, read_dates=False):
    gray, clean, geometry = grid(gray)
    h, w = gray.shape
    page = {"status": "unknown", "geometry": geometry, "rows": [],
            "image": save_crop(gray, [0, 0, w, h], directory, "page"),
            "coordinate_system": "deskewed page pixels; xyxy half-open"}
    if len(geometry["candidates"]) != 1:
        page["reason"] = "missing_or_ambiguous_ruled_grid"
        return page
    table = geometry["candidates"][0]
    xs, ys = table["xs"], table["ys"]
    page["header"] = save_crop(gray, [xs[0], ys[0], table["bbox"][2], ys[2]], directory, "header")
    major = ocr(gray[ys[0]+8:ys[1]-8, xs[0]+8:table["bbox"][2]-8])
    labels = [ocr(cv2.rotate(gray[ys[1]+8:ys[2]-8, xs[i]+8:xs[i+1]-8],
                            cv2.ROTATE_90_CLOCKWISE), 7) for i in range(2, 5)]
    page["header_ocr"] = {"major": major, "directions": labels}
    directions = [v["text"].upper() for v in labels]
    headings_ok = ({"PURCHASE", "SALE", "EXCHANGE"} == set(directions)
                   and all(v["min_confidence"] >= 80 for v in labels)
                   and all(s in major["text"].upper() for s in ("ASSET", "NAME", "AMOUNT", "TRANSACTION")))
    if not headings_ok:
        page["reason"] = "source_headings_not_independently_read"
    headers = [save_crop(gray, [a, ys[1], b, ys[2]], directory, f"amount-header-{i+1}")
               for i, (a, b) in enumerate(pairwise(xs[7:]))]
    date_headers = [ocr(gray[ys[0]+8:ys[1]-8, xs[i]+8:xs[i+1]-8]) for i in (5, 6)]
    page["header_ocr"]["dates"] = date_headers
    page.update(status="partial_evidence" if headings_ok else "unknown", amount_headers=headers)
    for number, (top, bottom) in enumerate(pairwise(ys[2:]), 1):
        cells = [ink_observation(clean, [a, top, b, bottom]) for a, b in pairwise(xs)]
        row = {"band": number, "disposition": "unknown", "cells": cells,
               "image": save_crop(gray, [xs[0], top, table["bbox"][2], bottom], directory, f"row-{number}")}
        fields = {}
        for name, start, end in (("owner", 0, 1), ("asset", 1, 2), ("direction", 2, 5),
                                  ("trade_date", 5, 6), ("notification_date", 6, 7),
                                  ("amount", 7, len(xs)-1)):
            box = [xs[start], top, xs[end], bottom]
            fields[name] = {"state": "unknown", "evidence": save_crop(gray, box, directory, f"row-{number}-{name}")}
        row["fields"] = fields
        page["rows"].append(row)
        if not headings_ok:
            continue  # Retain detected bands even when text cannot establish semantics.
        fields["owner"]["state"] = "blank" if cells[0]["state"] == "blank" else "unknown"
        fields["owner"]["scope"] = "ruling-excluded interior only"
        fields["owner"]["value"] = None  # Blank is not JT, self, or another owner.
        chosen = selection(cells[2:5])
        fields["direction"].update(value=directions[chosen] if chosen is not None else None,
                                    state="observed" if chosen is not None else "unknown")
        chosen_amount = selection(cells[7:])
        fields["amount"].update(observed_mark_indices=[i+1 for i, c in enumerate(cells[7:]) if c["state"] == "ink"],
            selected_index=chosen_amount+1 if chosen_amount is not None and table["amount_complete"] else None,
            state="observed_column" if chosen_amount is not None and table["amount_complete"] else "unknown",
            complete_columns=table["amount_complete"], numeric_range=None,
            numeric_range_state="unknown", header_evidence=headers)
        asset = ocr(gray[top+7:bottom-7, xs[1]+7:xs[2]-7])
        fields["asset"]["ocr_candidate"] = asset
        example = bool(re.match(r"Example\s*:", asset["text"], re.IGNORECASE))
        if example:
            second_read = ocr(gray[top+7:bottom-7, xs[1]+7:xs[2]-7], 7)
            fields["asset"]["example_label_second_read"] = second_read
            example = bool(re.match(r"Example\s*:", second_read["text"], re.IGNORECASE))
        if example:
            row["disposition"] = "printed_example"
        elif chosen is not None or chosen_amount is not None:
            row["disposition"] = "marked_row_not_certified_transaction"
        elif all(c["state"] == "blank" for c in cells):
            row["disposition"] = "blank_interiors_not_certified_empty"
        if read_dates:
            for name, index in (("trade_date", 5), ("notification_date", 6)):
                crop = gray[top+7:bottom-7, xs[index]+7:xs[index+1]-7]
                reads = [ocr(crop, psm) for psm in (6, 7)]
                fields[name]["ocr_reads"] = reads
                header = re.sub(r"[^A-Z]", "", date_headers[index-5]["text"].upper())
                expected = "DATEOFTRANSACTION" if index == 5 else "DATENOTIFIEDOFTRANSACTION"
                value = date_candidate(reads)
                if value is not None and header == expected:
                    fields[name].update(state="ocr_candidate_not_certified", value=value)
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_dir", type=Path, help="new directory; existing output is never overwritten")
    parser.add_argument("--read-dates", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"schema": "house-grid-evidence-v1", "status": "partial_evidence_only",
              "approval": False, "artifact_sha256": None, "pages": [],
              "limits": {"max_pages": 32, "max_pdf_bytes": 67108864},
              "render": {"dpi": 200, "size": 2200, "grayscale": True, "thread_count": 1,
                         "timeout_seconds": 120, "omp_threads": 1},
              "versions": {"opencv": cv2.__version__, "tesseract": str(pytesseract.get_tesseract_version())},
              "script_sha256": digest(__file__)}
    try:
        if args.pdf.stat().st_size > result["limits"]["max_pdf_bytes"]:
            raise ValueError("PDF exceeds offline byte budget")
        result["artifact_sha256"] = digest(args.pdf)
        count = pdfinfo_from_path(str(args.pdf), timeout=20)["Pages"]
        result["page_count"] = count
        if not 1 <= count <= result["limits"]["max_pages"]:
            raise ValueError("PDF exceeds offline page budget")
        for number in range(1, count+1):
            directory = args.output_dir / f"page-{number:03d}"
            directory.mkdir()
            page = {"number": number, "status": "unknown", "rows": []}
            images = convert_from_path(str(args.pdf), dpi=200, size=2200, grayscale=True,
                                       first_page=number, last_page=number, thread_count=1, timeout=120)
            oriented = None
            try:
                if len(images) != 1:
                    raise ValueError("renderer did not return exactly one page")
                image = images[0]
                raw = np.array(image.convert("L"))
                rendered = save_crop(raw, [0, 0, raw.shape[1], raw.shape[0]], directory, "rendered")
                page["rendered"] = rendered
                oriented = _orient_image(image, pytesseract)
                upright = np.array(oriented.convert("L"))
                rotation = next(k*90 for k in range(4) if np.array_equal(np.rot90(raw, k), upright))
                page = analyze_page(upright, directory, args.read_dates)
                page.update(number=number, rendered=rendered, rotation_ccw=rotation)
            except RuntimeError as exc:
                page["reason"] = str(exc)
            finally:
                if oriented is not None and all(oriented is not im for im in images):
                    oriented.close()
                for image in images:
                    image.close()
            page["directory"] = directory.name
            result["pages"].append(page)
        if all(p["status"] == "unknown" for p in result["pages"]):
            result["status"] = "unknown"
        if digest(args.pdf) != result["artifact_sha256"]:
            raise ValueError("source bytes changed during run; observations invalid")
    except (OSError, ValueError, RuntimeError, PDFPageCountError, PDFPopplerTimeoutError) as exc:
        result.update(status="unknown", error=str(exc))
    path = args.output_dir / "evidence.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"evidence": str(path), "status": result["status"],
                      "pages_recorded": len(result["pages"])}))
    return 0 if result["status"] == "partial_evidence_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
