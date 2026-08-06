"""Top-level metadata normalization and transaction consolidation.

`normalize_house_metadata` parses the House disclosure metadata TSV (member
info, filing date). `consolidate_transactions` joins per-PDF transaction
dictionaries with member metadata into one tidy DataFrame.
"""

import logging
from pathlib import Path

import pandas as pd

from analyzer.exceptions import ParsingError

logger = logging.getLogger(__name__)


def normalize_house_metadata(content: str) -> pd.DataFrame:
    if not content:
        raise ParsingError("Empty metadata content")

    lines = content.strip().split("\n")
    if len(lines) < 2:
        raise ParsingError("Insufficient metadata lines")

    header = [col.strip() for col in lines[0].lstrip("\ufeff").split("\t")]
    required = {"DocID", "First", "Last", "FilingDate"}
    missing = required.difference(header)
    if missing:
        raise ParsingError(
            f"Missing required metadata column(s): {', '.join(sorted(missing))}"
        )
    if len(header) != len(set(header)):
        raise ParsingError("Duplicate metadata column names")
    data = _parse_metadata_rows(lines[1:], header)

    if not data:
        raise ParsingError("No data rows in metadata")

    df = pd.DataFrame(data, columns=header)
    doc_ids = df["DocID"].astype(str).str.strip()
    if (doc_ids == "").any():
        raise ParsingError("Blank DocID in metadata")
    duplicate_mask = doc_ids.duplicated(keep=False)
    unique_duplicate_rows = (
        df.loc[duplicate_mask].assign(DocID=doc_ids[duplicate_mask]).drop_duplicates()
    )
    conflicting_ids = unique_duplicate_rows.loc[
        unique_duplicate_rows["DocID"].duplicated(keep=False), "DocID"
    ].unique()
    if len(conflicting_ids):
        sample = ", ".join(conflicting_ids[:5])
        raise ParsingError(f"Duplicate DocID(s) in metadata: {sample}")
    df["DocID"] = doc_ids
    if duplicate_mask.any():
        logger.warning(
            "Removed %d identical duplicate metadata row(s)", int(df.duplicated().sum())
        )
        df = df.drop_duplicates()
    return _validate_filing_dates(df)


def _parse_metadata_rows(lines: list[str], header: list[str]) -> list[list[str]]:
    data: list[list[str]] = []
    malformed: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        row = [col.strip() for col in line.split("\t")]
        if len(row) == len(header):
            data.append(row)
        else:
            malformed.append(
                row[header.index("DocID")]
                if len(row) > header.index("DocID")
                else "<unknown>"
            )
    if malformed:
        logger.warning(
            "Dropped %d metadata row(s) with a column count different from header (%d); doc IDs: %s",
            len(malformed),
            len(header),
            ", ".join(malformed[:10]),
        )
    return data


def _validate_filing_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "FilingDate" not in df.columns:
        raise ParsingError("Missing FilingDate column in metadata")

    invalid = pd.to_datetime(df["FilingDate"], errors="coerce").isna()
    if invalid.any():
        logger.warning(
            "Dropped %d metadata row(s) with invalid filing dates; doc IDs: %s",
            int(invalid.sum()),
            ", ".join(df.loc[invalid, "DocID"].astype(str).head(10)),
        )
    df["FilingDate"] = pd.to_datetime(df["FilingDate"], errors="coerce")
    df = df.dropna(subset=["FilingDate"])

    if df.empty:
        raise ParsingError("No valid filing dates in metadata")

    return df


def consolidate_transactions(
    pdf_transactions: dict[Path, list[dict]], member_metadata: dict[str, dict]
) -> pd.DataFrame:
    if not pdf_transactions:
        return pd.DataFrame()

    all_transactions = []
    for pdf_path, transactions in pdf_transactions.items():
        if not transactions:
            continue

        doc_id = pdf_path.stem
        member_info = member_metadata.get(doc_id)
        if not member_info:
            logger.warning(
                f"No metadata found for doc_id={doc_id}, skipping {len(transactions)} transaction(s)"
            )
            continue

        first = member_info.get("First", "")
        last = member_info.get("Last", "")
        if not first and not last:
            logger.warning(
                f"No 'First'/'Last' keys in metadata for doc_id={doc_id}, skipping {len(transactions)} transaction(s)"
            )
            continue

        for tx in transactions:
            all_transactions.append(
                {
                    "doc_id": doc_id,
                    "member": f"{first} {last}".strip(),
                    "transaction_date": tx["transaction_date"],
                    "disclosure_date": member_info["FilingDate"],
                    "ticker": tx["ticker"],
                    "transaction_type": tx["transaction_type"],
                    "owner_code": tx.get("owner_code"),
                    "amount_raw": tx.get("amount_raw"),
                    "amount_midpoint": tx.get("amount_midpoint"),
                    "instrument_type": tx.get("instrument_type", "stock"),
                    "strike_price": tx.get("strike_price"),
                    "expiry_date": tx.get("expiry_date"),
                }
            )

    if not all_transactions:
        return pd.DataFrame()

    df = pd.DataFrame(all_transactions)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"], errors="coerce")
    invalid = df[["transaction_date", "disclosure_date"]].isna().any(axis=1)
    if invalid.any():
        logger.warning(
            "Dropped %d consolidated transaction(s) with invalid dates; doc IDs: %s",
            int(invalid.sum()),
            ", ".join(df.loc[invalid, "doc_id"].astype(str).unique()[:10]),
        )
    return df.loc[~invalid].copy()
