from enum import StrEnum
from pydantic import BaseModel


class TransactionType(StrEnum):
    PURCHASE = "Purchase"
    SALE = "Sale"
    EXCHANGE = "Exchange"


class FilingType(StrEnum):
    PTR = "P"
    AMENDMENT = "A"


class DownloadStatus(StrEnum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"
    ERROR = "error"


class DownloadResult(BaseModel):
    doc_id: str
    status: DownloadStatus
    error_message: str = ""
    status_code: int = 0


# NOTE: A Transaction value object (frozen dataclass matching the DB schema)
# was considered but is intentionally deferred.  The entire codebase passes
# transactions as pd.DataFrame — from datasources.py through pipeline.py
# to analysis.py — and introducing a typed value object would require a
# large-scale refactor across all modules.  The DataFrame convention is
# documented here as the intentional representation.
