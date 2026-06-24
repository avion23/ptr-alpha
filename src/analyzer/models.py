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
