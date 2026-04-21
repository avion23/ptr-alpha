from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

class TransactionType(str, Enum):
    PURCHASE = "Purchase"
    SALE = "Sale"

class FilingType(str, Enum):
    PTR = "P"
    AMENDMENT = "A"

class DownloadStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"
    ERROR = "error"

class Transaction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    member: str
    ticker: str
    transaction_date: date
    disclosure_date: date
    transaction_type: TransactionType

class Filing(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_id: str = Field(alias="DocID")
    first_name: str = Field(alias="First")
    last_name: str = Field(alias="Last")
    filing_date: datetime = Field(alias="FilingDate")
    filing_type: FilingType = Field(alias="FilingType")

class Signal(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    member: str
    ticker: str
    disclosure_date: date
    signal_type: TransactionType
    horizon_days: int
    entry_price: float
    peak_potential_pct: float

class DownloadResult(BaseModel):
    doc_id: str
    status: DownloadStatus
    error_message: str = ""
    status_code: int = 0
