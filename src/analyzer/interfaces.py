from abc import ABC, abstractmethod
import pandas as pd


class TransactionSource(ABC):
    @abstractmethod
    def get_transactions(self, year: int) -> pd.DataFrame:
        pass


class PriceSource(ABC):
    @abstractmethod
    def get_prices(self, tickers: list[str], start, end) -> pd.DataFrame:
        pass
