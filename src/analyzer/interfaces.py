from abc import ABC, abstractmethod
import pandas as pd

class TransactionSource(ABC):
    @abstractmethod
    def get_transactions(self, year):
        pass

class PriceSource(ABC):
    @abstractmethod
    def get_prices(self, tickers, start, end):
        pass
