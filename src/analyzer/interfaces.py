from abc import ABC, abstractmethod

import pandas as pd


class TransactionSource(ABC):
    @abstractmethod
    def get_transactions(self, year: int) -> pd.DataFrame:
        pass
