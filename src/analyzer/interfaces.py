from abc import ABC, abstractmethod

import pandas as pd


class TransactionSource(ABC):
    @abstractmethod
    def get_transactions(self, year: int) -> pd.DataFrame:
        """Read this provider's canonical cache for ``year`` without side effects.

        Return a schema-bearing empty frame when no cached rows match. This
        method must never acquire or persist data; database/read failures
        propagate. Acquisition belongs to the provider's explicit fetch API.
        A source that cannot provide canonical transactions may raise its
        documented provider error instead (for example, reconciliation-only
        sources).
        """
        raise NotImplementedError
