"""Smoke tests for analyzer.interfaces module."""
import unittest

import pandas as pd

from analyzer.interfaces import PriceSource, TransactionSource


class TestInterfaces(unittest.TestCase):

    def test_transaction_source_is_abstract(self):
        # Cannot instantiate the abstract base class directly
        with self.assertRaises(TypeError):
            TransactionSource()

    def test_price_source_is_abstract(self):
        with self.assertRaises(TypeError):
            PriceSource()

    def test_concrete_subclass_works(self):
        class MySource(TransactionSource):
            def get_transactions(self, year):
                return pd.DataFrame({"x": [1]})

        # Concrete subclass can be instantiated
        s = MySource()
        df = s.get_transactions(2024)
        self.assertEqual(len(df), 1)

    def test_missing_abstract_method_raises(self):
        # Subclass that does not implement the abstract method can't be instantiated
        class IncompleteSource(TransactionSource):
            pass

        with self.assertRaises(TypeError):
            IncompleteSource()


if __name__ == "__main__":
    unittest.main()