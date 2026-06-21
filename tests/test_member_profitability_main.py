"""Smoke tests for member_profitability.main module.

`main.py` is an entry-point script that runs the full pipeline at import
time only when invoked via `python -m member_profitability.main`. When
imported normally, it just imports submodules and defines `main()`. We
test that it imports cleanly and that `main` is callable.
"""
import unittest
from pathlib import Path


class TestMainImports(unittest.TestCase):

    def test_module_imports(self):
        import member_profitability.main
        self.assertTrue(callable(member_profitability.main.main))

    def test_module_file_exists(self):
        path = Path(__file__).resolve().parent.parent / "member_profitability" / "main.py"
        self.assertTrue(path.exists())


class TestMainReExports(unittest.TestCase):

    def test_imports_orchestration_helpers(self):
        import member_profitability.main
        # main.py imports from sibling modules; if those fail the import
        # above would have raised. Spot-check a couple of expected names
        # exist as attributes (re-imported for convenience).
        self.assertTrue(hasattr(member_profitability.main, "pd"))
        from member_profitability import walk_forward
        self.assertTrue(callable(walk_forward.generate_windows))
        self.assertTrue(callable(walk_forward.collect_window_results))


if __name__ == "__main__":
    unittest.main()
