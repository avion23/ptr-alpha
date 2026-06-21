"""Smoke tests for scripts.docling_stragglers module."""
import unittest


class TestDoclingStragglers(unittest.TestCase):

    def test_module_imports(self):
        from scripts import docling_stragglers
        assert callable(docling_stragglers.find_zero_row_pdfs)
        assert callable(docling_stragglers.run_docling_on_pdf)
        assert callable(docling_stragglers.parse_markdown_to_txs)
        assert callable(docling_stragglers.main)

    def test_module_constants(self):
        from scripts import docling_stragglers
        # REPO is the repo root (possibly a worktree path)
        self.assertTrue(docling_stragglers.REPO.exists())
        self.assertTrue((docling_stragglers.REPO / "src").exists())


if __name__ == "__main__":
    unittest.main()