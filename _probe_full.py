
import sys, time, os
sys.path.insert(0, ".")
sys.path.insert(0, "src")
os.environ["OCR2_WORKERS"]="1"
from scripts import ocr_local_sweep as ocr
pdf = '/Users/avion/Documents.nosync/projects/insider-trading/.worktrees/luna-rebuild2/data/.staging/rebuild2/gen-live-20260809/2015/pdfs/9107576.pdf'
db = '/Users/avion/Documents.nosync/projects/insider-trading/.worktrees/luna-rebuild2/data/.staging/rebuild2/gen-live-20260809/congress.duckdb'
meta = ocr.load_metadata(db)
t0=time.time()
result = ocr.process_document("9107576", 2015, pdf, meta.get("9107576", {}))
print("elapsed", round(time.time()-t0,1))
print("status:", result.get("status"))
print("row_count:", result.get("row_count"))
print("covered_pages:", result.get("covered_pages"), "uncovered:", result.get("uncovered_pages"))
print("engines:", result.get("engines"))
for tx in result.get("rows", [])[:12]:
    print("  ", repr(tx.get("asset_description"))[:45], "|", tx.get("transaction_type"), "|", tx.get("transaction_date"), "|", tx.get("amount_midpoint"), "| engine", tx.get("_engine"))
print("reasons:", result.get("reasons")[:5])
