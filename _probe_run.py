
import sys, time
sys.path.insert(0, ".")
sys.path.insert(0, "src")
from scripts import ocr_local_sweep as ocr
pdf = '/Users/avion/Documents.nosync/projects/insider-trading/.worktrees/luna-rebuild2/data/.staging/rebuild2/gen-live-20260809/2015/pdfs/9107576.pdf'
t0=time.time()
pages, err = ocr.docling_pages(pdf)
print("err:", err, "elapsed:", round(time.time()-t0,1))
for p in pages:
    print("page", p["page"], "rows:", len(p["rows"]))
    for tx in p["rows"][:20]:
        print("  ", repr(tx.get("asset_description"))[:50], "|", tx.get("transaction_type"), "|", tx.get("transaction_date"), "|", tx.get("amount_midpoint"))
