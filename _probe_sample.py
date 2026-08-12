
import sys, time, json
sys.path.insert(0, ".")
sys.path.insert(0, "src")
from docling.document_converter import DocumentConverter
conv = DocumentConverter()
pdfs = ['/Users/avion/Documents.nosync/projects/insider-trading/.worktrees/luna-rebuild2/data/.staging/rebuild2/gen-live-20260809/2015/pdfs/20000883.pdf', '/Users/avion/Documents.nosync/projects/insider-trading/.worktrees/luna-rebuild2/data/.staging/rebuild2/gen-live-20260809/2015/pdfs/20001253.pdf', '/Users/avion/Documents.nosync/projects/insider-trading/.worktrees/luna-rebuild2/data/.staging/rebuild2/gen-live-20260809/2015/pdfs/20001382.pdf', '/Users/avion/Documents.nosync/projects/insider-trading/.worktrees/luna-rebuild2/data/.staging/rebuild2/gen-live-20260809/2015/pdfs/20002307.pdf', '/Users/avion/Documents.nosync/projects/insider-trading/.worktrees/luna-rebuild2/data/.staging/rebuild2/gen-live-20260809/2015/pdfs/20002313.pdf', '/Users/avion/Documents.nosync/projects/insider-trading/.worktrees/luna-rebuild2/data/.staging/rebuild2/gen-live-20260809/2015/pdfs/20002315.pdf']
out = {}
for pdf in pdfs:
    t0=time.time()
    result = conv.convert(pdf)
    doc = result.document
    tabs = []
    for t in doc.tables:
        df = t.export_to_dataframe(doc=doc)
        tabs.append({
            "pages": sorted({p.page_no for p in t.prov}),
            "columns": [str(c) for c in df.columns],
            "rows": [[str(df.iat[i,j]) for j in range(df.shape[1])] for i in range(len(df))],
        })
    out[pdf.split("/")[-1]] = tabs
    print(pdf.split("/")[-1], "tables:", len(tabs), "elapsed", round(time.time()-t0,1), flush=True)
json.dump(out, open("_sample_dfs.json","w"), indent=0)
print("saved")
