
import sys, time, json
sys.path.insert(0, ".")
sys.path.insert(0, "src")
import pandas as pd
from docling.document_converter import DocumentConverter
from scripts import ocr_local_sweep as ocr
docs = json.load(open("_sample_docs.json"))
root = '/Users/avion/Documents.nosync/projects/insider-trading/.worktrees/luna-rebuild2/data/.staging/rebuild2/gen-live-20260809'
conv = DocumentConverter()
hits = []
for y, d in docs:
    pdf = f"{root}/{y}/pdfs/{d}.pdf"
    try:
        result = conv.convert(pdf)
    except Exception as e:
        print(d, "convert err", type(e).__name__, flush=True); continue
    doc = result.document
    for ti, t in enumerate(doc.tables):
        df = t.export_to_dataframe(doc=doc)
        rows = ocr._dataframe_rows(df, 1)
        # also record fallback header derivation info
        header_row, names = ocr._derive_old_form_columns(df)
        if len(rows) or header_row is not None or names is not None:
            hits.append({"doc": d, "year": y, "table": ti, "shape": list(df.shape),
                          "rows": len(rows), "header_row": header_row, "names": names,
                          "cols": [str(c)[:30] for c in df.columns]})
            print("HIT", d, y, "T%d"%ti, df.shape, "rows:", len(rows), "header_row:", header_row, flush=True)
    print("done", d, flush=True)
json.dump(hits, open("_sample_hits.json","w"))
print("saved hits:", len(hits))
