
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "src")
import pandas as pd, json
from scripts import ocr_local_sweep as ocr
data = json.load(open("_tmp_dfs2.json"))
for doc, tables in data.items():
    for ti, t in enumerate(tables):
        df = pd.DataFrame(t["rows"], columns=t["columns"])
        rows = ocr._dataframe_rows(df, 1)
        print(doc, "T%d"%ti, df.shape, "-> rows:", len(rows))
        for r in rows[:3]:
            print("    ", repr(r.get("asset_description"))[:40], "|", r.get("transaction_type"), "|", r.get("transaction_date"), "|", r.get("amount_midpoint"))
