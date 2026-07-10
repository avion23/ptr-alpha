# House parser corpus audit (2026-07-10)

## Scope and command

This is a non-destructive parser-only audit of every locally cached House PDF. It did not fetch metadata or PDFs and did not write to `data/` or the database. The production `_parse_pdf_worker` cascade was invoked in an 8-process pool with Docling disabled, matching the bulk first-pass mode documented by the parser:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_house_parser.py \
  data /tmp/house_parse_audit.json --workers 8
```

The audit runner enumerated `data/*/pdfs/*.pdf`, validated each file with `_is_valid_pdf`, called `_parse_pdf_worker`, and recorded the transaction count, attempted engines, and uncaught exception for each PDF. The source corpus was outside the audit worktree; paths below are relative to `data/`.

## Results

- Documents: **2,971**; structurally invalid PDFs: **0**; uncaught exceptions: **0**.
- Parsed transactions: **13,168** from **2,531** documents.
- Potential parse failures: **440** valid PDFs produced zero transactions.
- Winning engines: pdfplumber **1,684**; pdftotext **845**; Tesseract OCR **2**. The remaining **440** exhausted the enabled cascade without a result.
- Runtime: **346.63 seconds**.
- Camelot emitted **5,178** `UserWarning` occurrences that image-based pages cannot be processed as text. These warnings do not include the source PDF path, so they cannot be reliably assigned to individual documents after a parallel run. They are expected parser fallback signals, but remain potential extraction risks.

| Year | PDFs | Transactions | Zero-row PDFs | pdfplumber | pdftotext | OCR |
|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 674 | 2757 | 111 | 403 | 160 | 0 |
| 2022 | 613 | 1844 | 110 | 313 | 190 | 0 |
| 2023 | 457 | 2114 | 70 | 256 | 131 | 0 |
| 2024 | 450 | 1398 | 57 | 237 | 156 | 0 |
| 2025 | 516 | 3890 | 66 | 315 | 134 | 1 |
| 2026 | 261 | 1165 | 26 | 160 | 74 | 1 |

## Potential errors and limitations

This section records limitations of this specific run. The canonical, severity-ranked list of
potential ingestion errors is
[`house-ingestion-error-catalog.md`](house-ingestion-error-catalog.md).

- Every zero-row PDF is listed below. A zero result can mean a non-PTR or amendment document, an unsupported layout, a scanned disclosure that Tesseract could not read, or a real parser defect. Metadata was intentionally not consulted because this audit covered every cached PDF.
- Docling was disabled to avoid its approximately 2 GB per-process footprint and 13-300 second per-document cost. Therefore these 440 files are candidates for the production second-pass Docling/Gemini OCR workflows, not confirmed permanent failures.
- Parser-engine exceptions are caught internally and logged only at debug level. The outer audit observed no exceptions, but cannot distinguish a clean zero result from internally suppressed engine failures. This diagnostic gap is itself a potential observability error.
- The run did not call `parse_cached_pdfs`, consolidate metadata, or persist transactions. Database joins, member attribution, and write behavior were not exercised; this was required to avoid changing the live corpus.
- Transaction counts measure emitted rows, not semantic correctness. Amounts, dates, owners, tickers, duplicate rows, and metadata linkage require separate validation.

### All zero-row documents

#### 2021 (111)

```text
20018152.pdf
20019320.pdf
8217820.pdf
8217840.pdf
8217844.pdf
8217846.pdf
8217852.pdf
8217856.pdf
8217868.pdf
8217869.pdf
8217870.pdf
8217876.pdf
8217884.pdf
8217886.pdf
8217887.pdf
8217894.pdf
8217902.pdf
8217905.pdf
8217906.pdf
8217912.pdf
8217925.pdf
8217926.pdf
8217927.pdf
8217928.pdf
8217934.pdf
8217937.pdf
8217938.pdf
8217939.pdf
8217977.pdf
8218027.pdf
8218028.pdf
8218029.pdf
8218030.pdf
8218031.pdf
8218040.pdf
8218054.pdf
8218059.pdf
8218062.pdf
8218078.pdf
8218082.pdf
8218083.pdf
8218108.pdf
8218109.pdf
8218110.pdf
8218111.pdf
8218112.pdf
8218113.pdf
8218114.pdf
8218126.pdf
8218132.pdf
8218139.pdf
8218140.pdf
8218144.pdf
8218147.pdf
8218157.pdf
8218167.pdf
8218182.pdf
8218183.pdf
8218196.pdf
8218220.pdf
8218261.pdf
8218262.pdf
8218296.pdf
8218297.pdf
8218311.pdf
8218312.pdf
8218321.pdf
8218328.pdf
8218329.pdf
8218330.pdf
8218338.pdf
8218358.pdf
8218359.pdf
8218370.pdf
8218371.pdf
8218372.pdf
8218373.pdf
8218374.pdf
8218375.pdf
8218376.pdf
8218379.pdf
8218380.pdf
8218381.pdf
8218386.pdf
8218397.pdf
8218398.pdf
8218405.pdf
8218408.pdf
8218410.pdf
8218411.pdf
8218413.pdf
8218414.pdf
8218417.pdf
8218424.pdf
8218426.pdf
8218433.pdf
8218436.pdf
8218441.pdf
8218442.pdf
8218460.pdf
8218477.pdf
8218478.pdf
8218481.pdf
8218489.pdf
8218490.pdf
8218491.pdf
8218500.pdf
8218505.pdf
8218506.pdf
8218507.pdf
8218512.pdf
```

#### 2022 (110)

```text
20021049.pdf
20021148.pdf
8218518.pdf
8218527.pdf
8218534.pdf
8218535.pdf
8218540.pdf
8218543.pdf
8218545.pdf
8218548.pdf
8218557.pdf
8218558.pdf
8218560.pdf
8218561.pdf
8218564.pdf
8218565.pdf
8218568.pdf
8218569.pdf
8218579.pdf
8218580.pdf
8218587.pdf
8218597.pdf
8218598.pdf
8218599.pdf
8218604.pdf
8218615.pdf
8218620.pdf
8218621.pdf
8218624.pdf
8218628.pdf
8218631.pdf
8218637.pdf
8218640.pdf
8218641.pdf
8218642.pdf
8218645.pdf
8218652.pdf
8218659.pdf
8218661.pdf
8218662.pdf
8218698.pdf
8218730.pdf
8218753.pdf
8218757.pdf
8218777.pdf
8218855.pdf
8218856.pdf
8218894.pdf
8218930.pdf
8218933.pdf
8218936.pdf
8218937.pdf
8218938.pdf
8218940.pdf
8219010.pdf
8219025.pdf
8219043.pdf
8219044.pdf
8219054.pdf
8219056.pdf
8219057.pdf
8219067.pdf
8219068.pdf
8219082.pdf
8219090.pdf
8219092.pdf
8219130.pdf
8219131.pdf
8219135.pdf
8219155.pdf
8219156.pdf
8219165.pdf
8219166.pdf
8219185.pdf
8219196.pdf
8219202.pdf
8219205.pdf
8219207.pdf
8219209.pdf
8219218.pdf
8219219.pdf
8219222.pdf
8219228.pdf
8219232.pdf
8219241.pdf
8219242.pdf
8219244.pdf
8219250.pdf
8219256.pdf
8219258.pdf
8219259.pdf
8219265.pdf
8219266.pdf
8219269.pdf
8219272.pdf
8219273.pdf
8219275.pdf
8219276.pdf
8219277.pdf
8219281.pdf
8219286.pdf
8219288.pdf
8219289.pdf
8219290.pdf
8219291.pdf
8219292.pdf
8219294.pdf
8219297.pdf
8219298.pdf
8219309.pdf
```

#### 2023 (70)

```text
20023060.pdf
8219339.pdf
8219352.pdf
8219353.pdf
8219356.pdf
8219359.pdf
8219362.pdf
8219383.pdf
8219384.pdf
8219412.pdf
8219414.pdf
8219415.pdf
8219417.pdf
8219420.pdf
8219422.pdf
8219425.pdf
8219430.pdf
8219432.pdf
8219436.pdf
8219442.pdf
8219444.pdf
8219447.pdf
8219455.pdf
8219469.pdf
8219470.pdf
8219483.pdf
8219485.pdf
8219521.pdf
8219522.pdf
8219524.pdf
8219525.pdf
8219527.pdf
8219742.pdf
8219743.pdf
8219748.pdf
8219783.pdf
8219799.pdf
8219808.pdf
8219814.pdf
8219826.pdf
8219834.pdf
8219843.pdf
8219845.pdf
8219854.pdf
8219876.pdf
8219893.pdf
8219904.pdf
8219905.pdf
8219919.pdf
8219921.pdf
8219922.pdf
8219944.pdf
8219953.pdf
8219959.pdf
8219971.pdf
8219972.pdf
8219992.pdf
8220010.pdf
8220023.pdf
8220037.pdf
8220039.pdf
8220046.pdf
8220049.pdf
8220051.pdf
8220060.pdf
8220067.pdf
8220068.pdf
8220074.pdf
8220078.pdf
8220079.pdf
```

#### 2024 (57)

```text
20025152.pdf
8220118.pdf
8220119.pdf
8220127.pdf
8220147.pdf
8220155.pdf
8220162.pdf
8220173.pdf
8220176.pdf
8220177.pdf
8220188.pdf
8220192.pdf
8220203.pdf
8220205.pdf
8220214.pdf
8220252.pdf
8220271.pdf
8220312.pdf
8220317.pdf
8220320.pdf
8220431.pdf
8220451.pdf
8220508.pdf
8220516.pdf
8220529.pdf
8220534.pdf
8220536.pdf
8220551.pdf
8220567.pdf
8220570.pdf
8220589.pdf
8220615.pdf
8220617.pdf
8220626.pdf
8220628.pdf
8220643.pdf
8220660.pdf
8220661.pdf
8220665.pdf
8220672.pdf
8220674.pdf
8220682.pdf
8220683.pdf
8220685.pdf
8220691.pdf
8220692.pdf
8220695.pdf
8220698.pdf
8220706.pdf
8220710.pdf
8220711.pdf
8220714.pdf
8220716.pdf
8220717.pdf
8220718.pdf
8220723.pdf
8220763.pdf
```

#### 2025 (66)

```text
8220731.pdf
8220747.pdf
8220750.pdf
8220753.pdf
8220754.pdf
8220755.pdf
8220757.pdf
8220764.pdf
8220765.pdf
8220768.pdf
8220770.pdf
8220780.pdf
8220782.pdf
8220783.pdf
8220796.pdf
8220799.pdf
8220809.pdf
8220824.pdf
8220827.pdf
8220828.pdf
8220834.pdf
8220836.pdf
8220844.pdf
8220845.pdf
8220902.pdf
8220903.pdf
8220904.pdf
8220906.pdf
8220958.pdf
8221120.pdf
8221123.pdf
8221124.pdf
8221173.pdf
8221176.pdf
8221177.pdf
8221223.pdf
8221228.pdf
8221231.pdf
8221233.pdf
8221237.pdf
8221238.pdf
8221263.pdf
8221264.pdf
8221270.pdf
8221276.pdf
9115546.pdf
9115549.pdf
9115562.pdf
9115575.pdf
9115599.pdf
9115616.pdf
9115623.pdf
9115635.pdf
9115641.pdf
9115642.pdf
9115662.pdf
9115664.pdf
9115665.pdf
9115670.pdf
9115671.pdf
9115676.pdf
9115677.pdf
9115679.pdf
9115684.pdf
9115686.pdf
9115689.pdf
```

#### 2026 (26)

```text
8221321.pdf
8221322.pdf
8221326.pdf
8221332.pdf
8221358.pdf
8221359.pdf
8221360.pdf
9115704.pdf
9115711.pdf
9115728.pdf
9115762.pdf
9115808.pdf
9115809.pdf
9115811.pdf
9115812.pdf
9115813.pdf
9115814.pdf
9115815.pdf
9115816.pdf
9115820.pdf
9115821.pdf
9115822.pdf
9115901.pdf
9116141.pdf
9116142.pdf
9116146.pdf
```
