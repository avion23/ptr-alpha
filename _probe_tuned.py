
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "src")
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.document import InputDocument, ConversionOptions
pdf = '/Users/avion/Documents.nosync/projects/insider-trading/.worktrees/luna-rebuild2/data/.staging/rebuild2/gen-live-20260809/2015/pdfs/9107576.pdf'
opts = PdfPipelineOptions()
opts.do_table_structure = True
opts.table_structure_options.mode = "accurate"
opts.images_scale = 2.0
conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
result = conv.convert(pdf)
doc = result.document
print("tables:", len(doc.tables))
for t in doc.tables:
    df = t.export_to_dataframe(doc=doc)
    print("TABLE pages:", sorted({p.page_no for p in t.prov}), "shape:", df.shape)
    print("columns:", [str(c) for c in df.columns])
    for idx in range(len(df)):
        print("  r%d:"%idx, [str(df.iat[idx,j])[:38] for j in range(df.shape[1])])
