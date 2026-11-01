from pathlib import Path
from docx import Document
from docx.shared import Inches
import re

md_file = Path("README.md")
docx_file = Path("readme.docx")

doc = Document()
doc.sections[0].top_margin = Inches(0.7)
doc.sections[0].bottom_margin = Inches(0.7)

for line in md_file.read_text(encoding="utf-8").splitlines():
    line = line.rstrip()

    if not line:
        continue
    elif line.startswith("### "):
        doc.add_heading(line[4:], level=3)
    elif line.startswith("## "):
        doc.add_heading(line[3:], level=2)
    elif line.startswith("# "):
        doc.add_heading(line[2:], level=1)
    elif line.startswith("- ") or line.startswith("* "):
        doc.add_paragraph(line[2:], style="List Bullet")
    elif re.match(r"^\d+\.\s", line):
        doc.add_paragraph(re.sub(r"^\d+\.\s", "", line), style="List Number")
    else:
        doc.add_paragraph(line)

doc.save(docx_file)
print(f"Converted successfully: {docx_file.resolve()}")