"""Markdown -> DOCX for the manuscript (python-docx; no pandoc needed).

Handles the subset ``manuscript.py`` writes: headings (#..####), paragraphs, bullet
lists, pipe tables, images on their own line (paths relative to the Markdown file), and
inline **bold**, *italic* and `code`.
"""

import re
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt

_INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*\s][^*]*?\*)")
_IMAGE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)$")


def _runs(par, text: str) -> None:
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = par.add_run(part[1:-1])
            run.font.name = "Courier New"
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table(doc, lines: list[str]) -> None:
    rows = [_cells(x) for x in lines if not re.fullmatch(r"\|?(\s*-+\s*\|)+\s*-*\s*", x.strip())]
    width = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=width)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j in range(width):
            cell = table.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            _runs(par, row[j] if j < len(row) else "")
            for run in par.runs:
                run.font.size = Pt(8)
                if i == 0:
                    run.bold = True


def markdown_to_docx(markdown: str, path: str | Path, *, base: str | Path) -> Path:
    path, base = Path(path), Path(base)
    doc = Document()
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            _table(doc, block)
            continue
        if m := re.match(r"^(#{1,4})\s+(.*)$", s):
            doc.add_heading(m.group(2), level=len(m.group(1)) - 1)  # "#" = title
        elif m := _IMAGE.match(s):
            img = base / m.group("path")
            if img.exists():
                doc.add_picture(str(img), width=Inches(6))
                cap = doc.add_paragraph()
                cap.add_run(m.group("alt")).italic = True
            else:
                doc.add_paragraph(f"[missing figure: {m.group('path')}]")
        elif s.startswith("- "):
            _runs(doc.add_paragraph(style="List Bullet"), s[2:])
        else:
            # Consecutive text lines form one paragraph (Markdown semantics).
            buf = [s]
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and not re.match(r"^(#|\||- |!\[)", lines[i + 1].strip())):
                i += 1
                buf.append(lines[i].strip())
            _runs(doc.add_paragraph(), " ".join(buf))
        i += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return path
