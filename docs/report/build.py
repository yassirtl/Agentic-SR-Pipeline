"""Technical report: docs/report/report.md -> report.html -> report.pdf.

    .venv/bin/pip install -e ".[report]"
    .venv/bin/python docs/report/build.py

The PDF is printed by a Chromium browser in headless mode (Chrome, Chromium, Brave or
Edge; set REPORT_BROWSER to its executable to choose one). No LaTeX or pandoc needed.
Images are referenced relative to report.md (e.g. figures/prisma_T3.png).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome", "chromium", "chromium-browser", "brave-browser",
]

CSS = """
@page { size: A4; margin: 18mm 17mm; }
body { font-family: "Times New Roman", Georgia, serif; font-size: 10.5pt; line-height: 1.38;
       color: #111; background: #fff; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; border-bottom: 1px solid #999; }
h3 { font-size: 11pt; margin: 11pt 0 4pt; }
p, li { text-align: justify; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9pt; width: 100%; }
th, td { border: 1px solid #888; padding: 2pt 4pt; vertical-align: top; text-align: left; }
th { background: #eee; }
code, pre { font-family: Menlo, Consolas, monospace; font-size: 8.5pt; }
pre { background: #f4f4f4; padding: 5pt; white-space: pre-wrap; }
img { max-width: 100%; display: block; margin: 6pt auto; }
.pending { color: #a00; }
"""


def browser() -> str:
    for cand in [os.environ.get("REPORT_BROWSER")] + CANDIDATES:
        if cand and (Path(cand).exists() or shutil.which(cand)):
            return cand
    raise SystemExit("no Chromium browser found: set REPORT_BROWSER to its executable")


def build(src: Path = HERE / "report.md") -> Path:
    body = markdown.markdown(src.read_text(), extensions=["tables", "fenced_code"])
    html = src.with_suffix(".html")
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'>"
                    f"<title>Technical report</title><style>{CSS}</style></head>"
                    f"<body>{body}</body></html>")
    pdf = src.with_suffix(".pdf")
    subprocess.run([browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()],
                   check=True, capture_output=True, timeout=120)
    return pdf


if __name__ == "__main__":
    print(build(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "report.md"))
