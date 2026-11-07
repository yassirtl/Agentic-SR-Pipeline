"""Open-access full text for a list of papers (used at full-text screening and extraction).

Routes, tried in order for each paper (checked live on the development data):

1. **PMC id -> NCBI efetch (db=pmc)**: the article as JATS XML, already split into
   sections ("Methods > Participants", tables), so no PDF parsing is needed.
   ``PMC_BATCH`` articles per request. Some publishers forbid XML full text: NCBI
   then returns the front matter only, with a comment saying so; such papers
   fall through to route 2. (PMC's own PDF links answer with a bot challenge
   page, so they are never used.)
2. **Open-access PDF link** (publisher, repository, arXiv): downloaded, accepted
   only if it really is a PDF (``%PDF-`` magic bytes; publishers often answer
   200 with an HTML login page), text extracted page by page with ``pypdf``.
3. Otherwise **abstract only**, with the reason (no open-access copy, HTTP 403,
   not a PDF, no text layer...). The project decision: such papers are screened
   on their abstract and flagged; PRISMA counts them as "reports not retrieved"
   at the full-text stage.

Every answer goes through ``HttpClient`` (disk cache, retries, rate limit), so a
second run is free and gives the same text.
"""

import io
import logging
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, Field

from .base import clean_text
from .http import HttpClient, HttpError
from .pubmed import EUTILS, HOST as NCBI_HOST, TOOL

PMC_BATCH = 20
MAX_PDF_BYTES = 40_000_000
UNUSABLE_PDF_HOSTS = ("pmc.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov", "europepmc.org")
NO_XML_NOTE = "does not allow downloading of the full text"

log = logging.getLogger(__name__)


class Section(BaseModel):
    title: str = Field(description="Section path, e.g. 'Methods > Participants', 'Table 2', 'Page 3'.")
    text: str


class FullText(BaseModel):
    paper_id: str
    status: Literal["pmc_xml", "pdf", "abstract_only"]
    source_url: str | None = None
    reason: str | None = Field(None, description="Why the full text is missing (abstract_only).")
    sections: list[Section] = Field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(len(s.text.split()) for s in self.sections)


# --- JATS (PMC XML) ------------------------------------------------------------

def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return clean_text(" ".join(el.itertext())) or ""


def _table_text(wrap: ET.Element) -> str:
    """Caption + one line per row, cells joined by ' | ' (keeps numbers next to their labels)."""
    lines = [_text(wrap.find("caption"))]
    for row in wrap.iter("tr"):
        cells = [_text(c) for c in row if c.tag in ("td", "th")]
        if any(cells):
            lines.append(" | ".join(cells))
    lines.append(_text(wrap.find("table-wrap-foot")))
    return "\n".join(line for line in lines if line)


def _pop_tables(article: ET.Element) -> list[Section]:
    """Remove every table from the article and return them as sections.

    Tables sit inside paragraphs, sections or a separate <floats-group>; taking
    them out first means paragraph text never repeats a table's cells.
    """
    parent = {child: el for el in article.iter() for child in el}
    tables = []
    for wrap in list(article.iter("table-wrap")):
        parent[wrap].remove(wrap)
        text = _table_text(wrap)
        if text:
            tables.append(Section(title=_text(wrap.find("label")) or "Table", text=text))
    return tables


def _sections(container: ET.Element, path: str) -> list[Section]:
    """Paragraphs directly in ``container`` form one section; nested <sec> recurse."""
    out: list[Section] = []
    paras = [_text(el) for el in container if el.tag in ("p", "list", "disp-quote")]
    if any(paras):
        out.append(Section(title=path or "Body", text="\n".join(p for p in paras if p)))
    for sec in container.findall("sec"):
        title = _text(sec.find("title")) or "Untitled section"
        out.extend(_sections(sec, f"{path} > {title}" if path else title))
    return out


def _pmcid_of(article: ET.Element) -> str | None:
    for el in article.iter("article-id"):
        kind, value = el.get("pub-id-type"), (el.text or "").strip()
        if kind == "pmcid" and value:
            return value.upper()
        if kind in ("pmc", "pmcaid") and value:
            return value.upper() if value.upper().startswith("PMC") else f"PMC{value}"
    return None


def parse_pmc_articles(xml_text: str) -> dict[str, tuple[list[Section], str | None]]:
    """efetch db=pmc answer -> {PMCID: (sections, reason if no body)}."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(xml_text, parser=parser)
    out: dict[str, tuple[list[Section], str | None]] = {}
    for article in root.iter("article"):
        pmcid = _pmcid_of(article)
        if not pmcid:
            continue
        body = article.find("body")
        tables = _pop_tables(article) if body is not None else []
        sections = _sections(body, "") + tables if body is not None else []
        if sections:
            out[pmcid] = (sections, None)
            continue
        comments = " ".join(c.text or "" for c in article.iter(ET.Comment))
        out[pmcid] = ([], "PMC: publisher does not allow full text in XML form"
                      if NO_XML_NOTE in comments else "PMC: no article body")
    return out


def normalize_pmcid(pmcid: str) -> str:
    pmcid = pmcid.strip().upper()
    return pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"


# --- PDF ----------------------------------------------------------------------------

def pdf_sections(data: bytes) -> list[Section]:
    """Text of each page (pypdf). Empty list when the PDF has no text layer (scanned)."""
    from pypdf import PdfReader  # imported lazily: only needed when a PDF is downloaded

    reader = PdfReader(io.BytesIO(data))
    out = []
    for n, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            out.append(Section(title=f"Page {n}", text=text))
    return out


def _fetch_pdf(paper: dict, http: HttpClient) -> FullText:
    url = paper.get("oa_pdf_url")
    pid = paper["paper_id"]
    try:
        data, ctype = http.get_bytes(url)
    except HttpError as exc:
        return FullText(paper_id=pid, status="abstract_only", source_url=url,
                        reason=f"PDF: HTTP {exc.status or 'network error'}")
    if not data.startswith(b"%PDF-"):
        kind = ctype.split(";")[0] or "unknown type"
        return FullText(paper_id=pid, status="abstract_only", source_url=url,
                        reason=f"PDF link returned {kind}, not a PDF")
    if len(data) > MAX_PDF_BYTES:
        return FullText(paper_id=pid, status="abstract_only", source_url=url,
                        reason="PDF too large")
    try:
        sections = pdf_sections(data)
    except Exception as exc:  # pypdf raises many types on broken files
        return FullText(paper_id=pid, status="abstract_only", source_url=url,
                        reason=f"PDF unreadable ({type(exc).__name__})")
    if not sections:
        return FullText(paper_id=pid, status="abstract_only", source_url=url,
                        reason="PDF has no text layer")
    return FullText(paper_id=pid, status="pdf", source_url=url, sections=sections)


# --- Public entry point --------------------------------------------------------------

def fetch_fulltexts(papers: list[dict], http: HttpClient, *, contact_email: str | None = None,
                    env: Mapping[str, str] | None = None, batch: int = PMC_BATCH
                    ) -> dict[str, FullText]:
    """Full text (or the reason there is none) for each paper, keyed by ``paper_id``."""
    api_key = (os.environ if env is None else env).get("NCBI_API_KEY") or None
    http.set_min_interval(NCBI_HOST, 0.11 if api_key else 0.34)

    out: dict[str, FullText] = {}
    reasons: dict[str, str] = {}
    by_pmc = {normalize_pmcid(p["pmcid"]): p for p in papers if p.get("pmcid")}
    ids = list(by_pmc)
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        try:
            xml_text = http.get_text(f"{EUTILS}/efetch.fcgi", {
                "db": "pmc", "id": ",".join(chunk), "tool": TOOL, "email": contact_email,
                "api_key": api_key})
            parsed = parse_pmc_articles(xml_text)
        except (HttpError, ET.ParseError) as exc:
            log.warning("PMC batch failed: %s", exc)
            parsed = {}
            for pmcid in chunk:
                reasons[by_pmc[pmcid]["paper_id"]] = f"PMC: {type(exc).__name__}"
        for pmcid in chunk:
            pid = by_pmc[pmcid]["paper_id"]
            if pmcid not in parsed:
                reasons.setdefault(pid, "PMC: article not returned")
                continue
            sections, reason = parsed[pmcid]
            if sections:
                out[pid] = FullText(paper_id=pid, status="pmc_xml", sections=sections,
                                    source_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/")
            else:
                reasons[pid] = reason

    for p in papers:
        pid = p["paper_id"]
        if pid in out:
            continue
        url = p.get("oa_pdf_url")
        if url and url.split("/")[2] not in UNUSABLE_PDF_HOSTS:
            ft = _fetch_pdf(p, http)
            if ft.status == "abstract_only" and pid in reasons:
                ft.reason = f"{reasons[pid]}; {ft.reason}"
            out[pid] = ft
        else:
            out[pid] = FullText(paper_id=pid, status="abstract_only",
                                reason=reasons.get(pid, "no open-access full text"))
    return {p["paper_id"]: out[p["paper_id"]] for p in papers}
