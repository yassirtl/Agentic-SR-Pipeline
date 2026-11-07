"""Split a paper into passages ("chunks") for retrieval.

Why chunk: a full text is ~5,000 words. A screener asking "were participants
aged 60 or over?" needs the two or three paragraphs that answer it, not the
whole paper: shorter prompts cost less and the model is not distracted by
thousands of irrelevant words.

How:
- chunk 0 is always title + abstract, so papers without a full text are
  searchable too;
- the full text is cut **by section first** (JATS sections, or PDF pages), so a
  chunk rarely mixes "Methods" and "Results";
- a long section becomes windows of ``size`` words overlapping by ``overlap``
  words, so a sentence cut at a window boundary is still whole in one chunk;
- very short sections (headings with one sentence, e.g. "Ethics approval") are
  merged with the following ones until the chunk is reasonably long;
- each chunk starts with its section path, e.g. ``[Methods > Participants]``:
  that context helps both keyword and meaning-based search.

Words, not tokens: a word count needs no tokenizer download and is stable;
250 English words are about 330 tokens.
"""

from ..sources.fulltext import FullText

MIN_WORDS = 60  # sections shorter than this are merged with the next ones


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(words[start:start + size])
        if start + size >= len(words):
            break
    return out


def chunk_paper(paper: dict, fulltext: FullText | None = None, *, size: int = 250,
                overlap: int = 40) -> list[dict]:
    """Chunks ``{chunk_id, paper_id, section, text}`` of one paper, in reading order."""
    pid = paper["paper_id"]
    chunks: list[dict] = []

    def add(section: str, words: list[str]) -> None:
        chunks.append({"chunk_id": f"{pid}::{len(chunks)}", "paper_id": pid,
                       "section": section, "text": f"[{section}] " + " ".join(words)})

    title = (paper.get("title") or "").strip()
    head = " ".join(p for p in (title + ("." if title and not title.endswith((".", "?", "!"))
                                         else ""), (paper.get("abstract") or "").strip()) if p)
    for window in _windows(head.split() or ["(no title or abstract)"], size, overlap):
        add("Title and abstract", window)

    if fulltext is None or not fulltext.sections:
        return chunks
    pending_titles: list[str] = []
    pending: list[str] = []
    for sec in fulltext.sections:
        words = sec.text.split()
        if not words:
            continue
        if pending and sec.title.startswith("Table"):  # a table stays a chunk of its own
            add(" + ".join(pending_titles), pending)
            pending_titles, pending = [], []
        if len(pending) + len(words) < MIN_WORDS and not sec.title.startswith("Table"):
            pending_titles.append(sec.title)
            pending += words
            continue
        title = " + ".join(pending_titles + [sec.title]) if pending else sec.title
        for window in _windows(pending + words, size, overlap):
            add(title, window)
        pending_titles, pending = [], []
    if pending:
        add(" + ".join(pending_titles), pending)
    return chunks
