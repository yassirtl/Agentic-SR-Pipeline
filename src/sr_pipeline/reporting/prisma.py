"""PRISMA 2020 flow diagram (Page et al. 2021, databases-only template), from ``prisma_counts``.

``flow_counts`` turns the counts the stages wrote into the diagram's boxes and checks that
the arithmetic closes; ``flow_diagram`` draws them (matplotlib, PNG).

Project rules that show in the boxes:
- *Removed before screening* (PRISMA 2020's three sub-boxes): duplicates; records
  **marked as ineligible by automation tools** = the Part 6d D title rule (code, before any
  LLM call: the title names the record type, e.g. "a systematic review"), counted here and
  not in "records excluded" (Part 9b fix 2); other reasons: records a database counted but
  that were not downloaded (above the ``max_records`` ceiling, or counted but not returned;
  Part 5 keeps them apart from the full-text "reports not retrieved"), and records above
  the screening limit ("not screened (limit N)", Part 9b fix 5).
- *Full text*: a report with no open-access full text is "not retrieved" in PRISMA terms,
  but by project decision it is still assessed on its abstract and flagged, so it stays in
  the flow ("of which on abstract only") instead of leaving it.
- Exclusion reasons are short labels with the criterion code in brackets ("not a primary
  RCT (D): n = 21"), one per line (Part 9b fix 1).

Layout (Part 9b fix 1): text is wrapped by measuring its rendered width against the box,
and each row is as tall as its wrapped text, so no text leaves its box (tested on the
text extents). ``draw_flow`` returns the figure and its (box, text) pairs for that test.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ..agents.screening import criterion_labels

NBSP = " "   # textwrap-proof space: "n = 1" never splits across lines


def _n(v) -> str:
    return f"n{NBSP}={NBSP}{v}"


def _reasons(by_criterion: dict, labels: dict[str, str]) -> list[str]:
    """One line per reason, most frequent first: 'not a primary RCT (D): n = 21'."""
    return [f"  {labels.get(cid, cid)} ({cid}):{NBSP}{_n(v)}"
            for cid, v in sorted(by_criterion.items(), key=lambda kv: -kv[1]) if v > 0]


def flow_counts(prisma: dict, protocol: dict | None = None) -> dict:
    """Boxes of the diagram (numbers + lines of text), the numbers, and the arithmetic
    checks. ``protocol`` gives the exclusion labels (else the criterion codes are shown)."""
    labels = criterion_labels(protocol) if protocol else {}
    ident = prisma.get("identification") or {}
    ta = prisma.get("title_abstract") or {}
    ft = prisma.get("full_text") or {}
    dbs = ident.get("databases") or {}
    identified = ident.get("identified", 0)
    not_dl = ident.get("not_retrieved", 0)
    dups = ident.get("duplicates_removed", 0)
    # Code-rule exclusions (title rule; Part 10b arXiv rule) move from "records excluded"
    # to the automation-tools box, each subtracted from its own criterion.
    auto_by = dict(ta.get("automation_by_criterion")
                   or ({"D": ta["title_rule"]} if ta.get("title_rule") else {}))
    automation = sum(auto_by.values())
    arxiv = ta.get("arxiv_rule", 0)
    not_screened = ta.get("not_screened", 0)
    limit = ta.get("screen_limit")
    screened = ta.get("records_screened", ident.get("records_screened", 0)) - automation
    ta_excl = ta.get("records_excluded", 0) - automation
    ta_by = dict(ta.get("excluded_by_criterion") or {})
    for cid, k in auto_by.items():
        ta_by[cid] = ta_by.get(cid, 0) - k
        if ta_by[cid] <= 0:
            ta_by.pop(cid)
    sought = ft.get("reports_sought", ta.get("included", 0))
    abstract_only = ft.get("reports_not_retrieved", 0)
    ft_excl = ft.get("reports_excluded", 0)
    included = ft.get("studies_included", 0)

    # A development run that cut the records itself (screen_smoke --limit) has no
    # not_screened count: the first check then fails, and the diagram says so.
    checks = {
        "identified - not downloaded - duplicates - automation - not screened = screened":
            identified - not_dl - dups - automation - not_screened == screened,
        "screened - excluded = sought": screened - ta_excl == sought,
        "sought - excluded = included": sought - ft_excl == included,
    }
    errors = ident.get("errors") or {}
    removed = ["Records removed before screening:",
               f"  Duplicate records ({_n(dups)})",
               f"  Marked as ineligible by automation tools ({_n(automation)}): title "
               "rule, the title names a review, meta-analysis or protocol"
               + (f"; arXiv preprint, not peer reviewed ({_n(arxiv)})" if arxiv else ""),
               f"  Not downloaded ({_n(not_dl)}): above the ceiling of "
               f"{ident.get('max_records', '?')} records per database, or counted but not "
               "returned"]
    if limit or not_screened:
        removed.append(f"  Not screened ({_n(not_screened)}): above the screening limit of "
                       f"{limit} records (best relevance rank kept)")
    return {
        "identification": {
            "n": identified,
            "lines": [f"Records identified from databases ({_n(identified)}):"]
            + [f"  {db}: {_n(d['identified'])}" for db, d in dbs.items()]
            + ([f"  failed: {', '.join(errors)}"] if errors else [])},
        "removed": {"n": not_dl + dups + automation + not_screened, "lines": removed},
        "screened": {"n": screened, "lines": [f"Records screened (title/abstract) "
                                              f"({_n(screened)})"]},
        "ta_excluded": {"n": ta_excl,
                        "lines": [f"Records excluded ({_n(ta_excl)}):"]
                        + _reasons(ta_by, labels)},
        "sought": {"n": sought, "lines": [f"Reports sought for retrieval ({_n(sought)})"]},
        "not_retrieved": {
            "n": abstract_only,
            "lines": [f"Reports not retrieved as full text ({_n(abstract_only)});",
                      "  assessed on the abstract and flagged"]},
        "assessed": {
            "n": sought,
            "lines": [f"Reports assessed for eligibility ({_n(sought)}):",
                      f"  full text {_n(sought - abstract_only)}, "
                      f"abstract only {_n(abstract_only)}"]},
        "ft_excluded": {"n": ft_excl,
                        "lines": [f"Reports excluded ({_n(ft_excl)}):"]
                        + _reasons(ft.get("excluded_by_criterion") or {}, labels)},
        "included": {
            "n": included,
            "lines": [f"Studies included in review ({_n(included)})",
                      f"  of which on abstract only "
                      f"{_n(ft.get('included_on_abstract_only', 0))}"]},
        "numbers": {"identified": identified, "duplicates": dups, "automation": automation,
                    "not_downloaded": not_dl, "not_screened": not_screened,
                    "screen_limit": limit, "screened": screened, "ta_excluded": ta_excl,
                    "sought": sought, "not_retrieved": abstract_only,
                    "ft_excluded": ft_excl, "included": included},
        "checks": checks,
        "consistent": all(checks.values()),
    }


# --- Drawing (units = inches: the axes fill the figure, 1 data unit = 1 inch) ---------------

FONT = 7.5
PAD = 0.08            # text margin inside a box
GAP = 0.32            # vertical space between rows (arrows)
WIDTH = 9.0
BAR_X, BAR_W = 0.10, 0.30
LEFT_X, RIGHT_X, BOX_W = 0.55, 4.75, 3.95
TITLE_H = 0.45


class _Measure:
    """Rendered width / height (inches) of a text at the diagram's font."""

    def __init__(self, fig):
        self.fig = fig
        self.renderer = fig.canvas.get_renderer()
        self.probe = fig.text(0, 0, "", fontsize=FONT)

    def size(self, s: str, **font) -> tuple[float, float]:
        self.probe.set_text(s)
        self.probe.set(fontsize=font.get("fontsize", FONT), weight=font.get("weight", "normal"))
        bb = self.probe.get_window_extent(self.renderer)
        return bb.width / self.fig.dpi, bb.height / self.fig.dpi

    def wrap(self, lines: list[str], width: float) -> str:
        """Greedy word wrap on ASCII spaces (NBSP keeps 'n = 1' together); continuation lines
        indented 2 more spaces than the line they continue."""
        out = []
        for line in lines:
            indent = " " * (len(line) - len(line.lstrip(" ")))
            first, *rest = line.strip(" ").split(" ")
            cur = indent + first
            for w in rest:
                if self.size(f"{cur} {w}")[0] > width:
                    out.append(cur)
                    cur = f"{indent}  {w}"
                else:
                    cur = f"{cur} {w}"
            out.append(cur)
        return "\n".join(out)


def _arrow(ax, x0, y0, x1, y1, *, dashed=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": "#333333", "lw": 1,
                            "linestyle": "--" if dashed else "-"})


ROWS = [("identification", "removed"), ("screened", "ta_excluded"),
        ("sought", "not_retrieved"), ("assessed", "ft_excluded"), ("included", None)]
PHASES = [("Identification", 0, 0), ("Screening", 1, 3), ("Included", 4, 4)]
PHASE_FONT = {"fontsize": 9, "weight": "bold"}


def draw_flow(prisma: dict, protocol: dict | None = None):
    """(figure, [(box patch, text)], title text) of the diagram; the caller saves it. The
    pairs include the phase bars and their labels."""
    c = flow_counts(prisma, protocol)
    fig = plt.figure(figsize=(WIDTH, 10), dpi=150)
    m = _Measure(fig)
    inner = BOX_W - 2 * PAD
    texts = {k: m.wrap(c[k]["lines"], inner) for row in ROWS for k in row if k}
    heights = [max(m.size(texts[k])[1] for k in row if k) + 2 * PAD for row in ROWS]
    for name, first, last in PHASES:   # a phase bar is at least as tall as its label
        span = sum(heights[first:last + 1]) + GAP * (last - first)
        need = m.size(name, **PHASE_FONT)[0] + 2 * PAD
        heights[last] += max(0.0, need - span)
    m.probe.remove()

    total = TITLE_H + sum(heights) + GAP * (len(ROWS) - 1) + 0.1
    fig.set_size_inches(WIDTH, total)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, total)
    ax.axis("off")

    pairs, tops, y = [], [], total - TITLE_H
    for (left, right), h in zip(ROWS, heights):
        bottom = y - h
        tops.append((y, bottom))
        for key, x, face in ((left, LEFT_X, "#ffffff"), (right, RIGHT_X, "#f7f7f7")):
            if key is None:
                continue
            rect = Rectangle((x, bottom), BOX_W, h, facecolor=face, edgecolor="#333333",
                             lw=1)
            ax.add_patch(rect)
            t = ax.text(x + PAD, y - PAD, texts[key], ha="left", va="top", fontsize=FONT)
            pairs.append((rect, t))
        if right:
            # Dashed: reports without full text stay in the flow (assessed on abstract).
            _arrow(ax, LEFT_X + BOX_W, bottom + h / 2, RIGHT_X, bottom + h / 2,
                   dashed=right == "not_retrieved")
        y = bottom - GAP
    for (_, prev_bottom), (top, _) in zip(tops, tops[1:]):
        _arrow(ax, LEFT_X + BOX_W / 2, prev_bottom, LEFT_X + BOX_W / 2, top)
    for name, first, last in PHASES:
        top, bottom = tops[first][0], tops[last][1]
        bar = Rectangle((BAR_X, bottom), BAR_W, top - bottom, facecolor="#cfe2f3",
                        edgecolor="none")
        ax.add_patch(bar)
        pairs.append((bar, ax.text(BAR_X + BAR_W / 2, (top + bottom) / 2, name, rotation=90,
                                   ha="center", va="center", **PHASE_FONT)))
    note = "" if c["consistent"] else "  (counts do not add up: see the report)"
    title = ax.text(WIDTH / 2, total - TITLE_H / 2, "PRISMA 2020 flow diagram" + note,
                    ha="center", va="center", fontsize=11)
    return fig, pairs, title


def flow_diagram(prisma: dict, path: str | Path, protocol: dict | None = None) -> Path:
    """Draw the PRISMA 2020 flow diagram to ``path`` (PNG)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, _, _ = draw_flow(prisma, protocol)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
