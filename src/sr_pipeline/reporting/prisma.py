"""PRISMA 2020 flow diagram (Page et al. 2021, databases-only template), from ``prisma_counts``.

``flow_counts`` turns the counts the stages wrote into the diagram's boxes and checks that
the arithmetic closes; ``flow_diagram`` draws them (matplotlib, PNG).

Two project rules show in the boxes:
- *Identification*: records a database counted but that were not downloaded (above the
  ``max_records`` ceiling, or counted but not returned) are "removed before screening:
  not downloaded". Part 5 keeps them apart from the full-text "reports not retrieved".
- *Full text*: a report with no open-access full text is "not retrieved" in PRISMA terms,
  but by project decision it is still assessed on its abstract and flagged, so it stays in
  the flow ("of which on abstract only") instead of leaving it.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


def _by_criterion(d: dict) -> str:
    return "; ".join(f"{k} n = {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))


def flow_counts(prisma: dict) -> dict:
    """Boxes of the diagram (numbers + lines of text) and the arithmetic checks."""
    ident = prisma.get("identification") or {}
    ta = prisma.get("title_abstract") or {}
    ft = prisma.get("full_text") or {}
    dbs = ident.get("databases") or {}
    identified = ident.get("identified", 0)
    not_dl = ident.get("not_retrieved", 0)
    dups = ident.get("duplicates_removed", 0)
    screened = ta.get("records_screened", ident.get("records_screened", 0))
    ta_excl = ta.get("records_excluded", 0)
    sought = ft.get("reports_sought", ta.get("included", 0))
    abstract_only = ft.get("reports_not_retrieved", 0)
    ft_excl = ft.get("reports_excluded", 0)
    included = ft.get("studies_included", 0)

    # A development run with --limit screens a sample: the first check then fails, and
    # the diagram says so (it never hides a gap).
    checks = {
        "identified - not downloaded - duplicates = screened":
            identified - not_dl - dups == screened,
        "screened - excluded = sought": screened - ta_excl == sought,
        "sought - excluded = included": sought - ft_excl == included,
    }
    errors = ident.get("errors") or {}
    return {
        "identification": {
            "n": identified,
            "lines": [f"Records identified from databases (n = {identified}):"]
            + [f"  {db}: n = {d['identified']}" for db, d in dbs.items()]
            + ([f"  failed: {', '.join(errors)}"] if errors else [])},
        "removed": {
            "n": not_dl + dups,
            "lines": ["Records removed before screening:",
                      f"  duplicate records (n = {dups})",
                      f"  not downloaded (n = {not_dl}; above the max_records "
                      f"ceiling of {ident.get('max_records', '?')} per database, or counted "
                      "but not returned)"]},
        "screened": {"n": screened, "lines": [f"Records screened (title/abstract) "
                                              f"(n = {screened})"]},
        "ta_excluded": {
            "n": ta_excl,
            "lines": [f"Records excluded (n = {ta_excl})"]
            + ([f"  {_by_criterion(ta.get('excluded_by_criterion') or {})}"]
               if ta.get("excluded_by_criterion") else [])},
        "sought": {"n": sought, "lines": [f"Reports sought for retrieval (n = {sought})"]},
        "not_retrieved": {
            "n": abstract_only,
            "lines": [f"Reports not retrieved as full text (n = {abstract_only});",
                      "  assessed on the abstract and flagged"]},
        "assessed": {
            "n": sought,
            "lines": [f"Reports assessed for eligibility (n = {sought}):",
                      f"  full text n = {sought - abstract_only}, "
                      f"abstract only n = {abstract_only}"]},
        "ft_excluded": {
            "n": ft_excl,
            "lines": [f"Reports excluded (n = {ft_excl}):"]
            + ([f"  {_by_criterion(ft.get('excluded_by_criterion') or {})}"]
               if ft.get("excluded_by_criterion") else [])},
        "included": {
            "n": included,
            "lines": [f"Studies included in review (n = {included})",
                      f"  of which on abstract only n = "
                      f"{ft.get('included_on_abstract_only', 0)}"]},
        "checks": checks,
        "consistent": all(checks.values()),
    }


def _wrap(lines: list[str], width: int = 52) -> str:
    out = []
    for line in lines:
        indent = " " * (len(line) - len(line.lstrip()))
        out += textwrap.wrap(line, width, subsequent_indent=indent + "  ") or [""]
    return "\n".join(out)


def _box(ax, x, y, w, h, lines, *, face="#ffffff"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor="#333333", lw=1))
    ax.text(x + 0.012, y + h - 0.012, _wrap(lines), ha="left", va="top", fontsize=7.5)


def _arrow(ax, x0, y0, x1, y1, *, dashed=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": "#333333", "lw": 1,
                            "linestyle": "--" if dashed else "-"})


def flow_diagram(prisma: dict, path: str | Path) -> Path:
    """Draw the PRISMA 2020 flow diagram to ``path`` (PNG)."""
    c = flow_counts(prisma)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lx, rx, w, rw = 0.08, 0.56, 0.42, 0.42
    rows = {  # (left box, right box, y, height)
        "Identification": ("identification", "removed", 0.80, 0.15),
        "Screening": ("screened", "ta_excluded", 0.62, 0.10),
        "": ("sought", "not_retrieved", 0.46, 0.09),
        " ": ("assessed", "ft_excluded", 0.28, 0.12),
        "Included": ("included", None, 0.10, 0.09),
    }
    phases = {"Identification": (0.80, 0.15), "Screening": (0.28, 0.44),
              "Included": (0.10, 0.09)}
    for name, (y, h) in phases.items():
        ax.add_patch(Rectangle((0.0, y), 0.05, h, facecolor="#cfe2f3", edgecolor="none"))
        ax.text(0.025, y + h / 2, name, rotation=90, ha="center", va="center", fontsize=9,
                weight="bold")
    prev = None
    for left, right, y, h in rows.values():
        _box(ax, lx, y, w, h, c[left]["lines"])
        if right:
            _box(ax, rx, y, rw, h, c[right]["lines"], face="#f7f7f7")
            # Dashed: reports without full text stay in the flow (assessed on abstract).
            _arrow(ax, lx + w, y + h / 2, rx, y + h / 2, dashed=right == "not_retrieved")
        if prev is not None:
            _arrow(ax, lx + w / 2, prev, lx + w / 2, y + h)
        prev = y
    note = "" if c["consistent"] else "  (counts do not add up: see the report)"
    ax.set_title("PRISMA 2020 flow diagram" + note, fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
