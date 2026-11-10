"""Recall funnel figure for the report: gold studies still in at each stage, per run.

    .venv/bin/python docs/report/funnel.py   -> docs/report/figures/funnel.png

Reads the tracked evaluations in evaluation/results/ (no run folder needed).
One panel per gold topic (T1, T3), one line per pipeline version.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evaluation" / "results"
OUT = Path(__file__).resolve().parent / "figures" / "funnel.png"

STAGES = ["gold", "covered", "found", "screened", "ta_included", "included"]
STAGE_LABELS = ["gold list", "in OpenAlex\nor PubMed", "found by\nsearch",
                "screened", "kept at\ntitle/abstract", "kept at\nfull text"]
# (label, colour, marker): fixed order, same version = same colour in both panels
VERSIONS = [("v1 held-out, limit 100", "#2a78d6", "o"),
            ("v2 search fixes, limit 100", "#eb6834", "s"),
            ("v2, limit 500", "#1baf7a", "^"),
            ("v2 + D fix, limit 500", "#eda100", "D")]
RUNS = {"T1 smartwatches (Nazarian 2021)": ["20261108-173514-4dabd6", "20261108-181121-3bbddf",
                                            "20261108-183527-3491f8", "20261108-204319-7daa7f"],
        "T3 embedded ECG (Arredondo-Velázquez 2026)": ["20261108-164532-b6cfd7",
                                                       "20261108-181123-8431c1",
                                                       "20261108-183528-6c9efd",
                                                       "20261108-204320-92ab19"]}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def counts(run_id: str) -> list[int]:
    d = json.loads((RESULTS / f"{run_id}.json").read_text())
    c = d["completeness"]["gold_recall"]["counts"]
    return [c[s] for s in STAGES]


def main() -> Path:
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=False)
    x = range(len(STAGES))
    for ax, (title, runs) in zip(axes, RUNS.items()):
        series = [counts(r) for r in runs]
        for (label, colour, marker), ys in zip(VERSIONS, series):
            ax.plot(x, ys, color=colour, lw=2, marker=marker, ms=6, label=label,
                    markeredgecolor="#fcfcfb", markeredgewidth=1, zorder=3)
        # direct labels: final count per version, stacked so they never collide
        finals = sorted(((ys[-1], v) for v, ys in zip(VERSIONS, series)), reverse=True)
        top = series[0][0]
        for i, (val, (label, colour, _)) in enumerate(finals):
            ax.annotate(f"{val}/{top}", xy=(len(STAGES) - 1, val),
                        xytext=(len(STAGES) - 0.62, top * (0.62 - 0.11 * i)),
                        color=INK, fontsize=8, va="center",
                        arrowprops=dict(arrowstyle="-", color=colour, lw=1))
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xticks(list(x), STAGE_LABELS, fontsize=7.5, color=MUTED)
        ax.set_xlim(-0.3, len(STAGES) - 0.1)
        ax.set_ylim(0, top * 1.08)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_facecolor("#fcfcfb")
        ax.set_ylabel("gold studies still in", color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="#fcfcfb")
    return OUT


if __name__ == "__main__":
    print(main())
