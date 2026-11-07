"""Forest plots (matplotlib PNG, no display needed).

- ``forest_plot``: one effect per study (Hedges' g with 95% CI, square size = weight) and
  the pooled random-effects diamond;
- ``paired_forest``: diagnostic accuracy, sensitivity and specificity side by side (Wilson
  95% CI), pooled diamonds when given.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _diamond(ax, y: float, lo: float, est: float, hi: float) -> None:
    ax.fill([lo, est, hi, est], [y, y + 0.3, y, y - 0.3], color="black")


def forest_plot(rows: list[dict], pooled: dict | None, path: str | Path, *, xlabel: str,
                title: str = "", null: float = 0.0) -> Path:
    """``rows``: {label, est, lo, hi, weight (0-1 or None)}; ``pooled``: {est, lo, hi,
    label} or None."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows) + (2 if pooled else 0)
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.4 * n))
    for i, r in enumerate(rows):
        y = n - i
        ax.plot([r["lo"], r["hi"]], [y, y], color="black", lw=1)
        size = 30 + 250 * (r.get("weight") or 0.1)
        ax.scatter([r["est"]], [y], marker="s", s=size, color="tab:blue", zorder=3)
        ax.text(1.02, y, f"{r['est']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    labels = [r["label"] for r in rows]
    ys = [n - i for i in range(len(rows))]
    if pooled:
        _diamond(ax, 1, pooled["lo"], pooled["est"], pooled["hi"])
        ax.text(1.02, 1, f"{pooled['est']:.2f} [{pooled['lo']:.2f}, {pooled['hi']:.2f}]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                fontweight="bold")
        labels.append(pooled.get("label", "Random effects"))
        ys.append(1)
    ax.axvline(null, color="grey", ls="--", lw=0.8)
    ax.set_yticks(ys, labels, fontsize=8)
    ax.set_ylim(0.3, n + 0.7)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def paired_forest(rows: list[dict], pooled: dict | None, path: str | Path, *,
                  title: str = "") -> Path:
    """``rows``: {label, sens: (p, lo, hi), spec: (p, lo, hi)}; ``pooled``: {sens: (p, lo,
    hi), spec: (p, lo, hi)} or None."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows) + (2 if pooled else 0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 1.2 + 0.4 * n), sharey=True)
    for ax, key, name in ((axes[0], "sens", "Sensitivity"), (axes[1], "spec", "Specificity")):
        for i, r in enumerate(rows):
            p, lo, hi = r[key]
            y = n - i
            ax.plot([lo, hi], [y, y], color="black", lw=1)
            ax.scatter([p], [y], marker="s", s=40, color="tab:blue", zorder=3)
        if pooled:
            p, lo, hi = pooled[key]
            _diamond(ax, 1, lo, p, hi)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(f"{name} (95% CI)")
    labels = [r["label"] for r in rows] + (["Pooled (univariate)"] if pooled else [])
    ys = [n - i for i in range(len(rows))] + ([1] if pooled else [])
    axes[0].set_yticks(ys, labels, fontsize=8)
    axes[0].set_ylim(0.3, n + 0.7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
