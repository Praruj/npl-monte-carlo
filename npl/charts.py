"""Static charts for the README and article (light and dark surfaces, reference data-viz palette)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

THEMES = {
    "light": dict(SURFACE="#fcfcfb", INK="#0b0b0b", INK2="#52514e", MUTED="#a3a29c", GRID="#e6e5e0",
                  BLUE="#2a78d6", ORANGE="#eb6834",
                  SEQ=["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    # Dark steps sit on the site's article background; magnitude runs dark -> light.
    "dark": dict(SURFACE="#0b0d12", INK="#efece4", INK2="#a39d91", MUTED="#56514a", GRID="#23262d",
                 BLUE="#3987e5", ORANGE="#d95926",
                 SEQ=["#0b0d12", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]),
}


def use_theme(name: str) -> None:
    t = THEMES[name]
    g = globals()
    g.update({k: v for k, v in t.items() if k != "SEQ"})
    g["SEQ"] = LinearSegmentedColormap.from_list("seq", t["SEQ"])
    plt.rcParams.update({
        "figure.facecolor": t["SURFACE"], "axes.facecolor": t["SURFACE"], "savefig.facecolor": t["SURFACE"],
        "axes.edgecolor": t["GRID"], "axes.labelcolor": t["INK2"], "xtick.color": t["INK2"],
        "ytick.color": t["INK2"], "text.color": t["INK"], "legend.labelcolor": t["INK"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    })


use_theme("light")


def _label(v: float) -> str:
    if 0 < v < 0.5:
        return "<1%"
    if 99.5 <= v < 100:
        return ">99%"
    return f"{v:.0f}%"


def _title(fig, title, sub):
    fig.text(0.01, 0.985, title, fontsize=13, fontweight="bold", va="top", color=INK)
    fig.text(0.01, 0.94, sub, fontsize=9.5, va="top", color=INK2)


def odds(summary: pd.DataFrame, short: dict, path, as_of: str, n: int):
    """Two panels with the same team order: P(playoffs) and P(title)."""
    s = summary.sort_values(["p_title", "p_playoffs"])
    labels = [f"{t} ({short[t]})" for t in s.index]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, col, name, color in ((axes[0], "p_playoffs", "Makes top 4", BLUE),
                                 (axes[1], "p_title", "Wins the title", BLUE)):
        y = np.arange(len(s))
        ax.barh(y, s[col] * 100, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, s[col] * 100):
            ax.text(v + 1.2, yi, _label(v), va="center", fontsize=9, color=INK2)
        ax.set_xlim(0, 105 if col == "p_playoffs" else max(60, s[col].max() * 115))
        ax.set_title(name, loc="left", color=INK)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(left=False)
        ax.set_xlabel("probability (%)")
    axes[0].set_yticks(np.arange(len(s)), labels)
    _title(fig, "NPL 2026: who makes the playoffs and who wins it",
           f"Ensemble of Elo and ball-by-ball models, {n:,} simulated seasons each, as of {as_of}")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def positions(matrix: pd.DataFrame, short: dict, path, as_of: str):
    m = matrix.loc[matrix.mul(np.arange(1, matrix.shape[1] + 1), axis=1).sum(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.imshow(m.values * 100, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j] * 100
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                        color=SURFACE if v > 45 else INK)
    ax.set_xticks(range(m.shape[1]), [f"{k}" for k in m.columns])
    ax.set_yticks(range(m.shape[0]), [f"{t} ({short[t]})" for t in m.index])
    ax.set_xlabel("final league position")
    ax.tick_params(left=False, bottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(3.5, color=INK2, linewidth=1.2, linestyle=(0, (3, 2)))
    ax.text(3.55, -0.75, "playoff line", fontsize=8.5, color=INK2, va="bottom")
    _title(fig, "Where each team finishes the league stage",
           f"% of simulated seasons (ensemble), as of {as_of}; rows sorted by average position")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def elo_history(history: pd.DataFrame, teams: list, short: dict, path):
    """Small multiples: each panel highlights one team, the rest in muted gray."""
    h = history.copy()
    h["game"] = h.groupby("team").cumcount()
    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=False, sharey=True)
    lo, hi = h.rating.min() - 10, h.rating.max() + 10
    for ax, team in zip(axes.flat, sorted(teams, key=lambda t: -h[h.team == t].rating.iloc[-1])):
        for other in teams:
            g = h[h.team == other]
            ax.plot(g.game, g.rating, color=GRID, linewidth=1)
        g = h[h.team == team]
        ax.plot(g.game, g.rating, color=BLUE, linewidth=2)
        ax.scatter([g.game.iloc[-1]], [g.rating.iloc[-1]], s=22, color=BLUE, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        for season, gs in g.groupby("season"):
            ax.axvline(gs.game.iloc[0] - 0.5, color=GRID, linewidth=0.8, linestyle=":")
        ax.axhline(1500, color=MUTED, linewidth=0.8)
        ax.set_title(f"{short[team]}  {g.rating.iloc[-1]:.0f}", loc="left", fontsize=10.5)
        ax.set_ylim(lo, hi)
        ax.tick_params(left=False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    for ax in axes[1]:
        ax.set_xlabel("NPL games played (2024 → 2026)")
    _title(fig, "Team Elo ratings across three NPL seasons",
           "Blue = the named team, gray = everyone else; dotted lines mark season starts; 1500 = average")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def model_compare(by_model: dict, short: dict, path):
    """Dot plot of P(title) under each model; two series, legend + direct labels."""
    names = list(by_model)
    base = by_model[names[0]]
    order = (sum(by_model[k].p_title for k in names) / len(names)).sort_values().index
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(order))
    for k, color, off in zip(names, (BLUE, ORANGE), (-0.12, 0.12)):
        v = by_model[k].p_title.reindex(order).values * 100
        ax.scatter(v, y + off, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, label=k)
    for yi, t in enumerate(order):
        vals = [by_model[k].p_title[t] * 100 for k in names]
        ax.plot(vals, [yi - 0.12, yi + 0.12], color=GRID, linewidth=1.5, zorder=1)
    ax.set_yticks(y, [f"{t} ({short[t]})" for t in order])
    ax.set_xlabel("P(wins the title), %")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(left=False)
    ax.legend(frameon=False, loc="lower right")
    _title(fig, "Title chances under each model",
           "Elo uses team results only; ball-by-ball uses current XIs and player rates")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160)
    plt.close(fig)
