"""Team Elo ratings for the NPL.

* Every team starts at 1500 in 2024.
* Between seasons each rating is pulled part of the way back to 1500, because NPL squads
  turn over heavily from one auction to the next.
* The chasing side gets a bonus of H rating points, set from the share of NPL games won by
  the side batting second (all NPL games are played at Kirtipur).
* The update size grows with the margin: a 40-run (or run-equivalent) win counts double.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import TEAMS

BASE = 1500.0


def expected(r_chase: float, r_bat: float, h: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_chase + h - r_bat) / 400.0))


def chase_bonus(results: pd.DataFrame) -> float:
    """Elo points equivalent to the observed chase win share (ties excluded)."""
    decided = results[~results.tie]
    p = (decided.winner == decided.team2).mean()
    p = float(np.clip(p, 0.35, 0.65))
    return 400.0 * np.log10(p / (1 - p))


def margin_runs_equivalent(row) -> float:
    if row.tie:
        return 0.0
    if row.winner == row.team1:
        return float(row.margin_runs)
    # Chases: a ball to spare is worth ~1.25 runs, a wicket in hand ~3 runs.
    return 1.25 * float(row.chase_balls_left) + 3.0 * float(row.chase_wkts_left)


@dataclass
class EloResult:
    ratings: dict
    history: pd.DataFrame          # one row per team per match (after the update)
    predictions: pd.DataFrame      # pre-match P(team batting second wins)
    season_start: dict = field(default_factory=dict)  # season -> ratings at its start


def run_elo(results: pd.DataFrame, k: float = 32.0, regress: float = 0.5,
            h: float | None = None) -> EloResult:
    h = chase_bonus(results) if h is None else h
    ratings = {t: BASE for t in TEAMS}
    hist, preds, starts = [], [], {}
    season = None
    for row in results.itertuples():
        if row.season != season:
            if season is not None:
                ratings = {t: BASE + (r - BASE) * (1 - regress) for t, r in ratings.items()}
            season = row.season
            starts[season] = dict(ratings)
        rb, rc = ratings[row.team1], ratings[row.team2]
        p_chase = expected(rc, rb, h)
        preds.append(dict(date=row.date, season=row.season, team1=row.team1, team2=row.team2,
                          p_chase=p_chase, chase_won=float(row.winner == row.team2) if not row.tie else 0.5))
        s = 0.5 if row.tie else float(row.winner == row.team2)
        mult = 1.0 + min(margin_runs_equivalent(row) / 40.0, 1.0)
        delta = k * mult * (s - p_chase)
        ratings[row.team2] = rc + delta
        ratings[row.team1] = rb - delta
        for t in (row.team1, row.team2):
            hist.append(dict(date=row.date, season=row.season, team=t, rating=ratings[t]))
    return EloResult(ratings=ratings, history=pd.DataFrame(hist), predictions=pd.DataFrame(preds),
                     season_start=starts)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune(results: pd.DataFrame, ks=(16, 24, 32, 48), regresses=(0.25, 0.5, 0.75),
         score_from: int = 2025) -> tuple[dict, pd.DataFrame]:
    """Grid-search K and between-season regression on log loss for seasons >= score_from."""
    rows = []
    for k in ks:
        for rg in regresses:
            pr = run_elo(results, k=k, regress=rg).predictions
            pr = pr[(pr.season >= score_from) & (pr.chase_won != 0.5)]
            rows.append(dict(k=k, regress=rg, log_loss=log_loss(pr.p_chase.values, pr.chase_won.values),
                             n=len(pr)))
    grid = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    best = grid.iloc[0]
    return dict(k=float(best.k), regress=float(best.regress)), grid
