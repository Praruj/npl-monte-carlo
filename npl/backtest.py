"""Out-of-sample match predictions for NPL 2025 and 2026 (games with Cricsheet line-ups).

For each game, the ball-by-ball model is refitted on balls bowled before that date and
fed the two actual XIs and the actual batting order of the game. Elo predictions are its
online pre-match probabilities. Everything is scored as P(side batting second wins).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .elo import log_loss
from .models import BallByBallModel
from .players import RateParams, bowling_sequence, estimate, match_lineup


def ball_model_predictions(matches, balls, lineups, seasons=(2025, 2026), n=2000, seed=1,
                           params: RateParams = RateParams()) -> pd.DataFrame:
    games = matches[(matches.comp == "NPL") & matches.season.isin(seasons) & ~matches.no_result
                    & matches.r2.notna()].sort_values("date")
    rng = np.random.default_rng(seed)
    rows, cache = [], {}
    for g in games.itertuples():
        if g.date not in cache:
            cache = {g.date: estimate(balls, asof=g.date, params=params)}
        rates = cache[g.date]
        idx = rates.player_index
        lu = lineups[g.match_id]
        bat1, bowlers1 = match_lineup(lu[g.team1])
        bat2, bowlers2 = match_lineup(lu[g.team2])

        def rows_of(bat, bowlers):
            b = np.array([idx.get(p, 0) for p in bat] + [0] * (11 - len(bat)))
            w = np.array([idx.get(p, 0) for p in bowling_sequence(bowlers)])
            return np.tile(b, (n, 1)), np.tile(w, (n, 1))

        model = BallByBallModel.__new__(BallByBallModel)
        model.rates = rates
        b1, w1 = rows_of(bat1, bowlers1)
        b2, w2 = rows_of(bat2, bowlers2)
        overs = int(g.overs_limit) if not pd.isna(g.overs_limit) else 20
        res = model.play_xi(b1, w1, b2, w2, rng, overs=overs)
        y = 0.5 if g.tie else float(g.winner == g.team2)
        rows.append(dict(date=g.date, season=g.season, team1=g.team1, team2=g.team2,
                         p_ball=float(res["chase_won"].mean()), y=y,
                         sim_r1=float(res["r1"].mean()), real_r1=g.r1))
    return pd.DataFrame(rows)


def score(preds: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """cols: label -> probability column. Ties are dropped from scoring."""
    d = preds[preds.y != 0.5]
    out = []
    for label, c in cols.items():
        p, y = d[c].values, d.y.values
        out.append(dict(model=label, n=len(d), log_loss=log_loss(p, y),
                        brier=float(np.mean((p - y) ** 2)), accuracy=float(np.mean((p > 0.5) == y))))
    return pd.DataFrame(out)
