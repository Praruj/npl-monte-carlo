"""Points table and net run rate, following standard T20 league playing conditions.

* 2 points for a win, 1 for a no result, 0 for a loss.
* NRR = runs scored / overs faced - runs conceded / overs bowled.
  A side bowled out is counted as having faced its full quota of overs.
  No-result matches do not count towards NRR.
* Ranking: points, then NRR, then a coin flip (exact NRR ties do not happen in practice).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import NR_POINTS, TEAMS, WIN_POINTS


def overs_to_balls(o) -> int:
    """'16.5' (16 overs and 5 balls) -> 101 legal balls."""
    o = float(o)
    whole = int(o)
    return whole * 6 + int(round((o - whole) * 10))


def nrr_overs(overs, wkts, overs_limit) -> float:
    """Overs that count for NRR: full allocation if bowled out, else overs actually faced."""
    return float(overs_limit) if int(wkts) >= 10 else overs_to_balls(overs) / 6


def standings(fixtures: pd.DataFrame, teams=TEAMS) -> pd.DataFrame:
    rows = {t: dict(P=0, W=0, L=0, NR=0, Pts=0, RF=0.0, OF=0.0, RA=0.0, OA=0.0) for t in teams}
    for f in fixtures.itertuples():
        if f.status == "no_result":
            for t in (f.bat_first, f.bat_second):
                rows[t]["P"] += 1
                rows[t]["NR"] += 1
                rows[t]["Pts"] += NR_POINTS
            continue
        if f.status != "result":
            continue
        o1 = nrr_overs(f.o1, f.w1, f.overs_limit)
        o2 = nrr_overs(f.o2, f.w2, f.overs_limit)
        for t, rf, of, ra, oa in ((f.bat_first, f.r1, o1, f.r2, o2),
                                  (f.bat_second, f.r2, o2, f.r1, o1)):
            r = rows[t]
            r["P"] += 1
            r["RF"] += rf
            r["OF"] += of
            r["RA"] += ra
            r["OA"] += oa
            if f.winner == t:
                r["W"] += 1
                r["Pts"] += WIN_POINTS
            else:
                r["L"] += 1
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["NRR"] = np.where(df.OF > 0, df.RF / df.OF.where(df.OF > 0, 1), 0) - np.where(
        df.OA > 0, df.RA / df.OA.where(df.OA > 0, 1), 0)
    df = df.sort_values(["Pts", "NRR"], ascending=False)
    df.index.name = "Team"
    return df


def rank(points: np.ndarray, nrr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-wise ranking of (n_sims, n_teams) arrays. Returns team indices, best first."""
    coin = rng.random(points.shape)
    return np.lexsort((coin, -nrr, -points), axis=-1)
