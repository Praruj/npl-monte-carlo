"""Monte Carlo season simulation: remaining league games, points table with NRR, then the
IPL-style playoff (Qualifier 1: 1v2, Eliminator: 3v4, Qualifier 2: loser Q1 v winner E,
Final: winner Q1 v winner Q2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import NR_POINTS, PLAYOFF_TEAMS, TEAMS, TEAM_IDX, WIN_POINTS
from .table import nrr_overs, rank


@dataclass
class SeasonSim:
    order: np.ndarray        # (n, T) final league order, team indices best first
    points: np.ndarray       # (n, T)
    nrr: np.ndarray          # (n, T)
    finalists: np.ndarray    # (n, 2)
    champion: np.ndarray     # (n,)

    def summary(self, teams=TEAMS) -> pd.DataFrame:
        n, T = self.order.shape
        pos = np.argsort(self.order, axis=1)          # (n, T) league position of each team
        df = pd.DataFrame(index=teams)
        df["exp_points"] = self.points.mean(0)
        df["p_top"] = (pos == 0).mean(0)
        df["p_playoffs"] = (pos < PLAYOFF_TEAMS).mean(0)
        df["p_final"] = np.array([(self.finalists == t).any(1).mean() for t in range(T)])
        df["p_title"] = np.bincount(self.champion, minlength=T) / n
        df.index.name = "Team"
        return df

    def position_matrix(self, teams=TEAMS) -> pd.DataFrame:
        n, T = self.order.shape
        pos = np.argsort(self.order, axis=1)
        m = np.stack([np.bincount(pos[:, t], minlength=T) / n for t in range(T)])
        return pd.DataFrame(m, index=teams, columns=[f"{k + 1}" for k in range(T)])


def _apply(acc, sims, res, nr_mask=None):
    pts, wins, rf, of, ra, oa = acc
    first, chase, win = res["bat_first"], res["chase"], res["winner"]
    keep = np.ones(len(sims), bool) if nr_mask is None else ~nr_mask
    s, f, c, w = sims[keep], first[keep], chase[keep], win[keep]
    np.add.at(pts, (s, w), WIN_POINTS)
    np.add.at(wins, (s, w), 1)
    for team, r_for, o_for, r_ag, o_ag in ((f, "r1", "o1", "r2", "o2"), (c, "r2", "o2", "r1", "o1")):
        np.add.at(rf, (s, team), res[r_for][keep])
        np.add.at(of, (s, team), res[o_for][keep])
        np.add.at(ra, (s, team), res[r_ag][keep])
        np.add.at(oa, (s, team), res[o_ag][keep])
    if nr_mask is not None and nr_mask.any():
        np.add.at(pts, (sims[nr_mask], first[nr_mask]), NR_POINTS)
        np.add.at(pts, (sims[nr_mask], chase[nr_mask]), NR_POINTS)


def simulate_season(fixtures: pd.DataFrame, model, n: int = 10000, seed: int = 0,
                    p_no_result: float = 0.0, from_scratch: bool = False) -> SeasonSim:
    rng = np.random.default_rng(seed)
    T = len(TEAMS)
    acc = [np.zeros((n, T)) for _ in range(6)]
    pts, wins, rf, of, ra, oa = acc
    sims = np.arange(n)
    for f in fixtures.itertuples():
        a, b = TEAM_IDX[f.bat_first], TEAM_IDX[f.bat_second]
        if not from_scratch and f.status == "no_result":
            pts[:, [a, b]] += NR_POINTS
        elif not from_scratch and f.status == "result":
            o1 = nrr_overs(f.o1, f.w1, f.overs_limit)
            o2 = nrr_overs(f.o2, f.w2, f.overs_limit)
            res = dict(bat_first=np.full(n, a), chase=np.full(n, b), r1=np.full(n, f.r1),
                       o1=np.full(n, o1), r2=np.full(n, f.r2), o2=np.full(n, o2),
                       winner=np.full(n, TEAM_IDX[f.winner]))
            _apply(acc, sims, res)
        else:
            res = model.play(np.full(n, a), np.full(n, b), rng)
            _apply(acc, sims, res, nr_mask=rng.random(n) < p_no_result)
    nrr = np.divide(rf, of, out=np.zeros_like(rf), where=of > 0) - np.divide(
        ra, oa, out=np.zeros_like(ra), where=oa > 0)
    order = rank(pts, nrr, rng)
    finalists, champion = playoffs(order[:, :PLAYOFF_TEAMS], model, rng)
    return SeasonSim(order=order, points=pts, nrr=nrr, finalists=finalists, champion=champion)


def playoffs(seeds: np.ndarray, model, rng):
    """seeds (n, 4) team indices in league order. Knockouts cannot end in a no result here."""
    def winner(a, b):
        w = model.play(a, b, rng)["winner"]
        return w, np.where(w == a, b, a)

    q1_w, q1_l = winner(seeds[:, 0], seeds[:, 1])
    el_w, _ = winner(seeds[:, 2], seeds[:, 3])
    q2_w, _ = winner(q1_l, el_w)
    champ, _ = winner(q1_w, q2_w)
    return np.stack([q1_w, q2_w], 1), champ


def no_result_rate(n_nr: int, n_scheduled: int) -> float:
    """Observed share of NPL league games lost to weather, with a half-game prior."""
    return (n_nr + 0.5) / (n_scheduled + 1)
