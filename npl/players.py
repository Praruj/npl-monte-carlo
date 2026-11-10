"""Player-level ball outcome rates.

The baseline is the NPL itself (every NPL game is at Kirtipur), split by innings and phase.
Players are described by multipliers on that baseline, one per outcome (0,1,2,3,4,6,W):

    mult[o] = (observed[o] + k * b[o]) / (expected[o] + k * b[o])

`expected` sums, ball by ball, what an average player would have produced in the same
competition, innings and phase. So T20I balls are judged against the T20I baseline and NPL
balls against the NPL baseline, then the relative effect is applied at Kirtipur scoring
levels. T20I balls get a lower weight, all balls decay with age, and `k` pseudo-balls pull
players with little data back to average.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import OUTCOMES

N_OUT = len(OUTCOMES)


@dataclass
class RateParams:
    half_life_days: float = 730.0
    t20i_weight: float = 0.6
    k_bat: float = 120.0
    k_bowl: float = 150.0


@dataclass
class Rates:
    base: np.ndarray          # (2 innings, 3 phases, 7) NPL outcome probabilities
    extras: np.ndarray        # (2, 3) NPL wides+no-balls per legal ball
    player_index: dict        # player id -> row in the mult arrays (row 0 = unknown/average)
    bat_mult: np.ndarray      # (P, 7)
    bowl_mult: np.ndarray     # (P, 7)
    bat_balls: dict           # player id -> weighted balls faced (for reporting)


def _baselines(legal: pd.DataFrame) -> dict:
    out = {}
    for (comp, inn, ph), g in legal.groupby(["comp", "inning", "phase"]):
        cnt = np.bincount(g.outcome.values, minlength=N_OUT).astype(float)
        out[(comp, inn, ph)] = cnt / cnt.sum()
    return out


def estimate(balls: pd.DataFrame, asof=None, params: RateParams = RateParams()) -> Rates:
    """Fit rates from all balls strictly before `asof` (None = use everything)."""
    df = balls if asof is None else balls[balls.date < pd.Timestamp(asof)]
    legal = df[df.legal]
    bl = _baselines(legal)

    npl_legal = legal[legal.comp == "NPL"]
    npl_all = df[df.comp == "NPL"]
    base = np.zeros((2, 3, N_OUT))
    extras = np.zeros((2, 3))
    for inn in range(2):
        for ph in range(3):
            base[inn, ph] = bl[("NPL", inn, ph)]
            seg = npl_all[(npl_all.inning == inn) & (npl_all.phase == ph)]
            extras[inn, ph] = (~seg.legal).sum() / max(seg.legal.sum(), 1)

    ref = pd.Timestamp(asof) if asof is not None else df.date.max() + pd.Timedelta(days=1)
    age = (ref - legal.date).dt.days.values
    w = 0.5 ** (age / params.half_life_days)
    w = w * np.where(legal.comp.values == "T20I", params.t20i_weight, 1.0)

    keys = list(zip(legal.comp.values, legal.inning.values, legal.phase.values))
    key_codes, uniq = pd.factorize(pd.Series(keys))
    base_rows = np.stack([bl[k] for k in uniq])[key_codes]          # (n, 7)
    onehot = np.eye(N_OUT)[legal.outcome.values]                    # (n, 7)
    prior = npl_legal.outcome.value_counts(normalize=True).reindex(range(N_OUT), fill_value=0).values

    ids = pd.Index(pd.unique(np.concatenate([legal.batter.values, legal.bowler.values])))
    player_index = {pid: i + 1 for i, pid in enumerate(ids)}

    def mults(col: str, k: float):
        codes = ids.get_indexer(legal[col].values)
        obs = np.zeros((len(ids), N_OUT))
        exp = np.zeros((len(ids), N_OUT))
        np.add.at(obs, codes, onehot * w[:, None])
        np.add.at(exp, codes, base_rows * w[:, None])
        m = (obs + k * prior) / (exp + k * prior)
        return np.vstack([np.ones(N_OUT), m]), obs.sum(1)

    bat_mult, bat_n = mults("batter", params.k_bat)
    bowl_mult, _ = mults("bowler", params.k_bowl)
    return Rates(base=base, extras=extras, player_index=player_index, bat_mult=bat_mult,
                 bowl_mult=bowl_mult, bat_balls=dict(zip(ids, bat_n)))


# Bowler slots for overs 1..20. Each of five bowlers gets four overs, never two in a row;
# slot 0 (the side's main bowler) takes the 1st, 3rd, 17th and 20th.
OVER_PATTERN = [0, 1, 0, 1, 2, 3, 2, 3, 4, 2, 4, 3, 4, 1, 3, 4, 0, 2, 1, 0]


def bowling_sequence(bowlers: list) -> list:
    bowlers = list(bowlers)[:5]
    while len(bowlers) < 5:
        bowlers.append(bowlers[len(bowlers) % max(len(bowlers), 1)] if bowlers else None)
    return [bowlers[s] for s in OVER_PATTERN]


def match_lineup(lineup_team: dict) -> tuple[list, list]:
    """Batting order and five bowlers for a team in one recorded match."""
    xi = lineup_team["xi"]
    order = [p for p in lineup_team["bat_order"] if p in xi] + [p for p in xi if p not in lineup_team["bat_order"]]
    bowlers = sorted(lineup_team["bowl_balls"], key=lambda p: -lineup_team["bowl_balls"][p])[:5]
    return order[:11], bowlers


def current_lineups(matches: pd.DataFrame, lineups: dict, season: int, teams: list,
                    additions: dict | None = None, rates: Rates | None = None) -> dict:
    """Most recent recorded XI per team in `season`, with five bowlers ranked by balls bowled
    across the season. `additions` maps team -> list of {"id", "bat_pos"} for players who
    joined after the last recorded match; each replaces the non-bowler with the least
    batting data and bats at `bat_pos` (1-based)."""
    season_m = matches[(matches.comp == "NPL") & (matches.season == season)].sort_values("date")
    out = {}
    for team in teams:
        played = [m for m in season_m.itertuples() if team in (m.team1, m.team2) and team in lineups[m.match_id]]
        if not played:
            raise ValueError(f"no recorded {season} match for {team}")
        last = lineups[played[-1].match_id][team]
        order, _ = match_lineup(last)
        tot: dict = {}
        for m in played:
            for p, n in lineups[m.match_id][team]["bowl_balls"].items():
                tot[p] = tot.get(p, 0) + n
        bowlers = [p for p in sorted(tot, key=lambda p: -tot[p]) if p in order][:5]
        for p in sorted(tot, key=lambda p: -tot[p]):
            if len(bowlers) >= 5:
                break
            if p not in bowlers:
                bowlers.append(p)
        for add in (additions or {}).get(team, []):
            if add["id"] in order:
                continue
            cands = [p for p in order if p not in bowlers]
            balls = (lambda p: rates.bat_balls.get(p, 0.0)) if rates else (lambda p: 0.0)
            order.remove(min(cands, key=balls))
            order.insert(add["bat_pos"] - 1, add["id"])
        out[team] = dict(bat=order, bowl=bowlers)
    return out
